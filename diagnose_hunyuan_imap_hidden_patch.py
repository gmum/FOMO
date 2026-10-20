from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers.utils import export_to_video

from receler.training.token_utils import describe_token_window, find_concept_token_indices


DEFAULT_IMAP_DOUBLE_BLOCKS = (1, 2, 4, 5, 6, 9, 10, 13, 15, 16, 17, 18, 19)


@dataclass
class MaskRecord:
    timestep_index: int
    block_index: int
    mask: torch.Tensor


@dataclass
class HunyuanSoftPatchController:
    grid: Tuple[int, int, int]
    source_token_indices: Sequence[int]
    target_token_indices: Sequence[int]
    selected_blocks: frozenset[int]
    source_prompt: str
    target_prompt: str
    patch_alpha: float
    mask_mode: str
    mask_top_fraction: float
    current_timestep_index: int = -1
    records: Dict[str, List[MaskRecord]] = field(
        default_factory=lambda: {
            "baseline_dog_mask_dog": [],
            "baseline_cat_mask_cat": [],
            "patched_mask_cat_prepatch": [],
            "patched_mask_effective": [],
            "patched_mask_cat_after_patch": [],
            "patched_mask_dog_after_patch": [],
        }
    )

    @property
    def num_visual_tokens(self) -> int:
        frames, height, width = self.grid
        return frames * height * width

    def enter_block(self, block_index: int) -> None:
        if block_index == 0:
            self.current_timestep_index += 1

    def add_mask(self, name: str, block_index: int, mask: torch.Tensor) -> None:
        self.records[name].append(
            MaskRecord(
                timestep_index=self.current_timestep_index,
                block_index=block_index,
                mask=mask.detach().to(device="cpu", dtype=torch.float16),
            )
        )


def parse_blocks(value: str) -> Tuple[int, ...]:
    blocks = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not blocks:
        raise argparse.ArgumentTypeError("At least one double-block index is required.")
    invalid = [index for index in blocks if index < 0 or index >= 20]
    if invalid:
        raise argparse.ArgumentTypeError(f"Double-block indices must be in [0, 19], got {invalid}.")
    return blocks


def concept_key(encoder_key: torch.Tensor, sample_index: int, token_indices: Sequence[int]) -> torch.Tensor:
    indices = torch.as_tensor(token_indices, device=encoder_key.device, dtype=torch.long)
    return encoder_key[sample_index].index_select(1, indices).mean(dim=1)


def gramcol_soft_mask(
    visual_hidden_states: torch.Tensor,
    image_query: torch.Tensor,
    text_key: torch.Tensor,
    grid: Tuple[int, int, int],
) -> torch.Tensor:
    """Return an all-head GramCol mask in [0, 1] with shape [F, H, W]."""
    frames, grid_height, grid_width = grid
    num_heads, num_tokens, head_dim = visual_hidden_states.shape
    expected_tokens = frames * grid_height * grid_width
    if num_tokens != expected_tokens:
        raise ValueError(
            f"Visual-token mismatch: attention has {num_tokens}, but grid {grid} requires {expected_tokens}."
        )

    spatial_tokens = grid_height * grid_width
    query_by_frame = image_query.reshape(num_heads, frames, spatial_tokens, head_dim)
    hidden_by_frame = visual_hidden_states.reshape(num_heads, frames, spatial_tokens, head_dim)

    qk_scores = torch.einsum("hfpd,hd->hfp", query_by_frame.float(), text_key.float())
    qk_scores = qk_scores / (float(head_dim) ** 0.5)
    anchor_indices = qk_scores.argmax(dim=-1)

    gather_indices = anchor_indices[..., None, None].expand(-1, -1, 1, head_dim)
    anchors = hidden_by_frame.gather(dim=2, index=gather_indices).squeeze(2)
    gramcol = torch.einsum("hfd,hfpd->hfp", anchors.float(), hidden_by_frame.float())

    minimum = gramcol.amin(dim=-1, keepdim=True)
    maximum = gramcol.amax(dim=-1, keepdim=True)
    normalized = (gramcol - minimum) / (maximum - minimum).clamp_min(1e-6)
    mask = normalized.mean(dim=0)
    return mask.reshape(frames, grid_height, grid_width).clamp_(0.0, 1.0)


def binary_top_fraction_mask(mask: torch.Tensor, top_fraction: float) -> torch.Tensor:
    """Select exactly the strongest spatial fraction independently in every frame."""
    flat = mask.flatten(start_dim=1)
    num_selected = max(1, math.ceil(flat.shape[1] * top_fraction))
    top_indices = flat.topk(num_selected, dim=1, largest=True, sorted=False).indices
    binary = torch.zeros_like(flat)
    binary.scatter_(1, top_indices, 1.0)
    return binary.reshape_as(mask)


class HunyuanIMAPSoftPatchAttnProcessor:
    """Hunyuan double-stream attention with batched dog-to-cat activation patching."""

    def __init__(self, controller: HunyuanSoftPatchController, block_index: int):
        self.controller = controller
        self.block_index = block_index

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        self.controller.enter_block(self.block_index)

        if hidden_states.shape[0] != 3:
            raise ValueError(
                "The IMAP patch processor expects batch [dog baseline, cat baseline, cat patched] (batch size 3)."
            )
        if attn.add_q_proj is None or encoder_hidden_states is None:
            raise ValueError("IMAP patching is supported only in HunyuanVideo double-stream blocks.")

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        image_query = query

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([query, encoder_query], dim=2)
        key = torch.cat([key, encoder_key], dim=2)
        value = torch.cat([value, encoder_value], dim=2)

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        visual_tokens = self.controller.num_visual_tokens
        if attention_output.shape[2] - encoder_hidden_states.shape[1] != visual_tokens:
            raise ValueError(
                "The dynamically derived visual grid does not match the double-block attention output: "
                f"grid={self.controller.grid}, attention={tuple(attention_output.shape)}."
            )

        if self.block_index in self.controller.selected_blocks:
            visual_output = attention_output[:, :, :visual_tokens]
            dog_key = concept_key(encoder_key, 0, self.controller.target_token_indices)
            baseline_cat_key = concept_key(encoder_key, 1, self.controller.source_token_indices)
            patched_cat_key = concept_key(encoder_key, 2, self.controller.source_token_indices)

            dog_mask = gramcol_soft_mask(
                visual_output[0], image_query[0], dog_key, self.controller.grid
            )
            baseline_cat_mask = gramcol_soft_mask(
                visual_output[1], image_query[1], baseline_cat_key, self.controller.grid
            )
            patched_cat_mask_pre = gramcol_soft_mask(
                visual_output[2], image_query[2], patched_cat_key, self.controller.grid
            )

            if self.controller.mask_mode == "binary_topk":
                effective_mask = binary_top_fraction_mask(
                    patched_cat_mask_pre, self.controller.mask_top_fraction
                )
            elif self.controller.mask_mode == "soft_unclamped":
                effective_mask = self.controller.patch_alpha * patched_cat_mask_pre
            else:
                effective_mask = (self.controller.patch_alpha * patched_cat_mask_pre).clamp(0.0, 1.0)
            flat_mask = effective_mask.reshape(1, 1, visual_tokens, 1).to(
                device=visual_output.device, dtype=visual_output.dtype
            )
            patched_visual = (1.0 - flat_mask) * visual_output[2:3] + flat_mask * visual_output[0:1]

            attention_output = attention_output.clone()
            attention_output[2:3, :, :visual_tokens] = patched_visual

            patched_cat_mask_after = gramcol_soft_mask(
                patched_visual[0], image_query[2], patched_cat_key, self.controller.grid
            )
            patched_dog_mask_after = gramcol_soft_mask(
                patched_visual[0], image_query[2], dog_key, self.controller.grid
            )

            self.controller.add_mask("baseline_dog_mask_dog", self.block_index, dog_mask)
            self.controller.add_mask("baseline_cat_mask_cat", self.block_index, baseline_cat_mask)
            self.controller.add_mask("patched_mask_cat_prepatch", self.block_index, patched_cat_mask_pre)
            self.controller.add_mask("patched_mask_effective", self.block_index, effective_mask)
            self.controller.add_mask("patched_mask_cat_after_patch", self.block_index, patched_cat_mask_after)
            self.controller.add_mask("patched_mask_dog_after_patch", self.block_index, patched_dog_mask_after)

        attention_output = attention_output.transpose(1, 2).flatten(2, 3).to(query.dtype)
        attention_output, encoder_attention_output = (
            attention_output[:, : -encoder_hidden_states.shape[1]],
            attention_output[:, -encoder_hidden_states.shape[1] :],
        )

        if getattr(attn, "to_out", None) is not None:
            attention_output = attn.to_out[0](attention_output)
            attention_output = attn.to_out[1](attention_output)
        if getattr(attn, "to_add_out", None) is not None:
            encoder_attention_output = attn.to_add_out(encoder_attention_output)

        return attention_output, encoder_attention_output


def install_patch_processors(transformer, controller: HunyuanSoftPatchController) -> None:
    for block_index, block in enumerate(transformer.transformer_blocks):
        block.attn.set_processor(HunyuanIMAPSoftPatchAttnProcessor(controller, block_index))


def decode_latents_one_by_one(pipe: HunyuanVideoPipeline, latents: torch.Tensor) -> List[List]:
    videos = []
    for sample in latents:
        sample = sample.unsqueeze(0).to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(sample, return_dict=False)[0]
        processed = pipe.video_processor.postprocess_video(decoded, output_type="pil")[0]
        videos.append(processed)
        del decoded
        torch.cuda.empty_cache()
    return videos


def aggregate_records(records: List[MaskRecord]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    if not records:
        raise ValueError("No masks were recorded. Check selected blocks and the denoising run.")
    raw = torch.stack([record.mask.float() for record in records], dim=0)
    aggregate = raw.mean(dim=0)
    metadata = [
        {"timestep_index": record.timestep_index, "block_index": record.block_index} for record in records
    ]
    return raw.numpy().astype(np.float16), aggregate.numpy().astype(np.float32), metadata


def upsample_mask(mask: np.ndarray, num_frames: int, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(mask).float()[None, None]
    resized = F.interpolate(tensor, size=(num_frames, height, width), mode="trilinear", align_corners=False)
    return resized[0, 0].numpy()


def mask_to_heatmap(mask_frames: np.ndarray) -> List[np.ndarray]:
    mask_min = float(mask_frames.min())
    mask_max = float(mask_frames.max())
    scale = max(mask_max - mask_min, 1e-8)
    normalized = (mask_frames - mask_min) / scale
    heatmaps = []
    for frame in normalized:
        gray = np.round(frame * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        heatmaps.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return heatmaps


def save_masks(
    output_dir: Path,
    controller: HunyuanSoftPatchController,
    num_frames: int,
    height: int,
    width: int,
    fps: int,
) -> None:
    for name, records in controller.records.items():
        raw, aggregate, metadata = aggregate_records(records)
        np.save(output_dir / f"{name}_per_step_layer.npy", raw)
        np.save(output_dir / f"{name}.npy", aggregate)
        with (output_dir / f"{name}_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        full_resolution = upsample_mask(aggregate, num_frames, height, width)
        export_to_video(mask_to_heatmap(full_resolution), str(output_dir / f"{name}.mp4"), fps=fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic IMAP-style source-to-target activation patching for HunyuanVideo."
    )
    parser.add_argument("--model_path", default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--source_prompt", default="a video of a cat")
    parser.add_argument("--target_prompt", default="a video of a dog")
    parser.add_argument("--source_concept", default="cat")
    parser.add_argument("--target_concept", default="dog")
    parser.add_argument("--output_dir", default="imap/hunyuan_cat_to_dog_alpha2")
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--patch_alpha", type=float, default=2.0)
    parser.add_argument(
        "--mask_mode",
        choices=("soft", "soft_unclamped", "binary_topk"),
        default="soft",
        help="Use a soft alpha-scaled mask or a binary per-frame top-fraction mask.",
    )
    parser.add_argument(
        "--mask_top_fraction",
        type=float,
        default=0.10,
        help="Fraction of spatial patches selected per frame in binary_topk mode.",
    )
    parser.add_argument(
        "--videos_only",
        action="store_true",
        help="Save only the three generated videos and run_config.json; skip all mask files.",
    )
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument(
        "--double_blocks",
        type=parse_blocks,
        default=DEFAULT_IMAP_DOUBLE_BLOCKS,
        help="Comma-separated Hunyuan double-block indices. Defaults to the IMAP Hunyuan layer set.",
    )
    return parser


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("HunyuanVideo height and width must be divisible by 16.")
    if args.patch_alpha <= 0:
        raise ValueError("patch_alpha must be positive.")
    if not 0.0 < args.mask_top_fraction <= 1.0:
        raise ValueError("mask_top_fraction must be in (0, 1].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16
    device = torch.device("cuda")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.vae.enable_tiling()
    pipe.to(device)

    source_indices = find_concept_token_indices(
        pipe.tokenizer,
        args.source_prompt,
        args.source_concept,
        DEFAULT_PROMPT_TEMPLATE,
        args.max_sequence_length,
    )
    target_indices = find_concept_token_indices(
        pipe.tokenizer,
        args.target_prompt,
        args.target_concept,
        DEFAULT_PROMPT_TEMPLATE,
        args.max_sequence_length,
    )
    print("Source token window:", describe_token_window(
        pipe.tokenizer, args.source_prompt, source_indices, DEFAULT_PROMPT_TEMPLATE, args.max_sequence_length
    ))
    print("Target token window:", describe_token_window(
        pipe.tokenizer, args.target_prompt, target_indices, DEFAULT_PROMPT_TEMPLATE, args.max_sequence_length
    ))

    latent_frames = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latent_height = args.height // pipe.vae_scale_factor_spatial
    latent_width = args.width // pipe.vae_scale_factor_spatial
    grid = (
        latent_frames // transformer.config.patch_size_t,
        latent_height // transformer.config.patch_size,
        latent_width // transformer.config.patch_size,
    )
    controller = HunyuanSoftPatchController(
        grid=grid,
        source_token_indices=source_indices,
        target_token_indices=target_indices,
        selected_blocks=frozenset(args.double_blocks),
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        patch_alpha=args.patch_alpha,
        mask_mode=args.mask_mode,
        mask_top_fraction=args.mask_top_fraction,
    )
    install_patch_processors(transformer, controller)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_latent = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=transformer.config.in_channels,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    latents = initial_latent.repeat(3, 1, 1, 1, 1)
    prompts = [args.target_prompt, args.source_prompt, args.source_prompt]

    print(f"Visual grid: {grid}; tokens={controller.num_visual_tokens}")
    print(f"Patched double blocks: {sorted(controller.selected_blocks)}")
    if controller.mask_mode == "binary_topk":
        print(
            f"Mask mode: binary top {100.0 * controller.mask_top_fraction:.1f}% per frame "
            "(W is exactly 0 or 1)"
        )
    else:
        weight_formula = (
            "W=alpha*M"
            if controller.mask_mode == "soft_unclamped"
            else "W=clamp(alpha*M,0,1)"
        )
        print(f"Mask mode: {controller.mask_mode}; alpha={controller.patch_alpha}; {weight_formula}")
    print(
        f"Batch order: [baseline {args.target_concept}, baseline {args.source_concept}, "
        f"patched {args.source_concept}->{args.target_concept}]"
    )

    output = pipe(
        prompt=prompts,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        latents=latents,
        max_sequence_length=args.max_sequence_length,
        output_type="latent",
    )
    videos_list = decode_latents_one_by_one(pipe, output.frames)
    videos = {
        f"baseline_{args.target_concept}": videos_list[0],
        f"baseline_{args.source_concept}": videos_list[1],
        f"patched_{args.source_concept}_to_{args.target_concept}": videos_list[2],
    }
    for name, frames in videos.items():
        export_to_video(frames, str(output_dir / f"{name}.mp4"), fps=args.fps)

    if not args.videos_only:
        save_masks(
            output_dir=output_dir,
            controller=controller,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
        )

    run_config = vars(args).copy()
    if args.mask_mode == "binary_topk":
        run_config["patch_alpha"] = None
    run_config["double_blocks"] = list(args.double_blocks)
    run_config["visual_grid"] = list(grid)
    run_config["source_token_indices"] = list(source_indices)
    run_config["target_token_indices"] = list(target_indices)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    saved_kind = "three generated videos" if args.videos_only else "videos and masks"
    print(f"Saved {saved_kind} to {output_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
