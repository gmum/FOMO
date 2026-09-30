import argparse
import gc
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils import export_to_video


LORA_WEIGHT_NAMES = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")


@dataclass
class VBankState:
    inject_steps: int = 5
    block_min: int = 20
    mode: str = "off"
    current_step: int = 0
    bank: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def begin(self, mode: str) -> None:
        if mode not in {"off", "record", "inject"}:
            raise ValueError(f"Unknown V-bank mode: {mode}")
        self.mode = mode
        self.current_step = 0

    def finish(self) -> None:
        self.mode = "off"
        self.current_step = 0

    def active_for(self, block_index: int) -> bool:
        return (
            self.mode in {"record", "inject"}
            and self.current_step < self.inject_steps
            and block_index > self.block_min
        )

    def key(self, block_index: int) -> Tuple[int, int]:
        return (self.current_step, block_index)


class HunyuanVideoVBankAttnProcessor2_0:
    """Hunyuan attention processor with RF-Edit-style V record/inject for single blocks."""

    def __init__(self, state: VBankState, block_index: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("This processor requires torch.nn.functional.scaled_dot_product_attention.")
        self.state = state
        self.block_index = block_index

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

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
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                text_seq_length = encoder_hidden_states.shape[1]
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, :-text_seq_length], image_rotary_emb),
                        query[:, :, -text_seq_length:],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, :-text_seq_length], image_rotary_emb),
                        key[:, :, -text_seq_length:],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        if attn.add_q_proj is not None and encoder_hidden_states is not None:
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

        if self.state.active_for(self.block_index):
            bank_key = self.state.key(self.block_index)
            if self.state.mode == "record":
                self.state.bank[bank_key] = value.detach().to("cpu", copy=True)
            elif self.state.mode == "inject":
                if bank_key not in self.state.bank:
                    raise KeyError(f"Missing V-bank tensor for step/block {bank_key}")
                bank_value = self.state.bank[bank_key].to(device=value.device, dtype=value.dtype)
                if bank_value.shape != value.shape:
                    raise ValueError(
                        f"V-bank shape mismatch for step/block {bank_key}: "
                        f"bank={tuple(bank_value.shape)} current={tuple(value.shape)}"
                    )
                value = bank_value

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


def parse_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def has_lora_weights(path: Path) -> bool:
    return any((path / weight_name).is_file() for weight_name in LORA_WEIGHT_NAMES)


def checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def resolve_lora_path(path: str) -> str:
    lora_path = Path(path)
    if has_lora_weights(lora_path):
        return str(lora_path)

    checkpoints = [candidate for candidate in lora_path.glob("checkpoint-*") if candidate.is_dir() and has_lora_weights(candidate)]
    if checkpoints:
        resolved = sorted(checkpoints, key=checkpoint_sort_key)[-1]
        print(f"Resolved LoRA path to latest checkpoint: {resolved}")
        return str(resolved)

    expected = ", ".join(LORA_WEIGHT_NAMES)
    raise FileNotFoundError(f"LORA_PATH does not contain {expected}, and no checkpoint-* child does: {path}")


def strip_mp4_suffix(output_path: str) -> str:
    path = Path(output_path)
    if path.suffix.lower() == ".mp4":
        return str(path.with_suffix(""))
    return output_path


def install_vbank_processors(pipe: HunyuanVideoPipeline, state: VBankState) -> int:
    blocks = getattr(pipe.transformer, "single_transformer_blocks", None)
    if blocks is None:
        raise AttributeError("The Hunyuan transformer has no single_transformer_blocks.")

    for block_index, block in enumerate(blocks):
        block.attn.set_processor(HunyuanVideoVBankAttnProcessor2_0(state, block_index))
    return len(blocks)


def set_lora_enabled(pipe: HunyuanVideoPipeline, enabled: bool, lora_weight: float) -> None:
    if enabled:
        if hasattr(pipe, "enable_lora"):
            pipe.enable_lora()
        pipe.set_adapters(["erase_lora"], adapter_weights=[lora_weight])
    else:
        if hasattr(pipe, "disable_lora"):
            pipe.disable_lora()
        else:
            pipe.set_adapters(["erase_lora"], adapter_weights=[0.0])


def make_initial_latents(
    pipe: HunyuanVideoPipeline,
    *,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
) -> torch.Tensor:
    device = pipe._execution_device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.config.in_channels,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=None,
    )


def run_pipe_pass(
    pipe: HunyuanVideoPipeline,
    state: VBankState,
    *,
    mode: str,
    prompt: str,
    latents: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> list:
    state.begin(mode)

    def mark_next_step(_pipe, step, _timestep, callback_kwargs):
        state.current_step = step + 1
        return callback_kwargs

    callback = mark_next_step if mode in {"record", "inject"} else None
    try:
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            latents=latents.clone(),
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames[0]
    finally:
        state.finish()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return video


def generate_vbank_comparison(args) -> None:
    dtype = parse_dtype(args.dtype)
    output_prefix = strip_mp4_suffix(args.output_path)
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    resolved_lora_path = resolve_lora_path(args.lora_path)

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=torch.float16,
    )
    pipe.vae.enable_tiling()

    pipe.load_lora_weights(resolved_lora_path, adapter_name="erase_lora")
    pipe.set_adapters(["erase_lora"], adapter_weights=[args.lora_weight])
    print(f"Loaded LoRA: {resolved_lora_path}")
    print(f"LoRA adapter weight: {args.lora_weight}")

    pipe.to("cuda")

    state = VBankState(inject_steps=args.inject_steps, block_min=args.block_min)
    num_single_blocks = install_vbank_processors(pipe, state)
    target_blocks = [idx for idx in range(num_single_blocks) if idx > args.block_min]
    print(f"Installed V-bank processors on {num_single_blocks} single blocks.")
    print(f"Recording/injecting blocks: {target_blocks}")
    print(f"Recording/injecting first {args.inject_steps} denoising steps.")

    initial_latents = make_initial_latents(
        pipe,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
    )
    print(f"Initial z_T latents: shape={tuple(initial_latents.shape)}, dtype={initial_latents.dtype}")

    set_lora_enabled(pipe, False, args.lora_weight)
    print("Pass 1/3: base model, recording V-bank.")
    base_video = run_pipe_pass(
        pipe,
        state,
        mode="record",
        prompt=args.prompt,
        latents=initial_latents,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    export_to_video(base_video, output_prefix + "_base.mp4", fps=args.fps)
    del base_video
    gc.collect()
    print(f"Saved: {output_prefix}_base.mp4")
    print(f"Recorded V-bank tensors: {len(state.bank)}")

    set_lora_enabled(pipe, True, args.lora_weight)
    print("Pass 2/3: LoRA model without V injection.")
    lora_video = run_pipe_pass(
        pipe,
        state,
        mode="off",
        prompt=args.prompt,
        latents=initial_latents,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    export_to_video(lora_video, output_prefix + "_lora.mp4", fps=args.fps)
    del lora_video
    gc.collect()
    print(f"Saved: {output_prefix}_lora.mp4")

    injected_outputs = {}
    sweep_steps = range(1, args.inject_steps + 1) if args.sweep_inject_steps else (args.inject_steps,)
    for pass_index, inject_steps in enumerate(sweep_steps, start=3):
        state.inject_steps = inject_steps
        set_lora_enabled(pipe, True, args.lora_weight)
        if args.sweep_inject_steps:
            suffix = f"_lora_vinject_{inject_steps:02d}steps"
            print(
                f"Pass {pass_index}/{args.inject_steps + 2}: "
                f"LoRA model with base V injection for first {inject_steps} steps."
            )
        else:
            suffix = "_lora_vinject"
            print("Pass 3/3: LoRA model with base V injection.")

        injected_video = run_pipe_pass(
            pipe,
            state,
            mode="inject",
            prompt=args.prompt,
            latents=initial_latents,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        injected_output = output_prefix + suffix + ".mp4"
        export_to_video(injected_video, injected_output, fps=args.fps)
        del injected_video
        gc.collect()
        injected_outputs[str(inject_steps)] = injected_output
        print(f"Saved: {injected_output}")

    manifest = {
        "prompt": args.prompt,
        "model_path": args.model_path,
        "requested_lora_path": args.lora_path,
        "resolved_lora_path": resolved_lora_path,
        "lora_weight": args.lora_weight,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "inject_steps": args.inject_steps,
        "sweep_inject_steps": args.sweep_inject_steps,
        "block_min": args.block_min,
        "num_single_blocks": num_single_blocks,
        "target_blocks": target_blocks,
        "num_vbank_tensors": len(state.bank),
        "example_vbank_shapes": {
            f"step{step}_block{block}": list(tensor.shape)
            for (step, block), tensor in list(sorted(state.bank.items()))[:5]
        },
        "outputs": {
            "base": output_prefix + "_base.mp4",
            "lora": output_prefix + "_lora.mp4",
            "lora_vinject": injected_outputs,
        },
    }
    manifest_path = output_prefix + "_vbank_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Hunyuan base, LoRA, and LoRA with RF-Edit-style V injection.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--lora_weight", type=float, default=1.0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=75)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject_steps", type=int, default=5)
    parser.add_argument(
        "--sweep_inject_steps",
        action="store_true",
        help="Save one injected video for each prefix length from 1 to --inject_steps.",
    )
    parser.add_argument("--block_min", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    generate_vbank_comparison(args)


if __name__ == "__main__":
    main()
