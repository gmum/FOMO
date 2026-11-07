#!/usr/bin/env python3

import argparse
import copy
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.utils import export_to_video
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

from fomo.localization import gramcol_soft_mask

try:
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover - fallback only
    safe_save_file = None

from fomo.cogvideox_tokens import (
    find_concept_token_indices,
    frames_from_latents_shape,
    layer_group_indices,
)


DEFAULT_PROMPT_PAIRS = (
    ("A realistic video of a dog walking on grass.", "A realistic video of a cat walking on grass."),
)

# Zero-based IMAP selections (IMAP/scripts/MeViS/cog2b.sh and cog5b.sh).
DEFAULT_IMAP_H_LAYERS_2B = (
    4, 6, 7, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
)
DEFAULT_IMAP_H_LAYERS = (
    1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 23,
)


@dataclass
class PromptPair:
    source_prompt: str
    target_prompt: str
    source_concept: str
    target_concept: str
    source_token_indices: Sequence[int] = field(default_factory=tuple)
    source_embeds: Optional[torch.Tensor] = None
    target_embeds: Optional[torch.Tensor] = None
    positive_batch_index: int = 1


class CogVideoXHVLoraController:
    def __init__(
        self,
        h_layers: Sequence[int],
        v_layers: Sequence[int],
        source_token_indices: Sequence[int],
        token_grid: Tuple[int, int, int],
        alpha: float,
        clamp_mask: bool,
        lamb_v: float,
        v_mask_mode: str,
    ) -> None:
        self.h_layers = frozenset(int(x) for x in h_layers)
        self.v_layers = frozenset(int(x) for x in v_layers)
        self.source_token_indices = tuple(int(x) for x in source_token_indices)
        self.token_grid = tuple(int(x) for x in token_grid)
        self.alpha = float(alpha)
        self.clamp_mask = bool(clamp_mask)
        self.lamb_v = float(lamb_v)
        self.v_mask_mode = str(v_mask_mode)
        self.mode = "off"
        self.positive_batch_index = -1
        self.capture_h = False
        self.capture_v = False
        self.source_h: Dict[int, torch.Tensor] = {}
        self.source_weights: Dict[int, torch.Tensor] = {}
        self.source_masks: Dict[int, torch.Tensor] = {}
        self.target_h: Dict[int, torch.Tensor] = {}
        self.source_v: Dict[int, torch.Tensor] = {}
        self.h_losses: Dict[int, torch.Tensor] = {}
        self.v_losses: Dict[int, torch.Tensor] = {}
        self.mask_means: Dict[int, float] = {}
        self.weight_means: Dict[int, float] = {}
        self.weight_maxima: Dict[int, float] = {}
        self.v_mask_means: Dict[int, float] = {}

    def reset_all(self) -> None:
        self.source_h.clear()
        self.source_weights.clear()
        self.source_masks.clear()
        self.target_h.clear()
        self.source_v.clear()
        self.h_losses.clear()
        self.v_losses.clear()
        self.mask_means.clear()
        self.weight_means.clear()
        self.weight_maxima.clear()
        self.v_mask_means.clear()

    def begin_teacher_source(self, token_indices: Sequence[int], capture_v: bool) -> None:
        self.reset_all()
        self.mode = "teacher_source"
        self.capture_h = True
        self.capture_v = bool(capture_v)
        self.source_token_indices = tuple(int(x) for x in token_indices)

    def begin_teacher_target(self) -> None:
        self.mode = "teacher_target"
        self.capture_h = True
        self.capture_v = False

    def begin_student(self, capture_h: bool = True, capture_v: bool = True) -> None:
        self.mode = "student"
        self.capture_h = bool(capture_h)
        self.capture_v = bool(capture_v)
        self.h_losses.clear()
        self.v_losses.clear()

    def finish(self) -> None:
        self.mode = "off"
        self.capture_h = False
        self.capture_v = False

    def uses_h(self, layer_idx: int) -> bool:
        return int(layer_idx) in self.h_layers

    def uses_v(self, layer_idx: int) -> bool:
        return int(layer_idx) in self.v_layers and self.lamb_v > 0.0

    def mean_source_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.source_masks:
            raise RuntimeError("No source GramCol masks captured; cannot build V mask.")
        mask = torch.stack(list(self.source_masks.values())).float().mean(dim=0).clamp(0.0, 1.0)
        return mask.reshape(-1).to(device=device, dtype=dtype)

    def total_h_loss(self) -> Optional[torch.Tensor]:
        if not self.h_losses:
            return None
        return torch.stack([self.h_losses[k] for k in sorted(self.h_losses)]).mean()

    def total_v_loss(self) -> Optional[torch.Tensor]:
        if not self.v_losses:
            return None
        return torch.stack([self.v_losses[k] for k in sorted(self.v_losses)]).mean()


class CogVideoXHVLoraProcessor(CogVideoXAttnProcessor2_0):
    def __init__(self, layer_idx: int, controller: CogVideoXHVLoraController):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.controller = controller

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        combined_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = combined_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(combined_states)
        key = attn.to_k(combined_states)
        value = attn.to_v(combined_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        pos_idx = self.controller.positive_batch_index
        if pos_idx < 0:
            pos_idx = batch_size + pos_idx
        pos_idx = max(0, min(batch_size - 1, pos_idx))

        if self.controller.mode == "teacher_source" and self.controller.capture_v and self.controller.uses_v(self.layer_idx):
            self.controller.source_v[self.layer_idx] = value[
                pos_idx : pos_idx + 1, :, text_seq_length:, :
            ].detach().to("cpu")

        elif self.controller.mode == "student" and self.controller.capture_v and self.controller.uses_v(self.layer_idx):
            target_v = self.controller.source_v.get(self.layer_idx)
            student_v = value[pos_idx : pos_idx + 1, :, text_seq_length:, :]
            if target_v is not None and target_v.shape == student_v.shape:
                diff = student_v.float() - target_v.to(device=student_v.device, dtype=student_v.dtype).float()
                if self.controller.v_mask_mode == "full":
                    weighted = diff
                    mask_mean = 1.0
                else:
                    mask = self.controller.mean_source_mask(diff.device, diff.dtype)
                    if self.controller.v_mask_mode == "background":
                        mask = 1.0 - mask
                    elif self.controller.v_mask_mode == "target":
                        pass
                    else:
                        raise ValueError(f"Unknown v_mask_mode={self.controller.v_mask_mode!r}")
                    weighted = diff * mask.reshape(1, 1, -1, 1)
                    mask_mean = float(mask.detach().mean().cpu())
                self.controller.v_losses[self.layer_idx] = weighted.pow(2).mean()
                self.controller.v_mask_means[self.layer_idx] = mask_mean

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # Preserve [batch, heads, video_tokens, head_dim] for GramCol and L_H,
        # exactly as in the Hunyuan trainer, before the output projection.
        visual_h = attn_output[pos_idx : pos_idx + 1, :, text_seq_length:, :]

        if self.controller.mode == "teacher_source" and self.controller.capture_h and self.controller.uses_h(self.layer_idx):
            mask = self._compute_source_mask(visual_h[0], query, key, text_seq_length, pos_idx)
            weight = self.controller.alpha * mask
            if self.controller.clamp_mask:
                weight = weight.clamp(0.0, 1.0)
            self.controller.source_h[self.layer_idx] = visual_h.detach().to("cpu")
            self.controller.source_weights[self.layer_idx] = weight.detach().to("cpu")
            self.controller.source_masks[self.layer_idx] = mask.detach().to("cpu")
            self.controller.mask_means[self.layer_idx] = float(mask.detach().mean().cpu())
            self.controller.weight_means[self.layer_idx] = float(weight.detach().mean().cpu())
            self.controller.weight_maxima[self.layer_idx] = float(weight.detach().max().cpu())

        elif self.controller.mode == "teacher_target" and self.controller.capture_h and self.controller.uses_h(self.layer_idx):
            self.controller.target_h[self.layer_idx] = visual_h.detach().to("cpu")

        elif self.controller.mode == "student" and self.controller.capture_h and self.controller.uses_h(self.layer_idx):
            source_h = self.controller.source_h.get(self.layer_idx)
            target_h = self.controller.target_h.get(self.layer_idx)
            weight = self.controller.source_weights.get(self.layer_idx)
            current_h = visual_h
            if source_h is not None and target_h is not None and weight is not None:
                if source_h.shape == current_h.shape and target_h.shape == current_h.shape:
                    weight_seq = weight.reshape(1, 1, -1, 1).to(device=current_h.device, dtype=current_h.dtype)
                    h_star = source_h.to(current_h.device, current_h.dtype) + weight_seq * (
                        target_h.to(current_h.device, current_h.dtype) - source_h.to(current_h.device, current_h.dtype)
                    )
                    self.controller.h_losses[self.layer_idx] = F.mse_loss(current_h.float(), h_star.float())

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)
        encoder_output, video_output = attn_output.split(
            [text_seq_length, attn_output.size(1) - text_seq_length], dim=1
        )
        return video_output, encoder_output

    def _compute_source_mask(
        self,
        visual_h: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        text_seq_length: int,
        pos_idx: int,
    ) -> torch.Tensor:
        token_idx = torch.tensor(self.controller.source_token_indices, device=key.device, dtype=torch.long)
        if token_idx.numel() == 0 or ((token_idx < 0) | (token_idx >= text_seq_length)).any():
            raise RuntimeError("No valid concept token indices for mask computation.")
        # Average multi-token concept keys exactly as in Hunyuan's _concept_key.
        concept_key = key[pos_idx, :, :text_seq_length, :].index_select(1, token_idx).mean(dim=1)
        return gramcol_soft_mask(
            visual_h,
            query[pos_idx, :, text_seq_length:, :],
            concept_key,
            self.controller.token_grid,
        )


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def resolve_h_layers(num_layers: int, groups: Sequence[str]) -> List[int]:
    layers = set()
    for group in groups:
        if group.lower() == "imap":
            if num_layers == 30:
                layers.update(DEFAULT_IMAP_H_LAYERS_2B)
            elif num_layers == 42:
                layers.update(DEFAULT_IMAP_H_LAYERS)
            else:
                raise ValueError(
                    f"IMAP preset expects CogVideoX-2B (30 blocks) or 5B (42 blocks), got {num_layers}. "
                    "Select early,middle,late,all explicitly for other architectures."
                )
        else:
            layers.update(layer_group_indices(num_layers, group))
    if not layers or any(idx < 0 or idx >= num_layers for idx in layers):
        raise ValueError(f"Invalid H layers for a {num_layers}-block model: {sorted(layers)}")
    return sorted(layers)


def target_modules_from_preset(preset: str) -> List[str]:
    presets = {
        "text_qk": ["to_q", "to_k"],
        "text_qkv": ["to_q", "to_k", "to_v"],
        "video_qk": ["to_q", "to_k"],
        "video_qkv": ["to_q", "to_k", "to_v"],
    }
    if preset in presets:
        return presets[preset]
    return parse_csv_strings(preset)


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def find_matching_linear_suffixes(model: torch.nn.Module, target_modules: Sequence[str]) -> List[str]:
    suffixes = tuple(target_modules)
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any(name.endswith(suffix) for suffix in suffixes)
    ]


def iter_trainable_parameters(model: torch.nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_summary(model: torch.nn.Module) -> dict:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"trainable": trainable, "total": total, "percent": 100.0 * trainable / total if total else 0.0}


def add_lora(transformer: torch.nn.Module, target_modules: Sequence[str], rank: int, alpha: int, adapter_name: str):
    matches = find_matching_linear_suffixes(transformer, target_modules)
    if not matches:
        raise ValueError(f"No CogVideoX LoRA target modules found for suffixes: {list(target_modules)}")
    freeze_model(transformer)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
    )
    transformer.add_adapter(config, adapter_name=adapter_name)
    return matches


@contextmanager
def adapter_mode(transformer, enabled: bool):
    if enabled:
        transformer.enable_adapters()
    else:
        transformer.disable_adapters()
    try:
        yield
    finally:
        transformer.enable_adapters()


def install_processors(pipe: CogVideoXPipeline, controller: CogVideoXHVLoraController, layers: Sequence[int]) -> Dict[int, object]:
    originals = {}
    for idx in sorted(set(int(x) for x in layers)):
        block = pipe.transformer.transformer_blocks[idx]
        originals[idx] = block.attn1.processor
        block.attn1.set_processor(CogVideoXHVLoraProcessor(idx, controller))
    return originals


def restore_processors(pipe: CogVideoXPipeline, originals: Dict[int, object]) -> None:
    for idx, processor in originals.items():
        pipe.transformer.transformer_blocks[idx].attn1.set_processor(processor)


def read_prompt_pairs(path: Optional[str], source_concept: str, target_concept: str) -> List[PromptPair]:
    if path is None:
        return [
            PromptPair(source, target, source_concept, target_concept)
            for source, target in DEFAULT_PROMPT_PAIRS
        ]
    pairs: List[PromptPair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            pairs.append(
                PromptPair(
                    source_prompt=str(item["source_prompt"]),
                    target_prompt=str(item["target_prompt"]),
                    source_concept=str(item.get("source_concept") or source_concept),
                    target_concept=str(item.get("target_concept") or target_concept),
                )
            )
    if not pairs:
        raise ValueError("Prompt pair JSONL is empty.")
    return pairs


def encode_prompts(pipe: CogVideoXPipeline, prompt: str, guidance_scale: float, max_sequence_length: int, device):
    do_cfg = guidance_scale > 1.0
    embeds, neg = pipe.encode_prompt(
        prompt,
        None,
        do_cfg,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_cfg:
        embeds = torch.cat([neg, embeds], dim=0)
        positive_batch_index = 1
    else:
        positive_batch_index = 0
    return embeds, positive_batch_index


def transformer_forward(
    pipe: CogVideoXPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    image_rotary_emb,
    guidance_scale: float,
) -> torch.Tensor:
    do_cfg = guidance_scale > 1.0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    t = timestep.expand(latent_model_input.shape[0])
    return pipe.transformer(
        hidden_states=latent_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=t,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
    )[0]


def cfg_noise_pred(noise_pred: torch.Tensor, guidance_scale: float, do_cfg: bool) -> torch.Tensor:
    noise_pred = noise_pred.float()
    if not do_cfg:
        return noise_pred
    uncond, text = noise_pred.chunk(2)
    return uncond + guidance_scale * (text - uncond)


def make_initial_latents(pipe: CogVideoXPipeline, args, dtype, device, generator):
    latent_frames = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    num_frames = int(args.num_frames)
    patch_size_t = pipe.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = int(patch_size_t) - latent_frames % int(patch_size_t)
        num_frames += additional_frames * pipe.vae_scale_factor_temporal
    latents = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels,
        num_frames,
        args.height,
        args.width,
        dtype,
        device,
        generator,
        None,
    )
    return latents, additional_frames


@torch.no_grad()
def rollout_to_index(
    pipe: CogVideoXPipeline,
    source_embeds: torch.Tensor,
    initial_latents: torch.Tensor,
    timesteps: Sequence[torch.Tensor],
    target_index: int,
    image_rotary_emb,
    guidance_scale: float,
    generator,
) -> torch.Tensor:
    latents = initial_latents
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    old_pred = None
    do_cfg = guidance_scale > 1.0
    for i, timestep in enumerate(timesteps[:target_index]):
        noise_pred = transformer_forward(pipe, latents, timestep, source_embeds, image_rotary_emb, guidance_scale)
        noise_pred = cfg_noise_pred(noise_pred, guidance_scale, do_cfg)
        if isinstance(pipe.scheduler, CogVideoXDPMScheduler):
            latents, old_pred = pipe.scheduler.step(
                noise_pred,
                old_pred,
                timestep,
                timesteps[i - 1] if i > 0 else None,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )
        else:
            latents = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
        latents = latents.to(source_embeds.dtype)
    return latents.detach()


def save_checkpoint(pipe: CogVideoXPipeline, output_dir: Path, adapter_name: str, config: dict, matches: Sequence[str], step: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(pipe.transformer, adapter_name=adapter_name)
    if hasattr(CogVideoXPipeline, "save_lora_weights"):
        CogVideoXPipeline.save_lora_weights(str(output_dir), transformer_lora_layers=state)
    else:
        if safe_save_file is not None:
            safe_save_file(state, str(output_dir / "pytorch_lora_weights.safetensors"))
        else:
            torch.save(state, output_dir / "pytorch_lora_weights.bin")
    metadata = dict(config)
    metadata["global_step"] = int(step)
    metadata["matched_lora_modules"] = list(matches)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


@torch.no_grad()
def save_training_videos(pipe, controller, pair: PromptPair, args, output_dir: Path, step: int) -> List[Path]:
    """Render base source/safe at step 0, then source with LoRA at checkpoints.

    Use cached embeddings and a separate scheduler so the text encoder can stay
    on CPU and preview sampling cannot change the training scheduler or RNG.
    """
    controller.finish()
    controller.reset_all()
    transformer = pipe.transformer
    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype
    vae_device = next(pipe.vae.parameters()).device
    was_training = transformer.training
    seed = args.seed if args.eval_seed is None else args.eval_seed
    do_cfg = args.guidance_scale > 1.0
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    items = (
        [("source_base", pair.source_prompt, pair.source_embeds),
         ("safe_base", pair.target_prompt, pair.target_embeds)]
        if step == 0 else
        [(f"step_{step:06d}_source_lora", pair.source_prompt, pair.source_embeds)]
    )
    written = []
    cuda_devices = [device.index] if device.type == "cuda" else []
    try:
        transformer.eval()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.vae.to(device)
        # Both registered models now use the same execution device. Excluding
        # the CPU text encoder also prevents Diffusers inferring a CPU device.
        preview = CogVideoXPipeline(
            tokenizer=None, text_encoder=None, transformer=transformer,
            vae=pipe.vae,
            scheduler=CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing"),
        )
        preview.set_progress_bar_config(disable=True)
        with torch.random.fork_rng(devices=cuda_devices), adapter_mode(transformer, enabled=step > 0):
            for name, prompt, cached in items:
                if cached is None:
                    raise RuntimeError(f"Missing cached embeddings for preview {name}.")
                embeds = cached.to(device=device, dtype=dtype)
                positive = embeds[pair.positive_batch_index : pair.positive_batch_index + 1]
                negative = embeds[:1] if do_cfg else None
                path = videos_dir / f"{name}.mp4"
                print(f"[video] Generating {path.name}: step={step}, seed={seed}, prompt={prompt}", flush=True)
                frames = preview(
                    prompt_embeds=positive, negative_prompt_embeds=negative,
                    height=args.height, width=args.width, num_frames=args.num_frames,
                    num_inference_steps=args.eval_inference_steps,
                    guidance_scale=args.guidance_scale, use_dynamic_cfg=False,
                    max_sequence_length=args.max_sequence_length,
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).frames[0]
                actual_size = frames[0].size if len(frames) else None
                if len(frames) != args.num_frames or actual_size != (args.width, args.height):
                    raise RuntimeError(
                        f"Preview shape mismatch: expected {args.num_frames} frames at "
                        f"{args.width}x{args.height}, got {len(frames)} frames at {actual_size}."
                    )
                export_to_video(frames, str(path), fps=args.eval_fps)
                with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
                    json.dump({
                        "step": step, "prompt": prompt, "seed": seed,
                        "lora_enabled": step > 0, "model_path": args.model_path,
                        "width": args.width, "height": args.height,
                        "num_frames": args.num_frames, "fps": args.eval_fps,
                        "num_inference_steps": args.eval_inference_steps,
                        "guidance_scale": args.guidance_scale, "use_dynamic_cfg": False,
                        "mask_source_concept": pair.source_concept,
                    }, handle, indent=2)
                written.append(path)
                print(f"[video] Saved {path}", flush=True)
                del frames, embeds, positive, negative
    finally:
        pipe.vae.to(vae_device)
        transformer.train(was_training)
        controller.finish()
        torch.cuda.empty_cache()
    return written


def sample_target_index(args) -> int:
    if args.lamb_v > 0.0 and args.v_steps > 0 and args.v_early_prob > 0.0:
        early_upper = min(args.v_steps, args.max_target_step + 1)
        if early_upper > args.min_target_step and random.random() < args.v_early_prob:
            return random.randint(args.min_target_step, early_upper - 1)
    return random.randint(args.min_target_step, args.max_target_step)


def build_parser():
    parser = argparse.ArgumentParser(description="Train CogVideoX LoRA with concept-local H target and optional V preservation.")
    parser.add_argument("--model_path", default="THUDM/CogVideoX-5b")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_pairs_jsonl", default=None)
    parser.add_argument("--source_concept", default="dog")
    parser.add_argument("--target_concept", default="cat")
    parser.add_argument(
        "--h_layer_groups", default="imap",
        help="imap: automatically use the 18 CogVideoX-2B or 22 CogVideoX-5B IMAP layers; alternatively early,middle,late,all.",
    )
    parser.add_argument("--v_layer_groups", default="early")
    parser.add_argument("--lora_target_preset", default="text_qk")
    parser.add_argument("--target_modules", default="")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--checkpointing_steps", type=int, default=10)
    parser.add_argument("--save_training_videos", action="store_true",
                        help="Save base source/safe videos before training and a source-with-LoRA video at every checkpoint (first prompt pair).")
    parser.add_argument("--eval_inference_steps", type=int, default=50)
    parser.add_argument("--eval_seed", type=int, default=None, help="Fixed preview seed; defaults to --seed.")
    parser.add_argument("--eval_fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--trajectory_steps", type=int, default=50)
    parser.add_argument("--min_target_step", type=int, default=0)
    parser.add_argument("--max_target_step", type=int, default=49)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--max_sequence_length", type=int, default=226)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--imap_alpha", type=float, default=5.0)
    parser.add_argument("--clamp_mask", action="store_true")
    parser.add_argument("--lamb_v", type=float, default=0.5)
    parser.add_argument("--v_steps", type=int, default=2)
    parser.add_argument("--v_early_prob", type=float, default=0.5)
    parser.add_argument("--v_mask_mode", choices=("full", "background", "target"), default="full")
    parser.add_argument("--split_hv_backward", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser


def enable_transformer_gradient_checkpointing(transformer: torch.nn.Module) -> None:
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        return
    if hasattr(transformer, "gradient_checkpointing_enable"):
        transformer.gradient_checkpointing_enable()
        return

    changed = False
    if hasattr(transformer, "gradient_checkpointing"):
        setattr(transformer, "gradient_checkpointing", True)
        changed = True
    for module in transformer.modules():
        if hasattr(module, "gradient_checkpointing"):
            setattr(module, "gradient_checkpointing", True)
            changed = True
    if not changed:
        raise RuntimeError("This CogVideoX transformer does not expose a gradient-checkpointing switch.")


def main(args):
    if args.eval_inference_steps <= 0 or args.eval_fps <= 0:
        raise ValueError("eval_inference_steps and eval_fps must be positive.")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16.")
    if args.trajectory_steps <= 0:
        raise ValueError("trajectory_steps must be positive.")
    if not 0 <= args.min_target_step <= args.max_target_step < args.trajectory_steps:
        raise ValueError("Require 0 <= min_target_step <= max_target_step < trajectory_steps.")
    if not 0.0 <= args.v_early_prob <= 1.0:
        raise ValueError("v_early_prob must be in [0,1].")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.transformer.train()
    for component in (pipe.vae, pipe.text_encoder):
        for parameter in component.parameters():
            parameter.requires_grad_(False)

    num_layers = len(pipe.transformer.transformer_blocks)
    h_groups = parse_csv_strings(args.h_layer_groups)
    v_groups = parse_csv_strings(args.v_layer_groups) if args.lamb_v > 0 else []
    h_layers = resolve_h_layers(num_layers, h_groups)
    v_layers = sorted({idx for group in v_groups for idx in layer_group_indices(num_layers, group)})
    install_layers = sorted(set(h_layers + v_layers))
    if not install_layers:
        raise ValueError("No layers selected.")

    raw_pairs = read_prompt_pairs(args.prompt_pairs_jsonl, args.source_concept, args.target_concept)
    for pair in raw_pairs:
        pair.source_token_indices = find_concept_token_indices(
            pipe.tokenizer,
            pair.source_prompt,
            pair.source_concept,
            args.max_sequence_length,
        )
        source_embeds, positive_index = encode_prompts(
            pipe, pair.source_prompt, args.guidance_scale, args.max_sequence_length, device
        )
        target_embeds, _ = encode_prompts(
            pipe, pair.target_prompt, args.guidance_scale, args.max_sequence_length, device
        )
        pair.source_embeds = source_embeds.detach().cpu()
        pair.target_embeds = target_embeds.detach().cpu()
        pair.positive_batch_index = int(positive_index)

    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    target_modules = parse_csv_strings(args.target_modules) if args.target_modules else target_modules_from_preset(args.lora_target_preset)
    adapter_name = "cogvideox_hv_lora"
    matches = add_lora(pipe.transformer, target_modules, args.rank, args.lora_alpha, adapter_name)
    pipe.transformer.to(device=device, dtype=dtype)
    if args.gradient_checkpointing:
        enable_transformer_gradient_checkpointing(pipe.transformer)
    summary = trainable_parameter_summary(pipe.transformer)
    if summary["trainable"] <= 0:
        raise RuntimeError("No trainable LoRA parameters.")
    optimizer = torch.optim.AdamW(
        iter_trainable_parameters(pipe.transformer),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    generator_for_shape = torch.Generator(device=device).manual_seed(args.seed)
    shape_latents, _ = make_initial_latents(pipe, args, dtype, device, generator_for_shape)
    token_grid = frames_from_latents_shape(pipe, shape_latents)
    del shape_latents

    controller = CogVideoXHVLoraController(
        h_layers=h_layers,
        v_layers=v_layers,
        source_token_indices=raw_pairs[0].source_token_indices,
        token_grid=token_grid,
        alpha=args.imap_alpha,
        clamp_mask=args.clamp_mask,
        lamb_v=args.lamb_v,
        v_mask_mode=args.v_mask_mode,
    )
    originals = install_processors(pipe, controller, install_layers)

    config = vars(args).copy()
    config.update(
        {
            "num_transformer_blocks": num_layers,
            "h_layers": h_layers,
            "v_layers": v_layers,
            "token_grid": list(token_grid),
            "target_modules_resolved": target_modules,
            "matched_lora_modules": matches,
            "prompt_pairs": [
                {
                    "source_prompt": pair.source_prompt,
                    "target_prompt": pair.target_prompt,
                    "source_concept": pair.source_concept,
                    "target_concept": pair.target_concept,
                    "source_token_indices": list(pair.source_token_indices),
                }
                for pair in raw_pairs
            ],
            "h_target_formula": "H_source + W * (H_target - H_source)",
            "h_capture": "attention output before to_out, shape [batch, heads, video_tokens, head_dim]",
            "mask_algorithm": "GramCol: per-head per-frame QK argmax anchor, H-H similarity, spatial min-max per head/frame, mean over heads",
            "mask_heads": "all attention heads, as in Hunyuan imap_mode=object",
            "mask_source": "source concept only, computed in base teacher_source; reused for target H mixing",
            "video_prompt_pair_index": 0,
            "weight_formula": "clamp(alpha*M_source,0,1)" if args.clamp_mask else "alpha*M_source",
            "v_loss_formula": {
                "full": "||V_student_source - stopgrad(V_base_source)||^2",
                "background": "||(1-M_source) * (V_student_source - stopgrad(V_base_source))||^2",
                "target": "||M_source * (V_student_source - stopgrad(V_base_source))||^2",
            }[args.v_mask_mode],
        }
    )
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"CogVideoX layers: {num_layers}; H groups={h_groups} layers={h_layers}; V groups={v_groups} layers={v_layers}")
    print("H/mask: pre-to_out attention heads; GramCol with per-head per-frame QK anchors (all heads)")
    print(f"Resolution: {args.width}x{args.height}; frames={args.num_frames}; trajectory_steps={args.trajectory_steps}")
    print(f"Token grid: {token_grid}; LoRA modules={len(matches)}; trainable={summary['trainable']}")
    print(f"L_H alpha={args.imap_alpha}, clamp={args.clamp_mask}; L_V lambda={args.lamb_v}, mode={args.v_mask_mode}, steps={args.v_steps}")

    log_path = output_dir / "loss.jsonl"
    try:
        if args.save_training_videos:
            save_training_videos(pipe, controller, raw_pairs[0], args, output_dir, step=0)
        progress = tqdm(range(1, args.max_train_steps + 1), desc="cogvideox-hv-lora")
        for step in progress:
            pair_index = random.randrange(len(raw_pairs))
            pair = raw_pairs[pair_index]
            controller.source_token_indices = pair.source_token_indices

            if pair.source_embeds is None or pair.target_embeds is None:
                raise RuntimeError("Prompt embeddings were not cached.")
            source_embeds = pair.source_embeds.to(device=device, dtype=dtype)
            target_embeds = pair.target_embeds.to(device=device, dtype=dtype)
            controller.positive_batch_index = pair.positive_batch_index

            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, args.trajectory_steps, device, None)
            pipe._num_timesteps = len(timesteps)
            generator = torch.Generator(device=device).manual_seed(args.seed + step)
            initial_latents, _ = make_initial_latents(pipe, args, dtype, device, generator)
            image_rotary_emb = (
                pipe._prepare_rotary_positional_embeddings(args.height, args.width, initial_latents.size(1), device)
                if pipe.transformer.config.use_rotary_positional_embeddings
                else None
            )
            controller.token_grid = frames_from_latents_shape(pipe, initial_latents)
            target_index = sample_target_index(args)
            capture_v = args.lamb_v > 0.0 and target_index < args.v_steps

            with adapter_mode(pipe.transformer, True), torch.no_grad():
                z_t = rollout_to_index(
                    pipe,
                    source_embeds,
                    initial_latents,
                    timesteps,
                    target_index,
                    image_rotary_emb,
                    args.guidance_scale,
                    generator,
                )
            timestep = timesteps[target_index]

            controller.begin_teacher_source(pair.source_token_indices, capture_v=capture_v)
            with adapter_mode(pipe.transformer, False), torch.no_grad():
                _ = transformer_forward(pipe, z_t, timestep, source_embeds, image_rotary_emb, args.guidance_scale)
            missing_h = set(h_layers).difference(controller.source_h).union(set(h_layers).difference(controller.source_weights))
            if missing_h:
                raise RuntimeError(f"Missing teacher source H/weights for layers: {sorted(missing_h)}")
            if capture_v:
                missing_v = set(v_layers).difference(controller.source_v)
                if missing_v:
                    raise RuntimeError(f"Missing teacher source V for layers: {sorted(missing_v)}")

            controller.begin_teacher_target()
            with adapter_mode(pipe.transformer, False), torch.no_grad():
                _ = transformer_forward(pipe, z_t, timestep, target_embeds, image_rotary_emb, args.guidance_scale)
            missing_target = set(h_layers).difference(controller.target_h)
            if missing_target:
                raise RuntimeError(f"Missing teacher target H for layers: {sorted(missing_target)}")

            optimizer.zero_grad(set_to_none=True)
            if args.split_hv_backward and capture_v:
                controller.begin_student(capture_h=True, capture_v=False)
                with adapter_mode(pipe.transformer, True):
                    _ = transformer_forward(pipe, z_t, timestep, source_embeds, image_rotary_emb, args.guidance_scale)
                loss_h = controller.total_h_loss()
                if loss_h is None or set(h_layers).difference(controller.h_losses):
                    raise RuntimeError("Missing student L_H losses.")
                loss_h.backward()
                loss_h_log = loss_h.detach()

                controller.begin_student(capture_h=False, capture_v=True)
                with adapter_mode(pipe.transformer, True):
                    _ = transformer_forward(pipe, z_t, timestep, source_embeds, image_rotary_emb, args.guidance_scale)
                loss_v = controller.total_v_loss()
                if loss_v is None or set(v_layers).difference(controller.v_losses):
                    raise RuntimeError("Missing student L_V losses.")
                (args.lamb_v * loss_v).backward()
                loss_v_log = loss_v.detach()
                loss_total_log = loss_h_log + args.lamb_v * loss_v_log
            else:
                controller.begin_student(capture_h=True, capture_v=capture_v)
                with adapter_mode(pipe.transformer, True):
                    _ = transformer_forward(pipe, z_t, timestep, source_embeds, image_rotary_emb, args.guidance_scale)
                loss_h = controller.total_h_loss()
                if loss_h is None or set(h_layers).difference(controller.h_losses):
                    raise RuntimeError("Missing student L_H losses.")
                loss_v = controller.total_v_loss() if capture_v else None
                if capture_v and (loss_v is None or set(v_layers).difference(controller.v_losses)):
                    raise RuntimeError("Missing student L_V losses.")
                loss_v_for_total = loss_v if loss_v is not None else loss_h.new_zeros(())
                loss_total = loss_h + args.lamb_v * loss_v_for_total
                loss_total.backward()
                loss_h_log = loss_h.detach()
                loss_v_log = loss_v_for_total.detach()
                loss_total_log = loss_total.detach()

            grad_norm = torch.nn.utils.clip_grad_norm_(list(iter_trainable_parameters(pipe.transformer)), args.max_grad_norm)
            optimizer.step()
            controller.finish()

            record = {
                "step": step,
                "loss_total": float(loss_total_log.detach().cpu()),
                "loss_h": float(loss_h_log.detach().cpu()),
                "loss_v": float(loss_v_log.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "target_step_index": int(target_index),
                "timestep": float(timestep.detach().cpu()),
                "prompt_pair_index": int(pair_index),
                "source_prompt": pair.source_prompt,
                "target_prompt": pair.target_prompt,
                "value_active": bool(capture_v),
                "value_layers": len(controller.v_losses),
                "mean_concept_mask": float(np.mean(list(controller.mask_means.values()))) if controller.mask_means else 0.0,
                "mean_effective_weight": float(np.mean(list(controller.weight_means.values()))) if controller.weight_means else 0.0,
                "max_effective_weight": float(max(controller.weight_maxima.values())) if controller.weight_maxima else 0.0,
                "mean_v_mask": float(np.mean(list(controller.v_mask_means.values()))) if controller.v_mask_means else 0.0,
                "v_mask_mode": args.v_mask_mode,
                "split_hv_backward": args.split_hv_backward,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            progress.set_postfix(
                loss=f"{record['loss_total']:.5f}",
                h=f"{record['loss_h']:.5f}",
                v=f"{record['loss_v']:.5f}",
                t=target_index,
            )

            checkpoint_due = args.checkpointing_steps > 0 and step % args.checkpointing_steps == 0
            if checkpoint_due:
                save_checkpoint(pipe, output_dir / f"checkpoint-{step:06d}", adapter_name, config, matches, step)

            del initial_latents, z_t, source_embeds, target_embeds
            controller.reset_all()
            torch.cuda.empty_cache()
            if args.save_training_videos and (checkpoint_due or step == args.max_train_steps):
                optimizer.zero_grad(set_to_none=True)
                save_training_videos(pipe, controller, raw_pairs[0], args, output_dir, step=step)

        save_checkpoint(pipe, output_dir, adapter_name, config, matches, args.max_train_steps)
        print(f"Saved final CogVideoX HV LoRA to {output_dir}")
    finally:
        restore_processors(pipe, originals)


if __name__ == "__main__":
    main(build_parser().parse_args())
