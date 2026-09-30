from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE

from .hunyuan_attention_capture import capture_hunyuan_double_block_attention
from .token_utils import find_concept_token_indices


def transformer_device(pipe) -> torch.device:
    return next(pipe.transformer.parameters()).device


def encode_prompt(pipe, prompt: str, max_sequence_length: int = 256):
    dtype = pipe.transformer.dtype
    device = transformer_device(pipe)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            device=device,
            max_sequence_length=max_sequence_length,
        )
    return (
        prompt_embeds.to(device=device, dtype=dtype),
        pooled_prompt_embeds.to(device=device, dtype=dtype),
        prompt_attention_mask.to(device=device, dtype=dtype),
    )


def make_guidance(pipe, batch_size: int, guidance_scale: float) -> torch.Tensor:
    return torch.tensor(
        [guidance_scale] * batch_size,
        dtype=pipe.transformer.dtype,
        device=transformer_device(pipe),
    ) * 1000.0


def prepare_flow_timesteps(scheduler, num_steps: int, device: torch.device) -> torch.Tensor:
    sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]
    scheduler.set_timesteps(num_steps, device=device, sigmas=sigmas)
    return scheduler.timesteps


def first_frame_latent_slice(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"Expected Hunyuan video tensor with shape B,C,T,H,W, got {tuple(tensor.shape)}")
    return tensor[:, :, :1, :, :]


def flow_add_noise(latents: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    sigma = (timestep.to(device=latents.device, dtype=latents.dtype) / float(num_train_timesteps)).flatten()
    while len(sigma.shape) < len(latents.shape):
        sigma = sigma.unsqueeze(-1)
    return sigma * noise + (1.0 - sigma) * latents


@contextmanager
def maybe_disable_adapters(transformer, enabled: bool):
    if not enabled:
        yield
        return
    transformer.disable_adapters()
    try:
        yield
    finally:
        transformer.enable_adapters()


@torch.no_grad()
def latent_sample_video(
    pipe,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    transformer = pipe.transformer
    scheduler = copy.deepcopy(pipe.scheduler)
    device = transformer_device(pipe)
    timesteps = prepare_flow_timesteps(scheduler, num_inference_steps, device)
    target_index = torch.randint(0, len(timesteps), (1,), device=device).item()
    latents = pipe.prepare_latents(
        batch_size=prompt_embeds.shape[0],
        num_channels_latents=transformer.config.in_channels,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=device,
        generator=None,
    )
    guidance = make_guidance(pipe, latents.shape[0], guidance_scale)

    for timestep_value in timesteps[:target_index]:
        latent_model_input = latents.to(transformer.dtype)
        timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
        model_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=pooled_prompt_embeds,
            guidance=guidance,
            return_dict=False,
        )[0]
        latents = scheduler.step(model_pred, timestep_value, latents, return_dict=False)[0]
    return latents.detach(), timesteps[target_index : target_index + 1].detach()


def predict_video_noise(
    pipe,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    device = transformer_device(pipe)
    guidance = make_guidance(pipe, latents.shape[0], guidance_scale)
    return pipe.transformer(
        hidden_states=latents.to(device=device, dtype=pipe.transformer.dtype),
        timestep=timestep.expand(latents.shape[0]).to(device=device, dtype=latents.dtype),
        encoder_hidden_states=prompt_embeds.to(device=device, dtype=pipe.transformer.dtype),
        encoder_attention_mask=prompt_attention_mask.to(device=device, dtype=pipe.transformer.dtype),
        pooled_projections=pooled_prompt_embeds.to(device=device, dtype=pipe.transformer.dtype),
        guidance=guidance,
        return_dict=False,
    )[0]


def compute_esd_losses_from_cached(
    pipe,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    neutral_embeds: torch.Tensor,
    neutral_pooled: torch.Tensor,
    neutral_attention_mask: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    latent_sample_steps: int,
    timestep_sample_steps: int,
    guidance_scale: float,
    negative_guidance: float,
    teacher_mode: str = "base",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    device = transformer_device(pipe)
    prompt_embeds = prompt_embeds.to(device=device, dtype=pipe.transformer.dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=pipe.transformer.dtype)
    prompt_attention_mask = prompt_attention_mask.to(device=device, dtype=pipe.transformer.dtype)
    neutral_embeds = neutral_embeds.to(device=device, dtype=pipe.transformer.dtype)
    neutral_pooled = neutral_pooled.to(device=device, dtype=pipe.transformer.dtype)
    neutral_attention_mask = neutral_attention_mask.to(device=device, dtype=pipe.transformer.dtype)

    if int(latent_sample_steps) != int(timestep_sample_steps):
        raise ValueError(
            "latent_sample_steps and timestep_sample_steps must match so ESD uses a paired z_t and timestep. "
            f"Got latent_sample_steps={latent_sample_steps}, timestep_sample_steps={timestep_sample_steps}."
        )

    z, timestep = latent_sample_video(
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        prompt_attention_mask,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=latent_sample_steps,
        guidance_scale=guidance_scale,
    )
    if teacher_mode not in {"base", "current"}:
        raise ValueError(f"teacher_mode must be 'base' or 'current', got {teacher_mode!r}")
    disable_teacher_lora = teacher_mode == "base"

    with torch.no_grad(), maybe_disable_adapters(pipe.transformer, disable_teacher_lora):
        e_0 = predict_video_noise(
            pipe,
            z,
            timestep,
            neutral_embeds,
            neutral_pooled,
            neutral_attention_mask,
            guidance_scale,
        )
        e_p = predict_video_noise(
            pipe,
            z,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            prompt_attention_mask,
            guidance_scale,
        )

    e_n = predict_video_noise(
        pipe,
        z,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
        prompt_attention_mask,
        guidance_scale,
    )
    target = e_0 - negative_guidance * (e_p - e_0)
    losses = {
        "full_video": F.mse_loss(e_n.float(), target.float()),
        "first_frame": F.mse_loss(first_frame_latent_slice(e_n).float(), first_frame_latent_slice(target).float()),
    }
    return losses, timestep.detach()


def compute_esd_loss_from_cached(
    pipe,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    neutral_embeds: torch.Tensor,
    neutral_pooled: torch.Tensor,
    neutral_attention_mask: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    latent_sample_steps: int,
    timestep_sample_steps: int,
    guidance_scale: float,
    negative_guidance: float,
    teacher_mode: str = "base",
) -> Tuple[torch.Tensor, torch.Tensor]:
    losses, timestep = compute_esd_losses_from_cached(
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        prompt_attention_mask,
        neutral_embeds,
        neutral_pooled,
        neutral_attention_mask,
        height,
        width,
        num_frames,
        latent_sample_steps,
        timestep_sample_steps,
        guidance_scale,
        negative_guidance,
        teacher_mode,
    )
    return losses["full_video"], timestep


def compute_esd_loss(
    pipe,
    prompt: str,
    neutral_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    latent_sample_steps: int,
    timestep_sample_steps: int,
    guidance_scale: float,
    negative_guidance: float,
    teacher_mode: str = "base",
    max_sequence_length: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = encode_prompt(pipe, prompt, max_sequence_length)
    neutral_embeds, neutral_pooled, neutral_attention_mask = encode_prompt(pipe, neutral_prompt, max_sequence_length)
    return compute_esd_loss_from_cached(
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        prompt_attention_mask,
        neutral_embeds,
        neutral_pooled,
        neutral_attention_mask,
        height,
        width,
        num_frames,
        latent_sample_steps,
        timestep_sample_steps,
        guidance_scale,
        negative_guidance,
        teacher_mode,
    )


@torch.no_grad()
def encode_video_latents(pipe, pixel_values: torch.Tensor) -> torch.Tensor:
    pixel_values = pixel_values.to(device=pipe._execution_device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    return latents * pipe.vae.config.scaling_factor


def compute_attention_loss(
    pipe,
    pixel_values: torch.Tensor,
    prompt: str,
    concept: str,
    timestep: torch.Tensor,
    guidance_scale: float,
    block_indices,
    attention_chunk_size: int,
    max_sequence_length: int = 256,
) -> Tuple[torch.Tensor, List[int]]:
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = encode_prompt(pipe, prompt, max_sequence_length)
    concept_indices = find_concept_token_indices(
        pipe.tokenizer,
        prompt=prompt,
        concept=concept,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length=max_sequence_length,
    )
    device = transformer_device(pipe)
    latents = encode_video_latents(pipe, pixel_values).to(device=device, dtype=pipe.transformer.dtype)
    noise = torch.randn_like(latents)
    noisy_latents = flow_add_noise(
        latents,
        noise,
        timestep,
        num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
    )
    guidance = make_guidance(pipe, noisy_latents.shape[0], guidance_scale)

    with capture_hunyuan_double_block_attention(
        pipe.transformer,
        concept_token_indices=concept_indices,
        block_indices=block_indices,
        chunk_size=attention_chunk_size,
    ) as recorder:
        pipe.transformer(
            hidden_states=noisy_latents.to(pipe.transformer.dtype),
            timestep=timestep.expand(noisy_latents.shape[0]).to(device=device, dtype=noisy_latents.dtype),
            encoder_hidden_states=prompt_embeds.to(device=device, dtype=pipe.transformer.dtype),
            encoder_attention_mask=prompt_attention_mask.to(device=device, dtype=pipe.transformer.dtype),
            pooled_projections=pooled_prompt_embeds.to(device=device, dtype=pipe.transformer.dtype),
            guidance=guidance,
            return_dict=False,
        )
        loss = recorder.total()
    return loss, concept_indices


def compute_attention_loss_from_cached(
    pipe,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    concept_token_indices: List[int],
    timestep: torch.Tensor,
    guidance_scale: float,
    block_indices,
    attention_chunk_size: int,
) -> Tuple[torch.Tensor, List[int]]:
    device = transformer_device(pipe)
    latents = latents.to(device=device, dtype=pipe.transformer.dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=pipe.transformer.dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=pipe.transformer.dtype)
    prompt_attention_mask = prompt_attention_mask.to(device=device, dtype=pipe.transformer.dtype)

    noise = torch.randn_like(latents)
    noisy_latents = flow_add_noise(
        latents,
        noise,
        timestep,
        num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
    )
    guidance = make_guidance(pipe, noisy_latents.shape[0], guidance_scale)

    with capture_hunyuan_double_block_attention(
        pipe.transformer,
        concept_token_indices=concept_token_indices,
        block_indices=block_indices,
        chunk_size=attention_chunk_size,
    ) as recorder:
        pipe.transformer(
            hidden_states=noisy_latents.to(pipe.transformer.dtype),
            timestep=timestep.expand(noisy_latents.shape[0]).to(device=device, dtype=noisy_latents.dtype),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=pooled_prompt_embeds,
            guidance=guidance,
            return_dict=False,
        )
        loss = recorder.total()
    return loss, concept_token_indices
