"""
L_ESD + L_V dla HunyuanVideo (Wariant A: bezpośrednie dopasowanie V).

L_V = || V_student, single − V_base, single ||²   dla t z wczesnych kroków,
                                                  single blocks > v_block_min

Kluczowa własność: L_V nie kosztuje ani jednego dodatkowego forwardu.
compute_esd_losses_from_cached i tak wykonuje na tym samym z_t:
    e_p  -> prompt z konceptem, LoRA WYŁĄCZONA  = model bazowy   -> V nauczyciela
    e_n  -> prompt z konceptem, LoRA WŁĄCZONA   = student        -> V studenta
Wystarczy obudować te dwa przebiegi procesorami przechwytującymi V.

Gradient: LoRA siedzi w double blocks, single blocks są zamrożone — ale V
w single blocks zależy od hidden_states wychodzących z double blocks, więc
    L_V -> V_single -> hidden_states -> double blocks -> LoRA
Nie trzeba trenować single blocks, żeby na nie wpływać.

Plik jest samodzielny — importuje pomocnicze funkcje z hunyuan_erase_losses.py
i niczego w nim nie zmienia.
Docelowa lokalizacja: receler/training/hunyuan_value_loss.py
"""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .hunyuan_erase_losses import (
    first_frame_latent_slice,
    make_guidance,
    maybe_disable_adapters,
    predict_video_noise,
    prepare_flow_timesteps,
    transformer_device,
)
from .hunyuan_value_capture import (
    ValueLossRecorder,
    capture_hunyuan_single_block_values,
)


@contextmanager
def _nullcontext():
    """Odpowiednik contextlib.nullcontext — wydzielony, żeby wyrażenie
    warunkowe przy `with` czytało się jednoznacznie."""
    yield


@contextmanager
def disable_gradient_checkpointing(module):
    """Tymczasowo wyłącza gradient checkpointing w całym modelu.

    DLACZEGO TO JEST KONIECZNE PRZY L_V:

    Gradient checkpointing nie zapamiętuje aktywacji — zamiast tego przelicza
    forward jeszcze raz podczas backwardu. torch.utils.checkpoint weryfikuje,
    czy oba przebiegi zapisały tyle samo tensorów, i przerywa, gdy się różnią.

    Nasz procesor przechwytujący V łamie to założenie: przy pierwszym forwardzie
    recorder jest w trybie "student" i wykonuje dodatkowe operacje (rzutowanie
    na float + mse_loss), a przy przeliczaniu jest już "off" i tych operacji nie
    ma. Efekt to CheckpointError w rodzaju:

        A different number of tensors was saved during the original forward
        and recomputation. Number of tensors saved during forward: 35
        Number of tensors saved during recomputation: 33

    Nauczyciel działa pod torch.no_grad(), więc gałąź z checkpointingiem
    (`if torch.is_grad_enabled() and self.gradient_checkpointing`) i tak się
    tam nie wykonuje — wyłączać trzeba tylko przebieg studenta.

    KOSZT: bez checkpointingu forward studenta trzyma pełne aktywacje. Przy
    768x432 i 33 klatkach (sekwencja ~11.9k tokenów) to rząd kilkudziesięciu GB.
    Jeśli zabraknie pamięci, kolejność ratunkowa:
        1. train_num_frames 33 -> 17   (sekwencja spada o połowę)
        2. rozdzielczosc 768x432 -> 512x288
        3. v_block_min wyżej, np. 30   (mniej bloków = mniej porównań,
           ale aktywacje i tak są trzymane dla całej sieci)
    """
    previous = {}
    for name, sub in module.named_modules():
        if hasattr(sub, "gradient_checkpointing"):
            previous[name] = sub.gradient_checkpointing
            sub.gradient_checkpointing = False
    try:
        yield
    finally:
        for name, sub in module.named_modules():
            if name in previous:
                sub.gradient_checkpointing = previous[name]


@torch.no_grad()
def latent_sample_video_with_index(
    pipe,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    early_steps: int = 0,
    early_prob: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Wariant latent_sample_video zwracający dodatkowo wylosowany indeks kroku.

    Z prawdopodobieństwem `early_prob` indeks losowany jest z [0, early_steps),
    w przeciwnym razie z pełnego zakresu. Skupienie na wczesnych krokach jest
    TAŃSZE — pętla poniżej wykonuje `target_index` iteracji, więc przy indeksie
    2 robi 2 kroki zamiast nawet 27.
    """
    transformer = pipe.transformer
    scheduler = copy.deepcopy(pipe.scheduler)
    device = transformer_device(pipe)
    timesteps = prepare_flow_timesteps(scheduler, num_inference_steps, device)

    num_timesteps = len(timesteps)
    upper = max(1, min(int(early_steps), num_timesteps))
    if early_steps > 0 and early_prob > 0.0 and random.random() < early_prob:
        target_index = random.randrange(0, upper)
    else:
        target_index = random.randrange(0, num_timesteps)

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

    return latents.detach(), timesteps[target_index : target_index + 1].detach(), target_index


def compute_esd_and_value_losses_from_cached(
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
    # --- parametry L_V ---
    v_enabled: bool = False,
    v_steps: int = 3,
    v_block_min: int | str = 20,
    v_video_only: bool = True,
    v_early_prob: float = 0.5,
    v_disable_gc: bool = True,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
    """Zwraca (straty, timestep, metryki).

    straty:
        "full_video"  - L_ESD na całym wideo
        "first_frame" - L_ESD na pierwszej klatce
        "value"       - L_V (obecna tylko, gdy policzona w tym kroku)

    metryki (do logowania):
        "timestep_index" - wylosowany indeks kroku denoisingu
        "v_active"       - 1.0 jeśli L_V liczone w tym kroku, inaczej 0.0
        "v_blocks"       - liczba bloków, z których zebrano V
    """
    device = transformer_device(pipe)
    dtype = pipe.transformer.dtype

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    prompt_attention_mask = prompt_attention_mask.to(device=device, dtype=dtype)
    neutral_embeds = neutral_embeds.to(device=device, dtype=dtype)
    neutral_pooled = neutral_pooled.to(device=device, dtype=dtype)
    neutral_attention_mask = neutral_attention_mask.to(device=device, dtype=dtype)

    if int(latent_sample_steps) != int(timestep_sample_steps):
        raise ValueError(
            "latent_sample_steps i timestep_sample_steps muszą być równe, żeby ESD "
            f"używał sparowanych z_t i t. Otrzymano {latent_sample_steps} vs {timestep_sample_steps}."
        )
    if teacher_mode not in {"base", "current"}:
        raise ValueError(f"teacher_mode musi być 'base' albo 'current', otrzymano {teacher_mode!r}")

    z, timestep, target_index = latent_sample_video_with_index(
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        prompt_attention_mask,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=latent_sample_steps,
        guidance_scale=guidance_scale,
        early_steps=v_steps if v_enabled else 0,
        early_prob=v_early_prob if v_enabled else 0.0,
    )

    # L_V liczymy tylko wtedy, gdy wylosowany krok mieści się we wczesnym oknie.
    capture_value = bool(v_enabled) and target_index < int(v_steps)
    # teacher_mode="current" oznacza brak osobnego modelu bazowego -> brak sensownego celu dla L_V
    if teacher_mode != "base":
        capture_value = False

    disable_teacher_lora = teacher_mode == "base"
    recorder = ValueLossRecorder(video_only=bool(v_video_only))

    metrics: Dict[str, float] = {
        "timestep_index": float(target_index),
        "v_active": 0.0,
        "v_blocks": 0.0,
    }
    losses: Dict[str, torch.Tensor] = {}

    if capture_value:
        with capture_hunyuan_single_block_values(pipe.transformer, recorder, block_min=v_block_min):
            # --- nauczyciel: LoRA wyłączona ---
            with torch.no_grad(), maybe_disable_adapters(pipe.transformer, disable_teacher_lora):
                e_0 = predict_video_noise(
                    pipe, z, timestep, neutral_embeds, neutral_pooled, neutral_attention_mask, guidance_scale
                )
                recorder.begin_teacher()
                e_p = predict_video_noise(
                    pipe, z, timestep, prompt_embeds, pooled_prompt_embeds, prompt_attention_mask, guidance_scale
                )
                recorder.finish()

            # --- student: LoRA włączona, z gradientem ---
            # gradient checkpointing MUSI być tu wyłączony, inaczej backward
            # przeliczy forward bez przechwytywania V i policzy inną liczbę
            # zapisanych tensorów -> CheckpointError (szczegóły w docstringu
            # disable_gradient_checkpointing)
            with disable_gradient_checkpointing(pipe.transformer) if v_disable_gc else _nullcontext():
                recorder.begin_student()
                e_n = predict_video_noise(
                    pipe, z, timestep, prompt_embeds, pooled_prompt_embeds, prompt_attention_mask, guidance_scale
                )
                recorder.finish()

            loss_value = recorder.total()
            if loss_value is not None:
                losses["value"] = loss_value
                metrics["v_active"] = 1.0
                metrics["v_blocks"] = float(len(recorder.losses))
        recorder.reset()
    else:
        with torch.no_grad(), maybe_disable_adapters(pipe.transformer, disable_teacher_lora):
            e_0 = predict_video_noise(
                pipe, z, timestep, neutral_embeds, neutral_pooled, neutral_attention_mask, guidance_scale
            )
            e_p = predict_video_noise(
                pipe, z, timestep, prompt_embeds, pooled_prompt_embeds, prompt_attention_mask, guidance_scale
            )
        e_n = predict_video_noise(
            pipe, z, timestep, prompt_embeds, pooled_prompt_embeds, prompt_attention_mask, guidance_scale
        )

    target = e_0 - negative_guidance * (e_p - e_0)
    losses["full_video"] = F.mse_loss(e_n.float(), target.float())
    losses["first_frame"] = F.mse_loss(
        first_frame_latent_slice(e_n).float(),
        first_frame_latent_slice(target).float(),
    )

    return losses, timestep.detach(), metrics