from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers.utils import export_to_video
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

IMAP_REPO_DIR = Path(__file__).resolve().parents[1] / "IMAP"
if IMAP_REPO_DIR.exists():
    sys.path.insert(0, str(IMAP_REPO_DIR))

from diagnose_hunyuan_imap_hidden_patch import DEFAULT_IMAP_DOUBLE_BLOCKS, gramcol_soft_mask
try:
    from imap.imap_utils import select_head as _imap_select_head
except ModuleNotFoundError as error:
    if error.name not in {"imap", "imap.imap_utils", "sklearn"}:
        raise
    _imap_select_head = None
from receler.loras.hunyuan_peft_lora import (
    DEFAULT_HUNYUAN_ERASE_TARGET_MODULES,
    add_hunyuan_erase_lora,
    iter_trainable_parameters,
    trainable_parameter_summary,
)
from receler.training.hunyuan_erase_losses import make_guidance, prepare_flow_timesteps
from receler.training.token_utils import find_concept_token_indices


DEFAULT_PROMPT_PAIRS = (
    ("a video of a dog", "a video of a cat"),
    ("a dog standing on a sofa", "a cat standing on a sofa"),
    ("a dog running across a grassy field", "a cat running across a grassy field"),
    ("a dog walking down a city street", "a cat walking down a city street"),
    ("a dog sitting beside a lake", "a cat sitting beside a lake"),
    ("a dog playing in a garden", "a cat playing in a garden"),
    ("a dog resting on a wooden floor", "a cat resting on a wooden floor"),
    ("a dog looking through a window", "a cat looking through a window"),
    ("a dog moving through fresh snow", "a cat moving through fresh snow"),
    ("a dog standing on a quiet beach", "a cat standing on a quiet beach"),
)


@dataclass
class PromptCondition:
    prompt: str
    concept_indices: Sequence[int]
    embeds: torch.Tensor
    pooled: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype) -> "PromptCondition":
        return PromptCondition(
            prompt=self.prompt,
            concept_indices=self.concept_indices,
            embeds=self.embeds.to(device=device, dtype=dtype),
            pooled=self.pooled.to(device=device, dtype=dtype),
            attention_mask=self.attention_mask.to(device=device, dtype=dtype),
        )


class IMAPHController:
    """Stores detached teacher targets and online student H/V losses for IMAP-HV."""

    def __init__(
        self,
        grid: Tuple[int, int, int],
        blocks: Sequence[int],
        alpha: float,
        clamp_mask: bool,
        lamb_v: float = 0.0,
        v_video_only: bool = True,
        v_mask_mode: str = "background",
        imap_mode: str = "object",
        motion_topk_heads: int = 5,
        motion_sep_score: str = "CHI",
    ):
        self.grid = grid
        self.blocks = frozenset(blocks)
        self.alpha = float(alpha)
        self.clamp_mask = bool(clamp_mask)
        self.lamb_v = float(lamb_v)
        self.v_video_only = bool(v_video_only)
        self.v_mask_mode = str(v_mask_mode)
        self.imap_mode = str(imap_mode)
        self.motion_topk_heads = int(motion_topk_heads)
        self.motion_sep_score = str(motion_sep_score)
        self.mode = "off"
        self.capture_value = False
        self.capture_h = False
        self.source_token_indices: Sequence[int] = ()
        self.source_states: Dict[int, torch.Tensor] = {}
        self.source_weights: Dict[int, torch.Tensor] = {}
        self.teacher_targets: Dict[int, torch.Tensor] = {}
        self.h_losses: List[torch.Tensor] = []
        self.h_loss_blocks: set[int] = set()
        self.teacher_values: Dict[int, torch.Tensor] = {}
        self.value_losses: Dict[int, torch.Tensor] = {}
        self.mask_means: Dict[int, float] = {}
        self.weight_means: Dict[int, float] = {}
        self.weight_maxima: Dict[int, float] = {}
        self.background_mask_means: Dict[int, float] = {}
        self.motion_heads: Dict[int, torch.Tensor] = {}
        self.motion_head_counts: Dict[int, int] = {}

    @property
    def num_visual_tokens(self) -> int:
        return int(prod(self.grid))

    def begin_teacher_source(self, source_token_indices: Sequence[int], capture_value: bool = False) -> None:
        self.mode = "teacher_source"
        self.capture_value = bool(capture_value)
        self.capture_h = False
        self.source_token_indices = tuple(source_token_indices)
        self.source_states.clear()
        self.source_weights.clear()
        self.teacher_targets.clear()
        self.h_losses.clear()
        self.h_loss_blocks.clear()
        self.teacher_values.clear()
        self.value_losses.clear()
        self.clear_source_masks()
        self.mask_means.clear()
        self.weight_means.clear()
        self.weight_maxima.clear()
        self.background_mask_means.clear()
        self.motion_heads.clear()
        self.motion_head_counts.clear()

    def begin_teacher_target(self, capture_value: bool = False) -> None:
        self.mode = "teacher_target"
        self.capture_value = bool(capture_value)
        self.capture_h = False

    def begin_student(self, capture_value: bool = False, capture_h: bool = True) -> None:
        self.mode = "student"
        self.capture_value = bool(capture_value)
        self.capture_h = bool(capture_h)
        self.h_losses.clear()
        self.h_loss_blocks.clear()
        self.value_losses.clear()

    def finish(self) -> None:
        self.mode = "off"
        self.capture_value = False
        self.capture_h = False

    def mean_source_mask(self) -> torch.Tensor:
        if not self.mask_means:
            raise RuntimeError("Cannot build a background mask before IMAP masks are captured.")
        masks = getattr(self, "_source_masks", None)
        if not masks:
            raise RuntimeError("No source IMAP masks were captured for L_V.")
        return torch.stack(list(masks.values())).mean(dim=0).clamp(0.0, 1.0)

    def store_source_mask(self, block_index: int, mask: torch.Tensor) -> None:
        if not hasattr(self, "_source_masks"):
            self._source_masks = {}
        self._source_masks[block_index] = mask.detach()

    def clear_source_masks(self) -> None:
        if hasattr(self, "_source_masks"):
            self._source_masks.clear()

    def total_value_loss(self) -> torch.Tensor | None:
        if not self.value_losses:
            return None
        return torch.stack([self.value_losses[index] for index in sorted(self.value_losses)]).mean()

    def total_h_loss(self) -> torch.Tensor | None:
        if not self.h_losses:
            return None
        return torch.stack(self.h_losses).mean()


def _concept_key(encoder_key: torch.Tensor, sample_index: int, token_indices: Sequence[int]) -> torch.Tensor:
    indices = torch.as_tensor(token_indices, device=encoder_key.device, dtype=torch.long)
    return encoder_key[sample_index].index_select(1, indices).mean(dim=1)


def select_motion_heads(
    hidden_state: torch.Tensor,
    text_seq_length: int,
    grid: Tuple[int, int, int],
    sep_score: str,
    topk: int,
) -> torch.Tensor:
    """Select motion heads like IMAP; fallback implements CHI without sklearn."""
    frames, grid_height, grid_width = grid
    if _imap_select_head is not None:
        return _imap_select_head(
            hidden_state=hidden_state,
            text_seq_length=text_seq_length,
            F=frames,
            H=grid_height,
            W=grid_width,
            sep_score=sep_score,
            topk=topk,
            text_seq_back=True,
        )

    if sep_score.strip().lower() != "chi":
        raise ModuleNotFoundError(
            "scikit-learn is required for IMAP head selection metrics other than CHI. "
            "Use MOTION_SEP_SCORE=CHI or install scikit-learn."
        )
    if hidden_state.ndim != 3:
        raise ValueError(f"hidden_state must be [heads, tokens, dim], got {tuple(hidden_state.shape)}")
    num_heads, total_tokens, dim = hidden_state.shape
    visual_tokens = frames * grid_height * grid_width
    if total_tokens - text_seq_length != visual_tokens:
        raise ValueError(
            f"Visual-token mismatch for CHI head selection: total={total_tokens}, "
            f"text={text_seq_length}, visual={total_tokens - text_seq_length}, expected={visual_tokens}."
        )

    visual = hidden_state[:, :-text_seq_length, :].float()
    patches_per_frame = grid_height * grid_width
    visual = visual.reshape(num_heads, frames, patches_per_frame, dim)
    frame_means = visual.mean(dim=2)
    global_mean = visual.reshape(num_heads, frames * patches_per_frame, dim).mean(dim=1)
    between = patches_per_frame * ((frame_means - global_mean[:, None, :]) ** 2).sum(dim=(1, 2))
    within = ((visual - frame_means[:, :, None, :]) ** 2).sum(dim=(1, 2, 3))
    n = frames * patches_per_frame
    chi = (between / max(frames - 1, 1)) / (within / max(n - frames, 1)).clamp_min(1e-12)
    k = min(int(topk), num_heads)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=hidden_state.device)
    return torch.topk(chi, k=k, largest=True, sorted=True).indices.to(dtype=torch.long)


class HunyuanIMAPHLossAttnProcessor:
    """Standard double-stream attention plus capture of pre-to_out visual H."""

    def __init__(self, controller: IMAPHController, block_index: int):
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
        if attn.add_q_proj is None or encoder_hidden_states is None:
            raise ValueError("IMAP H capture is only valid in Hunyuan double-stream blocks.")

        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        image_query = query

        encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([query, encoder_query], dim=2)
        key = torch.cat([key, encoder_key], dim=2)
        value = torch.cat([value, encoder_value], dim=2)
        attention_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        runtime_visual_tokens = attention_output.shape[2] - encoder_hidden_states.shape[1]
        capture_active = self.controller.mode in {"teacher_source", "teacher_target"} or (
            self.controller.mode == "student" and self.controller.capture_h
        )
        if self.block_index in self.controller.blocks and capture_active:
            visual_tokens = self.controller.num_visual_tokens
            if runtime_visual_tokens != visual_tokens:
                raise ValueError(
                    f"Visual grid {self.controller.grid} requires {visual_tokens} tokens, "
                    f"attention returned {runtime_visual_tokens}."
                )
            visual = attention_output[:, :, :visual_tokens]
            if self.controller.mode == "teacher_source":
                if visual.shape[0] != 1:
                    raise ValueError("Teacher source capture expects one source sample.")
                source_key = _concept_key(encoder_key, 0, self.controller.source_token_indices)
                if self.controller.imap_mode == "video":
                    selected_heads = select_motion_heads(
                        hidden_state=attention_output[0],
                        text_seq_length=encoder_hidden_states.shape[1],
                        grid=self.controller.grid,
                        sep_score=self.controller.motion_sep_score,
                        topk=self.controller.motion_topk_heads,
                    )
                    if selected_heads.numel() == 0:
                        raise RuntimeError(
                            f"No motion heads selected in block {self.block_index} "
                            f"with sep_score={self.controller.motion_sep_score}, "
                            f"topk={self.controller.motion_topk_heads}."
                        )
                    selected_heads = selected_heads.to(device=visual.device, dtype=torch.long)
                    source_visual_for_mask = visual[0].index_select(0, selected_heads)
                    source_query_for_mask = image_query[0].index_select(0, selected_heads)
                    source_key_for_mask = source_key.index_select(0, selected_heads)
                    source_mask = gramcol_soft_mask(
                        source_visual_for_mask,
                        source_query_for_mask,
                        source_key_for_mask,
                        self.controller.grid,
                    )
                    head_mask = torch.zeros(
                        visual.shape[1],
                        device=visual.device,
                        dtype=visual.dtype,
                    )
                    head_mask.scatter_(0, selected_heads, 1.0)
                    head_mask = head_mask.reshape(1, -1, 1, 1)
                    self.controller.motion_heads[self.block_index] = selected_heads.detach().cpu()
                    self.controller.motion_head_counts[self.block_index] = int(selected_heads.numel())
                else:
                    source_mask = gramcol_soft_mask(visual[0], image_query[0], source_key, self.controller.grid)
                    head_mask = 1.0
                weight_map = self.controller.alpha * source_mask
                if self.controller.clamp_mask:
                    weight_map = weight_map.clamp(0.0, 1.0)
                weight = weight_map.reshape(1, 1, visual_tokens, 1)
                weight = weight.to(device=visual.device, dtype=visual.dtype)
                self.controller.source_states[self.block_index] = visual.detach()
                self.controller.source_weights[self.block_index] = (head_mask * weight).detach()
                self.controller.store_source_mask(self.block_index, source_mask)
                self.controller.mask_means[self.block_index] = float(source_mask.detach().mean().cpu())
                self.controller.weight_means[self.block_index] = float(weight_map.detach().mean().cpu())
                self.controller.weight_maxima[self.block_index] = float(weight_map.detach().max().cpu())
            elif self.controller.mode == "teacher_target":
                if visual.shape[0] != 1:
                    raise ValueError("Teacher target capture expects one target sample.")
                source_visual = self.controller.source_states.pop(self.block_index, None)
                source_weight = self.controller.source_weights.pop(self.block_index, None)
                if source_visual is None or source_weight is None:
                    raise RuntimeError(f"Missing teacher source H/weight for block {self.block_index}.")
                source_visual = source_visual.to(device=visual.device, dtype=visual.dtype)
                source_weight = source_weight.to(device=visual.device, dtype=visual.dtype)
                h_star = source_visual + source_weight * (visual - source_visual)
                self.controller.teacher_targets[self.block_index] = h_star.detach()
            elif self.controller.mode == "student":
                if visual.shape[0] != 1:
                    raise ValueError("Student capture expects one dog sample.")
                target = self.controller.teacher_targets.get(self.block_index)
                if target is None:
                    raise RuntimeError(f"Missing teacher H target for block {self.block_index}.")
                target = target.to(device=visual.device, dtype=visual.dtype)
                if self.controller.imap_mode == "video":
                    selected_heads = self.controller.motion_heads.get(self.block_index)
                    if selected_heads is None or selected_heads.numel() == 0:
                        raise RuntimeError(
                            f"Missing selected motion heads for student H loss in block {self.block_index}."
                        )
                    selected_heads = selected_heads.to(device=visual.device, dtype=torch.long)
                    visual_for_loss = visual.index_select(1, selected_heads)
                    target_for_loss = target.index_select(1, selected_heads)
                else:
                    visual_for_loss = visual
                    target_for_loss = target
                self.controller.h_losses.append(
                    F.mse_loss(visual_for_loss.float(), target_for_loss.float())
                )
                self.controller.h_loss_blocks.add(self.block_index)

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


class HunyuanIMAPValueLossAttnProcessor:
    """Standard single-stream attention plus optional background-weighted V loss."""

    def __init__(self, controller: IMAPHController, block_index: int):
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
        text_seq_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0

        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
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
            encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        self._capture_value(value, text_seq_length)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)

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

    def _capture_value(self, value: torch.Tensor, text_seq_length: int) -> None:
        if not self.controller.capture_value or self.controller.mode not in {"teacher_source", "student"}:
            return

        captured = value
        if self.controller.v_video_only and text_seq_length > 0:
            captured = captured[:, :, :-text_seq_length]

        visual_tokens = self.controller.num_visual_tokens
        if captured.shape[2] != visual_tokens:
            raise ValueError(
                f"L_V expects {visual_tokens} visual tokens from grid {self.controller.grid}, "
                f"got {captured.shape[2]} in single block {self.block_index}."
            )

        if self.controller.mode == "teacher_source":
            if captured.shape[0] != 1:
                raise ValueError("Teacher source V capture expects one source sample.")
            # V_base can be very large at 720p. Keep the H targets on GPU for speed,
            # but offload only teacher V references to CPU; they are moved back
            # block-by-block during the student forward when L_V is computed.
            self.controller.teacher_values[self.block_index] = captured.detach().cpu()
            return

        if captured.shape[0] != 1:
            raise ValueError("Student V capture expects one source sample.")
        reference = self.controller.teacher_values.get(self.block_index)
        if reference is None:
            return
        if self.controller.v_mask_mode == "background":
            value_weight = 1.0 - self.controller.mean_source_mask()
        elif self.controller.v_mask_mode == "full":
            value_weight = torch.ones(
                visual_tokens,
                device=captured.device,
                dtype=captured.dtype,
            )
        else:
            raise ValueError(f"Unknown v_mask_mode: {self.controller.v_mask_mode}")
        value_weight = value_weight.reshape(1, 1, visual_tokens, 1).to(device=captured.device, dtype=captured.dtype)
        diff = value_weight * (captured - reference.to(device=captured.device, dtype=captured.dtype))
        self.controller.value_losses[self.block_index] = (diff.float() ** 2).mean()
        self.controller.background_mask_means[self.block_index] = float(value_weight.detach().mean().cpu())


def parse_blocks(value: str) -> Tuple[int, ...]:
    blocks = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not blocks or min(blocks) < 0 or max(blocks) >= 20:
        raise argparse.ArgumentTypeError("double_blocks must contain indices from 0 to 19")
    return blocks


def resolve_single_blocks(num_blocks: int, block_min: int | str | Iterable[int]) -> Tuple[int, ...]:
    if isinstance(block_min, str):
        value = block_min.strip()
        if value.lower() == "all":
            indices = list(range(num_blocks))
        elif "," in value:
            indices = [int(item.strip()) for item in value.split(",") if item.strip()]
        else:
            indices = [index for index in range(num_blocks) if index > int(value)]
    elif isinstance(block_min, int):
        indices = [index for index in range(num_blocks) if index > block_min]
    else:
        indices = [int(item) for item in block_min]

    if not indices:
        raise ValueError("L_V selected no single blocks.")
    invalid = [index for index in indices if index < 0 or index >= num_blocks]
    if invalid:
        raise IndexError(f"Single-block indices out of range 0..{num_blocks - 1}: {invalid}")
    return tuple(indices)


@contextmanager
def disable_gradient_checkpointing(module):
    previous = {}
    for name, submodule in module.named_modules():
        if hasattr(submodule, "gradient_checkpointing"):
            previous[name] = submodule.gradient_checkpointing
            submodule.gradient_checkpointing = False
    try:
        yield
    finally:
        for name, submodule in module.named_modules():
            if name in previous:
                submodule.gradient_checkpointing = previous[name]


@contextmanager
def maybe_disable_gradient_checkpointing(module, enabled: bool):
    if enabled:
        with disable_gradient_checkpointing(module):
            yield
    else:
        yield


def sample_target_index(args) -> int:
    if args.lamb_v > 0.0 and args.v_steps > 0 and args.v_early_prob > 0.0:
        early_upper = min(args.v_steps, args.max_target_step + 1)
        if early_upper > args.min_target_step and random.random() < args.v_early_prob:
            return random.randint(args.min_target_step, early_upper - 1)
    return random.randint(args.min_target_step, args.max_target_step)


def read_prompt_pairs(path: str | None) -> List[Tuple[str, str, str | None, str | None]]:
    if path is None:
        return [(source, target, None, None) for source, target in DEFAULT_PROMPT_PAIRS]
    pairs = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            pairs.append((
                str(item["source_prompt"]),
                str(item["target_prompt"]),
                str(item["source_concept"]) if item.get("source_concept") is not None else None,
                str(item["target_concept"]) if item.get("target_concept") is not None else None,
            ))
    if not pairs:
        raise ValueError("Prompt-pair JSONL is empty.")
    return pairs


def read_eval_prompts(path: str | None, fallback_prompt: str) -> List[Tuple[str, str]]:
    if path is None:
        return [("dog", fallback_prompt)]
    prompts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(item["name"])).strip("_")
            if not name:
                raise ValueError("Every evaluation prompt needs a non-empty filesystem-safe name.")
            prompts.append((name, str(item["prompt"])))
    if not prompts:
        raise ValueError("Evaluation-prompt JSONL is empty.")
    names = [name for name, _ in prompts]
    if len(names) != len(set(names)):
        raise ValueError("Evaluation-prompt names must be unique.")
    return prompts


def encode_condition(pipe, prompt: str, concept: str | None, max_length: int) -> PromptCondition:
    indices = ()
    if concept is not None:
        indices = find_concept_token_indices(
            pipe.tokenizer, prompt, concept, DEFAULT_PROMPT_TEMPLATE, max_length
        )
    with torch.no_grad():
        embeds, pooled, mask = pipe.encode_prompt(
            prompt=prompt,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            device=next(pipe.transformer.parameters()).device,
            max_sequence_length=max_length,
        )
    return PromptCondition(
        prompt=prompt,
        concept_indices=indices,
        embeds=embeds.detach().cpu(),
        pooled=pooled.detach().cpu(),
        attention_mask=mask.detach().cpu(),
    )


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


def transformer_forward(pipe, latents, timestep, condition: PromptCondition, guidance_scale: float):
    device = next(pipe.transformer.parameters()).device
    condition = condition.to(device, pipe.transformer.dtype)
    latents = latents.to(device=device)
    batch = latents.shape[0]
    return pipe.transformer(
        hidden_states=latents.to(dtype=pipe.transformer.dtype),
        timestep=timestep.expand(batch).to(device=device, dtype=latents.dtype),
        encoder_hidden_states=condition.embeds,
        encoder_attention_mask=condition.attention_mask,
        pooled_projections=condition.pooled,
        guidance=make_guidance(pipe, batch, guidance_scale),
        return_dict=False,
    )[0]


@torch.no_grad()
def student_rollout(pipe, condition, initial_latents, scheduler, target_index, guidance_scale):
    latents = initial_latents
    for t in scheduler.timesteps[:target_index]:
        pred = transformer_forward(pipe, latents, t, condition, guidance_scale)
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    return latents.detach(), scheduler.timesteps[target_index]


def save_checkpoint(pipe, output_dir, adapter_name, config, matches, step):
    output_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(pipe.transformer, adapter_name=adapter_name)
    HunyuanVideoPipeline.save_lora_weights(
        save_directory=str(output_dir),
        transformer_lora_layers=state,
    )
    metadata = dict(config)
    metadata["global_step"] = step
    metadata["matched_lora_modules"] = list(matches)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


@torch.inference_mode()
def generate_eval(pipe, controller, condition, args, adapters_enabled, output_path):
    controller.finish()
    device = next(pipe.transformer.parameters()).device
    condition = condition.to(device, pipe.transformer.dtype)
    generator = torch.Generator(device=device).manual_seed(args.eval_seed)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, args.eval_height, args.eval_width,
        args.eval_num_frames, torch.float32, device, generator
    )
    scheduler = copy.deepcopy(pipe.scheduler)
    prepare_flow_timesteps(scheduler, args.eval_inference_steps, device)
    with adapter_mode(pipe.transformer, adapters_enabled):
        for timestep in scheduler.timesteps:
            noise_pred = transformer_forward(
                pipe, latents, timestep, condition, args.guidance_scale
            )
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    pipe.vae.to(device)
    decoded = pipe.vae.decode(
        latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    frames = pipe.video_processor.postprocess_video(decoded, output_type="pil")[0]
    export_to_video(frames, str(output_path), fps=args.fps)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser(description="Train HunyuanVideo motion LoRA with CHI-selected heads.")
    parser.add_argument("--model_path", default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_pairs_jsonl", default=None)
    parser.add_argument("--source_concept", default="dog")
    parser.add_argument("--target_concept", default="cat")
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--checkpointing_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--imap_alpha", type=float, default=3.0)
    parser.add_argument(
        "--imap_mode",
        choices=("video",),
        default="video",
        help="Use motion-sensitive heads for the IMAP mask, target and H loss.",
    )
    parser.add_argument("--motion_topk_heads", type=int, default=5)
    parser.add_argument(
        "--motion_sep_score",
        choices=("Silhouette", "DBI", "CHI", "Fisher"),
        default="CHI",
    )
    parser.add_argument(
        "--clamp_mask",
        action="store_true",
        help="Use W=clamp(alpha*M,0,1). Without this flag, use unclamped W=alpha*M.",
    )
    parser.add_argument("--double_blocks", type=parse_blocks, default=DEFAULT_IMAP_DOUBLE_BLOCKS)
    parser.add_argument("--lamb_v", type=float, default=0.0)
    parser.add_argument("--v_steps", type=int, default=1)
    parser.add_argument("--v_block_min", default="20")
    parser.add_argument("--v_early_prob", type=float, default=0.5)
    parser.add_argument(
        "--v_video_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only visual/video tokens for L_V_bg. Use --no-v_video_only to include text tokens.",
    )
    parser.add_argument(
        "--v_mask_mode",
        choices=("background", "full"),
        default="background",
        help=(
            "How to weight the V preservation loss: 'background' uses (1-M_source), "
            "while 'full' uses all visual tokens with weight 1."
        ),
    )
    parser.add_argument(
        "--v_disable_gc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily disable gradient checkpointing for the student forward when L_V_bg is active.",
    )
    parser.add_argument(
        "--split_hv_backward",
        action="store_true",
        help=(
            "When L_V is active, compute L_H and L_V in two separate student forward/backward passes "
            "and accumulate gradients before one optimizer step. This reduces peak VRAM at the cost of speed."
        ),
    )
    parser.add_argument("--target_modules", default=",".join(DEFAULT_HUNYUAN_ERASE_TARGET_MODULES))
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--trajectory_steps", type=int, default=30)
    parser.add_argument("--min_target_step", type=int, default=0)
    parser.add_argument("--max_target_step", type=int, default=29)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Checkpoint Hunyuan transformer blocks during the student forward to reduce VRAM usage.",
    )
    parser.add_argument("--eval_prompt", default="a video of a dog")
    parser.add_argument(
        "--eval_prompts_jsonl",
        default=None,
        help="Optional JSONL with unique `name` and `prompt` fields. Generates every prompt at each evaluation.",
    )
    parser.add_argument("--eval_height", type=int, default=432)
    parser.add_argument("--eval_width", type=int, default=768)
    parser.add_argument("--eval_num_frames", type=int, default=49)
    parser.add_argument("--eval_inference_steps", type=int, default=30)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--skip_eval", action="store_true")
    return parser


def main(args):
    if args.imap_alpha <= 0:
        raise ValueError("imap_alpha must be positive")
    if args.motion_topk_heads <= 0:
        raise ValueError("motion_topk_heads must be positive")
    if args.lamb_v < 0:
        raise ValueError("lamb_v must be non-negative")
    if args.v_steps < 0:
        raise ValueError("v_steps must be non-negative")
    if not 0.0 <= args.v_early_prob <= 1.0:
        raise ValueError("v_early_prob must be in [0, 1]")
    if args.height % 16 or args.width % 16 or args.eval_height % 16 or args.eval_width % 16:
        raise ValueError("All heights and widths must be divisible by 16.")
    if not 0 <= args.min_target_step <= args.max_target_step < args.trajectory_steps:
        raise ValueError("Require 0 <= min_target_step <= max_target_step < trajectory_steps.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval").mkdir(exist_ok=True)

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=dtype
    )
    pipe = HunyuanVideoPipeline.from_pretrained(args.model_path, transformer=transformer, torch_dtype=dtype)
    pipe.vae.enable_tiling()
    pipe.to(device)
    for component in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        component.eval()
        for parameter in component.parameters():
            parameter.requires_grad_(False)

    raw_pairs = read_prompt_pairs(args.prompt_pairs_jsonl)
    cached_pairs = []
    for source_prompt, target_prompt, source_concept, target_concept in raw_pairs:
        source = encode_condition(
            pipe, source_prompt,
            source_concept if source_concept is not None else args.source_concept,
            args.max_sequence_length,
        )
        # An empty target concept is treated as no concept at all, which
        # allows an empty target prompt: erasure toward nothing rather than
        # toward another concept. Only source_token_indices are read
        # downstream, so the target side never needs token indices. The
        # source side keeps the strict behaviour on purpose and still fails
        # loudly when its concept cannot be located.
        target = encode_condition(
            pipe, target_prompt,
            (target_concept if target_concept is not None else args.target_concept) or None,
            args.max_sequence_length,
        )
        cached_pairs.append((source, target))
    raw_eval_prompts = read_eval_prompts(args.eval_prompts_jsonl, args.eval_prompt)
    eval_conditions = [
        (name, encode_condition(pipe, prompt, None, args.max_sequence_length))
        for name, prompt in raw_eval_prompts
    ]

    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    latent_frames = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    grid = (
        latent_frames // transformer.config.patch_size_t,
        (args.height // pipe.vae_scale_factor_spatial) // transformer.config.patch_size,
        (args.width // pipe.vae_scale_factor_spatial) // transformer.config.patch_size,
    )
    controller = IMAPHController(
        grid, args.double_blocks, args.imap_alpha, args.clamp_mask,
        lamb_v=args.lamb_v,
        v_video_only=args.v_video_only,
        v_mask_mode=args.v_mask_mode,
        imap_mode=args.imap_mode,
        motion_topk_heads=args.motion_topk_heads,
        motion_sep_score=args.motion_sep_score,
    )
    for block_index, block in enumerate(transformer.transformer_blocks):
        block.attn.set_processor(HunyuanIMAPHLossAttnProcessor(controller, block_index))
    selected_single_blocks: Tuple[int, ...] = ()
    if args.lamb_v > 0.0:
        selected_single_blocks = resolve_single_blocks(
            len(transformer.single_transformer_blocks), args.v_block_min
        )
        for block_index in selected_single_blocks:
            block = transformer.single_transformer_blocks[block_index]
            block.attn.set_processor(HunyuanIMAPValueLossAttnProcessor(controller, block_index))

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    adapter_name = "imap_h_lora"
    _, matches = add_hunyuan_erase_lora(
        transformer, rank=args.rank, alpha=args.lora_alpha,
        target_modules=target_modules, adapter_name=adapter_name
    )
    # PEFT can create newly injected adapter tensors on the CPU.  Move the
    # complete transformer once more after adapter construction so that base
    # and LoRA parameters have one unambiguous execution device.
    transformer.to(device=device, dtype=dtype)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    wrong_parameter_devices = [
        (name, str(parameter.device))
        for name, parameter in transformer.named_parameters()
        if parameter.device != device
    ]
    wrong_buffer_devices = [
        (name, str(buffer.device))
        for name, buffer in transformer.named_buffers()
        if buffer.device != device
    ]
    if wrong_parameter_devices or wrong_buffer_devices:
        preview = (wrong_parameter_devices + wrong_buffer_devices)[:10]
        raise RuntimeError(f"Transformer still spans multiple devices: {preview}")
    summary = trainable_parameter_summary(transformer)
    if summary["trainable"] == 0:
        raise RuntimeError("No trainable LoRA parameters.")
    optimizer = torch.optim.AdamW(
        iter_trainable_parameters(transformer), lr=args.learning_rate,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )

    config = vars(args).copy()
    config["double_blocks"] = list(args.double_blocks)
    config["prompt_pairs"] = raw_pairs
    config["eval_prompts"] = [
        {"name": name, "prompt": prompt} for name, prompt in raw_eval_prompts
    ]
    config["visual_grid"] = list(grid)
    weight_formula = "clamp(alpha*M_source,0,1)" if args.clamp_mask else "alpha*M_source"
    config["target_formula"] = f"H_source + W * (H_target - H_source), W={weight_formula}"
    if args.lamb_v > 0.0 and args.v_mask_mode == "background":
        config["value_loss_formula"] = (
            "L_V_bg = mean_s || (1-M_source) * "
            "(V_student_source - stopgrad(V_base_source)) ||^2"
        )
    elif args.lamb_v > 0.0 and args.v_mask_mode == "full":
        config["value_loss_formula"] = (
            "L_V_full = mean_s || V_student_source - stopgrad(V_base_source) ||^2"
        )
    else:
        config["value_loss_formula"] = "disabled"
    config["value_single_blocks"] = list(selected_single_blocks)
    config["hidden_loss_heads"] = (
        f"only the same top-{args.motion_topk_heads} source motion heads selected per double block"
    )
    config["mask_heads"] = "all attention heads; no CHI motion-head selection"
    if args.imap_mode == "video":
        config["mask_heads"] = (
            f"top-{args.motion_topk_heads} motion heads per selected double block, "
            f"selected with original IMAP select_head sep_score={args.motion_sep_score}"
        )
    config["imap_mode"] = args.imap_mode
    config["motion_topk_heads"] = args.motion_topk_heads
    config["motion_sep_score"] = args.motion_sep_score
    config["trajectory_owner"] = "student source-prompt rollout, detached"
    config["shared_zt"] = True
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Visual grid: {grid}; selected double blocks: {list(args.double_blocks)}")
    if args.imap_mode == "video":
        print(
            "Video/motion mask: "
            f"top-{args.motion_topk_heads} heads per layer via original IMAP select_head "
            f"({args.motion_sep_score})"
        )
    else:
        print("Object mask: all attention heads; no CHI motion-head selection")
    print("H loss: restricted to the same selected motion heads")
    print(f"IMAP target: H_source + W * (H_target - H_source), W={weight_formula}, alpha={args.imap_alpha}")
    if args.lamb_v > 0.0:
        print(
            "V loss: "
            f"lambda={args.lamb_v}, active when target_index < {args.v_steps}, "
            f"single blocks={list(selected_single_blocks)}, video_only={args.v_video_only}, "
            f"mask_mode={args.v_mask_mode}"
        )
    else:
        print("Background V loss: disabled")
    print(f"LoRA modules: {len(matches)}; trainable parameters: {summary['trainable']}")
    print(f"Gradient checkpointing: {transformer.is_gradient_checkpointing}")

    if not args.skip_eval:
        for eval_name, eval_condition in eval_conditions:
            generate_eval(
                pipe, controller, eval_condition, args, False,
                output_dir / "eval" / f"baseline_{eval_name}.mp4"
            )

    transformer.train()
    log_path = output_dir / "loss.jsonl"
    progress = tqdm(range(1, args.max_train_steps + 1), desc="imap-h-lora")
    for step in progress:
        pair_index = random.randrange(len(cached_pairs))
        source_cpu, target_cpu = cached_pairs[pair_index]
        source = source_cpu.to(device, dtype)
        target = target_cpu.to(device, dtype)
        target_index = sample_target_index(args)
        capture_value = args.lamb_v > 0.0 and target_index < args.v_steps

        scheduler = copy.deepcopy(pipe.scheduler)
        prepare_flow_timesteps(scheduler, args.trajectory_steps, device)
        generator = torch.Generator(device=device).manual_seed(args.seed + step)
        initial = pipe.prepare_latents(
            1, transformer.config.in_channels, args.height, args.width,
            args.num_frames, torch.float32, device, generator
        )
        controller.finish()
        with adapter_mode(transformer, True), torch.no_grad():
            z_t, timestep = student_rollout(
                pipe, source, initial, scheduler, target_index, args.guidance_scale
            )

        controller.begin_teacher_source(source.concept_indices, capture_value=capture_value)
        with adapter_mode(transformer, False), torch.no_grad():
            transformer_forward(pipe, z_t, timestep, source, args.guidance_scale)

        missing_source = controller.blocks.difference(controller.source_states).union(
            controller.blocks.difference(controller.source_weights)
        )
        if missing_source:
            raise RuntimeError(f"Missing teacher source H/weight captures for blocks: {sorted(missing_source)}")
        if capture_value:
            missing_values = set(selected_single_blocks).difference(controller.teacher_values)
            if missing_values:
                raise RuntimeError(f"Missing teacher source V captures for single blocks: {sorted(missing_values)}")

        controller.begin_teacher_target(capture_value=capture_value)
        with adapter_mode(transformer, False), torch.no_grad():
            transformer_forward(pipe, z_t, timestep, target, args.guidance_scale)

        missing_targets = controller.blocks.difference(controller.teacher_targets)
        if missing_targets:
            raise RuntimeError(f"Missing teacher H targets for blocks: {sorted(missing_targets)}")

        optimizer.zero_grad(set_to_none=True)
        if args.split_hv_backward and capture_value:
            controller.begin_student(capture_value=False, capture_h=True)
            with adapter_mode(transformer, True):
                transformer_forward(pipe, z_t, timestep, source, args.guidance_scale)
            missing_h_losses = controller.blocks.difference(controller.h_loss_blocks)
            if missing_h_losses:
                raise RuntimeError(f"Missing online student H losses for blocks: {sorted(missing_h_losses)}")
            loss_h = controller.total_h_loss()
            if loss_h is None:
                raise RuntimeError("No online L_H losses were captured.")
            if not torch.isfinite(loss_h):
                raise FloatingPointError(f"Non-finite L_H at step {step}: {loss_h}")
            loss_h_for_log = loss_h.detach()
            loss_h.backward()

            controller.teacher_targets.clear()
            controller.h_losses.clear()
            controller.h_loss_blocks.clear()
            torch.cuda.empty_cache()

            controller.begin_student(capture_value=True, capture_h=False)
            with adapter_mode(transformer, True):
                with maybe_disable_gradient_checkpointing(
                    transformer, args.v_disable_gc and args.gradient_checkpointing
                ):
                    transformer_forward(pipe, z_t, timestep, source, args.guidance_scale)
            loss_v = controller.total_value_loss()
            if len(controller.value_losses) != len(selected_single_blocks):
                raise RuntimeError(
                    f"Expected {len(selected_single_blocks)} L_V block losses, got {len(controller.value_losses)}"
                )
            if loss_v is None:
                raise RuntimeError("No L_V losses were captured in split H/V mode.")
            if not torch.isfinite(loss_v):
                raise FloatingPointError(f"Non-finite L_V_bg at step {step}: {loss_v}")
            loss_v_for_log = loss_v.detach()
            (args.lamb_v * loss_v).backward()
            loss = loss_h_for_log + args.lamb_v * loss_v_for_log
        else:
            controller.begin_student(capture_value=capture_value, capture_h=True)
            with adapter_mode(transformer, True):
                with maybe_disable_gradient_checkpointing(
                    transformer, capture_value and args.v_disable_gc and args.gradient_checkpointing
                ):
                    transformer_forward(pipe, z_t, timestep, source, args.guidance_scale)
            missing_h_losses = controller.blocks.difference(controller.h_loss_blocks)
            if missing_h_losses:
                raise RuntimeError(f"Missing online student H losses for blocks: {sorted(missing_h_losses)}")
            loss_h = controller.total_h_loss()
            if loss_h is None:
                raise RuntimeError("No online L_H losses were captured.")
            loss_v = controller.total_value_loss() if capture_value else None
            if capture_value and len(controller.value_losses) != len(selected_single_blocks):
                raise RuntimeError(
                    f"Expected {len(selected_single_blocks)} L_V block losses, got {len(controller.value_losses)}"
                )
            if loss_v is None:
                loss_v_for_total = loss_h.new_zeros(())
            else:
                loss_v_for_total = loss_v
            loss = loss_h + args.lamb_v * loss_v_for_total
            if not torch.isfinite(loss_h):
                raise FloatingPointError(f"Non-finite L_H at step {step}: {loss_h}")
            if loss_v is not None and not torch.isfinite(loss_v):
                raise FloatingPointError(f"Non-finite L_V_bg at step {step}: {loss_v}")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite total loss at step {step}: {loss}")
            loss_h_for_log = loss_h.detach()
            loss_v_for_log = loss_v.detach() if loss_v is not None else loss_h.new_zeros(()).detach()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(list(iter_trainable_parameters(transformer)), args.max_grad_norm)
        optimizer.step()
        controller.finish()

        record = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "loss_v_bg": float(loss_v_for_log.detach().cpu()),
            "loss_h": float(loss_h_for_log.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "target_step_index": target_index,
            "timestep": float(timestep.detach().cpu()),
            "prompt_pair_index": pair_index,
            "source_prompt": source.prompt,
            "target_prompt": target.prompt,
            "mean_imap_mask": float(np.mean(list(controller.mask_means.values()))),
            "mean_effective_weight": float(np.mean(list(controller.weight_means.values()))),
            "max_effective_weight": float(max(controller.weight_maxima.values())),
            "mean_background_mask": (
                float(np.mean(list(controller.background_mask_means.values())))
                if controller.background_mask_means else 0.0
            ),
            "value_active": capture_value,
            "value_blocks": len(controller.value_losses),
            "lamb_v": args.lamb_v,
            "v_steps": args.v_steps,
            "v_mask_mode": args.v_mask_mode,
            "split_hv_backward": args.split_hv_backward,
            "imap_mode": args.imap_mode,
            "motion_topk_heads": args.motion_topk_heads,
            "motion_selected_head_counts": (
                {
                    str(index): count
                    for index, count in sorted(controller.motion_head_counts.items())
                }
                if controller.motion_head_counts else {}
            ),
            "clamp_mask": args.clamp_mask,
            "imap_alpha": args.imap_alpha,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        progress.set_postfix(
            loss=f"{record['loss_total']:.5f}",
            h=f"{record['loss_h']:.5f}",
            v=f"{record['loss_v_bg']:.5f}",
            t=target_index,
            pair=pair_index,
        )

        del loss, z_t, initial, source, target
        controller.source_states.clear()
        controller.source_weights.clear()
        controller.teacher_targets.clear()
        controller.h_losses.clear()
        controller.h_loss_blocks.clear()
        controller.teacher_values.clear()
        controller.value_losses.clear()
        controller.clear_source_masks()
        torch.cuda.empty_cache()

        if args.checkpointing_steps > 0 and step % args.checkpointing_steps == 0:
            save_checkpoint(
                pipe, output_dir / f"checkpoint-{step:06d}", adapter_name,
                config, matches, step
            )
        if not args.skip_eval and args.eval_steps > 0 and step % args.eval_steps == 0:
            transformer.eval()
            for eval_name, eval_condition in eval_conditions:
                generate_eval(
                    pipe, controller, eval_condition, args, True,
                    output_dir / "eval" / f"student_step_{step:06d}_{eval_name}.mp4"
                )
            transformer.train()

    transformer.eval()
    save_checkpoint(pipe, output_dir, adapter_name, config, matches, args.max_train_steps)
    print(f"Saved final IMAP H LoRA to {output_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
