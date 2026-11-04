from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, List

import torch
import torch.nn.functional as F

from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0


@dataclass
class AttentionLossRecorder:
    concept_token_indices: List[int]
    chunk_size: int = 256
    losses: List[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.losses.clear()

    def add(self, loss: torch.Tensor) -> None:
        self.losses.append(loss)

    def total(self) -> torch.Tensor:
        if not self.losses:
            raise RuntimeError("No attention losses were captured.")
        return torch.stack(self.losses).mean()


class HunyuanVideoAttnCaptureProcessor(HunyuanVideoAttnProcessor2_0):
    def __init__(self, recorder: AttentionLossRecorder):
        super().__init__()
        self.recorder = recorder

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
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
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        latent_length = query.shape[2]

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

            self._capture_video_to_text_loss(query[:, :, :latent_length], key, attention_mask, latent_length)

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

    def _capture_video_to_text_loss(
        self,
        query_video: torch.Tensor,
        key_all: torch.Tensor,
        attention_mask: torch.Tensor | None,
        latent_length: int,
    ) -> None:
        concept_key_indices = [latent_length + idx for idx in self.recorder.concept_token_indices]
        if not concept_key_indices:
            return

        scale = 1.0 / math.sqrt(query_video.shape[-1])
        square_sum = query_video.new_tensor(0.0, dtype=torch.float32)
        chunk_size = max(1, self.recorder.chunk_size)
        key_all_float = key_all.float()
        mask = None
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool, device=query_video.device)

        for start in range(0, query_video.shape[2], chunk_size):
            q_chunk = query_video[:, :, start : start + chunk_size].float()
            scores = torch.matmul(q_chunk, key_all_float.transpose(-1, -2)) * scale
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            concept_probs = probs.index_select(
                dim=-1,
                index=torch.tensor(concept_key_indices, device=probs.device, dtype=torch.long),
            )
            square_sum = square_sum + concept_probs.pow(2).sum()

        self.recorder.add(torch.sqrt(square_sum + 1e-12))


def _normalize_block_indices(num_blocks: int, block_indices: Iterable[int] | str) -> List[int]:
    if block_indices == "last":
        return [num_blocks - 1]
    if isinstance(block_indices, str):
        return [int(item.strip()) for item in block_indices.split(",") if item.strip()]
    return [int(item) for item in block_indices]


@contextmanager
def capture_hunyuan_double_block_attention(
    transformer,
    concept_token_indices: List[int],
    block_indices: Iterable[int] | str = "last",
    chunk_size: int = 256,
):
    blocks = transformer.transformer_blocks
    indices = _normalize_block_indices(len(blocks), block_indices)
    recorder = AttentionLossRecorder(concept_token_indices=list(concept_token_indices), chunk_size=chunk_size)
    originals = []
    try:
        for index in indices:
            if index < 0 or index >= len(blocks):
                raise IndexError(f"Hunyuan double block index {index} is out of range 0..{len(blocks) - 1}")
            attn = blocks[index].attn
            originals.append((attn, attn.get_processor()))
            attn.set_processor(HunyuanVideoAttnCaptureProcessor(recorder))
        yield recorder
    finally:
        for attn, processor in originals:
            attn.set_processor(processor)

