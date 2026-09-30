"""
Przechwytywanie tensorów V (value) z single blocks HunyuanVideo na potrzeby L_V.

Idea: te same forwardy, które i tak wykonuje L_ESD (nauczyciel z wyłączoną LoRA
oraz student z włączoną), obudowujemy procesorami uwagi, które wyciągają V
z wybranych single blocks. Nauczyciel zapisuje V (odpięte od grafu), student
od razu liczy MSE względem zapisanego V i akumuluje skalary.

Dlaczego liczymy stratę od razu w procesorze, zamiast zbierać wszystkie V
i porównywać na końcu: przy 768x432 i 33 klatkach jeden tensor V ma ~73 MB,
a bloków jest ~19. Trzymanie dwóch kompletów naraz to ~3 GB, podczas gdy
akumulacja skalarów kosztuje tyle, ile i tak zajmuje graf autograd.

Plik jest samodzielny — nie modyfikuje niczego w repo.
Docelowa lokalizacja: receler/training/hunyuan_value_capture.py
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class ValueLossRecorder:
    """Zbiera V od nauczyciela i liczy MSE dla studenta.

    Tryby:
        "off"     - procesory działają jak zwykle, nic nie zapisują
        "teacher" - zapisuje V (detach) dla każdego obserwowanego bloku
        "student" - porównuje bieżące V z zapisanym i akumuluje MSE
    """

    mode: str = "off"
    video_only: bool = True
    teacher_values: Dict[int, torch.Tensor] = field(default_factory=dict)
    losses: List[torch.Tensor] = field(default_factory=list)

    def begin_teacher(self) -> None:
        self.mode = "teacher"
        self.teacher_values.clear()
        self.losses.clear()

    def begin_student(self) -> None:
        self.mode = "student"
        self.losses.clear()

    def finish(self) -> None:
        self.mode = "off"

    def reset(self) -> None:
        self.mode = "off"
        self.teacher_values.clear()
        self.losses.clear()

    def num_blocks_captured(self) -> int:
        return len(self.teacher_values)

    def total(self) -> Optional[torch.Tensor]:
        """Średnia MSE po obserwowanych blokach. None, jeśli nic nie zebrano."""
        if not self.losses:
            return None
        return torch.stack(self.losses).mean()


class HunyuanVideoValueCaptureProcessor:
    """Procesor uwagi HunyuanVideo z przechwytywaniem V.

    Odtwarza zachowanie HunyuanVideoAttnProcessor2_0 (kolejność operacji jest
    taka sama jak w HunyuanVideoVBankAttnProcessor2_0 z test_hunyuan_vbank.py),
    dokładając wyłącznie zapis/porównanie tensora value.
    """

    def __init__(self, recorder: ValueLossRecorder, block_index: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Wymagany torch.nn.functional.scaled_dot_product_attention (PyTorch >= 2.0).")
        self.recorder = recorder
        self.block_index = block_index

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        text_seq_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0

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

        # --- w double blocks doklejane są projekcje tekstowe; single blocks ich nie mają
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

        # ----------------------- przechwycenie V -----------------------------
        self._capture_value(value, text_seq_length)
        # ---------------------------------------------------------------------

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

    def _capture_value(self, value: torch.Tensor, text_seq_length: int) -> None:
        mode = self.recorder.mode
        if mode == "off":
            return

        captured = value
        if self.recorder.video_only and text_seq_length > 0:
            # w single blocks kolejność to [tokeny wideo, tokeny tekstu]
            captured = captured[:, :, :-text_seq_length]

        if mode == "teacher":
            self.recorder.teacher_values[self.block_index] = captured.detach()
        elif mode == "student":
            reference = self.recorder.teacher_values.get(self.block_index)
            if reference is None:
                return
            if reference.shape != captured.shape:
                raise ValueError(
                    f"Niezgodny kształt V w bloku {self.block_index}: "
                    f"nauczyciel={tuple(reference.shape)} student={tuple(captured.shape)}"
                )
            self.recorder.losses.append(F.mse_loss(captured.float(), reference.float()))


def resolve_value_block_indices(num_blocks: int, block_min: int | str | Iterable[int]) -> List[int]:
    """Zamienia specyfikację bloków na listę indeksów.

    Akceptuje:
        int          -> bloki o indeksie WIĘKSZYM niż podany (semantyka block_min
                        z test_hunyuan_vbank.py)
        "all"        -> wszystkie
        "21,25,30"   -> jawna lista
        iterowalne   -> jawna lista
    """
    if isinstance(block_min, str):
        if block_min.strip().lower() == "all":
            return list(range(num_blocks))
        if "," in block_min:
            return [int(item.strip()) for item in block_min.split(",") if item.strip()]
        block_min = int(block_min)

    if isinstance(block_min, int):
        indices = [index for index in range(num_blocks) if index > block_min]
    else:
        indices = [int(item) for item in block_min]

    for index in indices:
        if index < 0 or index >= num_blocks:
            raise IndexError(f"Indeks single block {index} poza zakresem 0..{num_blocks - 1}")
    return indices


@contextmanager
def capture_hunyuan_single_block_values(
    transformer,
    recorder: ValueLossRecorder,
    block_min: int | str | Iterable[int] = 20,
):
    """Instaluje procesory przechwytujące V na wybranych single blocks.

    Po wyjściu z bloku `with` przywraca oryginalne procesory — także wtedy,
    gdy w środku poleci wyjątek.
    """
    blocks = getattr(transformer, "single_transformer_blocks", None)
    if blocks is None:
        raise AttributeError("Transformer HunyuanVideo nie ma single_transformer_blocks.")

    indices = resolve_value_block_indices(len(blocks), block_min)
    originals = []
    try:
        for index in indices:
            attn = blocks[index].attn
            originals.append((attn, attn.get_processor()))
            attn.set_processor(HunyuanVideoValueCaptureProcessor(recorder, index))
        yield recorder
    finally:
        for attn, processor in originals:
            attn.set_processor(processor)
        recorder.finish()