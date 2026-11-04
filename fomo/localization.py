"""Where in the video the erased concept lives.

One function and one constant, extracted from the diagnostic tool this started
as. The tool itself drew heat-maps and is gone; what remains is the mask the
training loop actually calls.

The mask answers a question the usual cross-modal attention map does not: not
"how strongly does this token attend to the concept" but "how much does this
token look like the token that attends to it most". The first lights up the few
discriminative patches; the second spreads from those to the whole object,
because similarity in representation space is what holds an object together.
"""

from __future__ import annotations

from typing import Tuple

import torch

# Double-stream blocks the alignment loss is applied to. Chosen once by
# sweeping blocks individually; the gaps are blocks whose hidden states barely
# moved under the concept and therefore carried no signal.
DEFAULT_IMAP_DOUBLE_BLOCKS = (1, 2, 4, 5, 6, 9, 10, 13, 15, 16, 17, 18, 19)


def gramcol_soft_mask(
    visual_hidden_states: torch.Tensor,
    image_query: torch.Tensor,
    text_key: torch.Tensor,
    grid: Tuple[int, int, int],
) -> torch.Tensor:
    """Return an all-head mask in [0, 1] with shape [frames, height, width].

    `text_key` is the mean key of the concept tokens only, not of the whole
    prompt: that restriction is what makes the anchor land on the concept
    rather than on whatever the sentence happens to emphasise.

    The Gram column is computed directly rather than by forming the full Gram
    matrix. At 3600 positions per frame the matrix would be thirteen million
    entries per head per frame, and only one of its rows is ever read.
    """
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

    # Anchor: the position whose query matches the concept key best. Chosen per
    # head and per frame, so a 24-head model on 5 frames picks 120 anchors that
    # are only averaged together at the end.
    qk_scores = torch.einsum("hfpd,hd->hfp", query_by_frame.float(), text_key.float())
    qk_scores = qk_scores / (float(head_dim) ** 0.5)
    anchor_indices = qk_scores.argmax(dim=-1)

    gather_indices = anchor_indices[..., None, None].expand(-1, -1, 1, head_dim)
    anchors = hidden_by_frame.gather(dim=2, index=gather_indices).squeeze(2)
    gramcol = torch.einsum("hfd,hfpd->hfp", anchors.float(), hidden_by_frame.float())

    # Normalised per head and per frame, so every frame has a maximum of one.
    # That is deliberate for a concept present throughout the clip, and is the
    # reason a concept that leaves the frame still produces a mask.
    minimum = gramcol.amin(dim=-1, keepdim=True)
    maximum = gramcol.amax(dim=-1, keepdim=True)
    normalized = (gramcol - minimum) / (maximum - minimum).clamp_min(1e-6)
    mask = normalized.mean(dim=0)
    return mask.reshape(frames, grid_height, grid_width).clamp_(0.0, 1.0)
