"""The three helpers the CogVideoX trainer needs from the diagnostic tool.

Extracted from `imap/cogvideox_hidden_patch.py`, which is a six-hundred-line
heat-map viewer the training loop never calls. Only these three are imported,
so only these three are here, unchanged.

The CogVideoX counterpart of `fomo/tokens.py`: the two backbones tokenise and
lay out their latents differently, so the lookup has to be written per model.
"""

from typing import List, Tuple

import torch
from diffusers import CogVideoXPipeline


def layer_group_indices(num_layers: int, group: str) -> List[int]:
    group = group.lower()
    if group == "all":
        return list(range(num_layers))
    first = num_layers // 3
    second = (2 * num_layers) // 3
    if group in {"early", "ear"}:
        return list(range(0, first))
    if group in {"middle", "mid", "idle"}:
        return list(range(first, second))
    if group == "late":
        return list(range(second, num_layers))
    raise ValueError(f"Unknown layer group {group!r}; use early,middle,late,all")

def find_concept_token_indices(tokenizer, prompt: str, concept: str, max_length: int) -> List[int]:
    """Find token indices for concept in CogVideoX/T5 prompt tokens.

    This first tries exact sub-sequence matching on token IDs. If that fails, it
    falls back to decoded token pieces containing the concept string.
    """
    prompt_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids[0].tolist()
    concept_ids = tokenizer(
        concept,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist()

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    usable_len = len(prompt_ids)
    if pad_id is not None and pad_id in prompt_ids:
        usable_len = min(usable_len, prompt_ids.index(pad_id))
    if eos_id is not None and eos_id in prompt_ids:
        usable_len = min(usable_len, prompt_ids.index(eos_id) + 1)

    hits: List[int] = []
    if concept_ids:
        for i in range(max(0, usable_len - len(concept_ids) + 1)):
            if prompt_ids[i : i + len(concept_ids)] == concept_ids:
                hits.extend(range(i, i + len(concept_ids)))
        if hits:
            return sorted(set(hits))

    concept_norm = concept.lower().replace(" ", "")
    for i in range(usable_len):
        piece = tokenizer.decode([prompt_ids[i]]).lower().replace(" ", "").replace("▁", "")
        if concept_norm and concept_norm in piece:
            hits.append(i)
    if hits:
        return sorted(set(hits))

    raise ValueError(
        f"Could not find concept {concept!r} in prompt {prompt!r}. "
        "Pass a simpler concept string matching tokenizer pieces."
    )

def frames_from_latents_shape(pipe: CogVideoXPipeline, latents: torch.Tensor) -> Tuple[int, int, int]:
    """Return token grid (frames, height, width) after CogVideoX patch embedding."""
    _, latent_frames, _, latent_h, latent_w = latents.shape
    p = int(pipe.transformer.config.patch_size)
    p_t = pipe.transformer.config.patch_size_t
    grid_h = latent_h // p
    grid_w = latent_w // p
    if p_t is None:
        grid_f = latent_frames
    else:
        grid_f = (latent_frames + int(p_t) - 1) // int(p_t)
    return int(grid_f), int(grid_h), int(grid_w)
