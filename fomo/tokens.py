from __future__ import annotations

from typing import Dict, List, Sequence


def _clean_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def find_concept_token_indices(
    tokenizer,
    prompt: str,
    concept: str,
    prompt_template: Dict,
    max_sequence_length: int = 256,
) -> List[int]:
    crop_start = prompt_template.get("crop_start", 0)
    max_length = max_sequence_length + crop_start
    formatted_prompt = prompt_template["template"].format(prompt)
    tokenized = tokenizer(
        formatted_prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    ids = tokenized.input_ids[0].tolist()
    mask = tokenized.attention_mask[0].tolist()
    cropped_ids = ids[crop_start:]
    cropped_mask = mask[crop_start:]
    valid_len = sum(int(v) for v in cropped_mask)
    valid_ids = cropped_ids[:valid_len]

    target = _clean_text(concept)
    best_match: Sequence[int] | None = None

    max_span = min(8, len(valid_ids))
    for span in range(1, max_span + 1):
        for start in range(0, len(valid_ids) - span + 1):
            decoded = tokenizer.decode(valid_ids[start : start + span], skip_special_tokens=True)
            if _clean_text(decoded) == target:
                best_match = range(start, start + span)
                break
        if best_match is not None:
            break

    if best_match is None:
        hits = []
        for index, token_id in enumerate(valid_ids):
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            if target in _clean_text(decoded):
                hits.append(index)
        best_match = hits

    indices = list(best_match or [])
    if not indices:
        preview = tokenizer.decode(valid_ids, skip_special_tokens=True)
        raise ValueError(
            f"Could not find concept '{concept}' in tokenized prompt. "
            f"Prompt after template crop decodes as: {preview!r}"
        )
    return indices


def describe_token_window(tokenizer, prompt: str, indices: Sequence[int], prompt_template: Dict, max_sequence_length: int = 256) -> str:
    crop_start = prompt_template.get("crop_start", 0)
    max_length = max_sequence_length + crop_start
    formatted_prompt = prompt_template["template"].format(prompt)
    tokenized = tokenizer(
        formatted_prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    ids = tokenized.input_ids[0].tolist()[crop_start:]
    start = max(0, min(indices) - 4)
    end = min(len(ids), max(indices) + 5)
    pieces = []
    for offset, token_id in enumerate(ids[start:end], start=start):
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        marker = "*" if offset in indices else " "
        pieces.append(f"{marker}{offset}:{piece!r}")
    return " ".join(pieces)

