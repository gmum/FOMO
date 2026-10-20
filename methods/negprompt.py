"""NegPrompt: the untouched model with the erased concept as a negative prompt.

No weights and no code of its own beyond the sampling arguments below, which is
why this file is short. It is here so that looking for NegPrompt finds
something.

HunyuanVideo is CFG-distilled: `guidance_scale` feeds a guidance embedding
rather than running two-branch guidance, so a negative prompt has nowhere to
go. Real CFG has to be switched on explicitly -- `guidance_scale=1.0` with
`true_cfg_scale` above it -- which also means this row is sampled differently
from the baseline it is compared against. That follows from what is being
reproduced and is not a choice made here.
"""


def sampling_kwargs(concept, guidance_scale):
    """Pipeline arguments that negate `concept` under real two-branch CFG."""
    return {
        "negative_prompt": concept,
        "true_cfg_scale": guidance_scale,
        "guidance_scale": 1.0,
    }
