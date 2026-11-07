"""NegPrompt: the untouched model with the erased concept as a negative prompt.

No weights and no code of its own beyond the sampling arguments below, which is
why this file is short. It is here so that looking for NegPrompt finds
something.

The two backbones need different handling. HunyuanVideo is CFG-distilled:
`guidance_scale` feeds a guidance embedding rather than running two-branch
guidance, so a negative prompt has nowhere to go. Real CFG has to be switched
on explicitly, which also means that row is sampled differently from the
baseline it is compared against. CogVideoX already runs two-branch CFG, so the
negative prompt takes effect without changing the sampler and no CFG-matched
control is needed.
"""


def sampling_kwargs(concept, guidance_scale, model="hunyuan"):
    """Pipeline arguments that negate `concept` on the given backbone."""
    if model == "hunyuan":
        return {
            "negative_prompt": concept,
            "true_cfg_scale": guidance_scale,
            "guidance_scale": 1.0,
        }
    return {
        "negative_prompt": concept,
        "guidance_scale": guidance_scale,
    }
