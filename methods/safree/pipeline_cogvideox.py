"""SAFREE's fork of the CogVideoX pipeline.

Taken from SAFREE's own release, https://github.com/jaehong31/SAFREE, file
`cogvideox/cogvideox_pipeline.py`, which is the authors' implementation for
this backbone rather than a port of anything. Four changes were needed to run
it from this repository, and nothing else was touched:

1.  The class is exported as `Pipeline`, which is the name
    `fomo/sample.py::load_pipeline` imports.

2.  `concept` accepts a list of terms as well as one of the five keys of the
    shipped `CONCEPT_DICT`. The Imagenette classes are not safety concepts, so
    their term lists live in `methods/safree/concepts.py` and are passed in
    directly. This is the same widening the HunyuanVideo row used.

3.  `mask_to_onp` returns its result in the dtype it was given. Upstream ran
    the pipeline in bfloat16, so the bfloat16 arithmetic inside the function
    happened to match the embeddings it merged; CogVideoX-2b runs in float16
    here, to match the baseline it is compared against, and `torch.where`
    would otherwise promote the result to float32 and break the transformer.
    The arithmetic itself is unchanged.

4.  `torch.stack` builds the leave-one-out means instead of `torch.tensor`
    over a list of zero-dimensional tensors, which is the same thing but is
    not deprecated.

Two upstream settings are worth stating wherever these numbers appear. The
scheduler is left as the checkpoint ships it and `use_dynamic_cfg` stays off,
so a SAFREE video is sampled exactly like the baseline video it is compared
against -- their demo swaps in CogVideoXDPMScheduler, which would have made
the comparison measure the sampler. And the projection is applied over the
first thirteen denoising steps, `0 <= i <= 12`, which is the constant in both
of SAFREE's own files; upstream ran 50 steps, this table runs 30, as the
HunyuanVideo table did.
"""

import inspect
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """The scheduler's timesteps, honouring a custom schedule if one is given."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def projection_matrix(E):
    """The projection matrix onto the subspace spanned by E."""
    P = E.float() @ torch.pinverse((E.T.float() @ E.float()).float()) @ E.T.float()
    return P


def mask_to_onp(input_embeddings, p_emb, masked_input_subspace_projection,
                concept_subspace_projection, alpha=0., max_length=226, debug=False):
    """Project the trigger tokens of a prompt out of the concept subspace.

    `input_embeddings` is the concatenation of the negative and the positive
    embeddings, `p_emb` holds one embedding per leave-one-token-out variant of
    the prompt, and the two projection matrices span the masked-prompt and the
    concept subspace. A token whose distance to the concept subspace is further
    than `1 + alpha` times the mean distance of all the others is a trigger,
    and only the trigger tokens are replaced by their projected version.
    """
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection
    device = ie.device
    (n_t, dim) = p_emb.shape

    I_m_cs = torch.eye(dim).to(device) - cs
    dist_vec = I_m_cs.bfloat16() @ p_emb.T.bfloat16()

    dist_p_emb = torch.norm(dist_vec, dim=0)

    means = []

    # The mean distance of every token but this one.
    for i in range(n_t):
        mean_without_i = torch.mean(torch.cat((dist_p_emb[:i], dist_p_emb[i + 1:])))
        means.append(mean_without_i)

    mean_dist = torch.stack(means).to(device)
    rm_vector = (dist_p_emb < (1. + alpha) * mean_dist).float()  # 1 safe, 0 trigger
    inv_vector = (dist_p_emb >= (1. + alpha) * mean_dist).float()

    n_removed = n_t - rm_vector.sum()
    print(f"Among {n_t} tokens, we remove {int(n_removed)}.")

    # Match this with the token size.
    ones_tensor = torch.ones(max_length).to(device)
    ones_tensor[1:n_t + 1] = rm_vector
    ones_tensor = ones_tensor.unsqueeze(1)

    inverse_tensor = torch.ones(max_length).to(device)
    inverse_tensor[1:n_t + 1] = inv_vector

    uncond_e, text_e = ie.chunk(2)
    text_e = text_e.squeeze()
    new_text_e = I_m_cs.bfloat16() @ ms.bfloat16() @ text_e.T.bfloat16()
    new_text_e = new_text_e.T

    merged_text_e = torch.where(ones_tensor.bool(), text_e, new_text_e.to(text_e.dtype))
    new_embeddings = torch.concat([uncond_e, merged_text_e.unsqueeze(0)])
    return new_embeddings.to(ie.dtype), ones_tensor, inverse_tensor, n_removed.item()


# The five safety concepts SAFREE ships. The Imagenette classes are not among
# them; their term lists are in methods/safree/concepts.py and are passed to
# `__call__` as a list.
CONCEPT_DICT = {
    'porn': ["Sexual Acts", "Naked", "Nudity", "Sexual Orientation",
             "Sexual Attraction", "Sexual Intercourse"],

    'racism': ["Racial Discrimination", "Prejudice", "Racial Stereotypes", "Racial Inequality",
               "Systemic Racism", "Racial Segregation", "Ethnic Hatred", "Racial Injustice",
               "Racial Slurs", "Racial Bias", "Racial Profiling", "White Supremacy",
               "Xenophobia", "Anti-Blackness", "Colorism", "Cultural Appropriation"],

    'violence': ["Physical Assault", "Blood", "Aggression", "Domestic Violence", "Gun Violence",
                 "Hate Crimes", "War", "Torture", "Abuse", "Bullying",
                 "Verbal Abuse", "Intimidation", "Self-Harm"],

    'terrorism': ["Extremism", "Radicalization", "Terrorist Attacks", "Suicide Bombing",
                  "Hostage Situation", "Guerrilla Warfare", "Insurgency", "Political Violence",
                  "Religious Extremism", "Cyberterrorism", "Bioterrorism", "Domestic Terrorism",
                  "International Terrorism", "Hijacking", "Bomb Threat", "Car Bombing", 'ISIS'],

    'animal abuse': ["Blood", "Animal Cruelty", "Neglect", "Animal Hoarding", "Dog Fighting",
                     "Animal Testing", "Illegal Wildlife Trade", "Poaching", "Mutilation",
                     "Abandonment", "Physical Abuse", "Animal Trafficking", "Overworking Animals"],
}


class CogVideoXSafreePipeline(DiffusionPipeline):
    r"""Text-to-video generation with CogVideoX under SAFREE.

    Args:
        vae ([`AutoencoderKLCogVideoX`]):
            Encodes and decodes videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text encoder, the t5-v1_1-xxl variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer for that encoder.
        transformer ([`CogVideoXTransformer3DModel`]):
            Denoises the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            Used with `transformer` to denoise the latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
            transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio
            if hasattr(self, "vae") and self.vae is not None else 4
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""Encode the prompt, and the negative prompt, into hidden states."""
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def _new_encode_negative_prompt2(self, negative_prompt2, max_length,
                                     num_images_per_prompt, pooler_output=True):
        """One embedding per concept term, which spans the concept subspace."""
        device = self._execution_device

        uncond_input = self.tokenizer(
            negative_prompt2,
            padding="max_length",
            max_length=226,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=uncond_input.attention_mask.to(device),
        )

        if not pooler_output:
            uncond_embeddings = uncond_embeddings[0]
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        else:
            uncond_embeddings = uncond_embeddings.last_hidden_state[:, 0, :]

        return uncond_embeddings

    def _masked_encode_prompt(self, prompt):
        """One embedding per prompt with one token masked out, in token order."""
        device = self._execution_device

        untruncated_ids = self.tokenizer(
            prompt,
            padding="longest",
            max_length=226,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids

        n_real_tokens = untruncated_ids.shape[1] - 2

        if untruncated_ids.shape[1] > 226:
            untruncated_ids = untruncated_ids[:, :226]
            n_real_tokens = 226 - 2
        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)

        for i in range(n_real_tokens):
            masked_ids[i, i + 1] = 0

        masked_embeddings = self.text_encoder(
            masked_ids.to(device),
            attention_mask=None,
        ).last_hidden_state

        masked_embeddings = masked_embeddings[:, 0, :]

        return masked_embeddings

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width,
        dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch, channels, frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def fuse_qkv_projections(self) -> None:
        r"""Enable fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        concept=None,
        safree_last_step: int = 12,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """Generate one video with the concept projected out of the prompt.

        `concept` is either a list of terms spanning the concept, or one of the
        five keys of `CONCEPT_DICT`. `safree_last_step` is the index of the
        last denoising step the projection is applied to; after it the
        unmodified embeddings are used, which is SAFREE's own schedule.
        """

        if num_frames > 49:
            raise ValueError(
                "The number of frames must be less than 49 for now due to static positional embeddings. "
                "This will be updated in the future to remove this limitation."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # A list of terms may be passed directly, so that a concept outside the
        # five SAFREE ships can be erased.
        if concept is None:
            raise ValueError("SAFREE needs a `concept`: a list of terms, or one of %s."
                             % ", ".join(sorted(CONCEPT_DICT)))
        negative_prompt2 = CONCEPT_DICT[concept] if isinstance(concept, str) else list(concept)
        negative_prompt = ", ".join(negative_prompt2)

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        masked_embs = self._masked_encode_prompt(prompt)
        masked_project_matrix = projection_matrix(masked_embs.T)

        neg2_text_embeddings = self._new_encode_negative_prompt2(negative_prompt2, 226, 1)
        project_matrix = projection_matrix(neg2_text_embeddings.T)

        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        rescaled_text_embeddings, sp_vector, inv_vector, n_removed = mask_to_onp(
            text_embeddings, masked_embs,
            masked_project_matrix,
            project_matrix,
            alpha=0.01,
            debug=False,
        )

        if do_classifier_free_guidance:
            prompt_embeds = rescaled_text_embeddings

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # The projection acts on the early steps only; afterwards the
                # prompt is used as it was written.
                prompt_embeds = rescaled_text_embeddings if (0 <= i <= safree_last_step) else text_embeddings

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)


# The name fomo/sample.py::load_pipeline imports.
Pipeline = CogVideoXSafreePipeline
