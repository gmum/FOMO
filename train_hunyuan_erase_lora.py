from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from receler.loras.hunyuan_peft_lora import (
    DEFAULT_HUNYUAN_ERASE_TARGET_MODULES,
    add_hunyuan_erase_lora,
    iter_trainable_parameters,
    trainable_parameter_summary,
)
from receler.training.hunyuan_dataset import (
    CachedHunyuanVideoConceptDataset,
    collate_cached_hunyuan_batch,
    default_hunyuan_cache_dir,
    ensure_hunyuan_training_cache,
    ensure_prompt_metadata,
    ensure_video_dataset,
)
from receler.training.hunyuan_erase_losses import (
    compute_attention_loss_from_cached,
    compute_esd_losses_from_cached,
)
from receler.training.hunyuan_value_loss import compute_esd_and_value_losses_from_cached
from receler.training.token_utils import describe_token_window
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE


def read_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    for key in [
        "model_path",
        "dataset_dir",
        "cache_dir",
        "output_dir",
        "concept",
        "neutral_prompt_override",
        "max_train_steps",
        "rank",
        "lora_alpha",
        "learning_rate",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "weight_decay",
        "lamb_esd",
        "lamb_attn",
        "esd_full_video_weight",
        "esd_first_frame_weight",
        "negative_guidance",
        "teacher_mode",
        "attn_blocks",
        "train_width",
        "train_height",
        "train_num_frames",
        "num_bootstrap_videos",
        "checkpointing_steps",
        "bootstrap_width",
        "bootstrap_height",
        "bootstrap_num_frames",
        "bootstrap_prompt_override",
        "bootstrap_neutral_prompt_override",
        "lamb_v",
        "v_steps",
        "v_block_min",
        "v_early_prob",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.target_modules is not None:
        config["target_modules"] = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if args.use_empty_neutral_prompt:
        config["neutral_prompt_override"] = ""
    if args.bootstrap_dataset_if_missing:
        config["bootstrap_dataset_if_missing"] = True
    if args.no_bootstrap_dataset:
        config["bootstrap_dataset_if_missing"] = False
    if args.gradient_checkpointing:
        config["gradient_checkpointing"] = True
    if args.prompt_only_esd:
        config["prompt_only_esd"] = True
    return config


def dtype_from_config(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported mixed_precision value: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_hunyuan_size(height: int, width: int, name: str) -> None:
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"{name} must be divisible by 16 for HunyuanVideo, got height={height}, width={width}. "
            "Use e.g. 512x288 for a safe 16:9 smoke run, or 768x432 for a larger 16:9 run."
        )


def load_pipe(config: Dict[str, Any], dtype: torch.dtype) -> HunyuanVideoPipeline:
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        config["model_path"],
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        config["model_path"],
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.vae.enable_tiling()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    return pipe


def save_training_config(output_dir: Path, config: Dict[str, Any], lora_matches) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(config)
    serializable["matched_lora_modules"] = list(lora_matches)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def save_hunyuan_lora_checkpoint(
    *,
    pipe: HunyuanVideoPipeline,
    output_dir: Path,
    adapter_name: str,
    config: Dict[str, Any],
    lora_matches,
    global_step: int | None = None,
) -> None:
    checkpoint_config = dict(config)
    if global_step is not None:
        checkpoint_config["global_step"] = global_step
    transformer_lora_layers = get_peft_model_state_dict(pipe.transformer, adapter_name=adapter_name)
    HunyuanVideoPipeline.save_lora_weights(
        save_directory=str(output_dir),
        transformer_lora_layers=transformer_lora_layers,
    )
    save_training_config(output_dir, checkpoint_config, lora_matches)


def freeze_non_transformer_components(pipe: HunyuanVideoPipeline) -> None:
    for component in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        component.eval()
        for param in component.parameters():
            param.requires_grad_(False)


def move_transformer_to_cpu_for_cache(pipe: HunyuanVideoPipeline) -> None:
    if torch.cuda.is_available():
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
        print("Moved transformer to CPU for VAE/text cache.")


def unload_non_transformer_components(pipe: HunyuanVideoPipeline) -> None:
    for name in ("vae", "text_encoder", "text_encoder_2"):
        component = getattr(pipe, name, None)
        if component is not None:
            component.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Moved VAE and text encoders to CPU. Training will use cached tensors.")


def move_transformer_to_training_device(pipe: HunyuanVideoPipeline) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.transformer.to(device)
    print(f"Moved transformer to {device} for LoRA training.")


def train(config: Dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    dtype = dtype_from_config(str(config.get("mixed_precision", "bf16")))
    validate_hunyuan_size(int(config.get("train_height", 288)), int(config.get("train_width", 512)), "train size")
    validate_hunyuan_size(
        int(config.get("bootstrap_height", config.get("train_height", 288))),
        int(config.get("bootstrap_width", config.get("train_width", 512))),
        "bootstrap size",
    )
    pipe = load_pipe(config, dtype)
    freeze_non_transformer_components(pipe)
    prompt_only_esd = bool(config.get("prompt_only_esd", False))
    if prompt_only_esd and float(config.get("lamb_attn", 0.0)) != 0.0:
        raise ValueError("prompt_only_esd can only be used with LAMB_ATTN=0 because attention loss needs video latents.")

    if config.get("bootstrap_dataset_if_missing", False):
        if config.get("bootstrap_prompt_override") is not None:
            shown_prompt = str(config["bootstrap_prompt_override"])
            print(f"Bootstrap prompt override: {shown_prompt}")
        if prompt_only_esd:
            ensure_prompt_metadata(
                dataset_dir=config["dataset_dir"],
                concept=config["concept"],
                num_prompts=int(config.get("num_bootstrap_videos", 5)),
                prompt_override=config.get("bootstrap_prompt_override"),
                neutral_prompt_override=config.get("bootstrap_neutral_prompt_override"),
            )
            print("Prompt-only ESD metadata prepared; no bootstrap videos were generated.")
        else:
            ensure_video_dataset(
                pipe=pipe,
                dataset_dir=config["dataset_dir"],
                concept=config["concept"],
                num_videos=int(config.get("num_bootstrap_videos", 5)),
                width=int(config.get("bootstrap_width", 640)),
                height=int(config.get("bootstrap_height", 360)),
                num_frames=int(config.get("bootstrap_num_frames", 33)),
                fps=int(config.get("fps", 15)),
                num_inference_steps=int(config.get("bootstrap_inference_steps", 30)),
                guidance_scale=float(config.get("bootstrap_guidance_scale", 6.0)),
                seed=int(config.get("seed", 42)),
                prompt_override=config.get("bootstrap_prompt_override"),
                neutral_prompt_override=config.get("bootstrap_neutral_prompt_override"),
            )

    cache_dir = Path(
        config.get("cache_dir")
        or default_hunyuan_cache_dir(
            config["dataset_dir"],
            int(config.get("train_height", 288)),
            int(config.get("train_width", 512)),
            int(config.get("train_num_frames", 33)),
            int(config.get("max_sequence_length", 256)),
            neutral_prompt_override=config.get("neutral_prompt_override"),
        )
    )
    config["cache_dir"] = str(cache_dir)
    print(f"Hunyuan training cache: {cache_dir}")
    neutral_prompt_override = config.get("neutral_prompt_override")
    if neutral_prompt_override is not None:
        shown = "<empty>" if neutral_prompt_override == "" else str(neutral_prompt_override)
        print(f"Neutral prompt override for ESD teacher e_0: {shown}")
    move_transformer_to_cpu_for_cache(pipe)
    ensure_hunyuan_training_cache(
        pipe=pipe,
        dataset_dir=config["dataset_dir"],
        cache_dir=cache_dir,
        model_path=str(config["model_path"]),
        height=int(config.get("train_height", 288)),
        width=int(config.get("train_width", 512)),
        num_frames=int(config.get("train_num_frames", 33)),
        max_sequence_length=int(config.get("max_sequence_length", 256)),
        dtype=dtype,
        neutral_prompt_override=neutral_prompt_override,
        require_video_latents=not prompt_only_esd,
    )
    unload_non_transformer_components(pipe)
    move_transformer_to_training_device(pipe)
    if bool(config.get("gradient_checkpointing", False)):
        pipe.transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled for Hunyuan transformer.")

    target_modules = config.get("target_modules") or DEFAULT_HUNYUAN_ERASE_TARGET_MODULES
    adapter_name = str(config.get("adapter_name", "erase_lora"))
    _, lora_matches = add_hunyuan_erase_lora(
        pipe.transformer,
        rank=int(config.get("rank", 8)),
        alpha=int(config.get("lora_alpha", config.get("rank", 8))),
        dropout=float(config.get("lora_dropout", 0.0)),
        target_modules=target_modules,
        adapter_name=adapter_name,
    )
    summary = trainable_parameter_summary(pipe.transformer)
    print(f"Matched LoRA modules: {len(lora_matches)}")
    for name in lora_matches[:20]:
        print(f"  {name}")
    if len(lora_matches) > 20:
        print(f"  ... {len(lora_matches) - 20} more")
    print(
        "Trainable params: "
        f"{summary['trainable']} / {summary['total']} ({summary['percent']:.4f}%)"
    )
    if summary["trainable"] == 0:
        raise RuntimeError("No trainable LoRA parameters found.")

    dataset = CachedHunyuanVideoConceptDataset(
        dataset_dir=config["dataset_dir"],
        cache_dir=cache_dir,
        model_path=str(config["model_path"]),
        height=int(config.get("train_height", 360)),
        width=int(config.get("train_width", 640)),
        num_frames=int(config.get("train_num_frames", 33)),
        max_sequence_length=int(config.get("max_sequence_length", 256)),
        neutral_prompt_override=neutral_prompt_override,
        require_video_latents=not prompt_only_esd,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_cached_hunyuan_batch,
    )
    if int(config.get("train_batch_size", 1)) != 1:
        raise ValueError("This EraseAnything-style baseline expects train_batch_size=1.")

    first = dataset[0]
    first_indices = list(first["concept_token_indices"])
    print(f"Concept token indices for first prompt: {first_indices}")
    print(
        describe_token_window(
            pipe.tokenizer,
            first["prompt"],
            first_indices,
            DEFAULT_PROMPT_TEMPLATE,
            max_sequence_length=int(config.get("max_sequence_length", 256)),
        )
    )

    optimizer = torch.optim.AdamW(
        iter_trainable_parameters(pipe.transformer),
        lr=float(config.get("learning_rate", 2e-5)),
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.99))),
        eps=float(config.get("adam_epsilon", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    output_dir = Path(config["output_dir"])
    save_training_config(output_dir, config, lora_matches)

    max_train_steps = int(config.get("max_train_steps", 200))
    checkpointing_steps = int(config.get("checkpointing_steps", 0))
    progress = tqdm(range(max_train_steps), desc="erase-lora")
    data_iter = iter(dataloader)
    pipe.transformer.train()
    print(
        "ESD setup: "
        f"teacher_mode={config.get('teacher_mode', 'base')}, "
        f"negative_guidance={float(config.get('negative_guidance', 2.0))}, "
        f"full_video_weight={float(config.get('esd_full_video_weight', 1.0))}, "
        f"first_frame_weight={float(config.get('esd_first_frame_weight', 1.0))}, "
        f"lamb_attn={float(config.get('lamb_attn', 0.0))}, "
        f"gradient_checkpointing={bool(config.get('gradient_checkpointing', False))}, "
        f"prompt_only_esd={prompt_only_esd}"
    )

    for global_step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        lamb_esd = float(config.get("lamb_esd", 1.0))
        lamb_attn = float(config.get("lamb_attn", 0.0))
        esd_full_video_weight = float(config.get("esd_full_video_weight", 1.0))
        esd_first_frame_weight = float(config.get("esd_first_frame_weight", 1.0))
        optimizer.zero_grad(set_to_none=True)
        lamb_v = float(config.get("lamb_v", 0.0))
        esd_losses, timestep, v_metrics = compute_esd_and_value_losses_from_cached(
            pipe,
            prompt_embeds=batch["prompt_embeds"],
            pooled_prompt_embeds=batch["pooled_prompt_embeds"],
            prompt_attention_mask=batch["prompt_attention_mask"],
            neutral_embeds=batch["neutral_prompt_embeds"],
            neutral_pooled=batch["neutral_pooled_prompt_embeds"],
            neutral_attention_mask=batch["neutral_prompt_attention_mask"],
            height=int(config.get("train_height", 360)),
            width=int(config.get("train_width", 640)),
            num_frames=int(config.get("train_num_frames", 33)),
            latent_sample_steps=int(config.get("latent_sample_steps", 28)),
            timestep_sample_steps=int(config.get("timestep_sample_steps", 28)),
            guidance_scale=float(config.get("guidance_scale", 6.0)),
            negative_guidance=float(config.get("negative_guidance", 2.0)),
            teacher_mode=str(config.get("teacher_mode", "base")),
            v_enabled=lamb_v != 0.0,
            v_steps=int(config.get("v_steps", 3)),
            v_block_min=config.get("v_block_min", 20),
            v_video_only=bool(config.get("v_video_only", True)),
            v_early_prob=float(config.get("v_early_prob", 0.5)),
        )
        loss_esd_full = esd_losses["full_video"]
        loss_esd_first_frame = esd_losses["first_frame"]
        loss_esd = esd_full_video_weight * loss_esd_full + esd_first_frame_weight * loss_esd_first_frame

        loss_v = esd_losses.get("value")
        v_value = 0.0
        if loss_v is not None and lamb_v != 0.0:
            if not torch.isfinite(loss_v):
                raise FloatingPointError(f"Niefinitarne L_V w kroku {global_step}: {loss_v}")
            v_value = loss_v.detach().item()
            loss_esd = loss_esd + lamb_v * loss_v

        if not torch.isfinite(loss_esd):
            raise FloatingPointError(f"Non-finite L_ESD at step {global_step}: {loss_esd}")
        if not torch.isfinite(loss_esd_full):
            raise FloatingPointError(f"Non-finite full-video L_ESD at step {global_step}: {loss_esd_full}")
        if not torch.isfinite(loss_esd_first_frame):
            raise FloatingPointError(f"Non-finite first-frame L_ESD at step {global_step}: {loss_esd_first_frame}")
        esd_value = loss_esd.detach().item()
        esd_full_value = loss_esd_full.detach().item()
        esd_first_frame_value = loss_esd_first_frame.detach().item()
        timestep_value = timestep.detach().float().item()
        (lamb_esd * loss_esd).backward()
        del esd_losses, loss_esd, loss_esd_full, loss_esd_first_frame
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        concept_indices = batch["concept_token_indices"][0]
        attn_value = 0.0
        if lamb_attn != 0.0:
            loss_attn, concept_indices = compute_attention_loss_from_cached(
                pipe,
                latents=batch["latents"],
                prompt_embeds=batch["prompt_embeds"],
                pooled_prompt_embeds=batch["pooled_prompt_embeds"],
                prompt_attention_mask=batch["prompt_attention_mask"],
                concept_token_indices=batch["concept_token_indices"][0],
                timestep=timestep,
                guidance_scale=float(config.get("guidance_scale", 6.0)),
                block_indices=config.get("attn_blocks", "last"),
                attention_chunk_size=int(config.get("attention_chunk_size", 256)),
            )
            if not torch.isfinite(loss_attn):
                raise FloatingPointError(f"Non-finite L_attn at step {global_step}: {loss_attn}")
            attn_value = loss_attn.detach().item()
            (lamb_attn * loss_attn).backward()
            del loss_attn
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss_value = lamb_esd * esd_value + lamb_attn * attn_value

        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            esd=f"{esd_value:.4f}",
            esd_full=f"{esd_full_value:.4f}",
            esd_first=f"{esd_first_frame_value:.4f}",
            attn=f"{attn_value:.4f}",
            t=f"{timestep_value:.1f}",
            token=str(concept_indices),
            v=f"{v_value:.4f}",
            ti=f"{int(v_metrics['timestep_index'])}",
        )

        step = global_step + 1
        if checkpointing_steps > 0 and step % checkpointing_steps == 0:
            checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
            save_hunyuan_lora_checkpoint(
                pipe=pipe,
                output_dir=checkpoint_dir,
                adapter_name=adapter_name,
                config=config,
                lora_matches=lora_matches,
                global_step=step,
            )
            progress.write(f"Saved checkpoint LoRA to {checkpoint_dir}")

    pipe.transformer.eval()
    save_hunyuan_lora_checkpoint(
        pipe=pipe,
        output_dir=output_dir,
        adapter_name=adapter_name,
        config=config,
        lora_matches=lora_matches,
        global_step=max_train_steps,
    )
    print(f"Saved Hunyuan erase LoRA to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EraseAnything-style HunyuanVideo erasure LoRA.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--concept", default=None)
    parser.add_argument("--neutral_prompt_override", default=None)
    parser.add_argument("--use_empty_neutral_prompt", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--adam_beta1", type=float, default=None)
    parser.add_argument("--adam_beta2", type=float, default=None)
    parser.add_argument("--adam_epsilon", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lamb_esd", type=float, default=None)
    parser.add_argument("--lamb_attn", type=float, default=None)
    parser.add_argument("--esd_full_video_weight", type=float, default=None)
    parser.add_argument("--esd_first_frame_weight", type=float, default=None)
    parser.add_argument("--negative_guidance", type=float, default=None)
    parser.add_argument("--teacher_mode", default=None)
    parser.add_argument("--attn_blocks", default=None)
    parser.add_argument("--target_modules", default=None)
    parser.add_argument("--train_width", type=int, default=None)
    parser.add_argument("--train_height", type=int, default=None)
    parser.add_argument("--train_num_frames", type=int, default=None)
    parser.add_argument("--num_bootstrap_videos", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=None)
    parser.add_argument("--bootstrap_width", type=int, default=None)
    parser.add_argument("--bootstrap_height", type=int, default=None)
    parser.add_argument("--bootstrap_num_frames", type=int, default=None)
    parser.add_argument("--bootstrap_prompt_override", default=None)
    parser.add_argument("--bootstrap_neutral_prompt_override", default=None)
    # The value-preservation term. These four keys were already read from the
    # config, but had no flags, so a sweep could not vary them without writing
    # a YAML per run. lamb_v = 0 is plain ESD.
    parser.add_argument("--lamb_v", type=float, default=None)
    parser.add_argument("--v_steps", type=int, default=None)
    parser.add_argument("--v_block_min", type=int, default=None)
    parser.add_argument("--v_early_prob", type=float, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--prompt_only_esd", action="store_true")
    parser.add_argument("--bootstrap_dataset_if_missing", action="store_true")
    parser.add_argument("--no_bootstrap_dataset", action="store_true")
    args = parser.parse_args()

    config = apply_overrides(read_config(args.config), args)
    train(config)


if __name__ == "__main__":
    main()
