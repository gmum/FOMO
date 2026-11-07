#!/usr/bin/env python3
"""CogVideoX L_H plus late-block background V preservation.

The L_H-only entry point is left unchanged. Attention capture, GramCol,
LoRA helpers and preview export are shared with that implementation.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from tqdm.auto import tqdm

from fomo.cogvideox import (
    CogVideoXHVLoraController, CogVideoXHVLoraProcessor,
    adapter_mode, add_lora, encode_prompts,
    enable_transformer_gradient_checkpointing, find_concept_token_indices,
    frames_from_latents_shape, install_processors, iter_trainable_parameters,
    make_initial_latents, parse_csv_strings, read_prompt_pairs,
    resolve_h_layers, restore_processors, rollout_to_index,
    save_checkpoint, save_training_videos, target_modules_from_preset,
    trainable_parameter_summary, transformer_forward,
)

DEFAULT_V_LAYERS = (24, 25, 26, 27, 28, 29)


def resolve_v_layers(num_layers, specification):
    if isinstance(specification, str):
        indices = [int(x) for x in parse_csv_strings(specification)]
    else:
        indices = [int(x) for x in specification]
    indices = sorted(set(indices))
    if not indices or any(x < 0 or x >= num_layers for x in indices):
        raise ValueError(f"Invalid V layers for a {num_layers}-block model: {indices}")
    return indices


def sample_target_index(args):
    """Hunyuan mixture sampler; retain the same sampler in lambda_V=0 controls."""
    if args.v_steps > 0 and args.v_early_prob > 0.0:
        early_upper = min(args.v_steps, args.max_target_step + 1)
        if early_upper > args.min_target_step and random.random() < args.v_early_prob:
            return random.randint(args.min_target_step, early_upper - 1)
    return random.randint(args.min_target_step, args.max_target_step)


def value_loss_active(args, index):
    return args.lamb_v > 0.0 and 0 <= index < args.v_steps


def early_timestep_probability(args):
    count = max(0, min(args.v_steps, args.max_target_step + 1) - args.min_target_step)
    if not count:
        return 0.0
    uniform = count / (args.max_target_step - args.min_target_step + 1)
    return args.v_early_prob + (1.0 - args.v_early_prob) * uniform


def combined_loss(loss_h, loss_v, lamb_h, lamb_v):
    loss = lamb_h * loss_h
    if loss_v is not None:
        loss = loss + lamb_v * loss_v
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite L_H+L_V loss: {loss.detach()}")
    return loss


def build_parser():
    parser = argparse.ArgumentParser(description="Train CogVideoX LoRA with GramCol H targets and late-block V preservation.")
    parser.add_argument("--model_path", default="THUDM/CogVideoX-2b")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_pairs_jsonl", default=None)
    parser.add_argument("--source_concept", default="dog")
    parser.add_argument("--target_concept", default="cat")
    parser.add_argument(
        "--h_layer_groups", default="imap",
        help="imap: automatically use the 17 CogVideoX-2B or 21 CogVideoX-5B IMAP layers; alternatively early,middle,late,all.",
    )
    parser.add_argument("--v_layers", default=",".join(map(str, DEFAULT_V_LAYERS)),
                        help="Explicit zero-based V block indices, default 24,25,26,27,28,29.")
    parser.add_argument("--lamb_h", type=float, default=1.0)
    parser.add_argument("--lora_target_preset", default="text_qk")
    parser.add_argument("--target_modules", default="")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--checkpointing_steps", type=int, default=10)
    parser.add_argument("--save_training_videos", action=argparse.BooleanOptionalAction, default=True,
                        help="Save base source/safe videos before training and a source-with-LoRA video at every checkpoint (first prompt pair).")
    parser.add_argument("--eval_inference_steps", type=int, default=30)
    parser.add_argument("--eval_seed", type=int, default=None, help="Fixed preview seed; defaults to --seed.")
    parser.add_argument("--eval_fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--trajectory_steps", type=int, default=30)
    parser.add_argument("--min_target_step", type=int, default=0)
    parser.add_argument("--max_target_step", type=int, default=29)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--max_sequence_length", type=int, default=226)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--imap_alpha", type=float, default=5.0)
    parser.add_argument("--clamp_mask", action="store_true")
    parser.add_argument("--lamb_v", type=float, default=0.5)
    parser.add_argument("--v_steps", type=int, default=1)
    parser.add_argument("--v_early_prob", type=float, default=0.5)
    parser.add_argument("--v_mask_mode", choices=("full", "background"), default="background")
    parser.add_argument("--split_hv_backward", action="store_true")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser



def main(args):
    if not all(math.isfinite(x) and x >= 0 for x in (args.lamb_h, args.lamb_v)) or args.lamb_h + args.lamb_v <= 0:
        raise ValueError("Loss weights must be finite, non-negative and not both zero.")
    if not math.isfinite(args.imap_alpha) or args.imap_alpha <= 0:
        raise ValueError("imap_alpha must be finite and positive.")
    if not 0 <= args.v_steps <= args.trajectory_steps:
        raise ValueError("Require 0 <= v_steps <= trajectory_steps.")
    if args.lamb_v > 0 and args.v_steps <= args.min_target_step:
        raise ValueError("L_V cannot become active in the selected timestep interval.")
    if args.eval_inference_steps <= 0 or args.eval_fps <= 0:
        raise ValueError("eval_inference_steps and eval_fps must be positive.")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16.")
    if args.trajectory_steps <= 0:
        raise ValueError("trajectory_steps must be positive.")
    if not 0 <= args.min_target_step <= args.max_target_step < args.trajectory_steps:
        raise ValueError("Require 0 <= min_target_step <= max_target_step < trajectory_steps.")
    if not 0.0 <= args.v_early_prob <= 1.0:
        raise ValueError("v_early_prob must be in [0,1].")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.transformer.train()
    for component in (pipe.vae, pipe.text_encoder):
        for parameter in component.parameters():
            parameter.requires_grad_(False)

    num_layers = len(pipe.transformer.transformer_blocks)
    h_groups = parse_csv_strings(args.h_layer_groups)
    h_layers = resolve_h_layers(num_layers, h_groups)
    v_layers = resolve_v_layers(num_layers, args.v_layers) if args.lamb_v > 0 else []
    install_layers = sorted(set(h_layers + v_layers))
    if not install_layers:
        raise ValueError("No layers selected.")

    raw_pairs = read_prompt_pairs(args.prompt_pairs_jsonl, args.source_concept, args.target_concept)
    for pair in raw_pairs:
        pair.source_token_indices = find_concept_token_indices(
            pipe.tokenizer,
            pair.source_prompt,
            pair.source_concept,
            args.max_sequence_length,
        )
        source_embeds, positive_index = encode_prompts(
            pipe, pair.source_prompt, args.guidance_scale, args.max_sequence_length, device
        )
        target_embeds, _ = encode_prompts(
            pipe, pair.target_prompt, args.guidance_scale, args.max_sequence_length, device
        )
        pair.source_embeds = source_embeds.detach().cpu()
        pair.target_embeds = target_embeds.detach().cpu()
        pair.positive_batch_index = int(positive_index)

    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    target_modules = parse_csv_strings(args.target_modules) if args.target_modules else target_modules_from_preset(args.lora_target_preset)
    adapter_name = "cogvideox_hv_lora"
    matches = add_lora(pipe.transformer, target_modules, args.rank, args.lora_alpha, adapter_name)
    pipe.transformer.to(device=device, dtype=dtype)
    if args.gradient_checkpointing:
        enable_transformer_gradient_checkpointing(pipe.transformer)
    summary = trainable_parameter_summary(pipe.transformer)
    if summary["trainable"] <= 0:
        raise RuntimeError("No trainable LoRA parameters.")
    optimizer = torch.optim.AdamW(
        iter_trainable_parameters(pipe.transformer),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    generator_for_shape = torch.Generator(device=device).manual_seed(args.seed)
    shape_latents, _ = make_initial_latents(pipe, args, dtype, device, generator_for_shape)
    token_grid = frames_from_latents_shape(pipe, shape_latents)
    del shape_latents

    controller = CogVideoXHVLoraController(
        h_layers=h_layers,
        v_layers=v_layers,
        source_token_indices=raw_pairs[0].source_token_indices,
        token_grid=token_grid,
        alpha=args.imap_alpha,
        clamp_mask=args.clamp_mask,
        lamb_v=args.lamb_v,
        v_mask_mode=args.v_mask_mode,
    )
    originals = install_processors(pipe, controller, install_layers)

    config = vars(args).copy()
    config.update(
        {
            "num_transformer_blocks": num_layers,
            "training_variant": "lh_plus_late_v",
            "trajectory_owner": "student source-prompt rollout, detached",
            "shared_zt": True,
            "v_reference": "base source prompt on the same z_t and timestep; detached",
            "v_mask_uses_imap_alpha": False,
            "v_video_only": True,
            "timestep_sampling_independent_of_lambda_v": True,
            "early_timestep_probability": early_timestep_probability(args),
            "v_active_probability": early_timestep_probability(args) if args.lamb_v > 0 else 0.0,
            "total_loss_formula": "lambda_H * L_H + lambda_V * L_V",
            "v_timing": "active iff target_step_index < v_steps; index 0 is before any denoising",
            "h_layers": h_layers,
            "v_layers": v_layers,
            "token_grid": list(token_grid),
            "target_modules_resolved": target_modules,
            "matched_lora_modules": matches,
            "prompt_pairs": [
                {
                    "source_prompt": pair.source_prompt,
                    "target_prompt": pair.target_prompt,
                    "source_concept": pair.source_concept,
                    "target_concept": pair.target_concept,
                    "source_token_indices": list(pair.source_token_indices),
                }
                for pair in raw_pairs
            ],
            "h_target_formula": "H_source + W * (H_target - H_source)",
            "h_capture": "attention output before to_out, shape [batch, heads, video_tokens, head_dim]",
            "mask_algorithm": "GramCol: per-head per-frame QK argmax anchor, H-H similarity, spatial min-max per head/frame, mean over heads",
            "mask_heads": "all attention heads, as in Hunyuan imap_mode=object",
            "mask_source": "source concept only, computed in base teacher_source; reused for target H mixing",
            "video_prompt_pair_index": 0,
            "weight_formula": "clamp(alpha*M_source,0,1)" if args.clamp_mask else "alpha*M_source",
            "v_loss_formula": {
                "full": "||V_student_source - stopgrad(V_base_source)||^2",
                "background": "||(1-M_source) * (V_student_source - stopgrad(V_base_source))||^2",
                "target": "||M_source * (V_student_source - stopgrad(V_base_source))||^2",
            }[args.v_mask_mode],
        }
    )
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"CogVideoX layers: {num_layers}; H groups={h_groups} layers={h_layers}; V explicit layers={v_layers}")
    print("H/mask: pre-to_out attention heads; GramCol with per-head per-frame QK anchors (all heads)")
    print(f"Resolution: {args.width}x{args.height}; frames={args.num_frames}; trajectory_steps={args.trajectory_steps}")
    print(f"Token grid: {token_grid}; LoRA modules={len(matches)}; trainable={summary['trainable']}")
    print(f"L_H weight={args.lamb_h}, alpha={args.imap_alpha}, clamp={args.clamp_mask}; L_V lambda={args.lamb_v}, mode={args.v_mask_mode}, steps={args.v_steps}")
    print(f"L_V uses raw M_source (no IMAP alpha); early sampling probability={early_timestep_probability(args):.6f}")

    log_path = output_dir / "loss.jsonl"
    try:
        if args.save_training_videos:
            save_training_videos(pipe, controller, raw_pairs[0], args, output_dir, step=0)
        progress = tqdm(range(1, args.max_train_steps + 1), desc="cogvideox-lh-late-lv")
        for step in progress:
            pair_index = random.randrange(len(raw_pairs))
            pair = raw_pairs[pair_index]
            controller.source_token_indices = pair.source_token_indices

            if pair.source_embeds is None or pair.target_embeds is None:
                raise RuntimeError("Prompt embeddings were not cached.")
            source_embeds = pair.source_embeds.to(device=device, dtype=dtype)
            target_embeds = pair.target_embeds.to(device=device, dtype=dtype)
            controller.positive_batch_index = pair.positive_batch_index

            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, args.trajectory_steps, device, None)
            pipe._num_timesteps = len(timesteps)
            generator = torch.Generator(device=device).manual_seed(args.seed + step)
            initial_latents, _ = make_initial_latents(pipe, args, dtype, device, generator)
            image_rotary_emb = (
                pipe._prepare_rotary_positional_embeddings(args.height, args.width, initial_latents.size(1), device)
                if pipe.transformer.config.use_rotary_positional_embeddings
                else None
            )
            controller.token_grid = frames_from_latents_shape(pipe, initial_latents)
            target_index = sample_target_index(args)
            capture_v = value_loss_active(args, target_index)

            with adapter_mode(pipe.transformer, True), torch.no_grad():
                z_t = rollout_to_index(
                    pipe,
                    source_embeds,
                    initial_latents,
                    timesteps,
                    target_index,
                    image_rotary_emb,
                    args.guidance_scale,
                    generator,
                )
            timestep = timesteps[target_index]

            controller.begin_teacher_source(pair.source_token_indices, capture_v=capture_v)
            with adapter_mode(pipe.transformer, False), torch.no_grad():
                _ = transformer_forward(pipe, z_t, timestep, source_embeds, image_rotary_emb, args.guidance_scale)
            missing_h = set(h_layers).difference(controller.source_h).union(set(h_layers).difference(controller.source_weights))
            if missing_h:
                raise RuntimeError(f"Missing teacher source H/weights for layers: {sorted(missing_h)}")
            if capture_v:
                missing_v = set(v_layers).difference(controller.source_v)
                if missing_v:
                    raise RuntimeError(f"Missing teacher source V for layers: {sorted(missing_v)}")

            controller.begin_teacher_target()
            with adapter_mode(pipe.transformer, False), torch.no_grad():
                _ = transformer_forward(pipe, z_t, timestep, target_embeds, image_rotary_emb, args.guidance_scale)
            missing_target = set(h_layers).difference(controller.target_h)
            if missing_target:
                raise RuntimeError(f"Missing teacher target H for layers: {sorted(missing_target)}")

            optimizer.zero_grad(set_to_none=True)
            if args.split_hv_backward and capture_v:
                controller.begin_student(capture_h=True, capture_v=False)
                with adapter_mode(pipe.transformer, True):
                    _ = transformer_forward(pipe, z_t, timestep, source_embeds, image_rotary_emb, args.guidance_scale)
                loss_h = controller.total_h_loss()
                if loss_h is None or set(h_layers).difference(controller.h_losses):
                    raise RuntimeError("Missing student L_H losses.")
                combined_loss(loss_h, None, args.lamb_h, 0.0).backward()
                loss_h_log = loss_h.detach()

                controller.begin_student(capture_h=False, capture_v=True)
                with adapter_mode(pipe.transformer, True):
                    _ = transformer_forward(pipe, z_t, timestep, source_embeds, image_rotary_emb, args.guidance_scale)
                loss_v = controller.total_v_loss()
                if loss_v is None or set(v_layers).difference(controller.v_losses):
                    raise RuntimeError("Missing student L_V losses.")
                combined_loss(loss_v, None, args.lamb_v, 0.0).backward()
                loss_v_log = loss_v.detach()
                loss_total_log = args.lamb_h * loss_h_log + args.lamb_v * loss_v_log
            else:
                controller.begin_student(capture_h=True, capture_v=capture_v)
                with adapter_mode(pipe.transformer, True):
                    _ = transformer_forward(pipe, z_t, timestep, source_embeds, image_rotary_emb, args.guidance_scale)
                loss_h = controller.total_h_loss()
                if loss_h is None or set(h_layers).difference(controller.h_losses):
                    raise RuntimeError("Missing student L_H losses.")
                loss_v = controller.total_v_loss() if capture_v else None
                if capture_v and (loss_v is None or set(v_layers).difference(controller.v_losses)):
                    raise RuntimeError("Missing student L_V losses.")
                loss_v_for_total = loss_v if loss_v is not None else loss_h.new_zeros(())
                loss_total = combined_loss(loss_h, loss_v_for_total, args.lamb_h, args.lamb_v)
                loss_total.backward()
                loss_h_log = loss_h.detach()
                loss_v_log = loss_v_for_total.detach()
                loss_total_log = loss_total.detach()

            grad_norm = torch.nn.utils.clip_grad_norm_(list(iter_trainable_parameters(pipe.transformer)), args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite LoRA gradient norm at update {step}: {grad_norm}")
            optimizer.step()
            controller.finish()

            record = {
                "step": step,
                "loss_total": float(loss_total_log.detach().cpu()),
                "loss_h": float(loss_h_log.detach().cpu()),
                "loss_v": float(loss_v_log.detach().cpu()),
                "weighted_loss_h": args.lamb_h * float(loss_h_log.detach().cpu()),
                "weighted_loss_v": args.lamb_v * float(loss_v_log.detach().cpu()),
                "lamb_h": args.lamb_h,
                "lamb_v": args.lamb_v,
                "early_timestep": bool(target_index < args.v_steps),
                "v_mask_uses_imap_alpha": False,
                "source_concept": pair.source_concept,
                "target_concept": pair.target_concept,
                "grad_norm": float(grad_norm.detach().cpu()),
                "target_step_index": int(target_index),
                "timestep": float(timestep.detach().cpu()),
                "prompt_pair_index": int(pair_index),
                "source_prompt": pair.source_prompt,
                "target_prompt": pair.target_prompt,
                "value_active": bool(capture_v),
                "value_layers": len(controller.v_losses),
                "mean_concept_mask": float(np.mean(list(controller.mask_means.values()))) if controller.mask_means else 0.0,
                "mean_effective_weight": float(np.mean(list(controller.weight_means.values()))) if controller.weight_means else 0.0,
                "max_effective_weight": float(max(controller.weight_maxima.values())) if controller.weight_maxima else 0.0,
                "mean_v_mask": float(np.mean(list(controller.v_mask_means.values()))) if controller.v_mask_means else 0.0,
                "v_mask_mode": args.v_mask_mode,
                "split_hv_backward": args.split_hv_backward,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            progress.set_postfix(
                loss=f"{record['loss_total']:.5f}",
                h=f"{record['loss_h']:.5f}",
                v=f"{record['loss_v']:.5f}",
                t=target_index,
            )

            checkpoint_due = args.checkpointing_steps > 0 and step % args.checkpointing_steps == 0
            if checkpoint_due:
                save_checkpoint(pipe, output_dir / f"checkpoint-{step:06d}", adapter_name, config, matches, step)

            del initial_latents, z_t, source_embeds, target_embeds
            controller.reset_all()
            torch.cuda.empty_cache()
            if args.save_training_videos and (checkpoint_due or step == args.max_train_steps):
                optimizer.zero_grad(set_to_none=True)
                save_training_videos(pipe, controller, raw_pairs[0], args, output_dir, step=step)

        save_checkpoint(pipe, output_dir, adapter_name, config, matches, args.max_train_steps)
        print(f"Saved final CogVideoX HV LoRA to {output_dir}")
    finally:
        restore_processors(pipe, originals)



if __name__ == "__main__":
    main(build_parser().parse_args())
