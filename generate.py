"""Generate a video from HunyuanVideo, with or without a trained adapter.

The adapter produced by train_hunyuan_imap_h_lora.py is a standard PEFT LoRA,
so loading it needs nothing beyond diffusers. Pass --lora to sample from the
unlearned model and leave it out to sample from the original one; with the same
--seed the two differ only by the adapter, which is what makes a side-by-side
comparison meaningful.

    python generate.py --prompt "A video of a tench fish." --output base.mp4
    python generate.py --prompt "A video of a tench fish." --output erased.mp4 \\
        --lora outputs/tench/unclamped_alpha7
"""

import argparse
import os

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

WEIGHT_NAMES = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")


def resolve_lora_dir(path):
    """Accept either a weight directory or a parent holding checkpoint-* dirs.

    Training writes one directory per checkpoint, so pointing at the run
    directory is the common case; the newest checkpoint is then used.
    """
    if any(os.path.isfile(os.path.join(path, n)) for n in WEIGHT_NAMES):
        return path
    checkpoints = sorted(
        os.path.join(path, d) for d in os.listdir(path)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(path, d))
        and any(os.path.isfile(os.path.join(path, d, n)) for n in WEIGHT_NAMES)
    )
    if not checkpoints:
        raise SystemExit("No LoRA weights found under %s" % path)
    return checkpoints[-1]


def main():
    ap = argparse.ArgumentParser(description="Sample a video, optionally through an adapter.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True, help="Path of the .mp4 to write.")
    ap.add_argument("--lora", default=None,
                    help="Adapter directory, or a parent containing checkpoint-* dirs.")
    ap.add_argument("--lora_weight", type=float, default=1.0)
    ap.add_argument("--model_path", default="hunyuanvideo-community/HunyuanVideo")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    # HunyuanVideo expects a frame count of the form 4k+1.
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    if (args.num_frames - 1) % 4:
        raise SystemExit("--num_frames must be of the form 4k+1 (17, 33, 49, 65).")

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    if args.lora:
        weights = resolve_lora_dir(args.lora)
        pipe.load_lora_weights(weights, adapter_name="erase")
        pipe.set_adapters(["erase"], adapter_weights=[args.lora_weight])
        print("Adapter : %s (weight %s)" % (weights, args.lora_weight))
    else:
        print("Adapter : none (original model)")

    frames = pipe(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).frames[0]

    directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(directory, exist_ok=True)
    export_to_video(frames, args.output, fps=args.fps)
    print("Wrote   : %s" % args.output)


if __name__ == "__main__":
    main()
