"""One video from one prompt typed on the command line.

Everything else in benchmark/ reads its prompts from a module, which is right
for a benchmark and useless when you just want to look at something. Generation
settings default to the ones the benchmark uses, so a video made here is
directly comparable with anything already on disk -- same resolution, same
frame count, same sampler, same seed schedule.

    python benchmark/generate_one.py \\
        --prompt "A video of Barack Obama standing in a bright room." \\
        --out /tmp/look.mp4

    python benchmark/generate_one.py \\
        --prompt "A video of Barack Obama standing in a bright room." \\
        --lora outputs/identity9_sweep/barack_obama_demo_lr1e3/unclamped_alpha7/checkpoint-000030 \\
        --out /tmp/look_erased.mp4

Pass --seed the same in both and the pair differs only by the adapter.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_DEFAULTS  # noqa: E402

# torch and the pipeline load inside main so --dry_run works on a login node.


def main():
    ap = argparse.ArgumentParser(description="Generate a single video.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, help="Path of the .mp4 to write.")
    ap.add_argument("--lora", default="",
                    help="LoRA directory, or a parent holding checkpoint-*. "
                         "Empty means the base model.")
    ap.add_argument("--lora_weight", type=float, default=1.0)
    ap.add_argument("--model_family", default="hunyuan")
    ap.add_argument("--model_path", default=None)
    # Defaults deliberately equal to the rest of benchmark/.
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print("Prompt : %s" % args.prompt)
    print("LoRA   : %s" % (args.lora or "(base model)"))
    print("Output : %s" % args.out)
    print("Size   : %dx%d, %d frames, %d steps, gs %s, seed %d"
          % (args.width, args.height, args.num_frames,
             args.num_inference_steps, args.guidance_scale, args.seed))
    if args.dry_run:
        return

    import torch
    from diffusers.utils import export_to_video
    from generate import load_pipeline, resolve_lora_dir, set_active_adapter

    if args.model_path is None:
        args.model_path = MODEL_DEFAULTS[args.model_family]["model_path"]
    print("\nLoading %s ..." % args.model_path)
    pipe = load_pipeline(args.model_family, args.model_path)

    adapter = None
    if args.lora:
        weights_dir = resolve_lora_dir(args.lora)
        adapter = "erase"
        pipe.load_lora_weights(weights_dir, adapter_name=adapter)
        print("Loaded LoRA from %s" % weights_dir)
    set_active_adapter(pipe, adapter, args.lora_weight)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    frames = pipe(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
    ).frames[0]

    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    export_to_video(frames, args.out, fps=args.fps)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
