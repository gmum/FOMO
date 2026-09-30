"""Generate a slice of a hand-written prompt list, numbered globally.

Separate from generate_celeb.py on purpose. That script is the benchmark's
generator and half a dozen pipelines depend on its numbering, so it stays as
it is; this one exists to do the thing the benchmark never needs -- take one
flat list, hand different slices of it to different jobs, and have every video
land in a single folder numbered by its line in the file.

Two consequences follow from that, and both are the point:

    videos/adhoc/<tag>/NN.mp4   NN is the line number, not the position
                                within whichever slice produced it.
    seed = --seed + NN          so every prompt gets its own noise, and the
                                same line generated twice -- once through the
                                base model, once through an adapter -- gets
                                the same noise both times.

    python benchmark/generate_adhoc.py --file configs/adhoc_prompts.txt \\
        --start 10 --count 10 --out videos/adhoc/base

Run from the repository root.
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_prompts(path):
    if not os.path.exists(path):
        raise SystemExit("No prompt list at %s" % path)
    lines = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    if not lines:
        raise SystemExit("%s has no prompts in it." % path)
    return lines


def main():
    ap = argparse.ArgumentParser(description="Generate one slice of a prompt list.")
    ap.add_argument("--file", default="configs/adhoc_prompts.txt")
    ap.add_argument("--out", required=True, help="Directory the videos go in.")
    ap.add_argument("--start", type=int, default=0, help="First line to render.")
    ap.add_argument("--count", type=int, default=0, help="How many; 0 means all.")
    ap.add_argument("--lora", default="", help="Checkpoint directory, or empty for base.")
    ap.add_argument("--lora_weight", type=float, default=1.0)
    ap.add_argument("--model_family", default="hunyuan")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    prompts = read_prompts(args.file)
    stop = len(prompts) if args.count <= 0 else min(len(prompts), args.start + args.count)
    slice_ = [(i, prompts[i]) for i in range(args.start, stop)]
    if not slice_:
        print("Nothing in lines %d..%d of %d." % (args.start, stop, len(prompts)))
        return

    print("List   : %s  (%d prompts)" % (args.file, len(prompts)))
    print("Slice  : lines %d..%d" % (args.start, stop - 1))
    print("Output : %s" % args.out)
    print("LoRA   : %s" % (args.lora or "(base model)"))
    print("Size   : %dx%d, %d frames, %d steps, gs %s"
          % (args.width, args.height, args.num_frames,
             args.num_inference_steps, args.guidance_scale))
    print()
    for index, prompt in slice_:
        print("  %03d  seed %d  %s" % (index, args.seed + index, prompt))
    if args.dry_run:
        return

    import torch
    from diffusers.utils import export_to_video
    from generate import load_pipeline, resolve_lora_dir, set_active_adapter
    from common import MODEL_DEFAULTS

    os.makedirs(args.out, exist_ok=True)
    todo = [(i, p) for i, p in slice_
            if not (args.skip_existing
                    and os.path.exists(os.path.join(args.out, "%03d.mp4" % i)))]
    if not todo:
        print("\nNothing to do; every file already exists.")
        return

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

    started = time.time()
    for done, (index, prompt) in enumerate(todo, start=1):
        out = os.path.join(args.out, "%03d.mp4" % index)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
        frames = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        export_to_video(frames, out, fps=args.fps)
        print("  [%d/%d] %s  seed %d  (%.1f min elapsed)"
              % (done, len(todo), os.path.basename(out),
                 args.seed + index, (time.time() - started) / 60))
        del frames
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone in %.1f min -> %s" % ((time.time() - started) / 60, args.out))


if __name__ == "__main__":
    main()
