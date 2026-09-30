"""Generate the ceiling videos for the mapping targets.

CLIP-tgt has no scale of its own. The floor is what the base model scores
against the target text when it was prompted for the source, which
clip_score.csv already stores as clip_tgt_base. The ceiling is what the same
base model scores when asked for the target directly. Between the two, the
method's own number becomes a percentage rather than a bare similarity.

Every generation setting here matches benchmark/generate.py exactly -- same
resolution, frame count, sampler steps, guidance and seed schedule -- because
the ceiling is only comparable to the floor if the two were produced the same
way. Change one of them and the normalisation stops meaning anything.

One task per (variant, target) cell, twenty videos each:

    python3 benchmark/generate_ceiling.py --cell 0
    python3 benchmark/generate_ceiling.py --list

Output: videos/_ceiling/<variant>/<target slug>/<NN>.mp4
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_DEFAULTS, slugify  # noqa: E402
import ceiling_prompts as CP  # noqa: E402

# torch and the pipeline are imported inside main, after --list and --dry_run
# have had their chance to return. Those two generate nothing, so they have to
# work on a login node, where the aarch64 conda environment does not exist.

VARIANTS = ("A", "B", "C")


def cells():
    """Every (variant, target) pair that has prompts, in a fixed order.

    Not every target carries every variant: the far targets only have C,
    because rewriting a v7 scene for a wooden box has no minimal repair. Cells
    are skipped rather than faked, so the array index is dense. Output paths
    are keyed by variant and target, not by index, so adding targets renumbers
    the array without invalidating anything already generated.
    """
    return [(variant, target)
            for target in CP.TARGETS
            for variant in VARIANTS
            if target in getattr(CP, variant)]


def prompts_for_cell(variant, target):
    """Twenty (seed, prompt) pairs for one cell.

    Variants A and B vary the prompt and hold the seed schedule; variant C
    holds one prompt and varies the seed, which is the whole point of it.
    """
    if variant == "A":
        texts = CP.A[target]
    elif variant == "B":
        texts = CP.B[target]
    else:
        texts = [CP.C[target]] * len(CP.SEEDS)
    return list(zip(CP.SEEDS, texts))


def main():
    ap = argparse.ArgumentParser(description="Generate ceiling videos.")
    ap.add_argument("--cell", type=int, default=None,
                    help="Index into the (variant, target) list. See --list.")
    ap.add_argument("--variant", default=None, choices=VARIANTS)
    ap.add_argument("--target", default=None)
    ap.add_argument("--out_root", default="videos/_ceiling")
    ap.add_argument("--model_family", default="hunyuan")
    ap.add_argument("--model_path", default=None)
    # Defaults deliberately equal to benchmark/generate.py for the base runs.
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--list", action="store_true", help="Print the cells and exit.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the prompts of the chosen cell, generate nothing.")
    args = ap.parse_args()

    plan = cells()
    if args.list:
        for index, (variant, target) in enumerate(plan):
            print("%2d  %s  %s" % (index, variant, target))
        print("\n%d cells, %d videos" % (len(plan), len(plan) * len(CP.SEEDS)))
        return

    if args.cell is not None:
        if not 0 <= args.cell < len(plan):
            raise SystemExit("cell must be in 0..%d" % (len(plan) - 1))
        variant, target = plan[args.cell]
    elif args.variant and args.target:
        variant, target = args.variant, args.target
        if target not in CP.TARGETS:
            raise SystemExit("unknown target %r; see --list" % target)
    else:
        raise SystemExit("give --cell, or both --variant and --target")

    pairs = prompts_for_cell(variant, target)
    out_dir = os.path.join(args.out_root, variant, slugify(target))
    print("Variant  : %s" % variant)
    print("Target   : %s" % target)
    print("Output   : %s" % out_dir)
    print("Size     : %dx%d, %d frames, %d steps, gs %s"
          % (args.width, args.height, args.num_frames,
             args.num_inference_steps, args.guidance_scale))
    print()
    for index, (seed, prompt) in enumerate(pairs):
        print("  %02d  seed %d  %s" % (index, seed, prompt))
    if args.dry_run:
        return

    import torch
    from diffusers.utils import export_to_video
    from generate import load_pipeline

    os.makedirs(out_dir, exist_ok=True)
    todo = [(i, s, p) for i, (s, p) in enumerate(pairs)
            if not (args.skip_existing
                    and os.path.exists(os.path.join(out_dir, "%02d.mp4" % i)))]
    if not todo:
        print("\nNothing to do; every file already exists.")
        return

    if args.model_path is None:
        args.model_path = MODEL_DEFAULTS[args.model_family]["model_path"]
    print("\nLoading %s ..." % args.model_path)
    pipe = load_pipeline(args.model_family, args.model_path)

    started = time.time()
    for done, (index, seed, prompt) in enumerate(todo, start=1):
        out = os.path.join(out_dir, "%02d.mp4" % index)
        generator = torch.Generator(device="cpu").manual_seed(seed)
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
        print("  [%d/%d] %s  (%.1f min elapsed)"
              % (done, len(todo), os.path.basename(out), (time.time() - started) / 60))
        del frames
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone in %.1f min -> %s" % ((time.time() - started) / 60, out_dir))


if __name__ == "__main__":
    main()
