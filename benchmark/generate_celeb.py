"""Generate the twenty prompts of one person, with or without a LoRA.

The directory layout mirrors the Imagenette sweep on purpose, so that the same
row-building and scoring tools can be pointed at it later:

    videos/celeb/_shared/base/<person>/<NN>.mp4
    videos/celeb/unlearn_<erased>/<person>/<method>/<NN>.mp4

One task generates one (column, person) cell. A column is the model that
erased a particular identity; inside it every person is generated, so the
effect on the other seven can be measured the way PSR measures it for objects.

Generation settings match benchmark/generate.py exactly. A column is only
comparable with the baseline if the two were produced the same way.

    python3 benchmark/generate_celeb.py --person "Taylor Swift" \\
        --output_dir videos/celeb/_shared/base/taylor_swift --method base
"""

import argparse
import gc
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_DEFAULTS, slugify  # noqa: E402

# torch and the pipeline are imported inside main, after --dry_run has had its
# chance to return, so listing prompts works on a login node.


def main():
    ap = argparse.ArgumentParser(description="Generate one subject's prompt set.")
    ap.add_argument("--person", required=True, help="Name as written in the prompt module.")
    ap.add_argument("--prompts_module", default="celeb_prompts",
                    help="Module exposing PROMPTS, PEOPLE and optionally GENDER. "
                         "Any subject with its own prompt list works, not only people.")
    ap.add_argument("--output_dir", required=True,
                    help="Directory that will hold <method>/<NN>.mp4.")
    ap.add_argument("--method", default="base", help="Subdirectory name for this model.")
    ap.add_argument("--lora", default="",
                    help="LoRA directory, or a parent holding checkpoint-*. "
                         "Empty means the base model.")
    ap.add_argument("--lora_weight", type=float, default=1.0)
    ap.add_argument("--model_family", default="hunyuan")
    ap.add_argument("--model_path", default=None)
    # Defaults deliberately equal to benchmark/generate.py.
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_prompts", type=int, default=0,
                    help="Use only the first N prompts. 0 means all of them. "
                         "The prompt list stays long so the set can be extended "
                         "later without renumbering what already exists.")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    CP = importlib.import_module(args.prompts_module)
    if args.person not in CP.PROMPTS:
        raise SystemExit("Unknown subject %r in %s. Known: %s"
                         % (args.person, args.prompts_module, ", ".join(CP.PEOPLE)))
    prompts = CP.PROMPTS[args.person]
    if args.num_prompts:
        prompts = prompts[:args.num_prompts]
    out_dir = os.path.join(args.output_dir, args.method)

    gender = getattr(CP, "GENDER", {}).get(args.person)
    print("Subject  : %s%s" % (args.person, " (%s)" % gender if gender else ""))
    print("Method   : %s" % args.method)
    print("LoRA     : %s" % (args.lora or "(base model)"))
    print("Output   : %s" % out_dir)
    print("Size     : %dx%d, %d frames, %d steps, gs %s"
          % (args.width, args.height, args.num_frames,
             args.num_inference_steps, args.guidance_scale))
    print()
    for index, prompt in enumerate(prompts):
        print("  %02d  %s" % (index, prompt))
    if args.dry_run:
        return

    import torch
    from diffusers.utils import export_to_video
    from generate import load_pipeline, resolve_lora_dir, set_active_adapter

    os.makedirs(out_dir, exist_ok=True)
    todo = [(i, p) for i, p in enumerate(prompts)
            if not (args.skip_existing
                    and os.path.exists(os.path.join(out_dir, "%02d.mp4" % i)))]
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
        out = os.path.join(out_dir, "%02d.mp4" % index)
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
        print("  [%d/%d] %s  (%.1f min elapsed)"
              % (done, len(todo), os.path.basename(out), (time.time() - started) / 60))
        del frames
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone in %.1f min -> %s" % ((time.time() - started) / 60, out_dir))


if __name__ == "__main__":
    main()
