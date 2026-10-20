"""Generate the videos of one experiment, or the baseline they are compared to.

One invocation produces one cell of the matrix: one subject inside one column.
A column is the model that erased a particular concept; inside it every subject
is generated, which is what makes the off-diagonal measurable -- PSR for
objects, Preserve for identities.

    # the untouched model, shared by every experiment on this prompt set
    python -m experiments.generate experiments/objects_mapping2.yaml \
        --base --subject church

    # one column of one experiment
    python -m experiments.generate experiments/objects_mapping2.yaml \
        --erased church --subject tench

    # everything in one process, which is what to do on a single long job
    python -m experiments.generate experiments/identities.yaml --all

The seed depends only on the prompt index, never on the column or the method,
so the same prompt gets the same noise everywhere and the comparison between
rows is paired rather than merely averaged.

A note on negative prompts. HunyuanVideo is CFG-distilled, so `guidance_scale`
feeds a guidance embedding instead of running two-branch guidance, and a
negative prompt has nowhere to go. The negprompt method therefore switches real
CFG on -- guidance_scale 1.0 with true_cfg_scale above it -- which also means
it samples differently from the baseline. That is a property of the baseline
being reproduced, not a choice made here.
"""

import argparse
import gc
import os
import time

from experiments import config as expconfig
from experiments import layout


def load_pipeline(model_path, safree=False):
    """The text-to-video pipeline on the GPU.

    Imported here rather than at module scope so that --dry-run and --help work
    in an environment without torch, such as a login node.
    """
    import torch
    from diffusers import HunyuanVideoTransformer3DModel

    if safree:
        # A separate pipeline class, not a flag: it cannot share a process with
        # the stock one, which is why a SAFREE experiment generates no baseline.
        from safree.pipeline import HunyuanVideoPipeline as Pipeline
    else:
        from diffusers import HunyuanVideoPipeline as Pipeline

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = Pipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    return pipe


def resolve_adapter(path, checkpoint=None):
    """A directory holding LoRA weights.

    Accepts the weight directory itself, or a parent holding checkpoint-*, in
    which case `checkpoint` picks one and the newest is used without it.
    Pinning matters: generating step 100 and then step 250 into the same place
    would silently keep the first, because existing files are skipped.
    """
    names = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")

    def has_weights(directory):
        return any(os.path.isfile(os.path.join(directory, n)) for n in names)

    if checkpoint:
        pinned = os.path.join(path, checkpoint)
        if not has_weights(pinned):
            raise SystemExit("No LoRA weights in %s" % pinned)
        return pinned
    if has_weights(path):
        return path
    found = sorted(os.path.join(path, d) for d in os.listdir(path)
                   if d.startswith("checkpoint-")
                   and has_weights(os.path.join(path, d)))
    if not found:
        raise SystemExit("No LoRA weights under %s" % path)
    return found[-1]


def plan(cfg, base, erased, subjects_wanted, root="videos"):
    """The (column directory, erased concept, subject) triples to generate.

    The baseline is keyed by the prompt set rather than by the experiment,
    because that is what it depends on: the three object experiments differ
    only in the mapping they train towards and all three are read against the
    same untouched videos.
    """
    everyone = expconfig.subjects(cfg)
    chosen = subjects_wanted or everyone
    for subject in chosen:
        if subject not in everyone:
            raise SystemExit("%r is not a subject of %s. Known: %s"
                             % (subject, cfg["prompts"], ", ".join(everyone)))
    if base:
        return [(layout.base_videos(cfg["prompts"], root=root), None, s)
                for s in chosen]

    columns = [r.concept for r in expconfig.rows(cfg)]
    if erased:
        if erased not in columns:
            raise SystemExit("%r is not a column of %s. Columns: %s"
                             % (erased, cfg["name"], ", ".join(columns)))
        columns = [erased]
    return [(layout.column_videos(cfg["name"], c, root=root), c, s)
            for c in columns for s in chosen]


def main():
    ap = argparse.ArgumentParser(description="Generate one experiment's videos.")
    ap.add_argument("config")
    ap.add_argument("--base", action="store_true",
                    help="Generate the untouched model instead of a column.")
    ap.add_argument("--erased", default=None,
                    help="Restrict to the column that erases this concept.")
    ap.add_argument("--subject", nargs="*", default=None,
                    help="Restrict to these subjects. Default: all of them.")
    ap.add_argument("--all", action="store_true",
                    help="No restriction; generate the whole matrix here.")
    ap.add_argument("--videos-root", default="videos")
    ap.add_argument("--adapters-root", default="adapters")
    ap.add_argument("--lora-weight", type=float, default=1.0,
                    help="Adapter strength at inference. Scaling it walks the "
                         "erasure/damage trade-off without retraining.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Exit zero when a column has no trained adapter, so "
                         "an array submitted as a dependency can be re-run to "
                         "fill gaps instead of failing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and the prompts, generate nothing.")
    args = ap.parse_args()

    cfg = expconfig.load(args.config)
    if args.all and (args.erased or args.subject):
        raise SystemExit("--all takes no --erased or --subject")
    if args.base and cfg["method"] == "safree":
        raise SystemExit("A SAFREE experiment has no baseline of its own: the "
                         "pipeline class is replaced, so the untouched model is "
                         "not available in the same process. Generate the "
                         "baseline from a fomo experiment on the same prompts.")

    work = plan(cfg, args.base, args.erased, args.subject,
                root=args.videos_root)

    sampling = cfg["sampling"]
    method = "base" if args.base else cfg["method"]

    print("experiment : %s (%s)" % (cfg["name"], cfg["kind"]))
    print("method     : %s" % method)
    print("size       : %dx%d, %d frames, %d steps, gs %s"
          % (sampling["width"], sampling["height"], sampling["num_frames"],
             sampling["num_inference_steps"], sampling["guidance_scale"]))
    print("cells      : %d" % len(work))
    print()

    if args.dry_run:
        for column, erased, subject in work:
            print("%s  (erased: %s)" % (layout.subject_videos(column, subject),
                                        erased or "none"))
            for index, prompt in enumerate(expconfig.prompts_for(cfg, subject)):
                print("   %02d  %s" % (index, prompt))
        return

    # Adapters, resolved before the model is loaded so that a missing one costs
    # nothing. One per column; a cell inside a column reuses it.
    adapters = {}
    if method == "fomo":
        for _, erased, _ in work:
            if erased in adapters:
                continue
            directory = layout.adapter_dir(
                cfg["name"], erased, cfg["train"]["mask_mode"],
                cfg["train"]["imap_alpha"], root=args.adapters_root)
            if not os.path.isdir(directory):
                if args.allow_missing:
                    print("SKIP %s: nothing trained under %s" % (erased, directory))
                    adapters[erased] = None
                    continue
                raise SystemExit("No adapter for %r under %s" % (erased, directory))
            adapters[erased] = resolve_adapter(directory, cfg["train"]["checkpoint"])
            print("adapter    : %s -> %s" % (erased, adapters[erased]))
        work = [cell for cell in work if adapters.get(cell[1]) is not None]
        if not work:
            print("Nothing to generate.")
            return

    import torch
    from diffusers.utils import export_to_video

    print("\nLoading %s ..." % sampling["model_path"])
    pipe = load_pipeline(sampling["model_path"], safree=(method == "safree"))

    loaded = set()
    extra_static = {}
    if method == "safree":
        from safree.concepts import CONCEPTS as SAFREE_TERMS
    elif method == "negprompt":
        extra_static = dict(true_cfg_scale=sampling["guidance_scale"],
                            guidance_scale=1.0)

    started, done, total = time.time(), 0, sum(
        len(expconfig.prompts_for(cfg, s)) for _, _, s in work)

    for column, erased, subject in work:
        out_dir = layout.subject_videos(column, subject)
        os.makedirs(out_dir, exist_ok=True)

        if method == "fomo":
            name = layout.slugify(erased)
            if name not in loaded:
                pipe.load_lora_weights(adapters[erased], adapter_name=name)
                loaded.add(name)
            pipe.enable_lora()
            pipe.set_adapters([name], adapter_weights=[args.lora_weight])
            extra = dict(guidance_scale=sampling["guidance_scale"])
        elif method == "negprompt":
            extra = dict(extra_static, negative_prompt=erased)
        elif method == "safree":
            if erased not in SAFREE_TERMS:
                raise SystemExit("No SAFREE term list for %r" % erased)
            extra = dict(guidance_scale=sampling["guidance_scale"],
                         concept=SAFREE_TERMS[erased])
        else:
            if hasattr(pipe, "disable_lora"):
                pipe.disable_lora()
            extra = dict(guidance_scale=sampling["guidance_scale"])

        print("\n=== erase %-22s generate %-22s ==="
              % (erased or "nothing", subject))
        for index, prompt in enumerate(expconfig.prompts_for(cfg, subject)):
            out = os.path.join(out_dir, "%02d.mp4" % index)
            done += 1
            if os.path.exists(out):
                print("  [%d/%d] exists, skipped" % (done, total))
                continue
            generator = torch.Generator(device="cpu").manual_seed(
                sampling["seed"] + index)
            frames = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=sampling["num_inference_steps"],
                num_frames=sampling["num_frames"],
                height=sampling["height"],
                width=sampling["width"],
                generator=generator,
                **extra
            ).frames[0]
            export_to_video(frames, out, fps=sampling["fps"])
            print("  [%d/%d] %s  (%.1f min)"
                  % (done, total, os.path.basename(out),
                     (time.time() - started) / 60))
            del frames
            gc.collect()
            torch.cuda.empty_cache()

    print("\nDone in %.1f min." % ((time.time() - started) / 60))


if __name__ == "__main__":
    main()
