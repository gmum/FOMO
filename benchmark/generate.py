"""Generate evaluation videos for one class of one table column.

Protocol: 20 prompts per concept, 17-frame videos, one model load per job.

Output layout:

    <output_dir>/<class_slug>/<method>/NN.mp4
    <output_dir>/prompts/<class_slug>.json

Methods:

    base        base model, no intervention
    negprompt   base model with a negative prompt (true CFG; see below)
    <label>     any LoRA passed via --models label=path

The seed depends only on the prompt index (base_seed + NN), never on the
method, so the same prompt gets the same noise in every method and the
comparison between rows is paired.

Note on negative prompts: HunyuanVideo is CFG-distilled, so `guidance_scale`
feeds a guidance embedding rather than performing real two-branch guidance.
A negative prompt therefore has nowhere to go unless real CFG is switched on.
--negprompt does exactly that: guidance_scale=1.0 plus true_cfg_scale>1. This
means the negprompt rows are sampled differently from the base row, which is
why a CFG-matched control (empty negative prompt) is worth generating too.

Run from the repository root.
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
from diffusers.utils import export_to_video

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safree_concepts import CONCEPTS as SAFREE_CONCEPTS  # noqa: E402
from common import (  # noqa: E402
    CLASS_SETS, IMAGENET_INDEX, MODEL_DEFAULTS, MODEL_FAMILIES,
    all_prompt_sets, prompts_for, slugify,
)


def load_pipeline(family, model_path):
    """Return a text-to-video pipeline on the GPU.

    Imported here rather than at module scope so that --dry_run and --help
    work in an environment that only has one of the two model families.
    """
    if family == "hunyuan":
        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_path, transformer=transformer, torch_dtype=torch.float16)
    elif family == "cogvideox":
        from diffusers import CogVideoXPipeline
        # CogVideoX-2b is released in float16; the 5b variants are bfloat16.
        dtype = torch.float16 if "2b" in model_path.lower() else torch.bfloat16
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.vae.enable_slicing()
    else:
        raise SystemExit("Unknown model family %r" % family)

    pipe.vae.enable_tiling()
    pipe.to("cuda")
    return pipe


def load_safree_pipeline(model_path):
    """SAFREE's own fork of the HunyuanVideo pipeline, from the repository root.

    It is a separate class rather than a flag on the stock pipeline, so it
    cannot share a process with the base model. Note that it hard-codes
    do_true_cfg = False, meaning it samples exactly like the base model and
    needs no CFG-matched control the way negprompt does.
    """
    import torch
    from diffusers import HunyuanVideoTransformer3DModel
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from safree_hunyuan_pipeline import HunyuanVideoPipeline as SafreePipeline

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = SafreePipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    return pipe


def resolve_lora_dir(path):
    """Return a directory holding LoRA weights.

    Accepts either the weight directory itself or a parent containing
    checkpoint-* subdirectories, in which case the newest one is used.
    """
    names = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")
    if any(os.path.isfile(os.path.join(path, n)) for n in names):
        return path
    checkpoints = sorted(
        os.path.join(path, d) for d in os.listdir(path)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(path, d))
        and any(os.path.isfile(os.path.join(path, d, n)) for n in names)
    )
    if checkpoints:
        return checkpoints[-1]
    raise SystemExit("No LoRA weights found under %s" % path)


def set_active_adapter(pipe, adapter_name, weight):
    if adapter_name is None:
        if hasattr(pipe, "disable_lora"):
            pipe.disable_lora()
        return
    if hasattr(pipe, "enable_lora"):
        pipe.enable_lora()
    pipe.set_adapters([adapter_name], adapter_weights=[weight])


def main():
    ap = argparse.ArgumentParser(description="Generate evaluation videos.")
    ap.add_argument("--concepts", nargs="+", required=True,
                    help='Classes to generate, e.g. --concepts "garbage truck" church')
    ap.add_argument("--output_dir", required=True,
                    help="One table column, e.g. videos/imagenette_v1/unlearn_church")
    ap.add_argument("--class_set", default="imagenette", choices=sorted(CLASS_SETS))
    ap.add_argument("--prompt_set", default="v1", choices=all_prompt_sets(),
                    help="Template set (v1-v4) or per-class scene set (v5-v7).")
    ap.add_argument("--num_prompts", type=int, default=20)
    ap.add_argument("--model_family", default="hunyuan", choices=MODEL_FAMILIES,
                    help="Which text-to-video model to sample from.")
    ap.add_argument("--model_path", default=None,
                    help="Defaults to the standard checkpoint for --model_family.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="LoRA methods as label=path, e.g. ours=outputs/imap_church/checkpoint-000150")
    ap.add_argument("--lora_weight", type=float, default=1.0)
    ap.add_argument("--no_base", action="store_true",
                    help="Skip the base model (e.g. when it is already symlinked in).")
    ap.add_argument("--negprompt", action="store_true",
                    help="Add a 'negprompt' method: base model with a negative prompt.")
    ap.add_argument("--negprompt_text", default=None,
                    help="Negative prompt text. Empty string is valid and means "
                         "the CFG-matched control. Unset falls back to the concept.")
    ap.add_argument("--negprompt_label", default="negprompt",
                    help="Method directory name. Use e.g. cfgbase for the control "
                         "so it does not overwrite the negprompt videos.")
    ap.add_argument("--safree", action="store_true",
                    help="Generate with the SAFREE pipeline instead of the stock one. "
                         "Cannot be combined with --models or --negprompt: SAFREE "
                         "replaces the pipeline class, not the conditioning.")
    ap.add_argument("--safree_label", default="safree")
    ap.add_argument("--true_cfg_scale", type=float, default=6.0)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the plan and the prompts, generate nothing.")
    args = ap.parse_args()

    if args.model_path is None:
        args.model_path = MODEL_DEFAULTS[args.model_family]["model_path"]

    # CogVideoX decodes in groups of four, so the frame count has to satisfy
    # (num_frames - 1) % 4 == 0. Catch it here rather than after a model load.
    if args.model_family == "cogvideox" and (args.num_frames - 1) % 4 != 0:
        raise SystemExit(
            "CogVideoX needs (num_frames - 1) %% 4 == 0; got %d. "
            "Nearby valid values: 17, 21, 25, 49." % args.num_frames)

    native = MODEL_DEFAULTS[args.model_family]
    if (args.width, args.height) != (native["width"], native["height"]):
        print("NOTE: %s is trained at %dx%d; generating at %dx%d is outside "
              "that distribution and low accuracy may reflect the resolution "
              "rather than the model."
              % (args.model_family, native["width"], native["height"],
                 args.width, args.height))

    classes = CLASS_SETS[args.class_set]
    for concept in args.concepts:
        if concept not in classes:
            print("WARNING: '%s' is not in class set '%s'" % (concept, args.class_set))

    # Resolved per concept: template sets render the same wording for every
    # class, per-class sets have their own scenes.
    prompts_by_concept = {
        concept: prompts_for(args.prompt_set, concept)[:args.num_prompts]
        for concept in args.concepts
    }
    n_prompts = max(len(v) for v in prompts_by_concept.values())

    # (label, adapter_name, kind); kind is base | lora | negprompt
    methods = []
    if not args.no_base:
        methods.append(("base", None, "base"))
    model_specs = []
    if args.models:
        for spec in args.models:
            if "=" not in spec:
                raise SystemExit("--models expects label=path, got %r" % spec)
            label, path = spec.split("=", 1)
            model_specs.append((label, resolve_lora_dir(path)))
    if args.negprompt:
        methods.append((args.negprompt_label, None, "negprompt"))
    if args.safree:
        if model_specs or args.negprompt:
            raise SystemExit("--safree cannot be combined with --models or --negprompt")
        # SAFREE swaps the pipeline class, so the base model is not available
        # in the same process and would otherwise be generated from it.
        methods = [(args.safree_label, None, "safree")]
        missing = [c for c in args.concepts if c not in SAFREE_CONCEPTS]
        if missing:
            raise SystemExit("No SAFREE term list for: %s" % ", ".join(missing))

    total = sum(len(prompts_by_concept[c]) for c in args.concepts) * (len(methods) + len(model_specs))
    print("Model    : %s (%s)" % (args.model_family, args.model_path))
    print("Concepts : %s" % ", ".join(args.concepts))
    print("Methods  : %s" % ", ".join([m for m, _, _ in methods] + [l for l, _ in model_specs]))
    print("Prompts  : %d (set %s)" % (n_prompts, args.prompt_set))
    print("Size     : %dx%d, %d frames" % (args.width, args.height, args.num_frames))
    print("Total    : %d videos" % total)
    print()

    # One prompt map per class. A single shared file would be overwritten by
    # each array task writing into the same output directory.
    prompts_dir = os.path.join(args.output_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    settings = {
        "height": args.height, "width": args.width,
        "num_frames": args.num_frames, "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale, "base_seed": args.seed,
    }
    for concept in args.concepts:
        prompts = prompts_by_concept[concept]
        payload = {
            "concept": concept,
            "class_set": args.class_set,
            "prompt_set": args.prompt_set,
            "model_family": args.model_family,
            "model_path": args.model_path,
            "imagenet_index": IMAGENET_INDEX.get(concept),
            "prompts": {"%02d" % i: p for i, p in enumerate(prompts)},
            "seeds": {"%02d" % i: args.seed + i for i in range(len(prompts))},
            "settings": settings,
        }
        path = os.path.join(prompts_dir, slugify(concept) + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print("Wrote prompt map: %s" % path)

    if args.dry_run:
        for concept in args.concepts:
            print("\n%s:" % concept)
            for i, prompt in enumerate(prompts_by_concept[concept]):
                print("  %02d  %s" % (i, prompt))
        return

    print("\nLoading %s (%s%s) ..."
          % (args.model_path, args.model_family, ", SAFREE" if args.safree else ""))
    if args.safree:
        pipe = load_safree_pipeline(args.model_path)
    else:
        pipe = load_pipeline(args.model_family, args.model_path)

    for label, weights_dir in model_specs:
        pipe.load_lora_weights(weights_dir, adapter_name=label)
        methods.append((label, label, "lora"))
        print("Loaded LoRA '%s' from %s" % (label, weights_dir))

    started, done = time.time(), 0
    for concept in args.concepts:
        slug = slugify(concept)
        for method, adapter, kind in methods:
            method_dir = os.path.join(args.output_dir, slug, method)
            os.makedirs(method_dir, exist_ok=True)

            # negprompt runs on the base model, so any LoRA must be off.
            set_active_adapter(pipe, adapter, args.lora_weight)

            if kind == "negprompt":
                # An empty string is a valid value: it selects the CFG-matched
                # control. Distinguish it from "not given" with `is None`,
                # otherwise "" would silently fall back to negating the concept.
                negative = concept if args.negprompt_text is None else args.negprompt_text
                if args.model_family == "hunyuan":
                    # CFG-distilled: real two-branch guidance has to be turned
                    # on explicitly, which also changes how the row is sampled.
                    extra = dict(negative_prompt=negative,
                                 true_cfg_scale=args.true_cfg_scale,
                                 guidance_scale=1.0)
                    print("\n=== %s | %s (negative_prompt=%r, true_cfg=%s) ==="
                          % (concept, method, negative, args.true_cfg_scale))
                else:
                    # CogVideoX already does two-branch CFG, so the negative
                    # prompt takes effect without changing the sampler and no
                    # CFG-matched control is needed.
                    extra = dict(negative_prompt=negative,
                                 guidance_scale=args.guidance_scale)
                    print("\n=== %s | %s (negative_prompt=%r, gs=%s) ==="
                          % (concept, method, negative, args.guidance_scale))
            elif kind == "safree":
                terms = SAFREE_CONCEPTS[concept]
                extra = dict(guidance_scale=args.guidance_scale, concept=terms)
                print("\n=== %s | %s (%d SAFREE terms) ==="
                      % (concept, method, len(terms)))
            else:
                extra = dict(guidance_scale=args.guidance_scale)
                print("\n=== %s | %s ===" % (concept, method))

            for index, prompt in enumerate(prompts_by_concept[concept]):
                out = os.path.join(method_dir, "%02d.mp4" % index)
                done += 1
                if args.skip_existing and os.path.exists(out):
                    print("  [%d/%d] [skipped] %s" % (done, total, os.path.basename(out)))
                    continue
                generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
                frames = pipe(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=args.num_inference_steps,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    **extra,
                ).frames[0]
                export_to_video(frames, out, fps=args.fps)
                elapsed = time.time() - started
                print("  [%d/%d] %s  (%.1f min elapsed)" % (done, total, prompt, elapsed / 60))
                del frames
                gc.collect()
                torch.cuda.empty_cache()

    print("\nDone in %.1f min -> %s" % ((time.time() - started) / 60, args.output_dir))


if __name__ == "__main__":
    main()
