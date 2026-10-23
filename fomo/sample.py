"""Generate one video, with or without a trained adapter.

For looking at something quickly: give it a prompt, say which adapter (or
none), get an mp4. Nothing here knows about
experiments or prompt sets.

    python3 -m fomo.sample --prompt "A video of a tench fish." --out base.mp4

    python3 -m fomo.sample --prompt "A video of a tench fish." --out erased.mp4 \
        --adapter adapters/objects_mapping2/tench/unclamped_alpha7

With the same --seed the two differ only by the adapter, which is what makes
putting them side by side mean anything.

The adapter the trainer writes is a standard PEFT LoRA, so loading it needs
nothing beyond diffusers. `load_pipeline` and `resolve_adapter` live here
rather than in the experiment code, so the method stays usable on its own.
"""

import argparse
import os

WEIGHT_NAMES = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")


def resolve_adapter(path, checkpoint=None):
    """A directory holding LoRA weights.

    Accepts the weight directory itself, or a parent holding checkpoint-*
    subdirectories. In that case `checkpoint` picks one by name and the newest
    is used without it. Pinning matters once a run has several: generating
    step 100 and then step 250 into the same place would silently keep the
    first, because existing files are skipped.
    """
    def has_weights(directory):
        return any(os.path.isfile(os.path.join(directory, n)) for n in WEIGHT_NAMES)

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


def load_pipeline(model_path="hunyuanvideo-community/HunyuanVideo", safree=False):
    """The text-to-video pipeline on the GPU.

    Imported inside the function so that --help and the dry runs elsewhere work
    in an environment without torch, such as a cluster login node.
    """
    import torch
    from diffusers import HunyuanVideoTransformer3DModel

    if safree:
        # A separate pipeline class, not a flag: it cannot share a process with
        # the stock one, which is why a SAFREE run generates no baseline.
        from methods.safree.pipeline import HunyuanVideoPipeline as Pipeline
    else:
        from diffusers import HunyuanVideoPipeline as Pipeline

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = Pipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    return pipe


def main():
    ap = argparse.ArgumentParser(description="Sample one video, with or without an adapter.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, help="Path of the .mp4 to write.")
    ap.add_argument("--adapter", default=None,
                    help="Adapter directory, or a parent holding checkpoint-*. "
                         "Leave it out to sample the untouched model.")
    ap.add_argument("--checkpoint", default=None,
                    help="Pick one checkpoint by name, e.g. checkpoint-000006. "
                         "Default: the newest one under --adapter.")
    ap.add_argument("--adapter-weight", type=float, default=1.0,
                    help="Adapter strength. Scaling it walks the "
                         "erasure/damage trade-off without retraining.")
    ap.add_argument("--negative", default=None,
                    help="Negative prompt. HunyuanVideo is CFG-distilled, so "
                         "this also switches real two-branch guidance on, "
                         "which changes how the video is sampled.")
    ap.add_argument("--model-path", default="hunyuanvideo-community/HunyuanVideo")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--num-frames", type=int, default=17,
                    help="HunyuanVideo expects a count of the form 4k+1.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    if (args.num_frames - 1) % 4:
        raise SystemExit("--num-frames must be of the form 4k+1 (17, 33, 49, 65).")

    import torch
    from diffusers.utils import export_to_video

    pipe = load_pipeline(args.model_path)

    extra = dict(guidance_scale=args.guidance)
    if args.adapter:
        weights = resolve_adapter(args.adapter, args.checkpoint)
        pipe.load_lora_weights(weights, adapter_name="erase")
        pipe.set_adapters(["erase"], adapter_weights=[args.adapter_weight])
        print("adapter : %s (weight %s)" % (weights, args.adapter_weight))
    else:
        print("adapter : none (untouched model)")
    if args.negative is not None:
        extra = dict(negative_prompt=args.negative,
                     true_cfg_scale=args.guidance, guidance_scale=1.0)
        print("negative: %r (real CFG on)" % args.negative)

    print("prompt  : %s" % args.prompt)
    print("size    : %dx%d, %d frames, %d steps, seed %d"
          % (args.width, args.height, args.num_frames, args.steps, args.seed))

    frames = pipe(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        num_inference_steps=args.steps,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        **extra
    ).frames[0]

    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    export_to_video(frames, args.out, fps=args.fps)
    print("wrote   : %s" % args.out)


if __name__ == "__main__":
    main()
