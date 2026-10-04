import argparse
import csv
import json
import re
from pathlib import Path


def slugify(text, max_len=72):
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len].strip("_") or "prompt"


def read_prompts(path, prompt_field=None, start_index=0, limit=None):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(handle)
            keys = [prompt_field] if prompt_field else [
                "prompt", "sensitive prompt", "source_prompt", "text", "caption",
            ]
            field = next((key for key in keys if key in (reader.fieldnames or [])), None)
            if field is None:
                raise ValueError(f"No prompt column in {path}")
            rows = list(reader)
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
            keys = [prompt_field] if prompt_field else [
                "prompt", "source_prompt", "sensitive prompt", "text", "caption",
            ]
            field = None
        else:
            rows = [{"prompt": line.strip()} for line in handle if line.strip()]
            field, keys = "prompt", ["prompt"]

    prompts = []
    for row in rows:
        prompt = row.get(field) if field else next((row[key] for key in keys if row.get(key)), None)
        if prompt:
            index = int(row["index"]) if row.get("index") is not None else len(prompts)
            prompts.append({"index": index, "prompt": str(prompt)})
    prompts = prompts[start_index:] if limit is None else prompts[start_index:start_index + limit]
    if not prompts:
        raise ValueError(f"No prompts selected from {path}")
    indices = [row["index"] for row in prompts]
    if len(set(indices)) != len(indices) or any(index < 0 or index > 999 for index in indices):
        raise ValueError("Prompt indices must be unique integers between 0 and 999")
    return prompts


def resolve_lora_dir(path):
    path = Path(path)
    weight_names = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")
    if any((path / name).is_file() for name in weight_names):
        return path
    checkpoints = sorted(
        directory for directory in path.glob("checkpoint-*")
        if directory.is_dir() and any((directory / name).is_file() for name in weight_names)
    )
    if not checkpoints:
        raise FileNotFoundError(f"No LoRA weights under {path}")
    return checkpoints[-1]


def load_pipeline(args, lora_path):
    import torch
    from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
        revision=args.revision,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model_path, transformer=transformer, torch_dtype=torch.bfloat16,
        revision=args.revision,
    )
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    if lora_path:
        weight_name = "pytorch_lora_weights.safetensors"
        if not (lora_path / weight_name).is_file():
            weight_name = "pytorch_lora_weights.bin"
        pipe.load_lora_weights(str(lora_path), weight_name=weight_name, adapter_name=args.adapter_name)
        pipe.set_adapters([args.adapter_name], adapter_weights=[args.lora_weight])
        if hasattr(pipe, "enable_lora"):
            pipe.enable_lora()
        print(f"LoRA: {lora_path} (weight {args.lora_weight})", flush=True)
    return pipe


def generate_one(pipe, prompt, output_path, *, seed, height, width, num_frames,
                 num_inference_steps, guidance_scale, fps):
    import torch
    from diffusers.utils import export_to_video

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        frames = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
    export_to_video(frames, str(output_path), fps=fps)
    torch.cuda.empty_cache()


def save_metadata(path, record):
    import fcntl

    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    previous = json.loads(line)
                    records[previous["index"]] = previous
        previous = records.get(record["index"])
        if previous and any(previous.get(key) != value for key, value in record.items() if key != "status"):
            raise ValueError(f"Generation settings differ for index {record['index']}; use a new output directory")
        records[record["index"]] = record
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(records[index]) + "\n" for index in sorted(records)), encoding="utf-8",
        )
        temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HunyuanVideo videos with or without LoRA.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompts_file", type=Path, help="CSV, JSONL or TXT prompt list.")
    parser.add_argument("--output", type=Path, help="MP4 path for a single prompt.")
    parser.add_argument("--output_root", type=Path, help="Dataset output root.")
    parser.add_argument("--variant_name", help="Dataset variant, e.g. baseline or checkpoint_000050.")
    parser.add_argument("--prompt_field")
    parser.add_argument("--limit", type=int, help="Number of prompts; default: all.")
    parser.add_argument("--start_index", type=int, default=0, help="Skip this many prompt-list entries.")
    parser.add_argument("--lora", "--lora_path", dest="lora_path", default="")
    parser.add_argument("--lora_weight", type=float, default=1.0)
    parser.add_argument("--adapter_name", default="imap_h_lora")
    parser.add_argument("--model_path", default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--revision", default="e8c2aaa66fe3742a32c11a6766aecbf07c56e773")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    if args.prompt is not None:
        if args.output is None or args.output_root is not None or args.variant_name is not None:
            parser.error("Use --prompt with --output, without --output_root or --variant_name")
    elif args.output is not None or args.output_root is None or not args.variant_name:
        parser.error("Use --prompts_file with --output_root and --variant_name, without --output")
    if min(args.height, args.width, args.num_frames, args.num_inference_steps, args.fps) <= 0:
        parser.error("Dimensions, frames, inference steps and FPS must be positive")
    if (args.num_frames - 1) % 4:
        parser.error("--num_frames must be of the form 4k+1")
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--start_index must be nonnegative and --limit must be positive")
    return args


def main():
    args = parse_args()
    lora_text = args.lora_path.strip()
    lora_path = None
    if lora_text and lora_text.lower() not in {"none", "baseline", "null"}:
        lora_path = resolve_lora_dir(lora_text)

    settings = {key: getattr(args, key) for key in (
        "seed", "height", "width", "num_frames", "num_inference_steps", "guidance_scale", "fps",
    )}
    if args.prompt is not None:
        if args.skip_existing and args.output.exists():
            print(f"Skipped: {args.output}")
            return
        pipe = load_pipeline(args, lora_path)
        generate_one(pipe, args.prompt, args.output, **settings)
        print(f"Wrote: {args.output}")
        return

    prompts = read_prompts(args.prompts_file, args.prompt_field, args.start_index, args.limit)
    variant = slugify(args.variant_name, max_len=96)
    output_dir = args.output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    records = {}
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["index"]] = record

    jobs = []
    for row in prompts:
        output_path = output_dir / "videos" / f"{row['index']:03d}_seed{args.seed}_{slugify(row['prompt'])}.mp4"
        record = {
            **row, **settings, "variant": variant,
            "lora_path": str(lora_path) if lora_path else "", "output_path": str(output_path),
            "model_path": args.model_path, "model_revision": args.revision,
            "lora_weight": args.lora_weight, "adapter_name": args.adapter_name,
        }
        previous = records.get(row["index"])
        if previous and any(previous.get(key) != value for key, value in record.items()):
            raise ValueError(f"Generation settings differ for index {row['index']}; use a new output directory")
        jobs.append((output_path, record))

    pipe = None
    for position, (output_path, record) in enumerate(jobs, start=1):
        if args.skip_existing and output_path.exists():
            record["status"] = "skipped"
        else:
            if pipe is None:
                pipe = load_pipeline(args, lora_path)
            generate_one(pipe, record["prompt"], output_path, **settings)
            record["status"] = "generated"
        save_metadata(metadata_path, record)
        print(f"[{position}/{len(jobs)}] {record['status']}: {output_path.name}", flush=True)


if __name__ == "__main__":
    main()
