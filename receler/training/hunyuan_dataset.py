from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers.utils import export_to_video, load_video

from .token_utils import find_concept_token_indices


@dataclass
class VideoConceptExample:
    file: str
    prompt: str
    concept: str
    neutral_prompt: str


DEFAULT_BOOTSTRAP_SCENES = [
    ("a {concept} walking in a garden, realistic video", "a quiet garden path, realistic video"),
    ("a {concept} sitting on a sofa, realistic video", "an empty sofa in a living room, realistic video"),
    ("a {concept} running through grass, realistic video", "wind moving through green grass, realistic video"),
    ("a close-up video of a {concept} face, realistic", "a close-up video of soft indoor light and fabric texture, realistic"),
    ("a {concept} sleeping on a blanket, realistic video", "an empty blanket on a bed, realistic video"),
    ("a {concept} playing near a lake, realistic video", "ripples on a quiet lake shore, realistic video"),
    ("a {concept} standing on a city sidewalk, realistic video", "an empty city sidewalk, realistic video"),
    ("a {concept} jumping over a small log in a forest, realistic video", "a small log on a forest path, realistic video"),
    ("a {concept} looking out of a car window, realistic video", "a car window with passing scenery outside, realistic video"),
    ("a {concept} walking along a beach at sunset, realistic video", "waves moving along a beach at sunset, realistic video"),
    ("a {concept} playing with a ball in a park, realistic video", "a ball resting on grass in a park, realistic video"),
    ("a {concept} resting on a wooden porch, realistic video", "an empty wooden porch in soft daylight, realistic video"),
    ("a {concept} running through shallow snow, realistic video", "fresh tracks in shallow snow, realistic video"),
    ("a {concept} sitting beside a window indoors, realistic video", "sunlight falling beside a window indoors, realistic video"),
    ("a {concept} walking on a mountain trail, realistic video", "an empty mountain trail, realistic video"),
    ("a {concept} sniffing flowers in a meadow, realistic video", "flowers moving gently in a meadow, realistic video"),
    ("a {concept} crossing a quiet street, realistic video", "a quiet empty street, realistic video"),
    ("a {concept} lying on a rug in a living room, realistic video", "an empty rug in a living room, realistic video"),
    ("a {concept} running beside a person in a park, realistic video", "a person running alone in a park, realistic video"),
    ("a {concept} turning its head toward the camera, realistic video", "a steady camera view of an indoor room, realistic video"),
]


def read_metadata(dataset_dir: str | Path) -> List[VideoConceptExample]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return []
    examples = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            examples.append(VideoConceptExample(**item))
    return examples


def write_metadata(dataset_dir: str | Path, examples: Iterable[VideoConceptExample]) -> None:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.__dict__, ensure_ascii=False) + "\n")


def ensure_video_dataset(
    pipe,
    dataset_dir: str | Path,
    concept: str,
    num_videos: int = 5,
    width: int = 640,
    height: int = 360,
    num_frames: int = 33,
    fps: int = 15,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
    seed: int = 1234,
    prompt_override: str | None = None,
    neutral_prompt_override: str | None = None,
) -> List[VideoConceptExample]:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    examples = read_metadata(dataset_dir)
    existing_files = {example.file for example in examples if (dataset_dir / example.file).exists()}

    needed = max(0, num_videos - len(existing_files))
    if needed == 0:
        return examples

    start_index = 0
    while f"{start_index:03d}.mp4" in existing_files or (dataset_dir / f"{start_index:03d}.mp4").exists():
        start_index += 1

    initial_count = len(examples)
    for offset in range(needed):
        if prompt_override is None:
            scene_idx = (initial_count + offset) % len(DEFAULT_BOOTSTRAP_SCENES)
            prompt_tmpl, neutral_tmpl = DEFAULT_BOOTSTRAP_SCENES[scene_idx]
            prompt = prompt_tmpl.format(concept=concept)
            neutral_prompt = neutral_tmpl.format(concept=concept)
        else:
            prompt = prompt_override.format(concept=concept)
            neutral_prompt = "" if neutral_prompt_override is None else neutral_prompt_override.format(concept=concept)
        file_name = f"{start_index + offset:03d}.mp4"
        generator = torch.Generator(device=pipe._execution_device).manual_seed(seed + start_index + offset)
        with torch.no_grad():
            frames = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        export_to_video(frames, str(dataset_dir / file_name), fps=fps)
        examples.append(
            VideoConceptExample(
                file=file_name,
                prompt=prompt,
                concept=concept,
                neutral_prompt=neutral_prompt,
            )
        )
        existing_files.add(file_name)

    write_metadata(dataset_dir, examples)
    return examples


def ensure_prompt_metadata(
    dataset_dir: str | Path,
    concept: str,
    num_prompts: int = 5,
    prompt_override: str | None = None,
    neutral_prompt_override: str | None = None,
) -> List[VideoConceptExample]:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    examples = read_metadata(dataset_dir)
    if len(examples) >= num_prompts:
        return examples

    existing_names = {example.file for example in examples}
    start_index = 0
    while f"{start_index:03d}.prompt" in existing_names:
        start_index += 1

    initial_count = len(examples)
    needed = num_prompts - len(examples)
    for offset in range(needed):
        if prompt_override is None:
            scene_idx = (initial_count + offset) % len(DEFAULT_BOOTSTRAP_SCENES)
            prompt_tmpl, neutral_tmpl = DEFAULT_BOOTSTRAP_SCENES[scene_idx]
            prompt = prompt_tmpl.format(concept=concept)
            neutral_prompt = neutral_tmpl.format(concept=concept)
        else:
            prompt = prompt_override.format(concept=concept)
            neutral_prompt = "" if neutral_prompt_override is None else neutral_prompt_override.format(concept=concept)
        examples.append(
            VideoConceptExample(
                file=f"{start_index + offset:03d}.prompt",
                prompt=prompt,
                concept=concept,
                neutral_prompt=neutral_prompt,
            )
        )

    write_metadata(dataset_dir, examples)
    return examples


class HunyuanVideoConceptDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        height: int,
        width: int,
        num_frames: int,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.examples = read_metadata(self.dataset_dir)
        if not self.examples:
            raise ValueError(f"No examples found in {self.dataset_dir / 'metadata.jsonl'}")
        missing = [example.file for example in self.examples if not (self.dataset_dir / example.file).exists()]
        if missing:
            raise FileNotFoundError(f"Metadata references missing videos: {missing}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index % len(self.examples)]
        video_path = self.dataset_dir / example.file
        pixel_values = load_video_tensor(video_path, self.height, self.width, self.num_frames)
        return {
            "pixel_values": pixel_values,
            "prompt": example.prompt,
            "concept": example.concept,
            "neutral_prompt": example.neutral_prompt,
            "file": example.file,
        }


def _to_pil(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def load_video_tensor(video_path: str | Path, height: int, width: int, num_frames: int) -> torch.Tensor:
    frames = load_video(str(video_path))
    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)
    tensors = []
    for index in indices:
        image = _to_pil(frames[int(index)]).resize((width, height), resample=Image.BICUBIC)
        array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor)
    return torch.stack(tensors, dim=1).contiguous()


def default_hunyuan_cache_dir(
    dataset_dir: str | Path,
    height: int,
    width: int,
    num_frames: int,
    max_sequence_length: int,
    neutral_prompt_override: str | None = None,
) -> Path:
    neutral_tag = ""
    if neutral_prompt_override is not None:
        if neutral_prompt_override == "":
            neutral_tag = "_neutral_empty"
        else:
            digest = hashlib.sha1(neutral_prompt_override.encode("utf-8")).hexdigest()[:8]
            neutral_tag = f"_neutral_{digest}"
    return (
        Path(dataset_dir)
        / "cache"
        / f"hunyuan_{width}x{height}_f{num_frames}_seq{max_sequence_length}{neutral_tag}"
    )


def _cache_file_for_example(cache_dir: str | Path, example: VideoConceptExample) -> Path:
    return Path(cache_dir) / f"{Path(example.file).stem}.pt"


def _cache_metadata_matches(
    item: Dict,
    example: VideoConceptExample,
    *,
    model_path: str,
    height: int,
    width: int,
    num_frames: int,
    max_sequence_length: int,
    neutral_prompt_override: str | None = None,
) -> bool:
    effective_neutral_prompt = example.neutral_prompt if neutral_prompt_override is None else neutral_prompt_override
    return (
        item.get("file") == example.file
        and item.get("prompt") == example.prompt
        and item.get("neutral_prompt") == effective_neutral_prompt
        and item.get("concept") == example.concept
        and item.get("model_path") == model_path
        and int(item.get("height", -1)) == int(height)
        and int(item.get("width", -1)) == int(width)
        and int(item.get("num_frames", -1)) == int(num_frames)
        and int(item.get("max_sequence_length", -1)) == int(max_sequence_length)
    )


def _load_cache_if_valid(
    cache_path: Path,
    example: VideoConceptExample,
    *,
    model_path: str,
    height: int,
    width: int,
    num_frames: int,
    max_sequence_length: int,
    neutral_prompt_override: str | None = None,
    require_video_latents: bool = True,
) -> Dict | None:
    if not cache_path.exists():
        return None
    try:
        item = torch.load(cache_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(item, dict):
        return None
    if not _cache_metadata_matches(
        item,
        example,
        model_path=model_path,
        height=height,
        width=width,
        num_frames=num_frames,
        max_sequence_length=max_sequence_length,
        neutral_prompt_override=neutral_prompt_override,
    ):
        return None
    required = [
        "prompt_embeds",
        "pooled_prompt_embeds",
        "prompt_attention_mask",
        "neutral_prompt_embeds",
        "neutral_pooled_prompt_embeds",
        "neutral_prompt_attention_mask",
        "concept_token_indices",
    ]
    if require_video_latents:
        required.append("latents")
    if any(key not in item for key in required):
        return None
    return item


@torch.no_grad()
def _encode_prompt_for_cache(
    pipe,
    prompt: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        device=device,
        max_sequence_length=max_sequence_length,
    )
    return (
        prompt_embeds.to(dtype=dtype).detach().cpu(),
        pooled_prompt_embeds.to(dtype=dtype).detach().cpu(),
        prompt_attention_mask.to(dtype=dtype).detach().cpu(),
    )


@torch.no_grad()
def _encode_video_latents_for_cache(pipe, pixel_values: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    pixel_values = pixel_values.to(device=device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    return latents.detach().cpu()


def ensure_hunyuan_training_cache(
    pipe,
    dataset_dir: str | Path,
    cache_dir: str | Path,
    *,
    model_path: str,
    height: int,
    width: int,
    num_frames: int,
    max_sequence_length: int,
    dtype: torch.dtype,
    neutral_prompt_override: str | None = None,
    require_video_latents: bool = True,
) -> List[Path]:
    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    examples = read_metadata(dataset_dir)
    if not examples:
        raise ValueError(f"No examples found in {dataset_dir / 'metadata.jsonl'}")

    if require_video_latents:
        missing = [example.file for example in examples if not (dataset_dir / example.file).exists()]
        if missing:
            raise FileNotFoundError(f"Metadata references missing videos: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_paths: List[Path] = []
    for example in tqdm(examples, desc="hunyuan-cache"):
        cache_path = _cache_file_for_example(cache_dir, example)
        cached = _load_cache_if_valid(
            cache_path,
            example,
            model_path=model_path,
            height=height,
            width=width,
            num_frames=num_frames,
            max_sequence_length=max_sequence_length,
            neutral_prompt_override=neutral_prompt_override,
            require_video_latents=require_video_latents,
        )
        if cached is not None:
            cache_paths.append(cache_path)
            continue

        if require_video_latents:
            pixel_values = load_video_tensor(dataset_dir / example.file, height, width, num_frames).unsqueeze(0)
            latents = _encode_video_latents_for_cache(pipe, pixel_values, device=device)
        effective_neutral_prompt = example.neutral_prompt if neutral_prompt_override is None else neutral_prompt_override
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = _encode_prompt_for_cache(
            pipe,
            example.prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
        neutral_prompt_embeds, neutral_pooled_prompt_embeds, neutral_prompt_attention_mask = _encode_prompt_for_cache(
            pipe,
            effective_neutral_prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
        concept_token_indices = find_concept_token_indices(
            pipe.tokenizer,
            prompt=example.prompt,
            concept=example.concept,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            max_sequence_length=max_sequence_length,
        )

        cache_item = {
            "file": example.file,
            "prompt": example.prompt,
            "neutral_prompt": effective_neutral_prompt,
            "metadata_neutral_prompt": example.neutral_prompt,
            "neutral_prompt_override": neutral_prompt_override,
            "concept": example.concept,
            "model_path": model_path,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "max_sequence_length": int(max_sequence_length),
            "prompt_only": not require_video_latents,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "neutral_prompt_embeds": neutral_prompt_embeds,
            "neutral_pooled_prompt_embeds": neutral_pooled_prompt_embeds,
            "neutral_prompt_attention_mask": neutral_prompt_attention_mask,
            "concept_token_indices": concept_token_indices,
        }
        if require_video_latents:
            cache_item["latents"] = latents.squeeze(0)
        torch.save(cache_item, cache_path)
        cache_paths.append(cache_path)

        if require_video_latents:
            del pixel_values, latents
        del prompt_embeds, pooled_prompt_embeds, prompt_attention_mask
        del neutral_prompt_embeds, neutral_pooled_prompt_embeds, neutral_prompt_attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "model_path": model_path,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "max_sequence_length": int(max_sequence_length),
        "prompt_only": not require_video_latents,
        "files": [path.name for path in cache_paths],
    }
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return cache_paths


class CachedHunyuanVideoConceptDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        cache_dir: str | Path,
        *,
        model_path: str,
        height: int,
        width: int,
        num_frames: int,
        max_sequence_length: int,
        neutral_prompt_override: str | None = None,
        require_video_latents: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = Path(cache_dir)
        self.model_path = model_path
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.max_sequence_length = int(max_sequence_length)
        self.neutral_prompt_override = neutral_prompt_override
        self.require_video_latents = require_video_latents
        self.examples = read_metadata(self.dataset_dir)
        if not self.examples:
            raise ValueError(f"No examples found in {self.dataset_dir / 'metadata.jsonl'}")

        missing_cache = []
        for example in self.examples:
            cache_path = _cache_file_for_example(self.cache_dir, example)
            cached = _load_cache_if_valid(
                cache_path,
                example,
                model_path=self.model_path,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                max_sequence_length=self.max_sequence_length,
                neutral_prompt_override=self.neutral_prompt_override,
                require_video_latents=self.require_video_latents,
            )
            if cached is None:
                missing_cache.append(str(cache_path))
        if missing_cache:
            raise FileNotFoundError(
                "Missing or stale Hunyuan training cache files. "
                f"Run ensure_hunyuan_training_cache first. Bad files: {missing_cache}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index % len(self.examples)]
        return torch.load(_cache_file_for_example(self.cache_dir, example), map_location="cpu")


def collate_cached_hunyuan_batch(examples: List[Dict]) -> Dict:
    batch = {
        "prompt_embeds": torch.cat([example["prompt_embeds"] for example in examples], dim=0),
        "pooled_prompt_embeds": torch.cat([example["pooled_prompt_embeds"] for example in examples], dim=0),
        "prompt_attention_mask": torch.cat([example["prompt_attention_mask"] for example in examples], dim=0),
        "neutral_prompt_embeds": torch.cat([example["neutral_prompt_embeds"] for example in examples], dim=0),
        "neutral_pooled_prompt_embeds": torch.cat(
            [example["neutral_pooled_prompt_embeds"] for example in examples], dim=0
        ),
        "neutral_prompt_attention_mask": torch.cat(
            [example["neutral_prompt_attention_mask"] for example in examples], dim=0
        ),
        "prompts": [example["prompt"] for example in examples],
        "concepts": [example["concept"] for example in examples],
        "neutral_prompts": [example["neutral_prompt"] for example in examples],
        "files": [example["file"] for example in examples],
        "concept_token_indices": [list(example["concept_token_indices"]) for example in examples],
    }
    if all("latents" in example for example in examples):
        batch["latents"] = torch.stack([example["latents"] for example in examples], dim=0)
    return batch


def collate_video_concept_batch(examples: List[Dict]) -> Dict:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples], dim=0),
        "prompts": [example["prompt"] for example in examples],
        "concepts": [example["concept"] for example in examples],
        "neutral_prompts": [example["neutral_prompt"] for example in examples],
        "files": [example["file"] for example in examples],
    }
