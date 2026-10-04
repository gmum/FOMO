import json
import re

from qwen_prompts import REPAIR_PROMPT

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
NUM_FRAMES = 16
MAX_SIDE = 768
MAX_FORMAT_ATTEMPTS = 5
VIDEO_ID_PATTERN = re.compile(r"^(\d{3})_")
ANSWER_PATTERN = re.compile(
    r"ANS:\s*(Yes|No)\s*,\s*Yes:\s*(\d{1,3})%\s*,\s*No:\s*(\d{1,3})%\.?",
    flags=re.IGNORECASE,
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_inputs(directory, metadata, config):
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    prompts = {}
    for record in read_jsonl(metadata):
        video_id = int(record["index"])
        if video_id in prompts or not isinstance(record["prompt"], str):
            raise ValueError(f"Duplicate index or invalid prompt in {metadata}: {video_id}")
        prompts[video_id] = record["prompt"]

    inputs, ids = {}, set()
    for video in sorted(directory.glob("*.mp4")):
        match = VIDEO_ID_PATTERN.match(video.name)
        if match is None:
            raise ValueError(f"Video filename must start with a three-digit ID and _: {video}")
        video_id = int(match.group(1))
        if video_id in ids or video_id not in prompts:
            raise ValueError(f"Duplicate video ID or missing generation prompt for {video.name}")
        ids.add(video_id)
        inputs[video.name] = {
            "video": video.name, "video_index": video_id,
            "generation_prompt": prompts[video_id], **config,
        }
    if not inputs:
        raise ValueError(f"No MP4 videos found in {directory}")
    return inputs


def read_frames(path):
    import cv2
    import numpy as np
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"Cannot determine frame count: {path}")
        frames = []
        for index in np.linspace(0, total - 1, NUM_FRAMES, dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Cannot decode frame {index} from {path}")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            frames.append(image)
    finally:
        capture.release()
    return frames


def parse_answer(raw_answer):
    answer = " ".join(raw_answer.split())
    match = ANSWER_PATTERN.fullmatch(answer)
    if match is None:
        raise ValueError(f"Unexpected Qwen answer format: {answer!r}")
    label = match.group(1).capitalize()
    yes_score, no_score = int(match.group(2)), int(match.group(3))
    if (max(yes_score, no_score) > 100 or yes_score + no_score != 100
            or label != ("Yes" if yes_score >= 50 else "No")):
        raise ValueError(f"Inconsistent Yes/No answer: {answer!r}")
    return label, yes_score, no_score, f"ANS: {label}, Yes: {yes_score}%, No: {no_score}%."


def load_model(model_name):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen evaluation requires a CUDA GPU.")
    print(f"Loading Qwen: {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    return processor, model.eval()


def evaluate_frames(processor, model, frames, prompt):
    import torch

    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image", "image": frame} for frame in frames)
    base_messages = [{"role": "user", "content": content}]
    messages = base_messages
    for attempt in range(1, MAX_FORMAT_ATTEMPTS + 1):
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw_answer = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            return parse_answer(raw_answer)
        except ValueError as error:
            print(f"Invalid answer ({attempt}/{MAX_FORMAT_ATTEMPTS}): {raw_answer!r}", flush=True)
            if attempt == MAX_FORMAT_ATTEMPTS:
                raise RuntimeError(f"Qwen failed to return a valid answer after {attempt} attempts") from error
            messages = base_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": raw_answer}]},
                {"role": "user", "content": [{"type": "text", "text": REPAIR_PROMPT}]},
            ]


def load_results(path, inputs):
    records = {}
    if not path.exists():
        return records
    for record in read_jsonl(path):
        name = record["video"]
        if name in records or name not in inputs:
            raise ValueError(f"Duplicate or unknown video in {path}: {name}")
        if any(record.get(key) != value for key, value in inputs[name].items()):
            raise ValueError(f"Existing video, prompt, or evaluation settings differ for {name}; use a new output directory.")
        label, yes, no, _ = parse_answer(record["raw_normalized_answer"])
        if (record["answer"], record["yes_probability_percent"], record["no_probability_percent"]) != (label, yes, no):
            raise ValueError(f"Inconsistent result in {path}: {name}")
        records[name] = record
    return records


def write_summary(path, records, expected, config):
    yes_count = sum(record["answer"] == "Yes" for record in records.values())
    evaluated = len(records)
    mean = yes_count / evaluated if evaluated else None
    result = {
        **config,
        "evaluated_videos": evaluated,
        "expected_videos": expected,
        "complete": evaluated == expected,
        "yes_count": yes_count,
        "no_count": evaluated - yes_count,
        "binary_mean": mean,
        "yes_rate_percent": 100.0 * mean if mean is not None else None,
        "sampled_frames_per_video": NUM_FRAMES,
        "max_frame_side": MAX_SIDE,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result
