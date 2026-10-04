import argparse
import json
import re
from pathlib import Path

VIDEO_ID = re.compile(r"^(\d{3})_")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare videos with DINO and LPIPS.")
    parser.add_argument("--reference-dir", required=True, type=Path, help="Baseline videos.")
    parser.add_argument("--generated-dir", required=True, type=Path, help="Videos after unlearning.")
    parser.add_argument("--output", required=True, type=Path, help="Results JSON.")
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=392)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def indexed_videos(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {directory}")
    indexed = {}
    for path in sorted(directory.glob("*.mp4")):
        match = VIDEO_ID.match(path.name)
        if match is None:
            raise ValueError(f"Video filename must start with a three-digit ID and _: {path}")
        video_id = int(match.group(1))
        if video_id in indexed:
            raise ValueError(f"Duplicate video ID {video_id:03d} in {directory}")
        indexed[video_id] = path
    if not indexed:
        raise ValueError(f"No MP4 videos found in {directory}")
    return indexed


def load_frames(path, num_frames, height, width, device):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    capture = cv2.VideoCapture(str(path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            raise ValueError(f"{path} has {total_frames} frames, fewer than requested {num_frames}")
        frames = []
        for index in np.linspace(0, total_frames - 1, num_frames, dtype=np.int64):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index} from {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    tensor = torch.from_numpy(np.stack(frames)).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2).div_(255.0)
    return F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)


def dino_features(frames, model):
    import torch.nn.functional as F

    mean = frames.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = frames.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    features = model(pixel_values=(frames - mean) / std).last_hidden_state[:, 0]
    return F.normalize(features.float(), dim=-1)


def paired_scores(reference_frames, generated_frames, dino_model, lpips_model, batch_size):
    import numpy as np
    import torch

    dino_values, lpips_values = [], []
    for start in range(0, len(reference_frames), batch_size):
        reference_batch = reference_frames[start : start + batch_size]
        generated_batch = generated_frames[start : start + batch_size]
        with torch.inference_mode():
            reference_features = dino_features(reference_batch, dino_model)
            generated_features = dino_features(generated_batch, dino_model)
            similarities = (reference_features * generated_features).sum(dim=-1)
            distances = lpips_model(
                reference_batch.mul(2.0).sub(1.0), generated_batch.mul(2.0).sub(1.0), normalize=False,
            ).flatten()
        dino_values.extend(similarities.cpu().tolist())
        lpips_values.extend(distances.float().cpu().tolist())
    return float(np.mean(dino_values)), float(np.mean(lpips_values))


def main():
    args = parse_args()
    if min(args.num_frames, args.height, args.width, args.batch_size) <= 0:
        raise ValueError("Frame count, dimensions, and batch size must be positive")
    if args.height % 14 or args.width % 14:
        raise ValueError("DINOv2 input height and width must be divisible by patch size 14")
    reference = indexed_videos(args.reference_dir)
    generated = indexed_videos(args.generated_dir)
    if set(reference) != set(generated):
        raise ValueError(
            f"Video ID sets differ: missing generated={sorted(set(reference) - set(generated))}, "
            f"missing reference={sorted(set(generated) - set(reference))}"
        )

    import lpips
    import numpy as np
    import torch
    from transformers import AutoModel

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    print(f"Loading DINO {args.dino_model} and LPIPS AlexNet", flush=True)
    dino_model = AutoModel.from_pretrained(args.dino_model).to(device).eval()
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    per_video = []
    for position, video_id in enumerate(sorted(reference), start=1):
        clips = [
            load_frames(path, args.num_frames, args.height, args.width, device)
            for path in (reference[video_id], generated[video_id])
        ]
        dino_score, lpips_score = paired_scores(*clips, dino_model, lpips_model, args.batch_size)
        per_video.append({
            "video_id": f"{video_id:03d}",
            "reference_video": reference[video_id].name,
            "generated_video": generated[video_id].name,
            "dino_similarity": dino_score,
            "lpips": lpips_score,
        })
        print(f"Scored {position}/{len(reference)} video pairs", flush=True)

    dino_video_scores = np.asarray([item["dino_similarity"] for item in per_video])
    lpips_video_scores = np.asarray([item["lpips"] for item in per_video])
    result = {
        "metric": f"paired_full_frame_dinov2_lpips_{args.num_frames}f",
        "num_videos": len(per_video),
        "num_frames": args.num_frames,
        "sampling": "uniform_full_video_corresponding_frames",
        "pairing": "three_digit_video_id",
        "resize_height": args.height,
        "resize_width": args.width,
        "dino_model": args.dino_model,
        "dino_feature": "normalized_cls_token",
        "dino_similarity_mean": float(dino_video_scores.mean()),
        "dino_similarity_std_across_videos": float(dino_video_scores.std()),
        "lpips_backbone": "alex",
        "lpips_mean": float(lpips_video_scores.mean()),
        "lpips_std_across_videos": float(lpips_video_scores.std()),
        "reference_dir": str(args.reference_dir.resolve()),
        "generated_dir": str(args.generated_dir.resolve()),
        "reference_kind": "generated_baseline",
        "comparison_scope": "full_frame_no_mask",
        "per_video": per_video,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
