"""Classify every frame of every generated video and write per-class accuracy.

One evaluation run covers one directory of videos, that is one table column:

    <videos_dir>/<class_slug>/<method>/NN.mp4

Every frame is classified independently and the accuracies are pooled per
class, following the protocol: 20 prompts per concept, 17 frames each, so 340
frames per class per method.

Two backends, chosen by --class_set:

  imagenette   torchvision classifier over the 1000 ImageNet classes. All ten
               Table 1 classes are ImageNet classes, so this is direct.
               Second metric is Top-5.

  cifar10      CLIP zero-shot over the ten CIFAR-10 labels. Required because
               the CIFAR-10 labels are not ImageNet classes: "deer" has none,
               and "dog"/"bird"/"truck" each map to dozens. Second metric is
               the mean probability of the correct label — Top-5 out of ten
               candidates would carry almost no information.

Outputs, written to --output_dir:

    frames.csv       one row per frame, with the predicted class
    per_class.csv    method, concept, n_frames, top1_acc, top2_acc
    summary.json     ESR/PSR for the requested erased concept

per_class.csv is what make_row.py consumes. The column is named top2_acc in
both backends; what it holds is documented in summary.json.

Run from the repository root.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASS_SETS, IMAGENET_INDEX, slugify  # noqa: E402

TORCHVISION_CLASSIFIERS = ("resnet50", "resnet18", "vit_b_16", "convnext_tiny")

# Prompt ensembling is standard for CLIP zero-shot and gains a few points over
# a single template. Text embeddings are averaged per class.
CLIP_TEMPLATES = (
    "a photo of a {}.",
    "a video of a {}.",
    "a blurry photo of a {}.",
    "a photo of the {}.",
    "a bright photo of a {}.",
    "a low resolution photo of a {}.",
)

# "automobile" is rare in web captions compared with "car", which is what CLIP
# was trained on. This affects only the text side of the classifier.
CLIP_LABEL_ALIASES = {"automobile": "car"}


def read_frames(path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


class ImageNetBackend:
    """Torchvision classifier over the 1000 ImageNet classes."""

    second_metric = "Top-5 accuracy"

    def __init__(self, classifier, device):
        import torchvision.models as tvm

        table = {
            "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
            "vit_b_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1),
            "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        }
        if classifier not in table:
            raise SystemExit("Unknown classifier: %s. Available: %s"
                             % (classifier, ", ".join(table)))
        builder, weights = table[classifier]
        self.model = builder(weights=weights).to(device).eval()
        self.preprocess = weights.transforms()
        self.categories = weights.meta["categories"]
        self.device = device

    @torch.no_grad()
    def score(self, frames, concept, batch_size):
        """Return (hit_top1, hit_top2, predicted_index) per frame."""
        from PIL import Image

        target = IMAGENET_INDEX[concept]
        tensors = [self.preprocess(Image.fromarray(f)) for f in frames]
        hits1, hits2, preds = [], [], []
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(self.device)
            probs = torch.softmax(self.model(batch).float(), dim=-1)
            top5 = probs.topk(5, dim=-1).indices.cpu().numpy()
            for row in top5:
                hits1.append(int(row[0] == target))
                hits2.append(int(target in row))
                preds.append(int(row[0]))
        return hits1, hits2, preds

    def name_of(self, index):
        return self.categories[index]


class ClipBackend:
    """CLIP zero-shot over the ten CIFAR-10 labels."""

    second_metric = "mean probability of the correct label"

    def __init__(self, clip_model, classes, device):
        from transformers import CLIPModel, CLIPProcessor

        self.classes = classes
        self.device = device
        self.model = CLIPModel.from_pretrained(clip_model).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.text_features = self._build_text_features()

    @torch.no_grad()
    def _build_text_features(self):
        per_class = []
        for name in self.classes:
            label = CLIP_LABEL_ALIASES.get(name, name)
            prompts = [t.format(label) for t in CLIP_TEMPLATES]
            inputs = self.processor(text=prompts, return_tensors="pt",
                                    padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            per_class.append(mean / mean.norm())
        return torch.stack(per_class)

    @torch.no_grad()
    def score(self, frames, concept, batch_size):
        from PIL import Image

        target = self.classes.index(concept)
        images = [Image.fromarray(f) for f in frames]
        hits1, probs2, preds = [], [], []
        for start in range(0, len(images), batch_size):
            batch = self.processor(images=images[start:start + batch_size],
                                   return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = 100.0 * feats @ self.text_features.T   # CLIP's logit scale
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for row in probs:
                pred = int(row.argmax())
                hits1.append(int(pred == target))
                probs2.append(float(row[target]))
                preds.append(pred)
        return hits1, probs2, preds

    def name_of(self, index):
        return self.classes[index]


def main():
    ap = argparse.ArgumentParser(description="Per-frame ESR/PSR evaluation.")
    ap.add_argument("--videos_dir", required=True,
                    help="One table column, e.g. videos/imagenette_v1/unlearn_church")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--class_set", default="imagenette", choices=sorted(CLASS_SETS))
    ap.add_argument("--erased_concept", required=True,
                    help="Concept this column erases. Only affects summary.json; "
                         "per_class.csv always covers all ten classes.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Method subdirectories to score. Default: all present.")
    ap.add_argument("--prompt_ids", nargs="*", default=None,
                    help="Restrict to specific prompt indices, e.g. 00 03.")
    ap.add_argument("--classifier", default="resnet50", choices=TORCHVISION_CLASSIFIERS,
                    help="imagenette backend only.")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14",
                    help="cifar10 backend only.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    classes = CLASS_SETS[args.class_set]
    if args.erased_concept not in classes:
        raise SystemExit("'%s' is not in %s: %s"
                         % (args.erased_concept, args.class_set, ", ".join(classes)))

    available = defaultdict(dict)
    for concept in classes:
        class_dir = os.path.join(args.videos_dir, slugify(concept))
        if not os.path.isdir(class_dir):
            continue
        for method in sorted(os.listdir(class_dir)):
            method_dir = os.path.join(class_dir, method)
            if not os.path.isdir(method_dir):
                continue
            if any(f.endswith(".mp4") for f in os.listdir(method_dir)):
                available[method][concept] = method_dir
    if not available:
        raise SystemExit("No videos found under %s" % args.videos_dir)

    methods = args.methods or sorted(available)
    prompt_filter = {str(p).zfill(2) for p in args.prompt_ids} if args.prompt_ids else None

    if args.class_set == "imagenette":
        backend = ImageNetBackend(args.classifier, args.device)
        backend_name = args.classifier
    else:
        backend = ClipBackend(args.clip_model, classes, args.device)
        backend_name = args.clip_model

    print("Class set    : %s" % args.class_set)
    print("Backend      : %s (%s)" % (backend_name, args.device))
    print("Erased class : %s" % args.erased_concept)
    print("Methods      : %s" % ", ".join(methods))
    print("Prompts      : %s" % (", ".join(sorted(prompt_filter)) if prompt_filter else "all"))
    print()

    frame_rows = []
    # (method, concept) -> [sum top1, sum top2, frame count]
    stats = defaultdict(lambda: [0.0, 0.0, 0.0])

    for method in methods:
        if method not in available:
            print("[skipped] no videos for method '%s'" % method)
            continue
        for concept, method_dir in sorted(available[method].items()):
            videos = sorted(f for f in os.listdir(method_dir) if f.endswith(".mp4"))
            if prompt_filter is not None:
                videos = [v for v in videos if os.path.splitext(v)[0] in prompt_filter]
                if not videos:
                    print("  WARNING: no videos matching %s in %s"
                          % (sorted(prompt_filter), method_dir))
                    continue
            for name in videos:
                frames = read_frames(os.path.join(method_dir, name))
                if not frames:
                    print("  WARNING: empty file %s" % os.path.join(method_dir, name))
                    continue
                hits1, second, preds = backend.score(frames, concept, args.batch_size)
                for i in range(len(frames)):
                    stats[(method, concept)][0] += hits1[i]
                    stats[(method, concept)][1] += second[i]
                    stats[(method, concept)][2] += 1
                    frame_rows.append({
                        "method": method,
                        "concept": concept,
                        "prompt": os.path.splitext(name)[0],
                        "frame": i,
                        "pred": backend.name_of(preds[i]),
                        "pred_index": preds[i],
                        "top1_hit": hits1[i],
                        "second": second[i],
                    })
            done = stats[(method, concept)]
            if done[2]:
                print("  %14s | %18s | %2d videos, %4d frames | top1 %.3f  second %.3f"
                      % (method, concept, len(videos), int(done[2]),
                         done[0] / done[2], done[1] / done[2]))

    if not frame_rows:
        raise SystemExit("No frames were classified — check --videos_dir and --methods.")

    os.makedirs(args.output_dir, exist_ok=True)

    frames_csv = os.path.join(args.output_dir, "frames.csv")
    with open(frames_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)
    print("\nWrote %d rows -> %s" % (len(frame_rows), frames_csv))

    per_class_csv = os.path.join(args.output_dir, "per_class.csv")
    with open(per_class_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "concept", "n_frames", "top1_acc", "top2_acc"])
        for (method, concept), (h1, h2, n) in sorted(stats.items()):
            writer.writerow([method, concept, int(n), "%.6f" % (h1 / n), "%.6f" % (h2 / n)])
    print("Wrote per-class accuracy -> %s" % per_class_csv)

    erased = args.erased_concept
    others = [c for c in classes if c != erased]
    summary = {}
    for method in methods:
        if (method, erased) not in stats:
            continue
        h1, h2, n = stats[(method, erased)]
        preserved = [stats[(method, c)] for c in others if (method, c) in stats]
        summary[method] = {
            "ESR-1": 1.0 - h1 / n,
            "ESR-2": 1.0 - h2 / n,
            "PSR-1": float(np.mean([p[0] / p[2] for p in preserved])) if preserved else float("nan"),
            "PSR-2": float(np.mean([p[1] / p[2] for p in preserved])) if preserved else float("nan"),
            "n_frames_erased": int(n),
            "n_classes_preserved": len(preserved),
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "class_set": args.class_set,
            "backend": backend_name,
            "erased_concept": erased,
            "prompt_ids": sorted(prompt_filter) if prompt_filter else "all",
            "second_metric": backend.second_metric,
            "results": summary,
        }, handle, indent=2, ensure_ascii=False)

    print()
    print("=" * 66)
    print("Column: %s   (second metric: %s)" % (erased, backend.second_metric))
    print("=" * 66)
    print("%16s | %7s %7s | %7s %7s" % ("Method", "ESR-1", "ESR-2", "PSR-1", "PSR-2"))
    print("-" * 66)
    for method, v in summary.items():
        print("%16s | %7.3f %7.3f | %7.3f %7.3f"
              % (method, v["ESR-1"], v["ESR-2"], v["PSR-1"], v["PSR-2"]))
    print("-" * 66)
    print("\nWrote -> %s" % summary_path)


if __name__ == "__main__":
    main()
