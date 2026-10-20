"""Object erasure: classify every frame and write per-class accuracy.

One run scores one column -- a directory holding one subdirectory per class:

    <column>/<class_slug>/NN.mp4

All ten Imagenette classes are ImageNet-1k classes, so a torchvision
classifier can be used directly, with no zero-shot step in between. Every
frame is classified independently and pooled per class: twenty prompts times
seventeen frames is three hundred and forty frames behind each cell.

    ESR-k = 1 - Top-k on the class this column erased      erasure
    PSR-k = mean Top-k over the other nine                 collateral damage

Outputs, in --output-dir:

    frames.csv       one row per frame, with what it was predicted to be
    per_class.csv    concept, n_frames, top1_acc, top5_acc
    summary.json     ESR and PSR for this column

frames.csv is kept because the predicted label is what explains a surprising
cell. A class whose accuracy collapsed with no erasure applied is usually being
predicted as its ImageNet neighbour -- a cassette rather than a cassette
player -- and that is a property of the prompt, not of the method.

    python -m evaluation.resnet experiments/objects_mapping2.yaml \
        --erased church
"""

import argparse
import csv
import json
import os
from collections import defaultdict

from experiments import config as expconfig
from experiments import layout


def read_frames(path):
    """Every frame of one video as an array. Raises on a truncated file."""
    import imageio.v2 as imageio
    import numpy as np

    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


class Classifier(object):
    """ResNet-50 over the thousand ImageNet classes.

    IMAGENET1K_V2 weights, which is what the reported numbers were measured
    with. The choice matters more than it looks: the two weight sets differ by
    several points of Top-1 on these classes, so a table mixing them would
    compare methods through the classifier.
    """

    def __init__(self, device):
        import torch
        import torchvision.models as tvm

        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        self.torch = torch
        self.model = tvm.resnet50(weights=weights).to(device).eval()
        self.preprocess = weights.transforms()
        self.categories = weights.meta["categories"]
        self.device = device

    def score(self, frames, target_index, batch_size):
        """Per frame: hit at top 1, hit in top 5, and the predicted index."""
        from PIL import Image

        torch = self.torch
        tensors = [self.preprocess(Image.fromarray(f)) for f in frames]
        top1, top5, preds = [], [], []
        with torch.no_grad():
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start:start + batch_size]).to(self.device)
                probs = torch.softmax(self.model(batch).float(), dim=-1)
                for row in probs.topk(5, dim=-1).indices.cpu().numpy():
                    top1.append(int(row[0] == target_index))
                    top5.append(int(target_index in row))
                    preds.append(int(row[0]))
        return top1, top5, preds


def score_column(column_dir, classes, index, classifier, batch_size):
    """{concept: [hits@1, hits@5, frames]} plus the per-frame rows."""
    stats = defaultdict(lambda: [0, 0, 0])
    rows = []
    for concept in classes:
        directory = layout.subject_videos(column_dir, concept)
        if not os.path.isdir(directory):
            continue
        for name in sorted(f for f in os.listdir(directory) if f.endswith(".mp4")):
            path = os.path.join(directory, name)
            try:
                frames = read_frames(path)
            except Exception as error:
                # A generation interrupted mid-write leaves an mp4 with no
                # readable stream. One of those used to take a whole run down
                # near the end; a missing video is a hole in one cell, not a
                # reason to lose everything scored before it.
                print("  UNREADABLE %s (%s)" % (path, type(error).__name__))
                continue
            top1, top5, preds = classifier.score(frames, index[concept], batch_size)
            for i in range(len(frames)):
                stats[concept][0] += top1[i]
                stats[concept][1] += top5[i]
                stats[concept][2] += 1
                rows.append({"concept": concept,
                             "prompt": os.path.splitext(name)[0],
                             "frame": i,
                             "pred": classifier.categories[preds[i]],
                             "top1_hit": top1[i], "top5_hit": top5[i]})
        hit1, hit5, total = stats[concept]
        if total:
            print("  %18s | %4d frames | top1 %.3f  top5 %.3f"
                  % (concept, total, hit1 / total, hit5 / total))
    return stats, rows


def write_results(output_dir, stats, rows, erased, classes):
    os.makedirs(output_dir, exist_ok=True)

    if rows:
        with open(os.path.join(output_dir, "frames.csv"), "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(os.path.join(output_dir, "per_class.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["concept", "n_frames", "top1_acc", "top5_acc"])
        for concept in classes:
            if concept not in stats:
                continue
            hit1, hit5, total = stats[concept]
            writer.writerow([concept, total,
                             "%.6f" % (hit1 / total), "%.6f" % (hit5 / total)])

    summary = {"erased_concept": erased, "classifier": "resnet50 IMAGENET1K_V2"}
    if erased and erased in stats:
        hit1, hit5, total = stats[erased]
        others = [stats[c] for c in classes if c != erased and c in stats]
        summary.update({
            "ESR-1": 1.0 - hit1 / total,
            "ESR-5": 1.0 - hit5 / total,
            "PSR-1": sum(o[0] / o[2] for o in others) / len(others) if others else None,
            "PSR-5": sum(o[1] / o[2] for o in others) / len(others) if others else None,
            "n_frames_erased": total,
            "n_classes_preserved": len(others),
        })
    with open(os.path.join(output_dir, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Score one object column.")
    ap.add_argument("config")
    ap.add_argument("--erased", default=None,
                    help="The column to score. Omit with --base.")
    ap.add_argument("--base", action="store_true",
                    help="Score the untouched model instead of a column.")
    ap.add_argument("--videos-root", default="videos")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output-dir", default=None,
                    help="Overrides where the three files are written.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = expconfig.load(args.config)
    if cfg["kind"] != "objects":
        raise SystemExit("%s is a %s experiment; identities are scored with "
                         "evaluation.arcface." % (args.config, cfg["kind"]))
    if args.base == bool(args.erased):
        raise SystemExit("Pass either --base or --erased <concept>.")

    module = expconfig.prompt_module(cfg)
    classes = list(module.CLASSES)

    if args.base:
        column = layout.base_videos(cfg["prompts"], root=args.videos_root)
        results = layout.base_results(cfg["prompts"], root=args.results_root)
    else:
        if args.erased not in classes:
            raise SystemExit("%r is not an Imagenette class." % args.erased)
        column = layout.column_videos(cfg["name"], args.erased,
                                      root=args.videos_root)
        results = layout.column_results(cfg["name"], args.erased,
                                        root=args.results_root)
    results = args.output_dir or results

    if not os.path.isdir(column):
        # Not an error: a column may still be generating. Exiting zero keeps
        # one unfinished entry from marking a whole array as failed; the gap
        # shows up when the table is assembled, which is where it belongs.
        print("SKIPPED: %s does not exist yet." % column)
        return

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("videos  : %s" % column)
    print("results : %s" % results)
    print("erased  : %s" % (args.erased or "nothing (baseline)"))
    print("device  : %s" % device)
    print()

    classifier = Classifier(device)
    stats, rows = score_column(column, classes, module.INDEX,
                               classifier, args.batch_size)
    if not rows:
        raise SystemExit("No frames were classified under %s" % column)

    summary = write_results(results, stats, rows, args.erased, classes)
    if "ESR-1" in summary:
        print("\nESR-1 %.4f  ESR-5 %.4f | PSR-1 %.4f  PSR-5 %.4f"
              % (summary["ESR-1"], summary["ESR-5"],
                 summary["PSR-1"], summary["PSR-5"]))
    print("Wrote %s" % results)


if __name__ == "__main__":
    main()
