"""Place the method's target similarity between a floor and a ceiling.

CLIP-tgt is a bare cosine and carries no scale, so the same rise means
different things for different targets. Three numbers fix that, all measured
against the same text with the same encoder:

    floor    the base model, prompted for the source concept
             (clip_tgt_base, already in clip_score.csv)
    ours     the unlearned model, prompted for the source concept
             (clip_tgt_method, likewise)
    ceiling  the base model, prompted for the target concept directly
             (the videos under videos/_ceiling, scored here)

    score = 100 * (ours - floor) / (ceiling - floor)

Zero means nothing moved toward the target, one hundred means the method got
as far as asking the base model for the target outright. There is no single
honest ceiling, so all three prompt variants are reported side by side:

    A  the v7 prompt with the noun swapped and the contradictions left in
    B  the same scene with the contradicted attributes repaired
    C  a plain one-line prompt on twenty seeds

A is the strictest comparison, because the method also only ever sees a prompt
about the source. C is the loosest. If the score exceeds one hundred against a
ceiling, that ceiling was the weaker prompt, not a broken measurement.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import slugify  # noqa: E402
from clip_score import TEMPLATES, with_article  # noqa: E402

VARIANTS = ("A", "B", "C")


def main():
    ap = argparse.ArgumentParser(description="Normalise CLIP-tgt between floor and ceiling.")
    ap.add_argument("--clip_csv", required=True,
                    help="clip_score CSV holding clip_tgt_base and clip_tgt_method.")
    ap.add_argument("--method", required=True, help="Method label inside that CSV.")
    ap.add_argument("--ceiling_root", default="videos/_ceiling")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the plan and the file counts, load nothing.")
    args = ap.parse_args()

    cells = []
    for row in csv.DictReader(open(args.clip_csv, newline="", encoding="utf-8")):
        if row["clip_model"] != args.clip_model or row["method"] != args.method:
            continue
        if not row["clip_tgt_base"] or not row["clip_tgt_method"]:
            continue
        target = row["target_concept"]
        cells.append({
            "concept": row["concept"], "target": target,
            "text": with_article(target), "slug": slugify(target),
            "floor": float(row["clip_tgt_base"]),
            "ours": float(row["clip_tgt_method"]),
        })
    if not cells:
        raise SystemExit("No usable rows for method %r in %s" % (args.method, args.clip_csv))

    listing = lambda d: (sorted(os.path.join(d, f) for f in os.listdir(d)
                                if f.endswith(".mp4")) if os.path.isdir(d) else [])
    for cell in cells:
        cell["files"] = {v: listing(os.path.join(args.ceiling_root, v, cell["slug"]))
                         for v in args.variants}

    print("CSV      : %s" % args.clip_csv)
    print("Method   : %s" % args.method)
    print("Model    : %s" % args.clip_model)
    print("Variants : %s" % ", ".join(args.variants))
    print()
    for cell in cells:
        print("  %-18s -> %-22s floor %6.2f  ours %6.2f   %s"
              % (cell["concept"], cell["text"], cell["floor"], cell["ours"],
                 " ".join("%s:%d" % (v, len(cell["files"][v])) for v in args.variants)))
    empty = [(c["concept"], v) for c in cells for v in args.variants if not c["files"][v]]
    if empty:
        print("\n%d cells have no ceiling videos:" % len(empty))
        for concept, v in empty:
            print("   %s / %s" % (concept, v))
    if args.dry_run:
        return

    import torch
    from transformers import CLIPModel, CLIPProcessor
    from evaluate import read_frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    @torch.no_grad()
    def text_feature(phrase):
        inputs = processor(text=[t.format(phrase) for t in TEMPLATES],
                           return_tensors="pt", padding=True).to(device)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.mean(dim=0)
        return feats / feats.norm()

    @torch.no_grad()
    def score(files, text):
        total, seen = 0.0, 0
        for path in files:
            frames = read_frames(path)
            for start in range(0, len(frames), args.batch_size):
                batch = frames[start:start + args.batch_size]
                inputs = processor(images=batch, return_tensors="pt").to(device)
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                total += float((feats @ text).sum()) * 100.0
                seen += len(batch)
        return (total / seen) if seen else None

    print("\n%-18s %8s %8s %s"
          % ("concept", "floor", "ours",
             " ".join("%18s" % ("ceiling %s / %%" % v) for v in args.variants)))
    print("-" * (36 + 19 * len(args.variants)))

    rows, cache = [], {}
    for cell in cells:
        if cell["text"] not in cache:
            cache[cell["text"]] = text_feature(cell["text"])
        text = cache[cell["text"]]
        out = {"clip_model": args.clip_model, "method": args.method,
               "concept": cell["concept"], "target_concept": cell["target"],
               "clip_tgt_base": "%.4f" % cell["floor"],
               "clip_tgt_method": "%.4f" % cell["ours"]}
        parts = []
        for v in args.variants:
            ceiling = score(cell["files"][v], text) if cell["files"][v] else None
            out["ceiling_%s" % v] = "" if ceiling is None else "%.4f" % ceiling
            # A ceiling at or below the floor makes the ratio meaningless: the
            # base model was no better at the target than at the source, so
            # there is no interval to place anything inside.
            if ceiling is None or ceiling - cell["floor"] <= 0:
                out["score_%s" % v] = ""
                parts.append("%18s" % "----")
            else:
                pct = 100.0 * (cell["ours"] - cell["floor"]) / (ceiling - cell["floor"])
                out["score_%s" % v] = "%.2f" % pct
                parts.append("%10.2f %6.1f%%" % (ceiling, pct))
        rows.append(out)
        print("%-18s %8.2f %8.2f %s"
              % (cell["concept"], cell["floor"], cell["ours"], " ".join(parts)))

    for v in args.variants:
        vals = [float(r["score_%s" % v]) for r in rows if r["score_%s" % v]]
        print("%-18s %8s %8s %s"
              % ("AVG" if v == args.variants[0] else "",
                 "", "",
                 " ".join("%18s" % ("%.1f%%" % (sum(vals) / len(vals)) if vals else "----")
                          if w == v else "%18s" % "" for w in args.variants)))

    out_path = args.out or os.path.join(os.path.dirname(args.clip_csv), "ceiling_score.csv")
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nWrote %d rows -> %s" % (len(rows), out_path))


if __name__ == "__main__":
    main()
