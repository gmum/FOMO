"""CLIP similarity between generated frames and the two concepts of a mapping.

The frame classifier answers "is the erased class still there". It cannot
answer "did the object become the replacement", because the replacement is
usually not one of the ten Imagenette labels. This script answers the second
question, for the erased class only.

For every method and every erased class it reports four numbers:

    clip_src_base      base model      vs the source concept
    clip_src_method    unlearned model vs the source concept     should fall
    clip_tgt_base      base model      vs the target concept
    clip_tgt_method    unlearned model vs the target concept     should rise

Only comparisons down a column are meaningful. Two different texts have
different typical similarity ranges, so clip_src and clip_tgt must never be
compared with each other; base against method, for one fixed text, is the
comparison the numbers support.

The text is a neutral template ensemble, not the prompt the video was
generated from. Rewriting a v7 prompt to name the replacement produces
descriptions of things that do not exist, since those prompts carry
class-specific detail on purpose: swapping the name in "an English springer
spaniel, its long brown ears hanging beside its face" yields a German
shepherd with the wrong ears.

Run from the repository root.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASS_SETS, article_for, set_tag, slugify  # noqa: E402

# Templates take the concept with its article already attached, because a
# target such as "an empty park" carries one and "a photo of a an empty park"
# would not survive contact with the text encoder.
TEMPLATES = (
    "a photo of {}.",
    "a video of {}.",
    "a blurry photo of {}.",
    "a bright photo of {}.",
    "a low resolution photo of {}.",
)

DEFAULT_MODELS = ("openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32")


def with_article(phrase):
    """Prefix an article unless the phrase already begins with one."""
    first = phrase.strip().split()[0].lower()
    if first in ("a", "an", "the"):
        return phrase.strip()
    return "%s %s" % (article_for(phrase), phrase.strip())


def read_variants(path):
    """{(concept, mapping): target_concept} from a sweep table."""
    table = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise SystemExit("Expected 5 tab-separated columns, got %d: %r"
                                 % (len(parts), line))
            concept, mapping, target = parts[0], parts[1], parts[2]
            table[(concept, mapping)] = "" if target == "NONE" else target
    return table


def mapping_of(method):
    """imap_<mapping>_lr<tag> -> <mapping>."""
    if not method.startswith("imap_"):
        raise SystemExit("Method %r does not look like imap_<mapping>_lr<tag>" % method)
    rest = method[len("imap_"):]
    if "_lr" not in rest:
        raise SystemExit("Method %r has no _lr part" % method)
    return rest.rsplit("_lr", 1)[0]


def video_paths(root, tag, concept, method):
    """Unlearned videos for one class under one method, and the base videos.

    The base model does not depend on which concept is erased, so it lives once
    under _shared and is not duplicated into the columns.
    """
    slug = slugify(concept)
    method_dir = os.path.join(root, tag, "unlearn_%s" % slug, slug, method)
    base_dir = os.path.join(root, tag, "_shared", "base", slug)
    listing = lambda d: (sorted(os.path.join(d, f) for f in os.listdir(d)
                                if f.endswith(".mp4")) if os.path.isdir(d) else [])
    return listing(base_dir), listing(method_dir)


def main():
    ap = argparse.ArgumentParser(description="CLIP similarity for erased classes.")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="Method labels as written in the video directories.")
    ap.add_argument("--table", default="configs/in_variants.tsv")
    ap.add_argument("--class_set", default="imagenette", choices=sorted(CLASS_SETS))
    ap.add_argument("--prompt_set", default="v7")
    ap.add_argument("--videos_root", default="videos")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip_models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default=None, help="CSV path; defaults under results/<tag>.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the texts and the file counts, load nothing.")
    args = ap.parse_args()

    classes = CLASS_SETS[args.class_set]
    variants = read_variants(args.table)
    tag = set_tag(args.class_set, args.prompt_set, args.width, args.height,
                  args.guidance, args.seed)
    out_path = args.out or os.path.join("results", tag, "clip_score.csv")

    plan = []
    for method in args.methods:
        mapping = mapping_of(method)
        for concept in classes:
            if (concept, mapping) not in variants:
                print("WARNING: no row for %r / %r in %s" % (concept, mapping, args.table))
                continue
            target = variants[(concept, mapping)]
            base_files, method_files = video_paths(args.videos_root, tag, concept, method)
            plan.append({
                "method": method, "mapping": mapping, "concept": concept,
                "target_concept": target,
                "text_src": with_article(concept),
                "text_tgt": with_article(target) if target else "",
                "base_files": base_files, "method_files": method_files,
            })

    print("Tag     : %s" % tag)
    print("Methods : %s" % ", ".join(args.methods))
    print("Models  : %s" % ", ".join(args.clip_models))
    print("Cells   : %d" % len(plan))
    print()
    for item in plan:
        print("  %-22s %-18s src=%-28s tgt=%-30s base %2d, method %2d"
              % (item["method"], item["concept"], item["text_src"],
                 item["text_tgt"] or "(none)",
                 len(item["base_files"]), len(item["method_files"])))
    missing = [i for i in plan if not i["method_files"]]
    if missing:
        print("\n%d cells have no videos yet:" % len(missing))
        for item in missing:
            print("   %s / %s" % (item["method"], item["concept"]))
    if args.dry_run:
        return

    import torch
    from transformers import CLIPModel, CLIPProcessor
    from evaluate import read_frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for clip_model in args.clip_models:
        print("\n=== %s ===" % clip_model)
        model = CLIPModel.from_pretrained(clip_model).to(device).eval()
        processor = CLIPProcessor.from_pretrained(clip_model)

        @torch.no_grad()
        def text_feature(phrase):
            prompts = [t.format(phrase) for t in TEMPLATES]
            inputs = processor(text=prompts, return_tensors="pt",
                               padding=True).to(device)
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.mean(dim=0)
            return feats / feats.norm()

        @torch.no_grad()
        def score(files, text_feats):
            """Mean cosine similarity x100 over every frame of every file."""
            totals = [0.0] * len(text_feats)
            frames_seen = 0
            for path in files:
                frames = read_frames(path)
                for start in range(0, len(frames), args.batch_size):
                    batch = frames[start:start + args.batch_size]
                    inputs = processor(images=batch, return_tensors="pt").to(device)
                    feats = model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    for index, text in enumerate(text_feats):
                        totals[index] += float((feats @ text).sum()) * 100.0
                    frames_seen += len(batch)
            if not frames_seen:
                return [None] * len(text_feats), 0
            return [t / frames_seen for t in totals], frames_seen

        cache = {}
        for item in plan:
            texts, keys = [], []
            for key in ("text_src", "text_tgt"):
                if item[key]:
                    if item[key] not in cache:
                        cache[item[key]] = text_feature(item[key])
                    texts.append(cache[item[key]])
                    keys.append(key)
            if not texts:
                continue

            base_scores, base_frames = score(item["base_files"], texts)
            method_scores, method_frames = score(item["method_files"], texts)

            row = {
                "clip_model": clip_model, "method": item["method"],
                "mapping": item["mapping"], "concept": item["concept"],
                "target_concept": item["target_concept"],
                "n_frames_base": base_frames, "n_frames_method": method_frames,
                "clip_src_base": "", "clip_src_method": "",
                "clip_tgt_base": "", "clip_tgt_method": "",
            }
            for index, key in enumerate(keys):
                side = "src" if key == "text_src" else "tgt"
                if base_scores[index] is not None:
                    row["clip_%s_base" % side] = "%.4f" % base_scores[index]
                if method_scores[index] is not None:
                    row["clip_%s_method" % side] = "%.4f" % method_scores[index]
            rows.append(row)
            print("  %-22s %-18s src %s -> %s   tgt %s -> %s"
                  % (item["method"], item["concept"],
                     row["clip_src_base"] or "--", row["clip_src_method"] or "--",
                     row["clip_tgt_base"] or "--", row["clip_tgt_method"] or "--"))

        del model
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("Nothing was scored.")

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
