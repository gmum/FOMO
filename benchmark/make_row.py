"""Build one table row from per-class accuracies.

Column X of the table means "X is the erased concept", so

    ESR-k(X) = 1 - Top-k(X)                          erasure efficacy
    PSR-k(X) = mean Top-k over the other nine        collateral damage

Two modes, and picking the wrong one silently produces a plausible-looking but
meaningless row:

  --mode shared       one per_class.csv, all ten columns derived from it.
                      Valid only when the videos do NOT depend on the column,
                      i.e. the base model and the CFG-matched control. For
                      those there is a single set of 200 videos and the ten
                      columns are ten ways of slicing the same accuracies.

  --mode per-column   ten per_class.csv files, one cell taken from each.
                      Required for every method whose videos differ per column
                      (negprompt, esd, ours): accuracies measured in column X
                      say nothing about column Y.

All ten classes are always reported. There is no exclusion mechanism: a class
the generator cannot produce recognisably still gets a cell, and its ESR near
100 is simply what the measurement says. Dropping columns would make AVG
depend on a threshold, and any threshold chosen after seeing the numbers is
indistinguishable from tuning the metric.

Examples
--------
    python benchmark/make_row.py --mode shared \\
        --per_class results/imagenette_v1_1280x720/per_class.csv \\
        --class_set imagenette --method base --label "HunyuanVideo"

    python benchmark/make_row.py --mode per-column \\
        --results_root results/imagenette_v1_1280x720/negprompt \\
        --class_set imagenette --method negprompt --label "NegPrompt"
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASS_SETS, LATEX_ORDER, slugify  # noqa: E402

METRICS = ("ESR-1", "ESR-2", "PSR-1", "PSR-2")

# Which CLIP rows a block carries, per --clip_mode.
#   absolute  the unlearned model only, two rows
#   delta     method minus base, two rows, base row drops out
#   both      base and method side by side, four rows, base row drops out
CLIP_METRICS = {
    "absolute": ("CLIP-src", "CLIP-tgt"),
    "delta":    ("CLIP-src", "CLIP-tgt"),
    "both":     ("CLIP-src-base", "CLIP-src-ours",
                 "CLIP-tgt-base", "CLIP-tgt-ours"),
}
ALL_CLIP_METRICS = tuple(sorted({m for v in CLIP_METRICS.values() for m in v}))

# The normalised target score from ceiling_score.py: where the method sits on
# the interval between the floor and each of the three ceilings, in percent.
# Only mapping rows have one, because a row with no target has no interval.
CEILING_VARIANTS = ("A", "B", "C")
CEILING_METRICS = tuple("CEIL-%s" % v for v in CEILING_VARIANTS)
DISPLAY_CEILING = {m: "Target-%s" % v for m, v in zip(CEILING_METRICS, CEILING_VARIANTS)}

# The three raw quantities the percentage is built from, in CLIP units: where
# the base model already sits against the target text, where the method sits,
# and where the base model gets to when asked for the target outright. Worth
# printing next to the score, because the same percentage means very different
# things over a wide interval and over a narrow one.
RAW_METRICS = (("TGT-floor", "TGT-ours")
               + tuple("CEIL-raw-%s" % v for v in CEILING_VARIANTS))
DISPLAY_RAW = {"TGT-floor": "CLIP-tgt (floor)", "TGT-ours": "CLIP-tgt (ours)"}
DISPLAY_RAW.update({"CEIL-raw-%s" % v: "CLIP-tgt (ceiling %s)" % v
                    for v in CEILING_VARIANTS})

# ESR and PSR are stored as fractions and printed as percentages. CLIP is
# already a cosine similarity times one hundred, so it must not be scaled
# again.
SCALE = {"ESR-1": 100.0, "ESR-2": 100.0, "PSR-1": 100.0, "PSR-2": 100.0}
SCALE.update({metric: 1.0 for metric in ALL_CLIP_METRICS})
SCALE.update({metric: 1.0 for metric in CEILING_METRICS})
SCALE.update({metric: 1.0 for metric in RAW_METRICS})

# Higher is better for every metric except similarity to the concept that was
# supposed to disappear. The base columns are references, not results, so they
# carry no direction at all.
ARROWS = {"CLIP-src": "\\downarrow", "CLIP-src-ours": "\\downarrow",
          "CLIP-src-base": "", "CLIP-tgt-base": ""}
# The floor and the ceilings bracket the scale; they are what the method is
# measured against, not something it is trying to maximise.
ARROWS["TGT-floor"] = ""
ARROWS.update({"CEIL-raw-%s" % v: "" for v in CEILING_VARIANTS})

DISPLAY_CLIP = {"CLIP-src-base": "CLIP-src (base)",
                "CLIP-src-ours": "CLIP-src (ours)",
                "CLIP-tgt-base": "CLIP-tgt (base)",
                "CLIP-tgt-ours": "CLIP-tgt (ours)"}

# The second metric means different things per class set, and the CSV column is
# called top2_acc in both cases. Imagenette uses Top-5 over 1000 ImageNet
# classes; CIFAR-10 uses the mean probability of the correct label, because
# Top-5 out of ten candidates carries almost no information.
SECOND_METRIC_NAME = {"imagenette": "5", "cifar10": "p"}


def read_per_class(path, method):
    """Return {concept: (top1, top2)} for one method from a per_class.csv."""
    table = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method:
                continue
            table[row["concept"]] = (float(row["top1_acc"]), float(row["top2_acc"]))
    return table


def read_clip(path, method, clip_model, side, mode="absolute"):
    """{concept: {metric: value}} from clip_score.csv.

    side is "method" for an unlearned row or "base" for the reference row. The
    base model has no single target concept, since the target depends on the
    mapping, so no target similarity is ever available for a base row.

    Three modes:

      absolute  the unlearned model's similarity to each text. Readable only
                next to a reference, and averaging it across columns averages
                similarities to ten different texts, which have ten different
                natural ranges.
      delta     method minus base for the same text and the same prompt. The
                reference is inside the number, so the average is honest.
      both      base and method printed side by side, which keeps the raw
                similarities visible and still shows the reference.

    In delta and both modes the base row carries no CLIP rows: in the first its
    value is zero by construction, in the second every method block already
    prints the base column it is compared against.
    """
    out = {}
    if mode != "absolute" and side == "base":
        return out
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["clip_model"] != clip_model:
                continue
            if side == "method" and row["method"] != method:
                continue
            cell = out.setdefault(row["concept"], {})
            for name, stem in (("CLIP-src", "clip_src"), ("CLIP-tgt", "clip_tgt")):
                if name == "CLIP-tgt" and side != "method":
                    continue
                value = row.get("%s_%s" % (stem, side), "")
                reference = row.get("%s_base" % stem, "")
                if mode == "both":
                    if reference:
                        cell["%s-base" % name] = float(reference)
                    if value:
                        cell["%s-ours" % name] = float(value)
                elif value:
                    if mode == "absolute":
                        cell[name] = float(value)
                    elif reference:
                        cell[name] = float(value) - float(reference)
    return out


def read_ceiling(path, method, clip_model):
    """{concept: {CEIL-A, CEIL-B, CEIL-C}} from a ceiling_score CSV.

    The value is already a percentage of the floor-to-ceiling interval, so it
    is not scaled again. A variant whose ceiling was never generated, or whose
    ceiling fell at or below the floor, is absent rather than zero: there was
    no interval to place the method inside, and a zero would read as a result.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("clip_model") != clip_model or row.get("method") != method:
                continue
            cell = out.setdefault(row["concept"], {})
            for variant in CEILING_VARIANTS:
                value = row.get("score_%s" % variant, "")
                if value:
                    cell["CEIL-%s" % variant] = float(value)
                raw = row.get("ceiling_%s" % variant, "")
                if raw:
                    cell["CEIL-raw-%s" % variant] = float(raw)
            for column, name in (("clip_tgt_base", "TGT-floor"),
                                 ("clip_tgt_method", "TGT-ours")):
                value = row.get(column, "")
                if value:
                    cell[name] = float(value)
    return out


def cells_from_shared(table, classes):
    """All ten columns from a single set of per-class accuracies."""
    present = [c for c in classes if c in table]
    rows = {}
    for erased in present:
        others = [c for c in present if c != erased]
        if not others:
            continue
        rows[erased] = {
            "ESR-1": 1.0 - table[erased][0],
            "ESR-2": 1.0 - table[erased][1],
            "PSR-1": sum(table[c][0] for c in others) / len(others),
            "PSR-2": sum(table[c][1] for c in others) / len(others),
        }
    return rows


def cells_from_per_column(root, classes, method):
    """One cell per column, each from that column's own per_class.csv."""
    rows, missing = {}, []
    for erased in classes:
        path = os.path.join(root, slugify(erased), "per_class.csv")
        if not os.path.exists(path):
            missing.append("%s (no %s)" % (erased, path))
            continue
        table = read_per_class(path, method)
        if erased not in table:
            missing.append("%s (erased class absent from %s)" % (erased, path))
            continue
        others = [c for c in classes if c != erased and c in table]
        if not others:
            missing.append("%s (no preserved classes)" % erased)
            continue
        rows[erased] = {
            "ESR-1": 1.0 - table[erased][0],
            "ESR-2": 1.0 - table[erased][1],
            "PSR-1": sum(table[c][0] for c in others) / len(others),
            "PSR-2": sum(table[c][1] for c in others) / len(others),
        }
    return rows, missing


def main():
    ap = argparse.ArgumentParser(description="Build one table row.")
    ap.add_argument("--mode", required=True, choices=("shared", "per-column"))
    ap.add_argument("--class_set", default="imagenette", choices=sorted(CLASS_SETS))
    ap.add_argument("--method", default="base",
                    help="Method name as written in per_class.csv.")
    ap.add_argument("--label", default=None,
                    help="Row label in the LaTeX output. Defaults to --method.")
    ap.add_argument("--per_class", default=None,
                    help="shared mode: path to the single per_class.csv.")
    ap.add_argument("--results_root", default=None,
                    help="per-column mode: directory holding <class>/per_class.csv.")
    ap.add_argument("--clip_csv", default=None,
                    help="Optional clip_score.csv; adds CLIP rows to the block.")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14",
                    help="Which model's numbers to take from --clip_csv.")
    ap.add_argument("--clip_mode", default="absolute",
                    choices=("absolute", "delta", "both"),
                    help="absolute: the unlearned model only. delta: method "
                         "minus base, the only form that may be averaged over "
                         "columns. both: base and method side by side, four "
                         "CLIP rows per block.")
    ap.add_argument("--clip_delta", action="store_true",
                    help="Alias for --clip_mode delta.")
    ap.add_argument("--ceiling_csv", default=None,
                    help="Optional ceiling_score CSV; adds the normalised target "
                         "score as further rows. Only mapping rows have one.")
    ap.add_argument("--ceiling_raw", action="store_true",
                    help="With --ceiling_csv, also print the raw CLIP-tgt floor, "
                         "method and ceiling values the percentage is computed "
                         "from, so the width of each interval is visible.")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    if args.clip_delta:
        if args.clip_mode not in ("absolute", "delta"):
            raise SystemExit("--clip_delta contradicts --clip_mode %s" % args.clip_mode)
        args.clip_mode = "delta"

    classes = CLASS_SETS[args.class_set]
    label = args.label or args.method
    second = SECOND_METRIC_NAME[args.class_set]
    display = {"ESR-2": "ESR-" + second, "PSR-2": "PSR-" + second}
    missing = []

    if args.mode == "shared":
        if not args.per_class:
            raise SystemExit("shared mode needs --per_class")
        table = read_per_class(args.per_class, args.method)
        if not table:
            raise SystemExit("No rows for method '%s' in %s" % (args.method, args.per_class))
        rows = cells_from_shared(table, classes)
    else:
        if not args.results_root:
            raise SystemExit("per-column mode needs --results_root")
        rows, missing = cells_from_per_column(args.results_root, classes, args.method)

    if not rows:
        raise SystemExit("Nothing to report.")

    # The LaTeX writer appends "$<arrow>$", so a name closes its own math.
    if args.clip_mode == "delta":
        display["CLIP-src"] = "$\\Delta$CLIP-src"
        display["CLIP-tgt"] = "$\\Delta$CLIP-tgt"
    elif args.clip_mode == "both":
        display.update(DISPLAY_CLIP)

    metrics = list(METRICS)
    clip_shown = []
    if args.clip_csv:
        side = "base" if args.mode == "shared" and args.method == "base" else "method"
        clip = read_clip(args.clip_csv, args.method, args.clip_model, side,
                         mode=args.clip_mode)
        for name in CLIP_METRICS[args.clip_mode]:
            if any(name in cell for cell in clip.values()):
                metrics.append(name)
                clip_shown.append(name)
        for concept, cell in clip.items():
            if concept in rows:
                rows[concept].update(cell)

    if args.ceiling_csv:
        display.update(DISPLAY_CEILING)
        ceiling = read_ceiling(args.ceiling_csv, args.method, args.clip_model)
        wanted = list(CEILING_METRICS)
        if args.ceiling_raw:
            # Floor, method and ceilings first, then the percentages they
            # produce, so the block reads in the order the metric is defined.
            display.update(DISPLAY_RAW)
            wanted = list(RAW_METRICS) + wanted
        for name in wanted:
            if any(name in cell for cell in ceiling.values()):
                metrics.append(name)
        for concept, cell in ceiling.items():
            if concept in rows:
                rows[concept].update(cell)

    # A metric may be missing for some columns, so each average is taken
    # over the columns that actually carry it.
    avg = {}
    for metric in metrics:
        values = [r[metric] for r in rows.values() if metric in r]
        avg[metric] = sum(values) / len(values) if values else None

    print("=" * 78)
    print("%s   (%d of %d columns, %s)" % (label, len(rows), len(classes), args.class_set))
    print("=" * 78)
    print("%18s | %7s %7s | %7s %7s"
          % ("erased concept", "ESR-1", display["ESR-2"], "PSR-1", display["PSR-2"]))
    print("-" * 78)
    for erased in classes:
        if erased not in rows:
            print("%18s |    ----    ---- |    ----    ----" % erased)
            continue
        r = rows[erased]
        print("%18s | %7.3f %7.3f | %7.3f %7.3f"
              % (erased, r["ESR-1"], r["ESR-2"], r["PSR-1"], r["PSR-2"]))
    print("-" * 78)
    print("%18s | %7.3f %7.3f | %7.3f %7.3f"
          % ("AVG", avg["ESR-1"], avg["ESR-2"], avg["PSR-1"], avg["PSR-2"]))

    if clip_shown:
        mark = "d" if args.clip_mode == "delta" else ""
        width = 20 + 11 * len(clip_shown)
        print()
        print("%18s |" % "erased concept",
              " ".join("%10s" % (mark + m) for m in clip_shown))
        print("-" * width)
        for erased in classes:
            cell = rows.get(erased, {})
            print("%18s |" % erased,
                  " ".join("%10.3f" % cell[m] if m in cell else "      ----"
                           for m in clip_shown))
        print("-" * width)
        print("%18s |" % "AVG",
              " ".join("%10.3f" % avg[m] if avg.get(m) is not None else "      ----"
                       for m in clip_shown))

    if missing:
        print("\nMISSING (%d):" % len(missing))
        for item in missing:
            print("   %s" % item)
        print("Row computed from an incomplete set — AVG is not comparable.")

    print()
    print("%% --- LaTeX row: %s ---" % label)
    print("\\multirow{%d}{*}{%s}" % (len(metrics), label))
    for metric in metrics:
        scale = SCALE[metric]
        # A delta is worth an explicit sign: the direction is the whole point,
        # and a bare "1.42" reads as a level rather than a change.
        cell = "%+.2f" if args.clip_mode == "delta" and metric in clip_shown else "%.2f"
        body = " & ".join(
            cell % (rows[c][metric] * scale)
            if c in rows and metric in rows[c] else "--"
            for c in LATEX_ORDER[args.class_set]
        )
        tail = "--" if avg[metric] is None else cell % (avg[metric] * scale)
        # A base column is a reference, not a result, so it gets no arrow.
        arrow = ARROWS.get(metric, "\\uparrow")
        name = display.get(metric, metric) + ("$%s$" % arrow if arrow else "")
        print("& %s & %s & %s \\\\" % (name, body, tail))
    print("\\midrule")

    if args.out_json:
        parent = os.path.dirname(args.out_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "class_set": args.class_set,
            "method": args.method,
            "label": label,
            "mode": args.mode,
            "clip_mode": args.clip_mode if args.clip_csv else None,
            "ceiling_csv": args.ceiling_csv,
            "clip_model": args.clip_model if args.clip_csv else None,
            "columns_present": sorted(rows),
            "missing": missing,
            "results": {args.method: dict(rows, AVG=avg)},
        }
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print("\nWrote %s" % args.out_json)


if __name__ == "__main__":
    main()
