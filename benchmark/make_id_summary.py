"""One row per method: Original, Erase, Preserve, averaged over identities.

The per-identity table is the one to publish, but it is the wrong shape for
choosing between methods -- six tables of eleven columns cannot be compared by
eye. This collapses each to a single line so the trade-off is visible at a
glance: erasure strength against collateral damage, method by method.

Preserve is also shown as a fraction of Original, because that is the number
that travels between papers. A model that never drew a convincing face has
little to lose, so an absolute Preserve of 0.19 means one thing when Original
is 0.40 and quite another when it is 0.25.

    python benchmark/make_id_summary.py \\
        results/identity9/id_similarity_*_per_video.csv

    python benchmark/make_id_summary.py --people "Angela Merkel" ... -- FILES
"""

import argparse
import csv
import importlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def method_from_path(path):
    """The method label a per-video file was written for."""
    name = os.path.basename(path)
    match = re.match(r"id_similarity_(.+)_per_video\.csv$", name)
    return match.group(1) if match else name


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


def read_one(path, method, wanted):
    original, erase, preserve = [], [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["id_sim"] or row["person"] not in wanted:
                continue
            value = float(row["id_sim"])
            if row["method"] == "base" and row["erased"] == "none":
                original.append(value)
            elif row["method"] == method and row["erased"] in wanted:
                (erase if row["erased"] == row["person"] else preserve).append(value)
    return original, erase, preserve


def main():
    ap = argparse.ArgumentParser(description="Compare methods on one line each.")
    ap.add_argument("files", nargs="+", help="id_similarity_*_per_video.csv")
    ap.add_argument("--prompts_module", default="identity_prompts")
    ap.add_argument("--people", nargs="+", default=None)
    ap.add_argument("--label", default=None,
                    help="Prefix stripped from method names in the output.")
    args = ap.parse_args()

    people = args.people or importlib.import_module(args.prompts_module).PEOPLE
    wanted = set(people)

    print("=" * 84)
    print("%d identities: %s" % (len(people), ", ".join(people)))
    print("=" * 84)
    print("%-34s %18s %18s %10s" % ("method", "Erase v", "Preserve ^", "P/O"))
    print("-" * 84)

    rows = []
    shown_original = None
    for path in args.files:
        method = method_from_path(path)
        original, erase, preserve = read_one(path, method, wanted)
        o_mean, o_std = mean_std(original)
        e_mean, e_std = mean_std(erase)
        p_mean, p_std = mean_std(preserve)
        if shown_original is None and o_mean is not None:
            shown_original = (o_mean, o_std)
        ratio = "" if not (o_mean and p_mean) else "%.0f%%" % (100 * p_mean / o_mean)
        name = method
        if args.label and name.startswith(args.label):
            name = name[len(args.label):]
        print("%-34s %8.4f +/-%.4f %8.4f +/-%.4f %10s"
              % (name, e_mean or 0.0, e_std or 0.0, p_mean or 0.0, p_std or 0.0, ratio))
        rows.append((name, e_mean, e_std, p_mean, p_std, ratio))

    print("-" * 84)
    if shown_original:
        print("Original (no erasure): %.4f +/- %.4f" % shown_original)
    print("Erase is the erased identity; Preserve averages the others in that "
          "same model.\nP/O is Preserve over Original: how much of what the base "
          "model could do survives.")

    print("\n%% --- LaTeX ---")
    for name, e_mean, e_std, p_mean, p_std, ratio in rows:
        print("%s & %.4f$_{\\pm %.4f}$ & %.4f$_{\\pm %.4f}$ & %s \\\\"
              % (name.replace("_", r"\_"), e_mean or 0.0, e_std or 0.0,
                 p_mean or 0.0, p_std or 0.0, ratio))


if __name__ == "__main__":
    main()
