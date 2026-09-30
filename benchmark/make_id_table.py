"""The face-erasure table: Original, Erase and Preserve, per identity.

Three rows, following the layout the face-erasure literature uses.

    Original    the untouched model asked for that identity, scored against
                the reference photographs. This is the ceiling. Without it the
                other two rows have no scale, because a model that never drew
                a convincing Merkel makes her erasure look free.
    Erase       the model that erased identity P, asked for P. Lower is better.
    Preserve    the same model asked for everybody else, averaged. Higher is
                better; this is where collateral damage shows up.

The number after the plus-minus sign is the spread over individual videos, not
over identities. Averaging the per-identity means first would report a spread
of almost zero and say nothing about how unevenly the metric behaves, which is
the part worth knowing: ArcFace on generated faces is noisy, and a difference
smaller than this spread is not a difference.

    python benchmark/make_id_table.py \\
        --per_video results/identity9/id_similarity_per_video.csv \\
        --method imap_demo_lr1e3_000050

    python benchmark/make_id_table.py ... \\
        --people "Angela Merkel" "Barack Obama" "Donald Trump" \\
                 "Joe Biden" "Queen Elizabeth II"
"""

import argparse
import collections
import csv
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Short column headers. A table eleven columns wide has no room for
# "Queen Elizabeth II".
SHORT = {
    "Angela Merkel": "Merkel",
    "Queen Elizabeth II": "Elizabeth",
    "Joe Biden": "Biden",
    "Donald Trump": "Trump",
    "Barack Obama": "Obama",
    "LeBron James": "LeBron",
    "Lionel Messi": "Messi",
    "Cristiano Ronaldo": "Ronaldo",
    "Taylor Swift": "Swift",
}


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


def collect(path, method, people):
    """{(row, person): [per-video cosines]} for the three rows."""
    wanted = set(people)
    buckets = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["id_sim"]:
                continue
            person, erased = row["person"], row["erased"]
            if person not in wanted:
                continue
            value = float(row["id_sim"])
            if row["method"] == "base" and erased == "none":
                buckets[("Original", person)].append(value)
            elif row["method"] == method and erased in wanted:
                if erased == person:
                    buckets[("Erase", person)].append(value)
                else:
                    # Indexed by the identity that was erased, because the
                    # Preserve column asks what that model did to everyone else.
                    buckets[("Preserve", erased)].append(value)
    return buckets


def main():
    ap = argparse.ArgumentParser(description="Build the face-erasure table.")
    ap.add_argument("--per_video", required=True,
                    help="id_similarity per-video CSV.")
    ap.add_argument("--method", required=True,
                    help="Method label for the Erase and Preserve rows.")
    ap.add_argument("--prompts_module", default="identity_prompts")
    ap.add_argument("--people", nargs="+", default=None,
                    help="Subset and column order. Defaults to the module's PEOPLE.")
    ap.add_argument("--label", default=None, help="Model name in the first column.")
    ap.add_argument("--decimals", type=int, default=4)
    args = ap.parse_args()

    people = args.people or importlib.import_module(args.prompts_module).PEOPLE
    buckets = collect(args.per_video, args.method, people)
    label = args.label or "HunyuanVideo"

    rows = ("Original", "Erase", "Preserve")
    arrow = {"Original": "", "Erase": r"$\downarrow$", "Preserve": r"$\uparrow$"}
    fmt = "%%.%df" % args.decimals

    print("=" * 78)
    print("%s   method=%s" % (label, args.method))
    print("=" * 78)
    head = "%-10s" % "" + "".join("%12s" % SHORT.get(p, p)[:11] for p in people)
    print(head + "%18s" % "AVG")
    print("-" * len(head + " " * 18))
    table = {}
    for name in rows:
        cells, pooled = [], []
        for person in people:
            values = buckets.get((name, person), [])
            pooled.extend(values)
            mean, _ = mean_std(values)
            cells.append(mean)
        avg, std = mean_std(pooled)
        table[name] = (cells, avg, std)
        print("%-10s" % name
              + "".join("%12s" % ("----" if c is None else fmt % c) for c in cells)
              + "%18s" % ("----" if avg is None
                          else (fmt % avg) + " +/- " + (fmt % std)))
    print("-" * len(head + " " * 18))
    print("+/- is the standard deviation over videos, not over identities.")

    print("\n%% --- LaTeX ---")
    print(r"\multirow{3}{*}{%s}" % label)
    for name in rows:
        cells, avg, std = table[name]
        body = " & ".join("--" if c is None else fmt % c for c in cells)
        tail = "--" if avg is None else r"%s$_{\pm %s}$" % (fmt % avg, fmt % std)
        print("& %s%s & %s & %s \\\\" % (name, arrow[name], body, tail))
    print(r"\midrule")


if __name__ == "__main__":
    main()
