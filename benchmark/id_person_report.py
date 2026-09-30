"""Per-identity report: who is worth keeping in the face-erasure table.

Choosing the identities is not a cosmetic decision. An identity the base model
never rendered convincingly cannot be erased in any measurable sense, so its
column reports noise dressed up as a result -- and averaging it in drags the
whole table toward a conclusion nobody measured.

Four numbers per person, all from the same scored runs:

    Original     the base model asked for this person, against the reference
                 photographs. The ceiling. Below roughly 0.25 the model is not
                 really drawing this face at all.
    Face rate    how often a face was detected in those base videos. A high
                 Original on a low face rate means the few frames that had a
                 face happened to match, which is not the same thing.
    Erase        this person in the model that erased them. Lower is better.
    Caused       what erasing this person did to everyone else.
    Suffered     what happened to this person when somebody else was erased.

Caused and Suffered are the interesting pair. If damage were uniform they would
be flat across identities; where they are not, collateral damage is following
facial similarity rather than spreading evenly, and that is worth a sentence in
the paper.

    python benchmark/id_person_report.py results/identity9/*_per_video.csv
"""

import argparse
import collections
import csv
import importlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHORT = {
    "Angela Merkel": "Merkel", "Queen Elizabeth II": "Elizabeth",
    "Joe Biden": "Biden", "Donald Trump": "Trump", "Barack Obama": "Obama",
    "LeBron James": "LeBron", "Lionel Messi": "Messi",
    "Cristiano Ronaldo": "Ronaldo", "Taylor Swift": "Swift",
    "Michael Jackson": "Jackson",
}


def method_from_path(path):
    match = re.match(r"id_similarity_(.+)_per_video\.csv$", os.path.basename(path))
    return match.group(1) if match else os.path.basename(path)


def mean(values):
    return sum(values) / len(values) if values else None


def cell(value, width=9):
    return " " * width if value is None else "%*.4f" % (width, value)


def main():
    ap = argparse.ArgumentParser(description="Per-identity face-erasure report.")
    ap.add_argument("files", nargs="+", help="id_similarity_*_per_video.csv")
    ap.add_argument("--prompts_module", default="identity_prompts")
    ap.add_argument("--people", nargs="+", default=None)
    ap.add_argument("--floor", type=float, default=0.25,
                    help="Original below this means the identity is not usable.")
    args = ap.parse_args()

    people = args.people or importlib.import_module(args.prompts_module).PEOPLE
    wanted = set(people)

    original = collections.defaultdict(list)
    base_frames = collections.Counter()
    base_faces = collections.Counter()
    erase = collections.defaultdict(lambda: collections.defaultdict(list))
    caused = collections.defaultdict(lambda: collections.defaultdict(list))
    suffered = collections.defaultdict(lambda: collections.defaultdict(list))
    methods = []

    for path in args.files:
        method = method_from_path(path)
        if method not in methods:
            methods.append(method)
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                person, erased = row["person"], row["erased"]
                if person not in wanted:
                    continue
                if row["method"] == "base" and erased == "none":
                    base_frames[person] += int(row["n_frames"])
                    base_faces[person] += int(row["n_faces"])
                    if row["id_sim"]:
                        original[person].append(float(row["id_sim"]))
                    continue
                if row["method"] != method or erased not in wanted or not row["id_sim"]:
                    continue
                value = float(row["id_sim"])
                if erased == person:
                    erase[person][method].append(value)
                else:
                    caused[erased][method].append(value)
                    suffered[person][method].append(value)

    methods.sort()
    order = sorted(people, key=lambda p: -(mean(original[p]) or -1))

    print("=" * 96)
    print("RECOGNISABILITY  --  can the base model draw this person at all")
    print("=" * 96)
    print("%-12s %9s %10s   %s" % ("person", "Original", "face rate", "verdict"))
    print("-" * 96)
    for person in order:
        value = mean(original[person])
        rate = (base_faces[person] / base_frames[person]) if base_frames[person] else 0.0
        if value is None:
            verdict = "no base videos"
        elif value < args.floor:
            verdict = "TOO WEAK -- column measures noise"
        elif value < args.floor + 0.10:
            verdict = "borderline"
        else:
            verdict = "good"
        print("%-12s %s %9.1f%%   %s"
              % (SHORT.get(person, person), cell(value), 100 * rate, verdict))
    print("-" * 96)
    print("Ordered by Original. The threshold is %.2f; identities below it are "
          "not really\nbeing drawn, so erasing them proves nothing."
          % args.floor)

    print("\n" + "=" * 96)
    print("ERASURE  --  this person in the model that erased them (lower is better)")
    print("=" * 96)
    head = "%-12s" % "person" + "".join("%11s" % m.replace("imap_", "")[:10] for m in methods)
    print(head)
    print("-" * len(head))
    for person in order:
        print("%-12s" % SHORT.get(person, person)
              + "".join(cell(mean(erase[person][m]), 11) for m in methods))

    print("\n" + "=" * 96)
    print("COLLATERAL  --  averaged over methods")
    print("=" * 96)
    print("%-12s %9s %9s %9s %8s   %s"
          % ("person", "Original", "caused", "suffered", "surv %", "reading"))
    print("-" * 96)
    for person in order:
        o = mean(original[person])
        c = mean([v for m in methods for v in caused[person][m]])
        s = mean([v for m in methods for v in suffered[person][m]])
        pct = "" if not (o and s) else "%.0f%%" % (100 * s / o)
        if o is None or s is None:
            note = ""
        elif s / o > 0.6:
            note = "survives other erasures well"
        elif s / o < 0.25:
            note = "badly damaged by other erasures"
        else:
            note = ""
        print("%-12s %s %s %s %8s   %s"
              % (SHORT.get(person, person), cell(o), cell(c), cell(s), pct, note))
    print("-" * 96)
    print("caused   : mean similarity of everybody else in the model that erased "
          "this person.\nsuffered : mean similarity of this person across the "
          "models that erased somebody else.\nsurv %   : suffered over Original.")

    print("\n" + "=" * 96)
    print("SUGGESTED SET")
    print("=" * 96)
    usable = [p for p in order if (mean(original[p]) or 0) >= args.floor]
    weak = [p for p in order if (mean(original[p]) or 0) < args.floor]
    print("Usable (Original >= %.2f): %s" % (args.floor, ", ".join(SHORT.get(p, p) for p in usable)))
    if weak:
        print("Too weak to include:      %s" % ", ".join(SHORT.get(p, p) for p in weak))
    if len(usable) >= 5:
        print("\nTop five by Original: %s" % ", ".join(SHORT.get(p, p) for p in usable[:5]))


if __name__ == "__main__":
    main()
