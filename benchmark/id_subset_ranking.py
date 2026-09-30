"""Does the method ranking survive the choice of identities?

The face table can legitimately be published over five people or over nine, and
a reader who sees only one of those has no way to tell whether the ordering of
the methods is a property of the methods or a property of the five people that
happened to be chosen. This walks every subset from five identities up to nine
-- 256 of them -- recomputes Erase, Preserve and P/O on each, and reports how
often the ranking changes.

The aggregation matches make_id_summary.py exactly: means are pooled over
videos, not averaged over per-person means, so a subset's numbers are what that
subset's table would print. Restricting to a subset filters *both* sides of
Preserve -- only videos where the erased identity and the prompted identity are
both in the subset count, which is the only reading under which Preserve means
"everybody else in the table".

    python benchmark/id_subset_ranking.py results/identity9/*_per_video.csv

    # every subset printed, not just the summary
    python benchmark/id_subset_ranking.py --dump -- results/identity9/*_per_video.csv

    # count a video with no detected face as zero similarity
    python benchmark/id_subset_ranking.py --no_face_as_zero -- FILES
"""

import argparse
import collections
import csv
import importlib
import itertools
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


def label(method):
    return method.replace("imap_", "").replace("_lr1e3", "")


def short(person):
    return SHORT.get(person, person)


class Acc(object):
    """Running sum and count, so subset means are pooled rather than nested."""

    __slots__ = ("total", "n")

    def __init__(self):
        self.total = 0.0
        self.n = 0

    def add(self, value):
        self.total += value
        self.n += 1


def pool(accs):
    total = sum(a.total for a in accs)
    n = sum(a.n for a in accs)
    return (total / n) if n else None


def read(files, wanted, zero):
    """base[person], cell[method][(erased, person)], and the no-face tallies."""
    base = collections.defaultdict(Acc)
    cell = collections.defaultdict(lambda: collections.defaultdict(Acc))
    missing = collections.Counter()
    methods = []

    for path in files:
        method = method_from_path(path)
        if method not in methods:
            methods.append(method)
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                person, erased = row["person"], row["erased"]
                if person not in wanted:
                    continue
                blank = not row["id_sim"]
                if blank and not zero:
                    if row["method"] == method and erased in wanted:
                        missing[method] += 1
                    continue
                value = 0.0 if blank else float(row["id_sim"])
                if row["method"] == "base" and erased == "none":
                    base[person].add(value)
                elif row["method"] == method and erased in wanted:
                    cell[method][(erased, person)].add(value)

    return base, cell, missing, sorted(methods)


def score(subset, base, cell, methods):
    """Erase, Preserve, P/O per method for one subset of identities."""
    original = pool([base[p] for p in subset])
    out = {}
    for method in methods:
        table = cell[method]
        erase = pool([table[(p, p)] for p in subset])
        preserve = pool([table[(e, p)] for e in subset for p in subset if e != p])
        ratio = None
        if original and preserve is not None:
            ratio = preserve / original
        out[method] = (erase, preserve, ratio)
    return original, out


def order_by(scores, methods, index):
    """Method labels ranked, best first. Erase ascends; Preserve and P/O descend."""
    key = (lambda m: (scores[m][index] if scores[m][index] is not None else 1e9))
    reverse = index != 0
    if reverse:
        key = (lambda m: -(scores[m][index] if scores[m][index] is not None else -1e9))
    return tuple(sorted(methods, key=key))


def fmt(value, width=8, places=4):
    return " " * width if value is None else "%*.*f" % (width, places, value)


def main():
    ap = argparse.ArgumentParser(
        description="Method ranking over every identity subset of size 5..9.")
    ap.add_argument("files", nargs="+", help="id_similarity_*_per_video.csv")
    ap.add_argument("--prompts_module", default="identity_prompts")
    ap.add_argument("--people", nargs="+", default=None)
    ap.add_argument("--min_size", type=int, default=5)
    ap.add_argument("--max_size", type=int, default=None)
    ap.add_argument("--no_face_as_zero", action="store_true",
                    help="A video with no detected face scores 0 instead of "
                         "dropping out of the mean.")
    ap.add_argument("--rank_by", choices=["po", "preserve", "erase"], default="po")
    ap.add_argument("--dump", action="store_true",
                    help="Print every subset, not just the summary.")
    ap.add_argument("--top", type=int, default=8,
                    help="How many extreme subsets to list per size.")
    args = ap.parse_args()

    people = args.people or importlib.import_module(args.prompts_module).PEOPLE
    people = list(people)
    wanted = set(people)
    hi = min(args.max_size or len(people), len(people))
    index = {"erase": 0, "preserve": 1, "po": 2}[args.rank_by]

    base, cell, missing, methods = read(args.files, wanted, args.no_face_as_zero)
    if not methods:
        sys.exit("No per-video files matched id_similarity_<method>_per_video.csv.")

    labels = {m: label(m) for m in methods}
    subsets = [s for k in range(args.min_size, hi + 1)
               for s in itertools.combinations(people, k)]

    results = {}
    for subset in subsets:
        results[subset] = score(subset, base, cell, methods)

    # The full set is the reference ordering even when it is outside the
    # requested sizes, so it is scored here rather than looked up.
    full = tuple(people)
    if full not in results:
        results[full] = score(full, base, cell, methods)
    full_order = order_by(results[full][1], methods, index)

    print("=" * 100)
    print("%d identities, %d methods, %d subsets of size %d..%d"
          % (len(people), len(methods), len(subsets), args.min_size, hi))
    print("no-face videos: %s"
          % ("counted as 0.0" if args.no_face_as_zero
             else "dropped -- " + ", ".join("%s %d" % (labels[m], missing[m])
                                            for m in methods)))
    print("ranked by: %s" % {"po": "P/O (higher better)",
                             "preserve": "Preserve (higher better)",
                             "erase": "Erase (lower better)"}[args.rank_by])
    print("=" * 100)

    # ---- spread of each metric across subsets, by subset size -------------
    print("\nSPREAD  --  the same method scored on every subset of a given size")
    print("-" * 100)
    print("%-6s %-14s %23s %23s %19s"
          % ("size", "method", "Erase (min/med/max)", "Preserve (min/med/max)",
             "P/O (min..max)"))
    print("-" * 100)
    for size in range(args.min_size, hi + 1):
        group = [s for s in subsets if len(s) == size]
        for method in full_order:
            cols = []
            for idx in (0, 1, 2):
                vals = sorted(results[s][1][method][idx] for s in group
                              if results[s][1][method][idx] is not None)
                cols.append(vals)
            e, p, r = cols
            med = lambda v: v[len(v) // 2] if v else None
            print("%-6d %-14s %7s %7s %7s %7s %7s %7s %9s %9s"
                  % (size, labels[method],
                     fmt(e[0] if e else None, 7), fmt(med(e), 7), fmt(e[-1] if e else None, 7),
                     fmt(p[0] if p else None, 7), fmt(med(p), 7), fmt(p[-1] if p else None, 7),
                     fmt(r[0] if r else None, 9, 3), fmt(r[-1] if r else None, 9, 3)))
        if size != hi:
            print("")

    # ---- how often the ranking changes ------------------------------------
    orders = collections.Counter(order_by(results[s][1], methods, index)
                                 for s in subsets)
    print("\n" + "=" * 100)
    print("RANKING STABILITY")
    print("=" * 100)
    print("Ranking on all %d: %s" % (len(people), "  >  ".join(labels[m] for m in full_order)))
    agree = orders[full_order]
    print("That exact ordering holds on %d of %d subsets (%.1f%%)."
          % (agree, len(subsets), 100.0 * agree / len(subsets)))
    print("Distinct orderings seen: %d" % len(orders))
    print("-" * 100)
    for order, count in orders.most_common(10):
        mark = "  <- full set" if order == full_order else ""
        print("%5d  %5.1f%%   %s%s"
              % (count, 100.0 * count / len(subsets),
                 " > ".join(labels[m] for m in order), mark))

    # ---- pairwise dominance ----------------------------------------------
    print("\n" + "=" * 100)
    print("PAIRWISE  --  % of subsets where the row method beats the column method")
    print("=" * 100)
    head = "%-16s" % "" + "".join("%16s" % labels[m] for m in full_order)
    print(head)
    print("-" * len(head))
    better = (lambda a, b: a < b) if index == 0 else (lambda a, b: a > b)
    for a in full_order:
        line = "%-16s" % labels[a]
        for b in full_order:
            if a == b:
                line += "%16s" % "--"
                continue
            wins = sum(1 for s in subsets
                       if results[s][1][a][index] is not None
                       and results[s][1][b][index] is not None
                       and better(results[s][1][a][index], results[s][1][b][index]))
            line += "%15.1f%%" % (100.0 * wins / len(subsets))
        print(line)
    print("-" * len(head))
    print("A cell at 100% means that ordering is not an artefact of which\n"
          "identities were chosen; anything near 50% means it is.")

    # ---- which identities move the number --------------------------------
    lead = full_order[0]
    print("\n" + "=" * 100)
    print("LEVERAGE  --  what each identity does to %s, over subsets of size %d..%d"
          % (labels[lead], args.min_size, min(hi, len(people) - 1)))
    print("=" * 100)
    contrast = [s for s in subsets if len(s) < len(people)]
    print("%-12s %10s %10s %10s   %s"
          % ("person", "with", "without", "delta", "reading"))
    print("-" * 100)
    rows = []
    for person in people:
        inc = [results[s][1][lead][index] for s in contrast
               if person in s and results[s][1][lead][index] is not None]
        exc = [results[s][1][lead][index] for s in contrast
               if person not in s and results[s][1][lead][index] is not None]
        if not inc or not exc:
            continue
        a, b = sum(inc) / len(inc), sum(exc) / len(exc)
        rows.append((a - b, person, a, b))
    rows.sort(reverse=(index != 0))
    for delta, person, a, b in rows:
        good = (delta < 0) if index == 0 else (delta > 0)
        note = "flatters the method" if good else "drags the method down"
        print("%-12s %10.4f %10.4f %+10.4f   %s" % (short(person), a, b, delta, note))
    print("-" * 100)
    print("with    : mean over subsets containing this person.\n"
          "without : mean over subsets that leave them out.\n"
          "A large delta means the headline number depends on one identity.")

    # ---- the extreme subsets ---------------------------------------------
    print("\n" + "=" * 100)
    print("EXTREMES  --  best and worst subsets for %s" % labels[lead])
    print("=" * 100)
    # Every size actually generated except the full set, which has one member
    # and nothing to rank.
    for size in sorted({len(s) for s in subsets if len(s) < len(people)}):
        group = [s for s in subsets if len(s) == size
                 and results[s][1][lead][index] is not None]
        if not group:
            continue
        group.sort(key=lambda s: results[s][1][lead][index], reverse=(index != 0))
        print("\nsize %d  (%d subsets)" % (size, len(group)))
        print("%-8s %8s %8s %8s %7s   %s"
              % ("rank", "Original", "Erase", "Presv", "P/O", "identities"))
        print("-" * 100)
        show = list(range(min(args.top, len(group))))
        show += [i for i in range(max(len(group) - args.top, len(show)), len(group))]
        last = None
        for i in show:
            if last is not None and i != last + 1:
                print("%-8s %8s %8s %8s %7s   ..." % ("", "", "", "", ""))
            last = i
            s = group[i]
            original, sc = results[s]
            e, p, r = sc[lead]
            print("%-8d %s %s %s %7s   %s"
                  % (i + 1, fmt(original), fmt(e), fmt(p),
                     "" if r is None else "%6.1f%%" % (100 * r),
                     ", ".join(short(x) for x in s)))

    # ---- every subset, on request ----------------------------------------
    if args.dump:
        print("\n" + "=" * 100)
        print("EVERY SUBSET")
        print("=" * 100)
        head = "%-4s %8s" % ("size", "Orig") + \
               "".join("%22s" % labels[m] for m in full_order) + "   identities"
        print(head)
        print("%-4s %8s" % ("", "") +
              "".join("%22s" % "Erase / Presv / P/O" for _ in full_order))
        print("-" * len(head))
        for s in subsets:
            original, sc = results[s]
            line = "%-4d %s" % (len(s), fmt(original))
            for m in full_order:
                e, p, r = sc[m]
                line += "%22s" % ("%.4f %.4f %5.1f%%"
                                  % (e or 0.0, p or 0.0, 100 * (r or 0.0)))
            line += "   " + ", ".join(short(x) for x in s)
            print(line)


if __name__ == "__main__":
    main()
