"""Assemble the paper's tables from the scored results.

Two tables, one command each.

    python -m evaluation.tables \
        experiments/objects_null.yaml \
        experiments/objects_mapping1.yaml \
        experiments/objects_mapping2.yaml \
        --out tables/objects.tex

    python -m evaluation.tables experiments/identities.yaml \
        --out tables/identities.tex

The object table carries the untouched model as its first row, read from the
baseline results of the prompt set the experiments share. Without it the other
rows have no scale: a class the model never drew recognisably has an ESR near
a hundred before anything is unlearned, and a row of such classes would look
like a result.

Every class is always reported. There is no exclusion mechanism, because any
threshold chosen after seeing the numbers is indistinguishable from tuning the
metric, and an average that depends on one is not an average of anything.

Needs booktabs and multirow in the document preamble.
"""

import argparse
import csv
import os
import sys

from experiments import config as expconfig
from experiments import layout

OBJECT_METRICS = ("ESR-1", "ESR-5", "PSR-1", "PSR-5")
IDENTITY_ROWS = ("Original", "Erase", "Preserve")
IDENTITY_ARROW = {"Original": "", "Erase": r"$\downarrow$", "Preserve": r"$\uparrow$"}


# --------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------

def read_per_class(path):
    """{concept: (top1, top5)} from one per_class.csv."""
    table = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[row["concept"]] = (float(row["top1_acc"]), float(row["top5_acc"]))
    return table


def cells_from(table, classes, erased=None):
    """ESR and PSR for one column, or for all of them from a shared table.

    With `erased` given this is one cell of one column. Without it the whole
    row is derived from a single set of accuracies, which is valid only when
    the videos do not depend on the column -- that is, for the untouched model,
    where there is one set of two hundred videos and ten ways of slicing it.
    """
    present = [c for c in classes if c in table]
    targets = [erased] if erased else present
    out = {}
    for target in targets:
        if target not in table:
            continue
        others = [c for c in present if c != target]
        if not others:
            continue
        out[target] = {
            "ESR-1": 1.0 - table[target][0],
            "ESR-5": 1.0 - table[target][1],
            "PSR-1": sum(table[c][0] for c in others) / len(others),
            "PSR-5": sum(table[c][1] for c in others) / len(others),
        }
    return out


def object_row(cfg, classes, results_root, base=False):
    """One block of the object table: a dict of cells plus what is missing."""
    if base:
        path = os.path.join(layout.base_results(cfg["prompts"], root=results_root),
                            "per_class.csv")
        if not os.path.exists(path):
            return {}, ["the whole baseline (no %s)" % path]
        return cells_from(read_per_class(path), classes), []

    cells, missing = {}, []
    for row in expconfig.rows(cfg):
        path = os.path.join(
            layout.column_results(cfg["name"], row.concept, root=results_root),
            "per_class.csv")
        if not os.path.exists(path):
            missing.append("%s (no %s)" % (row.concept, path))
            continue
        got = cells_from(read_per_class(path), classes, erased=row.concept)
        if not got:
            missing.append("%s (its own class absent from %s)" % (row.concept, path))
            continue
        cells.update(got)
    return cells, missing


def average(cells, metrics):
    out = {}
    for metric in metrics:
        values = [c[metric] for c in cells.values() if metric in c]
        out[metric] = sum(values) / len(values) if values else None
    return out


def print_object_block(label, cells, order, missing):
    avg = average(cells, OBJECT_METRICS)
    print("=" * 78)
    print("%s   (%d of %d columns)" % (label, len(cells), len(order)))
    print("-" * 78)
    print("%18s | %7s %7s | %7s %7s"
          % ("erased concept", "ESR-1", "ESR-5", "PSR-1", "PSR-5"))
    for concept in order:
        cell = cells.get(concept)
        if not cell:
            print("%18s |    ----    ---- |    ----    ----" % concept)
            continue
        print("%18s | %7.3f %7.3f | %7.3f %7.3f"
              % (concept, cell["ESR-1"], cell["ESR-5"],
                 cell["PSR-1"], cell["PSR-5"]))
    print("-" * 78)
    shown = ["   ----" if avg[m] is None else "%7.3f" % avg[m]
             for m in OBJECT_METRICS]
    print("%18s | %s %s | %s %s" % tuple(["AVG"] + shown))
    if missing:
        print("\nMISSING (%d) -- AVG is not comparable:" % len(missing))
        for item in missing:
            print("   %s" % item)
    print()


def object_table(blocks, order, headers):
    """The LaTeX table. `blocks` is [(label, cells)] in printing order."""
    out = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Object erasure on the ten Imagenette classes. Every frame "
           r"of every generated video is classified with an ImageNet-1k "
           r"classifier; the column names the concept that was erased.}",
           r"\label{tab:objects}",
           r"\resizebox{\textwidth}{!}{",
           r"\begin{tabular}{ll%sc}" % ("c" * len(order)),
           r"\toprule",
           r"\textbf{Method} & \textbf{Metric}",
           r"& \multicolumn{%d}{c}{\textbf{Erased concept}}" % len(order),
           r"& \textbf{AVG} \\",
           r"\cmidrule(lr){3-%d}" % (2 + len(order)),
           "&" + "".join(" & %s" % h for h in headers) + r" & \\",
           r"\midrule"]
    for label, cells in blocks:
        avg = average(cells, OBJECT_METRICS)
        out.append(r"\multirow{%d}{*}{%s}" % (len(OBJECT_METRICS), label))
        for metric in OBJECT_METRICS:
            body = " & ".join("%.2f" % (cells[c][metric] * 100) if c in cells else "--"
                              for c in order)
            tail = "--" if avg[metric] is None else "%.2f" % (avg[metric] * 100)
            out.append(r"& %s$\uparrow$ & %s & %s \\" % (metric, body, tail))
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", "}", r"\end{table*}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Identities
# --------------------------------------------------------------------------

def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def identity_buckets(path, people):
    """{(row, person): [per-video cosines]} for Original, Erase and Preserve.

    Preserve is indexed by the identity that was erased, not by the identity
    that was generated: the question the column answers is what that model did
    to everybody else.
    """
    wanted = set(people)
    buckets = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["id_sim"] or row["person"] not in wanted:
                continue
            value = float(row["id_sim"])
            erased, person = row["erased"], row["person"]
            if erased == "none":
                key = ("Original", person)
            elif erased in wanted:
                key = ("Erase", person) if erased == person else ("Preserve", erased)
            else:
                continue
            buckets.setdefault(key, []).append(value)
    return buckets


def identity_table(label, buckets, people, short, decimals=4):
    fmt = "%%.%df" % decimals
    table = {}

    header = "%-10s" % "" + "".join("%12s" % short.get(p, p) for p in people)
    print("=" * len(header + " " * 20))
    print("%s" % label)
    print("-" * len(header + " " * 20))
    print(header + "%20s" % "AVG")
    for name in IDENTITY_ROWS:
        cells, pooled = [], []
        for person in people:
            values = buckets.get((name, person), [])
            pooled.extend(values)
            cells.append(mean_std(values)[0])
        avg, std = mean_std(pooled)
        table[name] = (cells, avg, std)
        print("%-10s" % name
              + "".join("%12s" % ("----" if c is None else fmt % c) for c in cells)
              + "%20s" % ("----" if avg is None
                          else (fmt % avg) + " +/- " + (fmt % std)))
    print("-" * len(header + " " * 20))
    print("The spread is over individual videos, not over identities: averaging")
    print("the per-identity means first reports a spread of almost zero and says")
    print("nothing about how unevenly the metric behaves, which is the part worth")
    print("knowing. A difference smaller than it is not a difference.")
    print()

    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Identity erasure. Original is the untouched model asked "
           r"for that identity; Erase is the model that removed it, asked for "
           r"it again; Preserve is that same model asked for the other four. "
           r"The spread is over videos.}",
           r"\label{tab:identities}",
           r"\resizebox{\columnwidth}{!}{",
           r"\begin{tabular}{ll%sc}" % ("c" * len(people)),
           r"\toprule",
           r"\textbf{Method} & \textbf{Metric}"
           + "".join(" & %s" % short.get(p, p) for p in people)
           + r" & \textbf{AVG} \\",
           r"\midrule",
           r"\multirow{%d}{*}{%s}" % (len(IDENTITY_ROWS), label)]
    for name in IDENTITY_ROWS:
        cells, avg, std = table[name]
        body = " & ".join("--" if c is None else fmt % c for c in cells)
        tail = "--" if avg is None else r"%s$_{\pm %s}$" % (fmt % avg, fmt % std)
        out.append(r"& %s%s & %s & %s \\" % (name, IDENTITY_ARROW[name], body, tail))
    out += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Assemble a table from scored results.")
    ap.add_argument("configs", nargs="+",
                    help="Experiment files, in the order they appear as rows.")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default=None, help="Write the LaTeX here.")
    ap.add_argument("--no-base", action="store_true",
                    help="Leave the untouched model out of the object table.")
    ap.add_argument("--base-label", default="HunyuanVideo")
    ap.add_argument("--decimals", type=int, default=4,
                    help="Identity table only; the object table is fixed at two.")
    args = ap.parse_args()

    configs = [expconfig.load(path) for path in args.configs]
    kinds = {cfg["kind"] for cfg in configs}
    if len(kinds) > 1:
        raise SystemExit("One table cannot mix %s: the columns differ."
                         % " and ".join(sorted(kinds)))
    kind = kinds.pop()

    if kind == "identities":
        if len(configs) > 1:
            raise SystemExit("The identity table holds one experiment; it is "
                             "already three rows wide.")
        cfg = configs[0]
        module = expconfig.prompt_module(cfg)
        path = os.path.join(
            layout.experiment_results(cfg["name"], root=args.results_root),
            "id_similarity_per_video.csv")
        if not os.path.exists(path):
            raise SystemExit("No scored results at %s -- run evaluation.arcface "
                             "first." % path)
        buckets = identity_buckets(path, module.PEOPLE)
        text = identity_table(cfg["label"], buckets, module.PEOPLE,
                              getattr(module, "SHORT", {}), args.decimals)
    else:
        sets = {cfg["prompts"] for cfg in configs}
        if len(sets) > 1:
            raise SystemExit("These experiments use different prompt sets (%s), "
                             "so their rows have different baselines and cannot "
                             "share a table." % ", ".join(sorted(sets)))
        module = expconfig.prompt_module(configs[0])
        classes, order = list(module.CLASSES), list(module.TABLE_ORDER)

        blocks = []
        if not args.no_base:
            cells, missing = object_row(configs[0], classes, args.results_root,
                                        base=True)
            print_object_block(args.base_label, cells, order, missing)
            blocks.append((args.base_label, cells))
        for cfg in configs:
            cells, missing = object_row(cfg, classes, args.results_root)
            print_object_block(cfg["label"], cells, order, missing)
            blocks.append((cfg["label"], cells))

        headers = [r"\shortstack{%s}" % c.replace(" ", r"\\") if " " in c else c
                   for c in order]
        text = object_table(blocks, order, headers)

    if args.out:
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("Wrote %s" % args.out)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
