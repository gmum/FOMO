"""Assemble a complete LaTeX table from rows produced by make_row.py.

Input is one or more JSON files written with `make_row.py --out_json`. Rows
appear in the order given on the command line.

Imagenette and CIFAR-10 cannot share a table: they are ten different classes,
so the column headers differ. This script refuses to mix them.

    python benchmark/make_table.py --class_set imagenette \\
        --rows results/imagenette_v1_1280x720/rows/base.json \\
               results/imagenette_v1_1280x720/rows/negprompt.json \\
        --out paper/tab_imagenette.tex

Requires booktabs, multirow and graphicx in the document preamble.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASS_SETS, LATEX_ORDER  # noqa: E402

METRICS = ("ESR-1", "ESR-2", "PSR-1", "PSR-2")
SECOND_METRIC_NAME = {"imagenette": "5", "cifar10": "p"}

HEADERS = {
    "imagenette": [
        r"\shortstack{cassette\\player}",
        r"\shortstack{chain\\saw}",
        "church",
        r"\shortstack{gas\\pump}",
        "tench",
        r"\shortstack{garbage\\truck}",
        r"\shortstack{English\\springer}",
        r"\shortstack{golf\\ball}",
        "parachute",
        r"\shortstack{French\\horn}",
    ],
    "cifar10": CLASS_SETS["cifar10"],
}

CAPTIONS = {
    "imagenette": ("Object erasure on the ten Imagenette classes, scored with an "
                   "ImageNet-1k classifier on every generated frame."),
    "cifar10": ("Object erasure on the ten CIFAR-10 classes. The CIFAR-10 labels "
                "are not ImageNet classes, so frames are scored with CLIP "
                "zero-shot over the ten labels; the second metric is the mean "
                "probability of the correct label rather than Top-5, which "
                "carries almost no information over ten candidates."),
}

LABELS = {"imagenette": "tab:erasure_imagenette", "cifar10": "tab:erasure_cifar10"}

PLACEHOLDERS = ("SAFREE", "ICE*", "Ours")


def load_row(path, class_set):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("class_set") != class_set:
        raise SystemExit("%s is for class set '%s', not '%s'"
                         % (path, payload.get("class_set"), class_set))
    method = payload["method"]
    cells = payload["results"][method]
    return payload.get("label", method), cells


def main():
    ap = argparse.ArgumentParser(description="Assemble the LaTeX results table.")
    ap.add_argument("--class_set", required=True, choices=sorted(CLASS_SETS))
    ap.add_argument("--rows", nargs="+", required=True,
                    help="JSON files from make_row.py --out_json, in table order.")
    ap.add_argument("--out", default=None, help="Write here instead of stdout.")
    ap.add_argument("--no_placeholders", action="store_true",
                    help="Do not append empty rows for methods not yet run.")
    args = ap.parse_args()

    cs = args.class_set
    second = SECOND_METRIC_NAME[cs]
    display = {"ESR-2": "ESR-" + second, "PSR-2": "PSR-" + second}
    order = LATEX_ORDER[cs]

    rows = [load_row(path, cs) for path in args.rows]

    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    out.append(r"\caption{%s}" % CAPTIONS[cs])
    out.append(r"\label{%s}" % LABELS[cs])
    out.append(r"\resizebox{\textwidth}{!}{")
    out.append(r"\begin{tabular}{llccccccccccc}")
    out.append(r"\toprule")
    out.append(r"\textbf{Methods} & \textbf{Metrics}")
    out.append(r"& \multicolumn{10}{c}{\textbf{Erased Concepts}}")
    out.append(r"& \textbf{AVG} \\")
    out.append(r"\cmidrule(lr){3-12}")
    out.append(r"&")
    for header in HEADERS[cs]:
        out.append("& %s" % header)
    out.append(r"& \\")
    out.append(r"\midrule")

    for label, cells in rows:
        out.append(r"\multirow{4}{*}{%s}" % label)
        avg = cells.get("AVG", {})
        for metric in METRICS:
            body = " & ".join(
                "%.2f" % (cells[c][metric] * 100) if c in cells else "--"
                for c in order
            )
            tail = "%.2f" % (avg[metric] * 100) if metric in avg else "--"
            out.append("& %s$\\uparrow$ & %s & %s \\\\"
                       % (display.get(metric, metric), body, tail))
        out.append(r"\midrule")

    if not args.no_placeholders:
        for name in PLACEHOLDERS:
            out.append(r"\multirow{4}{*}{%s}" % name)
            for metric in METRICS:
                out.append("& %s$\\uparrow$ & %s \\\\"
                           % (display.get(metric, metric), " & ".join([""] * 11)))
            out.append(r"\midrule")

    if out[-1] == r"\midrule":
        out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    out.append(r"}")
    out.append(r"\end{table*}")

    text = "\n".join(out) + "\n"
    if args.out:
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("Wrote %s (%d data rows)" % (args.out, len(rows)))
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
