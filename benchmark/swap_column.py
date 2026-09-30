"""Replace one column of a composite row with another method's result.

A table row is not always one training run. The near row of the Imagenette
table is assembled from three: the mapping for most classes, a replacement for
tench after every fish-to-fish target failed, and a different instrument for
French horn. make_row.py reads one directory and one method label, so the
composite has to exist on disk before it can be read.

This does that swap in two places at once, which is the whole point -- moving
per_class.csv without moving the CLIP row leaves a table whose accuracies come
from one model and whose similarities come from another, and nothing downstream
would notice.

    python benchmark/swap_column.py \\
        --results_root results/imagenette_v7_1280x720 \\
        --row imap_shark_lr1e3 --column "French horn" \\
        --from_method imap_near_lr1e3 \\
        --clip_from clip_score.csv --clip_base clip_score_fish.csv \\
        --clip_out clip_score_near.csv

The per_class.csv of the column is overwritten in place, with a .bak left
behind. The CLIP file is written to a new path and never edited in place.
"""

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import slugify  # noqa: E402


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Swap one column into a composite row.")
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--row", required=True,
                    help="Directory and method label of the composite row.")
    ap.add_argument("--column", required=True,
                    help="Erased concept whose cell is being replaced.")
    ap.add_argument("--from_method", required=True,
                    help="Method whose result is taken for that column.")
    ap.add_argument("--clip_base", default=None,
                    help="CLIP CSV holding the row as it stands.")
    ap.add_argument("--clip_from", default=None,
                    help="CLIP CSV holding --from_method for this column.")
    ap.add_argument("--clip_out", default=None,
                    help="Where the merged CLIP CSV is written.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    slug = slugify(args.column)
    src = os.path.join(args.results_root, args.from_method, slug, "per_class.csv")
    dst = os.path.join(args.results_root, args.row, slug, "per_class.csv")
    for path in (src, dst):
        if not os.path.exists(path):
            raise SystemExit("missing: %s" % path)

    rows, fields = read_rows(src)
    rows = [r for r in rows if r["method"] == args.from_method]
    if not rows:
        raise SystemExit("no rows for method %r in %s" % (args.from_method, src))
    for row in rows:
        row["method"] = args.row

    old, _ = read_rows(dst)
    print("per_class: %s" % dst)
    print("   was %d rows from %s" % (len(old), {r["method"] for r in old}))
    print("   now %d rows from %s, relabelled %s"
          % (len(rows), args.from_method, args.row))

    if not args.dry_run:
        shutil.copy(dst, dst + ".bak")
        write_rows(dst, rows, fields)
        print("   backup at %s.bak" % dst)

    if not (args.clip_base and args.clip_from and args.clip_out):
        print("\nCLIP untouched: give --clip_base, --clip_from and --clip_out "
              "to merge it too. Leaving it is a mistake unless the row has no "
              "CLIP columns -- the accuracies and the similarities would then "
              "describe two different models.")
        return

    base_path = os.path.join(args.results_root, args.clip_base)
    from_path = os.path.join(args.results_root, args.clip_from)
    out_path = os.path.join(args.results_root, args.clip_out)

    base_rows, base_fields = read_rows(base_path)
    from_rows, _ = read_rows(from_path)

    keep = [r for r in base_rows
            if r["method"] == args.row and r["concept"] != args.column]
    take = [dict(r, method=args.row) for r in from_rows
            if r["method"] == args.from_method and r["concept"] == args.column]
    if not take:
        raise SystemExit("no CLIP rows for (%s, %s) in %s"
                         % (args.from_method, args.column, from_path))

    merged = keep + take
    merged.sort(key=lambda r: (r["clip_model"], r["concept"]))
    print("\nCLIP: %s" % out_path)
    print("   %d rows kept from %s" % (len(keep), args.clip_base))
    print("   %d rows taken from %s" % (len(take), args.clip_from))
    for row in take:
        print("      %s -> %s" % (row["concept"], row.get("target_concept", "?")))

    if not args.dry_run:
        write_rows(out_path, merged, base_fields)


if __name__ == "__main__":
    main()
