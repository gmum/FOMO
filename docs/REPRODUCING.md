# Reproducing the object and identity experiments

Every number in these two families comes from one experiment file. The file is
the whole definition of a run: which prompts, which mapping, how long to train,
how to sample. Nothing that changes a number is passed on a command line.

| in the paper | experiment file | table |
|---|---|---|
| FOMO, Null mapping | `experiments/objects_null.yaml` | objects |
| FOMO, Safe Mapping 1 | `experiments/objects_mapping1.yaml` | objects |
| FOMO, Safe Mapping 2 | `experiments/objects_mapping2.yaml` | objects |
| NegPrompt | `experiments/objects_negprompt.yaml` | objects |
| SAFREE | `experiments/objects_safree.yaml` | objects |
| FOMO on public figures | `experiments/identities.yaml` | identities |

The object rows of the main table are Safe Mapping 2. The other two mappings
are the ablation that asks whether the method depends on the target being a
near neighbour of what was erased.

## One command

```bash
bash slurm/run.sh experiments/objects_mapping2.yaml
```

That submits four arrays, chained by dependency: training, the untouched model,
the columns, and scoring. `DRY=1` prints the `sbatch` lines without submitting
anything, which is worth doing first.

Running a second mapping afterwards should skip the baseline, because it is
keyed by the prompt set and is already there:

```bash
SKIP_BASE=1 bash slurm/run.sh experiments/objects_mapping1.yaml
```

The two rows we compare against train nothing, so they skip training by
themselves. The SAFREE row additionally has no baseline of its own: it
replaces the pipeline class, so the untouched model is not available in the
same process.

Then, once the arrays are done:

```bash
python3 -m evaluation.tables \
    experiments/objects_null.yaml \
    experiments/objects_mapping1.yaml \
    experiments/objects_mapping2.yaml \
    --out tables/objects.tex

python3 -m evaluation.tables experiments/identities.yaml \
    --out tables/identities.tex
```

The object table carries the untouched model as its first row. Both commands
print the numbers before writing the LaTeX, and name anything missing; a row
assembled from an incomplete set says so, because its average is not
comparable with a complete one.

## The same thing step by step

```bash
CONFIG=experiments/objects_mapping2.yaml

# 1. one adapter per concept. Array index = row of the experiment.
sbatch --array=0-9 --export=ALL,CONFIG=$CONFIG slurm/train.sbatch

# 2. the untouched model. Array index = subject.
sbatch --array=0-9 --export=ALL,CONFIG=$CONFIG,MODE=base slurm/generate.sbatch

# 3. the columns. Array index = row * subjects + subject.
sbatch --array=0-99 --export=ALL,CONFIG=$CONFIG slurm/generate.sbatch

# 4. scoring. Index 0 is the baseline, 1..10 are the columns.
sbatch --array=0-10 --export=ALL,CONFIG=$CONFIG slurm/evaluate.sbatch
```

`python3 -m experiments.config $CONFIG` prints the rows and how many subjects
there are, which is where the array bounds come from. For identities the shape
is 5 x 5, so the arrays are `0-4`, `0-4`, `0-24` and `0`.

Everything is resumable. Generation skips files that exist, scoring overwrites,
and the identity scorer appends to its per-video CSV under `--resume`, so a job
killed at the wall clock costs nothing beyond resubmitting it.

## What lands where

```
adapters/<experiment>/<concept>/<mask>_alpha<a>/checkpoint-NNNNNN/
videos/<prompts>/base/<subject>/NN.mp4
videos/<experiment>/<erased>/<subject>/NN.mp4
results/<prompts>/base/{per_class.csv,frames.csv,summary.json}
results/<experiment>/<erased>/{per_class.csv,frames.csv,summary.json}
results/<experiment>/id_similarity{,_per_video}.csv
tables/<name>.tex
```

A column is named after the concept it erases, and `base` is the column that
erases nothing. Both hold one directory per subject underneath, so a scorer
pointed at either does not need to know which it got.

The baseline is keyed by the prompt set rather than by the experiment, because
that is what it depends on: the three object experiments differ only in the
mapping they train towards, and all three are read against the same two hundred
untouched videos.

`frames.csv` is worth keeping. The predicted label is what explains a
surprising cell: a class whose accuracy collapsed with nothing erased is
usually being predicted as its ImageNet neighbour, a cassette rather than a
cassette player, and that is a property of the prompt and not of the method.

## How to read the metrics

Objects, where a column is the concept that was erased:

    ESR-k = 1 - Top-k on that concept        erasure
    PSR-k = mean Top-k over the other nine   collateral damage

Every class is always reported. There is no exclusion mechanism, because a
threshold chosen after seeing the numbers is indistinguishable from tuning the
metric, and an average that depends on one is not an average of anything.

Identities, three rows:

    Original    the untouched model asked for that person. The ceiling. Without
                it the other two have no scale, because a person the model
                never drew convincingly looks free to erase.
    Erase       the model that erased P, asked for P. Lower is better.
    Preserve    that same model asked for the other four. Higher is better.

The spread after the plus-minus sign is over individual videos, not over
identities. Averaging the per-identity means first reports a spread of almost
zero and says nothing about how unevenly the metric behaves, which is the part
worth knowing: ArcFace on generated faces is noisy, and a difference smaller
than that spread is not a difference.

**The face detection rate is part of the identity measurement.** A method that
destroys the face outright leaves nothing to embed, and averaging over the
frames that do contain a face reports exactly the frames where erasure failed --
so the number looks highest when the method worked best. Every row of
`id_similarity.csv` therefore carries `n_frames`, `n_faces` and
`detection_rate`, and a row whose detection rate collapsed is successful
erasure whatever its cosine says.

## Differences from the runs that produced the paper

Two, both deliberate.

The mapping table was rewritten to match the paper. Four object classes --
cassette player, church, garbage truck and tench -- had earlier targets in the
tables that were actually used for some of the development runs, and
`experiments/mappings/imagenette.tsv` now carries the ones the paper reports.

Directory names changed with the restructuring, so the result directories left
on the cluster from the original runs do not match the layout above. They were
not renamed, because nothing is gained by renaming files nobody will read
again; a fresh run writes to the new layout.
