# Object-erasure benchmark

ESR/PSR evaluation of concept erasure in HunyuanVideo, over the ten Imagenette
classes or the ten CIFAR-10 classes.

Protocol: for each target concept, generate 17-frame videos from 20 prompts,
classify every frame, and pool accuracies per class.

    ESR-k(X) = 1 - Top-k(X)                       erasure efficacy
    PSR-k(X) = mean Top-k over the other nine     collateral damage

Column X of the table means "X is the erased concept", so a full row needs all
ten classes generated under the model that erases X.

## Files

| file | purpose |
|---|---|
| `common.py` | class lists, prompt sets, article handling, directory naming. Shared by everything so generation and evaluation cannot disagree. Run it directly to validate the prompt sets. |
| `env.sh` | activates the conda environment and verifies the required packages. Sourced by every sbatch; not meant to be run on its own. |
| `generate.py` | generate videos for one class of one column |
| `evaluate.py` | classify frames, write `per_class.csv` |
| `make_row.py` | build one table row from per-class accuracies |
| `make_table.py` | assemble the LaTeX table from several rows |
| `run_generate.sbatch` | SLURM array wrapper around `generate.py` |
| `run_evaluate.sbatch` | SLURM array wrapper around `evaluate.py` |

Run everything from the repository root.

## Directory layout

Generated videos:

```
videos/<tag>/
├── _shared/
│   └── base/                          the ONLY real copies of base-model videos
│       ├── tench/00.mp4 … 19.mp4
│       ├── church/00.mp4 … 19.mp4
│       └── …ten classes x 20 videos = 200 files
│
├── unlearn_none/                      reference column
│   ├── tench/
│   │   ├── base     -> ../../_shared/base/tench    (symlink)
│   │   └── cfgbase/ 00.mp4 … 19.mp4                (CFG-matched control)
│   ├── church/  …same for all ten classes
│   └── prompts/<class>.json           prompt and seed map, one file per class
│
├── unlearn_church/                    column "church is erased"
│   ├── tench/
│   │   ├── base       -> ../../_shared/base/tench
│   │   ├── negprompt/ 00.mp4 … 19.mp4
│   │   └── ours/      00.mp4 … 19.mp4
│   ├── church/  …same for all ten classes
│   └── prompts/
│
└── unlearn_<other eight>/
```

Read a path as **what is erased / what is generated / with which method**:
`unlearn_church/tench/negprompt/03.mp4` is a tench video, generated with
`negative_prompt="church"`, from prompt 3.

`<tag>` encodes everything that changes the videos, so two runs with different
settings can never collide:

```
imagenette_v3                     class set, prompt set; 512x288, gs 6, seed 42
imagenette_v1_1280x720            non-default resolution spelled out
cifar10_v4_768x432_gs10_seed200   everything non-default spelled out
```

**Why `_shared`.** The base model does not depend on which concept is erased:
same prompts, same seeds, no intervention. Without sharing, eleven columns
would hold 2200 files of which 2000 are duplicates — roughly 150 GPU-hours
wasted at 1280x720. With sharing there are 200 real files and 110 symlinks.
The symlinks are relative, so the whole `<tag>` directory can be renamed or
moved. Use `find -L` when counting files, otherwise symlinked directories are
not followed.

Evaluation results:

```
results/<tag>/
├── per_class.csv                      column-independent methods (base, cfgbase)
├── frames.csv
├── summary.json
├── negprompt/                         one subdirectory per column
│   ├── tench/per_class.csv
│   └── …ten columns
└── rows/                              JSON rows for make_table.py
    ├── base.json
    └── negprompt.json
```

## Running

```bash
cd /path/to/repo
```

That is all that is needed. `benchmark/env.sh`, sourced by every sbatch,
locates and activates the conda environment on its own and checks that the
packages the job needs are importable. If it cannot, the job stops immediately
and prints every path it tried, rather than falling through to the system
python and failing minutes later inside `import torch`.

Two optional overrides, for an unusual install or a differently named
environment:

```bash
export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export ENV_NAME=eraser
```

### 1. Base model

Ten tasks, one per class. Videos land in `_shared/base` and are symlinked into
`unlearn_none`.

```bash
sbatch --array=0-9%5 --export=ALL,CLASS_SET=imagenette,PROMPT_SET=v1,\
WIDTH=1280,HEIGHT=720 benchmark/run_generate.sbatch
```

### 2. CFG-matched control

HunyuanVideo is CFG-distilled: `guidance_scale` feeds a guidance embedding
rather than performing real two-branch guidance, so a negative prompt has
nowhere to go. Enabling it requires `guidance_scale=1` plus `true_cfg_scale>1`,
which means the NegPrompt row is **sampled differently from the base row**.

This control uses that same sampling with an *empty* negative prompt, so
`negprompt - cfgbase` isolates the effect of the negation text. It is
column-independent, so it is generated once, next to `base`.

```bash
sbatch --array=0-9%5 --export=ALL,CLASS_SET=imagenette,PROMPT_SET=v1,\
WIDTH=1280,HEIGHT=720,NEGPROMPT=1,NEGPROMPT_TEXT=,NEGPROMPT_LABEL=cfgbase,\
NO_BASE=1 benchmark/run_generate.sbatch
```

`NEGPROMPT_TEXT=` with nothing after it is meaningful: it is the empty string,
not "unset". Unset would negate the column's concept instead.

### 3. NegPrompt

Full 10x10 matrix: every column needs all ten classes generated under that
column's negative prompt, otherwise PSR cannot be measured. Array index is
`column * 10 + class`.

```bash
sbatch --array=0-99%10 --export=ALL,MODE=matrix,CLASS_SET=imagenette,\
PROMPT_SET=v1,WIDTH=1280,HEIGHT=720,NEGPROMPT=1,NO_BASE=1 \
benchmark/run_generate.sbatch
```

### 4. A trained method

One column per submission, ten tasks. `MODELS` takes `label=path`; the path may
be a weights directory or a parent containing `checkpoint-*`, in which case the
newest is used.

```bash
sbatch --array=0-9%5 --export=ALL,CLASS_SET=imagenette,PROMPT_SET=v1,\
WIDTH=1280,HEIGHT=720,ERASED="church",\
MODELS="ours=outputs/imap_in_church_to_house/checkpoint-000150",\
NO_BASE=1 benchmark/run_generate.sbatch
```

Repeat for each of the ten concepts, changing `ERASED` and `MODELS`.

### 5. Evaluate

Column-independent methods, one task:

```bash
sbatch --array=0 --export=ALL,MODE=shared,CLASS_SET=imagenette,PROMPT_SET=v1,\
WIDTH=1280,HEIGHT=720,METHODS="base cfgbase" benchmark/run_evaluate.sbatch
```

Column-dependent methods, ten tasks:

```bash
sbatch --array=0-9%5 --export=ALL,MODE=per-column,CLASS_SET=imagenette,\
PROMPT_SET=v1,WIDTH=1280,HEIGHT=720,METHOD=negprompt \
benchmark/run_evaluate.sbatch
```

### 6. Build the table

```bash
TAG=imagenette_v1_1280x720

python3 benchmark/make_row.py --mode shared \
    --per_class results/$TAG/per_class.csv \
    --class_set imagenette --method base --label "HunyuanVideo" \
    --out_json results/$TAG/rows/base.json

python3 benchmark/make_row.py --mode shared \
    --per_class results/$TAG/per_class.csv \
    --class_set imagenette --method cfgbase --label "HunyuanVideo (CFG-matched)" \
    --out_json results/$TAG/rows/cfgbase.json

python3 benchmark/make_row.py --mode per-column \
    --results_root results/$TAG/negprompt \
    --class_set imagenette --method negprompt --label "NegPrompt" \
    --out_json results/$TAG/rows/negprompt.json

python3 benchmark/make_table.py --class_set imagenette \
    --rows results/$TAG/rows/base.json \
           results/$TAG/rows/cfgbase.json \
           results/$TAG/rows/negprompt.json \
    --out paper/tab_imagenette.tex
```

The document needs `booktabs`, `multirow` and `graphicx`.

## Two evaluation modes

Picking the wrong one produces a plausible-looking but meaningless row, so the
distinction is worth stating explicitly. It is **not** "baseline versus the
rest" but "do the videos depend on the column":

| | videos depend on column? | evaluation | row mode |
|---|---|---|---|
| `base`, `cfgbase` | no — one set of 200 videos | 1 run | `shared` |
| `negprompt`, `esd`, `ours` | yes — 200 videos per column | 10 runs | `per-column` |

In `shared` mode all ten columns come from a single `per_class.csv`, because
`ESR(X)` and `PSR(X)` are both derivable from the same per-class accuracies.
That shortcut is invalid as soon as each column has its own model.

## Notes

- **Prompt sets.** There are two kinds, both selected with `PROMPT_SET`.

  *Template sets* (`v1`-`v4`) share one wording across all ten classes, filled
  in with `{a} {c}`: `{c}` is the class name verbatim, `{a}` the agreeing
  article. `{a}` is only correct immediately before `{c}` — do not write
  `{a} nice {c}`. No template may contain concept-specific attributes, so that
  every column is asked for in the same way.

  *Per-class sets* (`v5`, `v6`, `v7`) are concrete scenes written separately for each
  class, which avoids making the prompt distribution artificially correlated
  between concepts. The class name still appears verbatim in every prompt, so a
  drop in accuracy after erasure is attributable to the model rather than to an
  ambiguous prompt. Both cover Imagenette only; requesting one for another
  class set is an error, not a silent fallback.

  `v5` mixes lengths on purpose — roughly a third short (8-15 words), the rest
  descriptive (25-40) — so erasure strength can be examined against prompt
  specificity. `v6` is `v5` rewritten for recognisability: uniform 10-20 words,
  the class filling the frame in its canonical ImageNet framing, plain daylight,
  no second object that is itself an ImageNet class. It exists because on `v5`
  the reference model scores 49.8 mean Top-1 against 64.4 on `v1`, with tench
  at 0.0 and garbage truck at 5.6 — columns whose reference accuracy is near
  zero cannot measure erasure at all. `v6` lifts the mean to 70.5, but two
  classes regress: describing the medium rather than the device sent cassette
  player to 10.0 (predicted `cassette`, a separate ImageNet class), and
  "liver-and-white" sent English springer to 0.0 (predicted Border collie).
  `v7` is `v6` with those two classes and tench rewritten and the other seven
  left identical, so the only difference between the sets is the change under
  test. It is defined in `common.py` as a diff, not as a fresh table.

  None of these sets was built by keeping the prompts that scored best. Each
  prompt is one video, so a per-prompt accuracy of 1.00 or 0.00 is a single
  coin flip, and selecting on it would fit the seed. Whichever set is used for
  the paper must be frozen before any erasure method is run, and reported as
  such — a prompt set re-picked after seeing method results is a tuned
  benchmark.

  `python3 benchmark/common.py` validates every set (size, duplicates, verbatim
  class name) and prints the length distribution.
- **Seeds.** `seed = base_seed + prompt_index`, independent of the method, so
  the same prompt gets the same noise in every row and the comparison is
  paired. Prompt set `v3` relies on this: twenty identical templates give
  twenty different videos.
- **Effective sample size.** Frames within one video are highly correlated, so
  340 frames per class are closer to 20 independent samples than 340. Fine for
  the point estimate; compute error bars over videos, not frames.
- **All ten classes are always reported.** There is no exclusion mechanism.
  Some concepts the generator cannot produce recognisably, so their `ESR` is
  near 100 even with no erasure — that is what the measurement says and it is
  reported as such. Dropping columns would make `AVG` depend on a threshold,
  and a threshold picked after seeing the numbers is indistinguishable from
  tuning the metric. Report the reference model's Top-1 per class alongside the
  table so a reader can see which columns are informative.
- **Resumable.** `--skip_existing` is on by default; re-running a generation
  array fills gaps only. Evaluation is cheap and simply overwrites.
- **Branches.** `sbatch` copies the `.sbatch` file at submission time, but
  Python files are read when each task *starts*. Switching branches while an
  array is queued silently swaps the code underneath it. Check `squeue` first.
