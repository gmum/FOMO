# Usage

Install first: [SETUP.md](SETUP.md). Run everything from the repository root.

## Structure

```
.
├── fomo/                       the method
│   ├── train.py                training loop and losses
│   ├── unlearn.py              unlearn one concept
│   ├── sample.py               generate one video
│   ├── localization.py         the concept mask
│   ├── lora.py                 adapter placement
│   ├── flow.py                 timesteps and guidance
│   ├── tokens.py               concept token lookup
│   └── attention.py            attention capture
├── experiments/                what is measured
│   ├── config.py               reads an experiment file
│   ├── layout.py               all output paths
│   ├── generate.py             generates one cell
│   ├── prompts/                evaluation prompts
│   ├── mappings/               what to unlearn, and onto what
│   └── *.yaml                  one file per row of a table in the paper
├── evaluation/
│   ├── resnet.py               objects: ESR and PSR
│   └── arcface.py              identities: ID-similarity
├── slurm/
│   ├── env.sh                  conda and caches
│   ├── train.sbatch
│   ├── generate.sbatch
│   ├── evaluate.sbatch
│   └── reproduce_experiment.sh chains all stages
└── docs/
```

Written when you run things, none of them in git:

```
data/        reference photographs you supply
adapters/    trained adapters
videos/      generated video
results/     csv and json from scoring
logs/        slurm output
.cache/
```

Everything for one family sits under one directory named after the prompt set,
so the untouched model and every variant are listed side by side:

```
videos/imagenette/
├── base/tench/00.mp4 … 19.mp4
└── unlearn_church/
    └── tench/
        ├── null/00.mp4 …
        ├── mapping1/…
        └── mapping2/…

results/imagenette/base/{per_class,frames}.csv  summary.json
results/imagenette/unlearn_church/mapping1/{per_class,frames}.csv  summary.json
results/identities/demographic/id_similarity{,_per_video}.csv
adapters/objects_mapping1/church/unclamped_alpha7/checkpoint-000250/
```

`unlearn_<concept>` says what the column erased; the directory below the
subject says which run made those videos. The untouched model has no such level
because it has no variants — one set of videos per prompt set, shared by
everything compared against it.

## Unlearn one concept

```bash
python3 -m fomo.unlearn --concept dog --target cat --out adapters/unlearn_dog
```

## Generate one video

```bash
python3 -m fomo.sample --prompt "a video of a dog" --out after.mp4 \
    --adapter adapters/unlearn_dog
```

Drop `--adapter` for the untouched model; with the same `--seed` the two differ
only by the adapter.

## Reproduce one row of a table

| in the paper | file |
|---|---|
| FOMO, Null mapping | `experiments/objects_null.yaml` |
| FOMO, Safe Mapping 1 | `experiments/objects_mapping1.yaml` |
| FOMO, Safe Mapping 2 | `experiments/objects_mapping2.yaml` |
| public figures | `experiments/identities.yaml` |

```bash
bash slurm/reproduce_experiment.sh experiments/objects_mapping2.yaml
```

Baseline, training, generation and scoring, chained by dependency. `DRY=1` in
front prints the `sbatch` lines without submitting anything. Pass `SKIP_BASE=1`
for the second and later object rows, since the three share one baseline.

One concept of that row, all stages:

```bash
CONCEPT=church bash slurm/reproduce_experiment.sh experiments/objects_mapping2.yaml
```

The numbers end up in `logs/eval_*.out`. Reprint them any time:

```bash
python3 -m evaluation.resnet experiments/objects_mapping2.yaml --report
```

`--help` on any of these lists the rest.
