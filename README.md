# FOMO — research branch

The published code is on `main`, together with its README. This branch adds
what is needed to reproduce the object and public-figure experiments: the
evaluation prompts, the mapping tables, the metrics and the SLURM layer.

- [docs/SETUP.md](docs/SETUP.md) — installation, running the trainer by hand
- [docs/REPRODUCING.md](docs/REPRODUCING.md) — paper table to command, and how to read the metrics

```bash
DRY=1 bash slurm/run.sh experiments/objects_mapping2.yaml   # look first
bash slurm/run.sh experiments/objects_mapping2.yaml         # train, generate, score
python3 -m evaluation.tables experiments/objects_*.yaml --out tables/objects.tex
```

## Layout

```
fomo/                         the method
    train.py                  training loop and losses
    localization.py           the concept mask
    lora.py                   adapter placement for HunyuanVideo
    flow.py tokens.py attention.py

experiments/                  one file per block of results in the paper
    config.py                 reads an experiment file, for Python and for bash
    layout.py                 where everything lives on disk
    generate.py               produces the videos of one cell
    prompts/                  the evaluation prompts
    mappings/                 what to unlearn, and what to put in its place
    *.yaml

evaluation/
    resnet.py                 ResNet-50 over every frame; ESR and PSR
    arcface.py                ArcFace against reference photographs
    tables.py                 the LaTeX tables

safree/                       the vendored method FOMO is compared against
slurm/                        env.sh, three sbatch scripts, and run.sh
```

Every entry point is a module, run from the repository root:
`python3 -m fomo.train`, `python3 -m experiments.generate`,
`python3 -m evaluation.tables`.

Adapters, videos, results and tables are written under `adapters/`, `videos/`,
`results/` and `tables/`, all ignored by git. So is `data/`, which is where the
reference photographs for the identity metric go; they are not distributed.

## What is not here

The unsafe-content and motion experiments live on their own branches. They need
things this one does not have at all -- a vision-language evaluator, DINO and
LPIPS, VBench, and several external prompt sets.
