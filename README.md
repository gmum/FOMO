# FOMO — research branch

The published code is on `main`, together with its README. This branch adds
what is needed to reproduce the object and public-figure experiments: the
evaluation prompts, the mapping tables, the metrics and the SLURM layer.

- [docs/SETUP.md](docs/SETUP.md) — installation and the GPU memory flags
- [docs/USAGE.md](docs/USAGE.md) — structure and what to run

```bash
# unlearn something
python3 -m fomo.unlearn --concept dog --target cat --out adapters/unlearn_dog

# look at one video
python3 -m fomo.sample --prompt "a video of a dog" --out after.mp4 \
    --adapter adapters/unlearn_dog

# reproduce one row of a table, or one concept of it
bash slurm/reproduce_experiment.sh experiments/objects_mapping2.yaml
CONCEPT=church bash slurm/reproduce_experiment.sh experiments/objects_mapping2.yaml
```

## Layout

```
fomo/                         the method
    train.py                  training loop and losses
    unlearn.py                unlearn one concept, short flags
    sample.py                 one video, with or without an adapter
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

methods/                      the methods FOMO is compared against
    negprompt.py
    safree/                   its own fork of the HunyuanVideo pipeline
slurm/                        env.sh, three sbatch scripts, reproduce_experiment.sh
```

Every entry point is a module, run from the repository root:
`python3 -m fomo.unlearn`, `python3 -m experiments.generate`,
`python3 -m evaluation.resnet`.

Adapters, videos and results are written under `adapters/`, `videos/` and
`results/`, all ignored by git. So is `data/`, which is where the reference
photographs for the identity metric go; they are not distributed.

## What is not here

The unsafe-content and motion experiments live on their own branches. They need
things this one does not have at all -- a vision-language evaluator, DINO and
LPIPS, VBench, and several external prompt sets.
