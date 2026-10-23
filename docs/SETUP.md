# Setup

## Installation

```bash
conda create -n fomo python=3.10 -y
conda activate fomo

# torch separately: the right index depends on the platform. On aarch64
# (for example GH200) the cu124 index stops at 2.5.1, so 2.6.0 has to come
# from cu126.
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Nothing is importable as an installed package. Every entry point is run as a
module from the repository root, which is what puts the root on the import
path:

```bash
python3 -m experiments.config experiments/objects_mapping2.yaml
```

## Model weights

The implementation targets
[HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo). The
weights are downloaded on first use and take tens of gigabytes. On a cluster
point the cache at something with room before the first run:

```bash
export HF_HOME=/path/with/space/huggingface
```

`slurm/env.sh` sets it to `.cache/huggingface` inside the repository if you do
not.

## The submitting host

`slurm/env.sh` loads `MODULES` (default `CUDA/12.8.0`), finds conda, activates
`ENV_NAME` (default `fomo`, a path works too), sets the caches, and stops a job
with a readable message if any of that fails rather than dying later on
`import torch`. All four are overridable:

```bash
export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export ENV_NAME=/path/to/conda/envs/some_env
export MODULES="CUDA/12.8.0 cuDNN/9.2.1.18-CUDA-12.8.0"   # ArcFace wants cuDNN
export HF_HOME=/path/with/space/huggingface
```

`slurm/reproduce_experiment.sh` runs on the host you submit from, which on
some clusters is a different architecture from the compute nodes — on Helios
the login node is x86 while the GH200 nodes are aarch64, so the conda
environment will not run there at all. That host needs PyYAML of its own;
everything else the script touches is in the standard library.

```bash
python3 -c "import yaml"        # on Helios this already passes
```

If it does not, and the login node has no `pip`, `python3 -m pip install --user
pyyaml` or a `module load` of a newer python is the way in.

## Reference photographs

The identity metric compares generated faces against photographs you supply,
one directory per person, three to five clear frontal shots each:

```
data/identity_refs/donald_trump/*.jpg
data/identity_refs/barack_obama/*.jpg
```

`data/` is ignored by git. Published photographs of public figures are
somebody's copyright, which is why none are distributed here.

```bash
python3 -m evaluation.arcface experiments/identities.yaml --check-refs
```

Reports how well each person's photographs agree. Two shots of one face land at
0.6-0.9 cosine; much lower usually means a different person, a hard profile, or
the wrong face from a group shot.

## GPU memory

At 720x1280 training needs close to 80 GB and will not fit without three flags.
`fomo.unlearn` and the SLURM scripts set them; calling `fomo.train` directly is
the one way to get this wrong.

| flag | why |
|---|---|
| `--no-v_disable_gc` | the default is the **opposite** and turns gradient checkpointing off for the student pass whenever the preservation loss is active, which exhausts a 95 GB card |
| `--split_hv_backward` | splits that pass in two, at the cost of one extra forward |
| `--gradient_checkpointing` | |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is also required: without it
the allocator fragments and a run that fits reports an out-of-memory error with
gigabytes reserved and unused. `slurm/env.sh` exports it; in a plain shell you
have to. Dropping to `--height 432 --width 768` is the other way out.

## Next

[USAGE.md](USAGE.md): structure, what to run, what lands where, file formats.
