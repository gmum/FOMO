# Setup

The implementation targets
[HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo). The
weights are downloaded on first use and take tens of gigabytes, so on a cluster
point the cache somewhere with room before the first run:

```bash
export HF_HOME=/path/with/space/huggingface
```

## Installation

```bash
conda create -n fomo python=3.10 -y
conda activate fomo

# torch is installed separately because the right index depends on the
# platform. On aarch64 (for example GH200) the cu124 index stops at 2.5.1,
# so 2.6.0 has to come from cu126.
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Nothing here is importable as an installed package; every entry point is run as
a module from the repository root, which is also what puts the root on the
import path:

```bash
python3 -m experiments.config experiments/objects_mapping2.yaml
```

The SLURM scripts do this for you. `slurm/env.sh` finds conda, activates the
environment named by `ENV_NAME` (default `fomo`), sets the caches under
`.cache/`, and stops the job with a readable message if the environment is
missing instead of failing later on `import torch`.

`slurm/run.sh` runs on the host you submit from, which on some clusters is a
different architecture from the compute nodes -- on Helios the login node is
x86 while the GH200 nodes are aarch64, so the conda environment cannot run
there at all. The submitting host therefore needs one thing of its own:

```bash
pip install --user pyyaml
```

Everything else `run.sh` touches is in the standard library.

## Unlearning one concept by hand

The reproduction path goes through the experiment files and is described in
[REPRODUCING.md](REPRODUCING.md). To erase something that is not in them, call
the trainer directly. A run is defined by the concept to erase, the concept
that replaces it, and a pair of prompts differing only in those two words.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo '{"source_prompt": "a video of a dog", "target_prompt": "a video of a cat",
       "source_concept": "dog", "target_concept": "cat"}' > pairs.jsonl

python3 -m fomo.train \
    --output_dir adapters/manual/dog \
    --source_concept "dog" \
    --target_concept "cat" \
    --prompt_pairs_jsonl pairs.jsonl \
    --max_train_steps 250 \
    --learning_rate 1e-3 \
    --imap_alpha 7 \
    --lamb_v 0.5 --v_steps 1 --v_video_only \
    --no-v_disable_gc --split_hv_backward --gradient_checkpointing \
    --height 720 --width 1280 --num_frames 17 \
    --skip_eval
```

The file may hold more than one pair, one JSON object per line, in which case
each training step samples one of them. An empty `target_prompt` erases the
concept toward nothing rather than toward a replacement.

### Key arguments

| argument | meaning |
|---|---|
| `--imap_alpha` | how far the representation is pushed toward the safe prompt; values above 1 extrapolate past it |
| `--lamb_v` | weight of the preservation term |
| `--v_steps` | how many early denoising steps the preservation term covers |
| `--double_blocks` | transformer blocks where the alignment loss is applied |
| `--imap_mode` | `object` for things and people, `video` for motions |
| `--clamp_mask` | clamp the localization weight to `[0, 1]`; off by default |

### Fitting on the GPU

At 720x1280 the run above needs close to 80 GB and will not fit without the
last three memory flags. `--no-v_disable_gc` matters most: the default is the
opposite, and it turns gradient checkpointing **off** for the student pass
whenever the preservation loss is active, which is enough to exhaust a 95 GB
card. `--split_hv_backward` splits that pass in two and lowers the peak
further, at the cost of one extra forward. `PYTORCH_CUDA_ALLOC_CONF` is not
optional either: without it the allocator fragments and a run that fits reports
an out-of-memory error with gigabytes reserved and unused. Dropping to
`--height 432 --width 768` is the other way out.

## Reference photographs

The identity metric compares generated faces against photographs you supply,
one directory per person, three to five clear frontal shots each:

```
data/identity_refs/donald_trump/*.jpg
data/identity_refs/barack_obama/*.jpg
...
```

`data/` is ignored by git and the photographs never leave the machine.
Published photographs of public figures are somebody's copyright, which is the
other reason they are not in this repository.

Check them before scoring anything:

```bash
python3 -m evaluation.arcface experiments/identities.yaml --check-refs
```

It reports how well each person's photographs agree with one another. Two shots
of the same face land around 0.6 to 0.9 apart in cosine; a pair far below that
is usually a different person, a hard profile, or the wrong face picked out of a
group shot. Catching it here costs a minute. Catching it afterwards costs a
rerun, and only if somebody happens to notice that one row came out uniformly
low.
