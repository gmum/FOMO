# FOMO

Official implementation of **FOMO: Forget the Concept, Don't Miss Out on the Scene in Selective Video Unlearning**.

FOMO removes a named concept from a text-to-video diffusion model while leaving
the rest of the scene close to what the original model would have produced. It
handles concepts that appear as objects or people, and also concepts that exist
only across frames, such as a motion.

## Installation

```bash
conda create -n fomo python=3.10 -y
conda activate fomo

# torch is installed separately because the right index depends on the
# platform. On aarch64 (for example GH200) the cu124 index stops at 2.5.1,
# so 2.6.0 has to come from cu126.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

The implementation targets
[HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo);
the weights are downloaded on first use.

## Unlearning a concept

A run is defined by the concept to erase, the concept that replaces it, and a
pair of prompts differing only in those two words.

```bash
echo '{"source_prompt": "a video of a dog", "target_prompt": "a video of a cat"}' > pairs.jsonl

python train_hunyuan_imap_h_lora.py \
    --output_dir outputs/dog \
    --source_concept "dog" \
    --target_concept "cat" \
    --prompt_pairs_jsonl pairs.jsonl \
    --max_train_steps 250 \
    --learning_rate 1e-3 \
    --imap_alpha 7 \
    --lamb_v 0.5
```

The file may hold more than one pair, one JSON object per line, in which case
each training step samples one of them. An empty `target_prompt` erases the
concept toward nothing rather than toward a replacement.

Training writes a checkpoint every `--checkpointing_steps` steps into
`outputs/<name>/unclamped_alpha<α>/checkpoint-XXXXXX/`.

### Key arguments

| argument | meaning |
|---|---|
| `--imap_alpha` | how far the representation is pushed toward the safe prompt; values above 1 extrapolate past it |
| `--lamb_v` | weight of the preservation term |
| `--v_steps` | how many early denoising steps the preservation term covers |
| `--double_blocks` | transformer blocks where the alignment loss is applied |
| `--imap_mode` | `object` for things and people, `video` for motions |
| `--clamp_mask` | clamp the localization weight to `[0, 1]`; off by default |

## Generating video

```bash
# original model
python generate.py --prompt "a video of a dog" --output base.mp4

# after unlearning
python generate.py --prompt "a video of a dog" --output erased.mp4 \
    --lora outputs/dog/unclamped_alpha7
```

With the same `--seed` the two runs differ only by the adapter, so they can be
placed side by side.

## What is in this repository

```
train_hunyuan_imap_h_lora.py              training
diagnose_hunyuan_imap_hidden_patch.py     concept localization
generate.py                               sampling, with or without an adapter
receler/loras/hunyuan_peft_lora.py        LoRA placement for HunyuanVideo
receler/training/                         flow timesteps, token lookup, attention capture
```

Training was run on HunyuanVideo at 720x1280, 17 frames, with LoRA rank 8 on
the text-stream query and key projections, α = 7 and λ_V = 0.5.

## Acknowledgements

The LoRA placement and training utilities under `receler/` are adapted from
[Receler](https://github.com/jasper0314-huang/Receler), used under its original
licence. The idea of ranking attention heads by their frame-wise separability
follows IMAP; the implementation here is our own.

## Citation

```bibtex
@inproceedings{fomo,
  title     = {FOMO: Forget the Concept, Don't Miss Out on the Scene in Selective Video Unlearning},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2027}
}
```
