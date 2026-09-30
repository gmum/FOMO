## TODO

- [x] Model checkpoint
- [x] Inference code
- [x] Hunyuan LoRA training code

## Installation

1. **Create a Conda environment with Python 3.10:**

```bash
conda create -n eraser python=3.10
conda activate eraser
```

2. **Install the required dependencies:**

Install specific versions of `torch`, `transformers`, and `accelerate` to ensure reproducibility. While other versions may work (as long as they are compatible with `diffusers`), we recommend using the versions below:

```bash
pip install torch==2.6.0 transformers==4.48.0 accelerate==1.1.0
```

3. **Install `diffusers` from source:**

We use `diffusers==0.33.0.dev0`. Please install it with the provided source file.

```bash
cd diffusers
pip install -e .
cd examples/cogvideo
pip install -r requirements.txt
pip install opencv-python omegaconf imageio imageio-ffmpeg
```

## Hunyuan LoRA training workflow

Use the `dev` branch for the training code:

```bash
git clone -b dev https://github.com/AgnieszkaPolowczyk/Video_Unlearning.git
cd Video_Unlearning
```

Create the environment as described above. On a SLURM cluster, first adjust `BASE_DIR`, `REPO_DIR`, and `ENV_DIR` at the top of the `slurm/*.sbatch` files to match your filesystem, then run:

```bash
sbatch slurm/create_t2v_env.sbatch
```

### 1. Train an erasing LoRA

The default config trains a small HunyuanVideo erasing LoRA for the concept `cat`:

```bash
python train_hunyuan_erase_lora.py \
  --config configs/hunyuan_erase_lora_cat.yaml \
  --concept cat \
  --model_path hunyuanvideo-community/HunyuanVideo \
  --dataset_dir data/cat \
  --output_dir outputs/hunyuan_lora_cat
```

On SLURM:

```bash
sbatch slurm/train_hunyuan_erase_lora_h100.sbatch
```

Useful SLURM overrides:

```bash
sbatch --export=ALL,CONCEPT=dog,MAX_TRAIN_STEPS=500,RANK=32,LEARNING_RATE=2e-5 slurm/train_hunyuan_erase_lora_h100.sbatch
```

The training script can bootstrap a small dataset if `bootstrap_dataset_if_missing: true` is set in the config. It writes generated training videos and `metadata.jsonl` to `DATASET_DIR`. The trained LoRA is saved to `OUTPUT_DIR` as `pytorch_lora_weights.safetensors` plus `training_config.json`. If `checkpointing_steps` is set, intermediate checkpoints are saved under `OUTPUT_DIR/checkpoint-*`.

Main training hyperparameters:

- `concept`: target concept to erase, for example `cat`, `dog`, `car`.
- `model_path`: HunyuanVideo model path or Hugging Face id.
- `dataset_dir`: folder with training videos and `metadata.jsonl`; can be bootstrapped.
- `output_dir`: folder where LoRA weights are saved.
- `rank`, `lora_alpha`, `target_modules`: LoRA capacity and target attention projections.
- `learning_rate`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay`: optimizer settings.
- `max_train_steps`, `checkpointing_steps`, `train_batch_size`, `gradient_checkpointing`.
- `train_width`, `train_height`, `train_num_frames`: training size; height and width must be divisible by 16.
- `bootstrap_width`, `bootstrap_height`, `bootstrap_num_frames`, `num_bootstrap_videos`: generated bootstrap dataset size.
- `lamb_esd`, `lamb_attn`, `esd_full_video_weight`, `esd_first_frame_weight`, `negative_guidance`, `teacher_mode`, `attn_blocks`.
- `neutral_prompt_override`, `use_empty_neutral_prompt`, `prompt_only_esd`.

Safe sizes to start with are `512x288` for training smoke tests, `768x432` for medium inference, and `1280x720` for larger Hunyuan inference when memory allows.

### 2. Run inference with the trained LoRA

Use `--lora_path` for LoRAs trained by this repo. Use `--lora_weight` to control the LoRA strength.

```bash
python test_hunyuan.py \
  --prompt "a dog sitting on the bed" \
  --model_path hunyuanvideo-community/HunyuanVideo \
  --lora_path outputs/hunyuan_lora_cat \
  --lora_weight 1.0 \
  --height 432 \
  --width 768 \
  --num_frames 75 \
  --fps 15 \
  --num_inference_steps 30 \
  --guidance_scale 6.0 \
  --output_path videos/hunyuan_lora_test \
  --seed 42 \
  --generate_clean
```

Outputs:

- `videos/hunyuan_lora_test_clean.mp4`: base model comparison, only when `--generate_clean` is used.
- `videos/hunyuan_lora_test_erase.mp4`: model with the trained LoRA.

Inference parameters:

- `height`, `width`: output resolution; use multiples of 16, for example `512x288`, `768x432`, `1280x720`.
- `num_frames`, `fps`: video length. `75` frames at `15` fps is about 5 seconds.
- `num_inference_steps`: denoising steps, usually `30`.
- `guidance_scale`: prompt guidance, usually around `6.0`.
- `lora_weight`: LoRA strength; try `0.5`, `0.75`, `1.0`, `1.25`.
- `seed`: reproducible generation seed.

### 3. Run LoRA + value injection comparison

`test_hunyuan_vbank.py` compares three outputs: base model, LoRA only, and LoRA with base-model value injection. It automatically uses the latest `checkpoint-*` inside `--lora_path` if the top-level folder has no LoRA weights.

```bash
python test_hunyuan_vbank.py \
  --prompt "a dog sitting on the bed" \
  --model_path hunyuanvideo-community/HunyuanVideo \
  --lora_path outputs/hunyuan_lora_cat \
  --lora_weight 1.0 \
  --height 432 \
  --width 768 \
  --num_frames 75 \
  --fps 15 \
  --num_inference_steps 30 \
  --guidance_scale 6.0 \
  --inject_steps 5 \
  --block_min 20 \
  --output_path videos/vbank_dog_bed \
  --seed 42
```

Outputs:

- `videos/vbank_dog_bed_base.mp4`
- `videos/vbank_dog_bed_lora.mp4`
- `videos/vbank_dog_bed_lora_vinject.mp4`
- `videos/vbank_dog_bed_vbank_manifest.json`

For a sweep over value-injection lengths, add:

```bash
--sweep_inject_steps
```

This saves one injected video for each prefix length from `1` to `--inject_steps`, for example `_lora_vinject_01steps.mp4`, `_lora_vinject_02steps.mp4`, etc.

## Inference

First, download the weights of the nudity erasure adapters via [Google Drive](https://drive.google.com/drive/folders/11r1dS2vzmbFeJZeDVZGsb2z9Tkrx64I1?usp=sharing).

We provide the inference scripts for T2VUnlearning. For CogVideoX-2B and 5B, please use the `test_cogvideo.sh`. 

```
CUDA_VISIBLE_DEVICES=0 python test_cogvideo.py \
--prompt=[Test prompt] \
--model_path=[Path of pretrained CogVideoX diffusers weight] \
--eraser_path=[Path of nudity erasure adapter] \
--eraser_rank=128 \
--num_frames=[Number of frames to generate. Default 49] \
--generate_clean \
--output_path=[Prefix for output videos] \
--seed=42
```

After running the script, you should find two output videos: `[output_path]_clean.mp4` and `[output_path]_erased.mp4`, corresponding to the results from the original model and the unlearned model, respectively.

For HunyuanVideo, please use the `test_hunyuanvideo.sh`

```
CUDA_VISIBLE_DEVICES=0 python test_hunyuan.py \
--prompt=[Test prompt] \
--model_path=[Path of pretrained HunyuanVideo diffusers weight] \
--eraser_path=[Path of nudity erasure adapter] \
--eraser_rank=128 \
--num_frames=[Number of frames to generate. Default 49] \
--generate_clean \
--output_path=[Prefix for output videos] \
--seed=42
```

We also include inference script of SAFREE (`test_safree_hunyuan.sh`)and negative prompting (`test_neg_hunyuan.sh`) for HunyuanVideo.

Evaluation prompt datasets can be found in `evaluation/data`.

## Exact LoRA used by A&A

The current LoRA used for the dog value-injection comparison is:

```text
/shared/results/gmpolowl/repozytorium/T2VUnlearning/outputs/dog_alone_esd_emptyneutral_ng1_lr3e3_rank8_1000steps_ckpt50_768x432_gc/checkpoint-000050
```

This checkpoint is not stored in GitHub; it must already exist on the cluster at this path, or `LORA_PATH` must be changed to the copied checkpoint location.

The folder name describes the training setup:

- erased concept / prompt family: `dog_alone`
- loss setup: `esd`
- neutral prompt: empty neutral prompt
- negative guidance: `ng1`
- learning rate: `3e-3`
- LoRA rank: `8`
- training budget: `1000` steps
- checkpoint interval: `50` steps
- checkpoint used for this run: `checkpoint-000050`
- resolution: `768x432`
- gradient checkpointing: enabled (`gc`)

To reproduce the value-injection sweep with this LoRA:

```bash
sbatch --export=ALL,MODEL_PATH=hunyuanvideo-community/HunyuanVideo,LORA_PATH=/shared/results/gmpolowl/repozytorium/T2VUnlearning/outputs/dog_alone_esd_emptyneutral_ng1_lr3e3_rank8_1000steps_ckpt50_768x432_gc/checkpoint-000050,LORA_WEIGHT=1,PROMPT="a dog sitting on the bed",WIDTH=768,HEIGHT=432,NUM_FRAMES=75,FPS=15,NUM_INFERENCE_STEPS=30,GUIDANCE_SCALE=6.0,SEED=42,INJECT_STEPS=30,SWEEP_INJECT_STEPS=1,BLOCK_MIN=20,OUTPUT_PATH=/shared/results/gmpolowl/repozytorium/T2VUnlearning/videos/vbank_dog_sitting_bed_ckpt50_768x432_seed42_sweep30 slurm/run_hunyuan_vbank_h100.sbatch
```

This run writes the base video, LoRA-only video, value-injected videos for injection steps `1..30`, and a `_vbank_manifest.json` file under:

```text
/shared/results/gmpolowl/repozytorium/T2VUnlearning/videos/
```

## Acknowledgements

This repository is built upon the excellent work of the following projects:

- [Receler](https://github.com/jasper0314-huang/Receler)
- [finetrainers](https://github.com/a-r-r-o-w/finetrainers)
- [diffusers](https://github.com/huggingface/diffusers)

We sincerely thank the authors and contributors of these projects for their valuable tools, insights, and open-source efforts. 
