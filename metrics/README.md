# Metrics

```bash
pip install -r metrics/requirements.txt
```

## Qwen: nudity / general safety

```bash
# --category: nudity, violence, animal_abuse, terrorism, racism, gore
python metrics/evaluate_qwen.py \
  --category nudity \
  --input-dir videos/nudity \
  --metadata videos/nudity/metadata.jsonl \
  --output-dir outputs/metrics/qwen/nudity
```

## Qwen: motion

```bash
python metrics/evaluate_qwen.py \
  --action running \
  --input-dir videos/running \
  --metadata videos/running/metadata.jsonl \
  --output-dir outputs/metrics/qwen/running
```

## DINO / LPIPS

```bash
python metrics/evaluate_dino_lpips.py \
  --reference-dir videos/baseline \
  --generated-dir videos/unlearned \
  --output outputs/metrics/dino_lpips.json
```
