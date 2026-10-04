import argparse
import json
from pathlib import Path

from qwen_definitions import MOTION_DEFINITIONS, SAFETY_DEFINITIONS
from qwen_prompts import MOTION_PROMPT, SAFETY_PROMPT
from qwen_utils import (
    MODEL_NAME, evaluate_frames, load_inputs, load_model,
    load_results, read_frames, write_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate videos with Qwen.")
    concept = parser.add_mutually_exclusive_group(required=True)
    concept.add_argument("--category", choices=sorted(SAFETY_DEFINITIONS), help="Unsafe-content category.")
    concept.add_argument("--action", choices=sorted(MOTION_DEFINITIONS), help="Source motion to detect.")
    parser.add_argument("--input-dir", required=True, type=Path, help="MP4 directory.")
    parser.add_argument("--metadata", required=True, type=Path, help="JSONL with index and prompt.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Results directory.")
    parser.add_argument("--model", default=MODEL_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.category:
        task, concept = "safety", args.category
        definition, template = SAFETY_DEFINITIONS[concept], SAFETY_PROMPT
    else:
        task, concept = "motion", args.action
        definition, template = MOTION_DEFINITIONS[concept], MOTION_PROMPT
    config = {"task": task, "concept": concept, "definition": definition, "model": args.model}
    inputs = load_inputs(args.input_dir, args.metadata, config)
    output = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    records = load_results(output, inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_config = {
        **config, "input_dir": str(args.input_dir.resolve()),
        "metadata": str(args.metadata.resolve()),
    }
    summary = write_summary(summary_path, records, len(inputs), summary_config)
    remaining = [name for name in inputs if name not in records]
    if not remaining:
        print(json.dumps(summary, indent=2))
        return

    processor, model = load_model(args.model)
    import torch

    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for name in remaining:
            info = inputs[name]
            prompt = template.format(**info)
            frames = read_frames(args.input_dir / name)
            label, yes_score, no_score, normalized = evaluate_frames(processor, model, frames, prompt)
            record = {
                **info,
                "answer": label, "yes_probability_percent": yes_score,
                "no_probability_percent": no_score, "raw_normalized_answer": normalized,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records[name] = record
            summary = write_summary(summary_path, records, len(inputs), summary_config)
            print(f"[{len(records)}/{len(inputs)}] {name}: {normalized}", flush=True)
            del frames
            torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
