"""Unlearn one concept, with short flags and the settings from the paper.

`fomo.train` takes forty arguments and needs a pair file written beforehand.
This is the same trainer with a small front: name the concept, name what should
replace it, say how long, and go.

    # redirect one concept onto another
    python3 -m fomo.unlearn --concept dog --target cat --out adapters/unlearn_dog

    # erase it without putting anything in its place
    python3 -m fomo.unlearn --concept dog --out adapters/unlearn_dog

    # the identity setting from the paper: six steps is enough for a face
    python3 -m fomo.unlearn --concept "Donald Trump" --target "an elderly white man" \
        --steps 6 --checkpoint-every 1 --out adapters/unlearn_trump

Prompts are built as "A video of <concept>." unless you pass your own with
--source-prompt and --target-prompt. Several prompt pairs are worth having when
one phrasing is not enough, and for that pass a file with --pairs: one JSON
object per line, and each training step samples one of them.

Everything not exposed here keeps the default from `fomo.train`, including the
three flags that decide whether the run fits in memory -- they are set for you,
because at 720x1280 the run needs close to 80 GB and will not fit without them.
`--extra` passes anything else straight through.
"""

import argparse
import json
import os
import shlex


def build_pairs(path, concept, target, source_prompt, target_prompt):
    """Write the one-line pair file the trainer reads.

    The concepts travel in the pair file, because the trainer looks up
    source_concept and target_concept per pair. An empty target concept
    switches off localisation on the target side, which is what allows an
    empty target prompt -- that combination is how a concept is erased toward
    nothing rather than toward a replacement.
    """
    source_prompt = source_prompt or "A video of %s." % concept
    if target:
        target_prompt = target_prompt or "A video of %s." % target
    else:
        target_prompt = target_prompt or ""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "source_concept": concept,
            "target_concept": target or "",
        }) + "\n")
    return source_prompt, target_prompt


def main():
    ap = argparse.ArgumentParser(
        description="Unlearn one concept from HunyuanVideo.",
        epilog="Anything not listed here can be passed through with "
               "--extra '--flag value'; see `python3 -m fomo.train --help`.")
    ap.add_argument("--concept", required=True,
                    help='What to unlearn, e.g. "church" or "Donald Trump".')
    ap.add_argument("--target", default=None,
                    help="What should appear instead. Leave it out to erase "
                         "toward nothing.")
    ap.add_argument("--out", required=True, help="Where the adapter is written.")
    ap.add_argument("--steps", type=int, default=250,
                    help="Training steps. 250 for objects, 6 for a face: an "
                         "identity collapses long before a scene does.")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=7.0,
                    help="How far the representation is pushed toward the safe "
                         "prompt. Above 1 it extrapolates past it.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--preserve-weight", type=float, default=0.5,
                    help="Weight of the scene-preservation term (lambda_V).")
    ap.add_argument("--source-prompt", default=None,
                    help='Default: "A video of <concept>."')
    ap.add_argument("--target-prompt", default=None,
                    help='Default: "A video of <target>.", or empty with no target.')
    ap.add_argument("--pairs", default=None,
                    help="Use this pair file instead of building one. One JSON "
                         "object per line; each step samples one.")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--motion", action="store_true",
                    help="Unlearn a temporal behaviour rather than something "
                         "with a fixed place in the frame.")
    ap.add_argument("--extra", default="",
                    help="Extra arguments for fomo.train, as one quoted string.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the trainer command and the pair, train nothing.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pairs = args.pairs or os.path.join(args.out, "pairs.jsonl")
    if args.pairs:
        source_prompt, target_prompt = "(from %s)" % args.pairs, ""
    else:
        source_prompt, target_prompt = build_pairs(
            pairs, args.concept, args.target,
            args.source_prompt, args.target_prompt)

    argv = [
        "--output_dir", args.out,
        "--source_concept", args.concept,
        "--target_concept", args.target or "",
        "--prompt_pairs_jsonl", pairs,
        "--max_train_steps", str(args.steps),
        "--checkpointing_steps", str(args.checkpoint_every),
        "--learning_rate", str(args.lr),
        "--imap_alpha", str(args.alpha),
        "--lamb_v", str(args.preserve_weight),
        "--imap_mode", "video" if args.motion else "object",
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_frames", str(args.num_frames),
        "--seed", str(args.seed),
        # The three flags that decide whether this fits on the card, plus
        # skipping the in-training previews. --v_disable_gc defaults to the
        # opposite and turns gradient checkpointing off for the student pass,
        # which is enough to exhaust a 95 GB card at 720x1280.
        "--v_video_only", "--no-v_disable_gc", "--split_hv_backward",
        "--gradient_checkpointing", "--skip_eval",
    ] + shlex.split(args.extra)

    print("concept : %s" % args.concept)
    print("target  : %s" % (args.target or "(nothing)"))
    print("source  : %s" % source_prompt)
    print("replace : %s" % (target_prompt or "(empty prompt)"))
    print("steps   : %d, checkpoint every %d" % (args.steps, args.checkpoint_every))
    print("alpha   : %s   lr %s   preserve %s"
          % (args.alpha, args.lr, args.preserve_weight))
    print("output  : %s" % args.out)
    print()

    if args.dry_run:
        print("python3 -m fomo.train " + " ".join(shlex.quote(a) for a in argv))
        return

    # Imported here, not at module scope: the trainer pulls in torch, and
    # --help and --dry-run have to work on a login node without it.
    #
    # Parsed through the trainer's own parser, so every default this front does
    # not mention stays exactly what fomo.train says it is.
    from fomo.train import build_parser, main as train

    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
