"""Read one experiment description and make it usable from Python and from bash.

An experiment is a YAML file in this directory. It is the whole definition of
one block of results in the paper: which prompts, which mapping, how long to
train, how to sample. Nothing that changes a number is passed on a command
line, so a run can be repeated from the file alone, and two runs that differ
have a diff to show for it.

The division of labour between the two kinds of file here is deliberate:

    mappings/*.tsv    the data -- many uniform rows, one per thing to unlearn.
                      A SLURM array index is a line number in this file.
    *.yaml            the settings for one experiment, which select rows from
                      a table by name and say what to do with them.

Usage from Python:

    cfg = load("experiments/objects_mapping2.yaml")
    for row in rows(cfg):
        ...

Usage from bash, which is how the sbatch scripts read it:

    eval "$(python3 -m experiments.config experiments/objects_mapping2.yaml --shell)"
"""

import argparse
import csv
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Settings every experiment has, with the values used in the paper. An
# experiment file overrides what it needs and stays silent about the rest, so
# the diff between two experiments is only what actually differs.
DEFAULTS = {
    "train": {
        "learning_rate": 1e-3,
        "imap_alpha": 7,
        "mask_mode": "unclamped",
        "max_train_steps": 250,
        "checkpointing_steps": 10,
        "rank": 8,
        "lora_alpha": 8,
        "target_modules": "attn.add_q_proj,attn.add_k_proj",
        "lamb_v": 0.5,
        "v_steps": 1,
        "v_block_min": 20,
        "v_early_prob": 0.5,
        "v_mask_mode": "background",
        "double_blocks": "1,2,4,5,6,9,10,13,15,16,17,18,19",
        "trajectory_steps": 30,
        "checkpoint": None,          # which checkpoint the tables are built on
    },
    "sampling": {
        "width": 1280,
        "height": 720,
        "num_frames": 17,
        "fps": 8,
        "num_inference_steps": 30,
        "guidance_scale": 6.0,
        "seed": 42,
        "num_prompts": 20,
        "model_path": "hunyuanvideo-community/HunyuanVideo",
    },
}

KINDS = ("objects", "identities")

# What produces the videos of a column.
#
#   fomo        a LoRA trained by this repository, one per concept
#   negprompt   the untouched model with the concept as a negative prompt
#   safree      the untouched model under SAFREE's pipeline
#
# The last two train nothing, so an experiment using them has no adapters and
# its `train` section is ignored. They still need a mapping table, because that
# is what enumerates the columns.
METHODS = ("fomo", "negprompt", "safree")


class Row(object):
    """One line of a mapping table: one concept to unlearn, one way to do it."""

    __slots__ = ("concept", "mapping", "target_concept",
                 "source_prompt", "target_prompt", "index")

    def __init__(self, index, concept, mapping, target_concept,
                 source_prompt, target_prompt):
        self.index = index
        self.concept = concept
        self.mapping = mapping
        # NONE is how the table spells an empty field. A genuinely empty column
        # would be swallowed, because a tab is IFS whitespace in bash and
        # consecutive tabs collapse into one.
        self.target_concept = "" if target_concept == "NONE" else target_concept
        self.source_prompt = source_prompt
        self.target_prompt = "" if target_prompt == "NONE" else target_prompt

    def __repr__(self):
        return "Row(%r, %r)" % (self.concept, self.mapping)


def _merge(defaults, given, where):
    out = dict(defaults)
    for key, value in (given or {}).items():
        if key not in defaults:
            raise SystemExit("%s: unknown setting %r. Known: %s"
                             % (where, key, ", ".join(sorted(defaults))))
        out[key] = value
    return out


def load(path):
    """Parse and validate one experiment file."""
    try:
        import yaml
    except ModuleNotFoundError:
        # This module is read on the submitting host as well as inside the
        # jobs, and on a cluster whose login node has a different architecture
        # from its compute nodes the conda environment cannot run there at all.
        # Saying so beats a traceback from `bash slurm/run.sh`.
        raise SystemExit(
            "PyYAML is missing from %s.\n"
            "It is needed wherever an experiment file is read, which includes "
            "the host you submit from:\n"
            "    pip install --user pyyaml" % sys.executable)

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    for key in ("name", "kind", "prompts", "mappings", "mapping"):
        if key not in raw:
            raise SystemExit("%s: missing required key %r" % (path, key))
    if raw["kind"] not in KINDS:
        raise SystemExit("%s: kind must be one of %s, got %r"
                         % (path, ", ".join(KINDS), raw["kind"]))
    method = raw.get("method", "fomo")
    if method not in METHODS:
        raise SystemExit("%s: method must be one of %s, got %r"
                         % (path, ", ".join(METHODS), method))

    cfg = {
        "path": path,
        "name": raw["name"],
        "kind": raw["kind"],
        "method": method,
        "prompts": raw["prompts"],
        "mappings": raw["mappings"],
        "mapping": raw["mapping"],
        "label": raw.get("label", raw["name"]),
        "train": _merge(DEFAULTS["train"], raw.get("train"), path),
        "sampling": _merge(DEFAULTS["sampling"], raw.get("sampling"), path),
    }
    if not rows(cfg):
        raise SystemExit("%s: no rows with mapping %r in %s"
                         % (path, cfg["mapping"], cfg["mappings"]))
    return cfg


def rows(cfg):
    """The lines of the mapping table this experiment trains on, in file order.

    The index is the line number in the table, counted from zero and over the
    whole file rather than over the filtered subset: it is what a SLURM array
    index means, and renumbering it per experiment would make the same index
    point at different concepts in different runs.
    """
    path = cfg["mappings"]
    if not os.path.isabs(path):
        path = os.path.join(REPO, path)
    out = []
    with open(path, newline="", encoding="utf-8") as handle:
        for index, fields in enumerate(csv.reader(handle, delimiter="\t")):
            if not fields or not fields[0].strip():
                continue
            if len(fields) != 5:
                raise SystemExit("%s line %d: %d columns, expected 5"
                                 % (path, index + 1, len(fields)))
            if fields[1] == cfg["mapping"]:
                out.append(Row(index, *fields))
    return out


def prompt_module(cfg):
    """The module holding this experiment's evaluation prompts."""
    return importlib.import_module("experiments.prompts.%s" % cfg["prompts"])


def subjects(cfg):
    """Everything generated inside every column.

    For objects these are the ten Imagenette classes, for identities the people.
    A column generates all of them, which is what makes the off-diagonal
    measurable: PSR for objects, Preserve for identities.
    """
    module = prompt_module(cfg)
    return list(module.CLASSES if cfg["kind"] == "objects" else module.PEOPLE)


def prompts_for(cfg, subject):
    """The prompts of one subject, truncated to `sampling.num_prompts`."""
    module = prompt_module(cfg)
    if subject not in module.PROMPTS:
        raise SystemExit("%s has no prompts for %r. Known: %s"
                         % (cfg["prompts"], subject, ", ".join(sorted(module.PROMPTS))))
    count = cfg["sampling"]["num_prompts"]
    return list(module.PROMPTS[subject])[:count or None]


def _shell(cfg):
    """Print the configuration as shell assignments.

    Flat, prefixed, and quoted, so that bash can read it with `eval` without
    knowing anything about YAML. Prompts and mappings stay on the Python side:
    a list of thirty sentences does not survive a trip through an environment
    variable intact, and the sbatch scripts never need them.
    """
    flat = [("EXP_NAME", cfg["name"]), ("EXP_KIND", cfg["kind"]),
            ("EXP_METHOD", cfg["method"]),
            ("EXP_LABEL", cfg["label"]), ("EXP_PROMPTS", cfg["prompts"]),
            ("EXP_MAPPINGS", cfg["mappings"]), ("EXP_MAPPING", cfg["mapping"]),
            ("EXP_ROWS", " ".join(str(r.index) for r in rows(cfg))),
            ("EXP_SUBJECTS", str(len(subjects(cfg))))]
    for section, prefix in (("train", "TRAIN_"), ("sampling", "GEN_")):
        for key, value in sorted(cfg[section].items()):
            flat.append((prefix + key.upper(), "" if value is None else value))
    for key, value in flat:
        print("%s=%s" % (key, _quote(str(value))))


def _quote(text):
    return "'" + text.replace("'", "'\\''") + "'"


def main():
    ap = argparse.ArgumentParser(description="Inspect one experiment file.")
    ap.add_argument("config")
    ap.add_argument("--shell", action="store_true",
                    help="Print shell assignments for `eval`.")
    ap.add_argument("--row", type=int, default=None,
                    help="Print one row's five fields, one per line, for the "
                         "Nth row of this experiment counted from zero. This "
                         "is what a SLURM array task reads to find its work.")
    ap.add_argument("--subject-at", type=int, default=None,
                    help="Print the Nth subject, counted from zero.")
    args = ap.parse_args()

    cfg = load(args.config)
    if args.shell:
        _shell(cfg)
        return
    if args.row is not None:
        selected = rows(cfg)
        if not 0 <= args.row < len(selected):
            raise SystemExit("row %d is out of range 0..%d for %s"
                             % (args.row, len(selected) - 1, cfg["name"]))
        row = selected[args.row]
        for field in (row.concept, row.target_concept,
                      row.source_prompt, row.target_prompt,
                      prompts_for(cfg, row.concept)[0]):
            print(field)
        return
    if args.subject_at is not None:
        everyone = subjects(cfg)
        if not 0 <= args.subject_at < len(everyone):
            raise SystemExit("subject %d is out of range 0..%d for %s"
                             % (args.subject_at, len(everyone) - 1, cfg["prompts"]))
        print(everyone[args.subject_at])
        return

    print("name     : %s" % cfg["name"])
    print("kind     : %s" % cfg["kind"])
    print("method   : %s" % cfg["method"])
    print("prompts  : %s (%d subjects, %d prompts each)"
          % (cfg["prompts"], len(subjects(cfg)),
             len(prompts_for(cfg, subjects(cfg)[0]))))
    print("mapping  : %s from %s" % (cfg["mapping"], cfg["mappings"]))
    print("adapters : %s" % cfg["train"]["checkpoint"])
    print()
    print("%5s  %-20s %-24s %s" % ("index", "concept", "target", "target prompt"))
    print("-" * 92)
    for row in rows(cfg):
        print("%5d  %-20s %-24s %s"
              % (row.index, row.concept, row.target_concept or "(none)",
                 row.target_prompt or "(empty)"))


if __name__ == "__main__":
    main()
