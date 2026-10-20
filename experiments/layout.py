"""Where everything lives on disk.

One module, imported by the generators, the scorers and the SLURM layer, so
that nothing has to agree with anything else by convention. Two sides that
disagree about a directory name do not fail; they quietly measure different
videos, and that is the kind of mistake that survives into a paper.

The tree, with `videos_root` and friends overridable but defaulted:

    adapters/<experiment>/<concept>/<mask>_alpha<a>/checkpoint-NNNNNN/
    videos/<prompts>/base/<subject>/NN.mp4
    videos/<experiment>/<erased>/<subject>/NN.mp4
    results/<prompts>/base/per_class.csv
    results/<experiment>/<erased>/per_class.csv
    results/<experiment>/id_similarity.csv
    tables/<experiment>.tex

A column is named after the concept it erases, and `base` is the column that
erases nothing. Both hold the same thing underneath -- one directory per
subject -- so a scorer pointed at either does not need to know which it got.

The baseline is keyed by the prompt set, not by the experiment, because that is
what it actually depends on: the three object experiments differ only in the
mapping they train towards, and all three are compared against the same two
hundred untouched videos. Keying it by experiment would regenerate them three
times and invite the question of why the three copies do not match.
"""

import os


def slugify(text):
    """Directory-safe form of a name: "gas pump" -> "gas_pump"."""
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower()


def adapter_dir(experiment, concept, mask_mode="unclamped", alpha=7,
                root="adapters"):
    """Where training writes the LoRA for one concept of one experiment."""
    return os.path.join(root, experiment, slugify(concept),
                        "%s_alpha%s" % (mask_mode, alpha))


def base_videos(prompts, root="videos"):
    """The untouched model's videos, shared by every experiment on this set."""
    return os.path.join(root, prompts, "base")


def column_videos(experiment, erased, root="videos"):
    """One column: the model that erased `erased`, asked for every subject."""
    return os.path.join(root, experiment, slugify(erased))


def subject_videos(column, subject):
    """The twenty or thirty videos of one subject inside a column."""
    return os.path.join(column, slugify(subject))


def base_results(prompts, root="results"):
    return os.path.join(root, prompts, "base")


def column_results(experiment, erased, root="results"):
    return os.path.join(root, experiment, slugify(erased))


def experiment_results(experiment, root="results"):
    return os.path.join(root, experiment)


def table_path(experiment, root="tables"):
    return os.path.join(root, "%s.tex" % experiment)
