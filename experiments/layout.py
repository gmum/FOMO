"""Where everything lives on disk.

One module, imported by the generator, the scorers and the SLURM layer, so
that nothing has to agree with anything else by convention. Two sides that
disagree about a directory name do not fail; they quietly measure different
videos, and that is the kind of mistake that survives into a paper.

The tree, with the roots overridable but defaulted:

    adapters/<experiment>/<concept>/<mask>_alpha<alpha>/checkpoint-NNNNNN/

    videos/<prompts>/base/<subject>/NN.mp4
    videos/<prompts>/unlearn_<erased>/<subject>/<tag>/NN.mp4

    results/<prompts>/base/{per_class,frames}.csv  summary.json
    results/<prompts>/unlearn_<erased>/<tag>/{per_class,frames}.csv  summary.json
    results/<prompts>/<tag>/id_similarity{,_per_video}.csv

Everything for one family of experiments sits under one directory named after
the prompt set -- `imagenette`, `identities` -- so the untouched model and
every variant are listed side by side. `unlearn_<concept>` says what the column
erased, and `<tag>` at the bottom says which run produced those videos: the
mapping it trained towards, or the name of a method that trains nothing.

The untouched model has no tag of its own because it has no variants: it is one
set of videos per prompt set, shared by everything compared against it. The
object experiments differ only in the mapping they train towards and all of
them are read against the same two hundred untouched videos.

`NN` is the prompt index. Its seed is `seed + NN`, independent of the column
and the tag, so the same prompt gets the same noise everywhere.
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
    """Where training writes the LoRA for one concept of one experiment.

    Keyed by the experiment rather than by the prompt set and tag: an adapter
    belongs to the run that trained it, and there is nothing here to list side
    by side.
    """
    return os.path.join(root, experiment, slugify(concept),
                        "%s_alpha%s" % (mask_mode, alpha))


# --- videos ---------------------------------------------------------------

def base_column(prompts, root="videos"):
    """The untouched model's videos for one prompt set."""
    return os.path.join(root, prompts, "base")


def unlearn_column(prompts, erased, root="videos"):
    """One column: the model that erased `erased`, asked for every subject."""
    return os.path.join(root, prompts, "unlearn_%s" % slugify(erased))


def videos_in(column, subject, tag=None):
    """The directory holding one subject's NN.mp4 inside a column.

    `tag` is omitted for the untouched model and given for a column, which is
    the one asymmetry in this layout: the baseline has no variants to keep
    apart, so adding a level there would name something that does not vary.
    """
    path = os.path.join(column, slugify(subject))
    return path if tag is None else os.path.join(path, tag)


# --- results --------------------------------------------------------------

def base_results(prompts, root="results"):
    return os.path.join(root, prompts, "base")


def column_results(prompts, erased, tag, root="results"):
    return os.path.join(root, prompts, "unlearn_%s" % slugify(erased), tag)


def run_results(prompts, tag, root="results"):
    """Results of one run that are not per column -- the identity CSVs."""
    return os.path.join(root, prompts, tag)
