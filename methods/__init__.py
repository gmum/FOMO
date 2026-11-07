"""The methods FOMO is compared against.

    negprompt.py          the untouched model with the concept as a negative
                          prompt; the CFG handling differs per backbone
    safree/concepts.py    SAFREE's per-concept term lists, plain text and the
                          same for every backbone
    safree/pipeline_<model>.py
                          SAFREE's own fork of that backbone's pipeline, taken
                          from their release and exporting the class as
                          `Pipeline`. Present for CogVideoX; for a backbone
                          they did not release one for, drop the file in under
                          this name.

Neither method trains anything, so neither produces adapters. Both are driven
from experiments/generate.py through the `method:` key of an experiment file.
"""
