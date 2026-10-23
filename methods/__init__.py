"""The methods FOMO is compared against in the paper.

    negprompt.py    the untouched model with the concept as a negative prompt
    safree/         training-free erasure through the text embedding, on its
                    own fork of the HunyuanVideo pipeline

Neither trains anything, so neither produces adapters. Both are driven from
`experiments/generate.py` through the `method:` key of an experiment file.
"""
