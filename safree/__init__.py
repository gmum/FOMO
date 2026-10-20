"""SAFREE: training-free erasure through the text embedding.

Somebody else's method, vendored because it ships its own fork of the
HunyuanVideo pipeline and cannot be driven through the stock one. It is one of
the methods FOMO is compared against in the paper.

NegPrompt, the other method compared against, needs no code of its own -- it is
a flag on the generator.
"""
