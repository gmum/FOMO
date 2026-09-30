"""The nine and the eleven as one list, for scoring across both sets.

Generation is organised as two square matrices, nine by nine and eleven by
eleven, because that is the cheap shape. Preservation, though, is more
convincing the wider it is measured: collateral damage to eight neighbours
from the same walk of life says less than damage to nineteen drawn from
politics, sport, film, music and technology.

This module exists so the scorer can walk a tree containing both sets. It adds
no prompts of its own -- the two source modules already share the thirty
close-up templates, so a merged list is consistent by construction.
`find_cells` skips directories that are not there, so pointing this at a tree
that holds only part of the matrix costs nothing and scores what exists.
"""

import identity_prompts_face as _nine
import identity11_prompts as _eleven

PEOPLE = list(_nine.PEOPLE) + list(_eleven.PEOPLE)
ALL_PEOPLE = list(PEOPLE)

PROMPTS = dict(_nine.PROMPTS)
PROMPTS.update(_eleven.PROMPTS)

GENDER = dict(_nine.GENDER)
GENDER.update(_eleven.GENDER)

DEMOGRAPHIC = dict(_nine.DEMOGRAPHIC)
DEMOGRAPHIC.update(_eleven.DEMOGRAPHIC)

assert len(PEOPLE) == len(set(PEOPLE)) == 20, "the two sets must not overlap"
assert set(PEOPLE) == set(PROMPTS) == set(GENDER) == set(DEMOGRAPHIC)
assert {len(v) for v in PROMPTS.values()} == {30}
