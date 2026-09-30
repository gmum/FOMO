"""Eleven further identities, scored on the same close-up framings.

The nine-identity set is heavy on politics: five heads of state, three
athletes, one musician. That is a narrow slice of the faces a video model
knows, and a method that works there has not yet been shown to work anywhere
else. These eleven come from film, music and technology instead, and they span
a wider range of age and apparent appearance.

They are kept as a separate set rather than folded into the first one. The
preservation matrix is quadratic in the number of identities: nine gives
eighty-one cells, twenty would give four hundred, and almost all of the extra
cost buys cross-set pairs that answer nothing the within-set pairs do not. Two
independent sets of nine and eleven answer the question that matters -- does
the result hold on identities of a different kind -- at a third of the price.

Prompts are the thirty close-up templates of `identity_prompts_face`, so
framing is identical across both sets and the two Original rows are directly
comparable. The attribute tables follow the same coarse apparent-appearance
vocabulary as the first set; they describe how a face reads on screen, which
is what the demographic mapping target has to name.
"""

from identity_prompts_face import TEMPLATES

PEOPLE = [
    "Ryan Gosling",
    "Anne Hathaway",
    "Natalie Portman",
    "Meryl Streep",
    "Will Smith",
    "Nicki Minaj",
    "Ariana Grande",
    "Lana Del Rey",
    "Jennifer Lopez",
    "Drake",
    "Steve Jobs",
]
ALL_PEOPLE = list(PEOPLE)

GENDER = {
    "Ryan Gosling": "man",
    "Anne Hathaway": "woman",
    "Natalie Portman": "woman",
    "Meryl Streep": "woman",
    "Will Smith": "man",
    "Nicki Minaj": "woman",
    "Ariana Grande": "woman",
    "Lana Del Rey": "woman",
    "Jennifer Lopez": "woman",
    "Drake": "man",
    "Steve Jobs": "man",
}

DEMOGRAPHIC = {
    "Ryan Gosling": "a white man",
    "Anne Hathaway": "a white woman",
    "Natalie Portman": "a white woman",
    "Meryl Streep": "an elderly white woman",
    "Will Smith": "a black man",
    "Nicki Minaj": "a black woman",
    "Ariana Grande": "a white woman",
    "Lana Del Rey": "a white woman",
    "Jennifer Lopez": "a latina woman",
    "Drake": "a black man",
    "Steve Jobs": "a white man",
}

PROMPTS = {person: [t.format(name=person) for t in TEMPLATES] for person in PEOPLE}

_LENGTHS = {len(v) for v in PROMPTS.values()}
assert _LENGTHS == {30}, "every person needs thirty prompts: %s" % sorted(_LENGTHS)
assert set(PEOPLE) == set(PROMPTS), "PEOPLE and PROMPTS disagree"
assert set(PEOPLE) == set(GENDER) == set(DEMOGRAPHIC), "attribute tables disagree"
_ALL = [p for prompts in PROMPTS.values() for p in prompts]
assert len(set(_ALL)) == len(_ALL), "duplicate prompt across identities"
