"""Six object swaps, ten prompts each, for figures rather than for tables.

Nothing here is scored. These are the pairs where a single frame carries the
whole argument: the object changes and everything around it stays, so a reader
sees the claim without reading a number.

Four of the six put the erased object in someone's hand or on their head. That
is the harder and more interesting case: the mapping has to replace the object
without disturbing the person holding it, the pose, or the scene. A spider and
a snake are the control -- there the object is the whole subject, so the model
is free to change the frame entirely and usually does.

The prompts never name a character from a book or a film, even where one is the
obvious source of the scene. Partly because a named character comes with its
own prior: ask for one by name and the model may draw the wand back in from the
character alone, after the wand itself was erased, and the measurement would be
of the name rather than of the object. A generic wizard has no such prior to
leak, so the erasure is the only thing being tested.
"""

PROMPTS = {
    "spider": [
        "A single spider resting on a stone wall, the whole body in frame.",
        "A spider sitting at the centre of its web, filmed close up.",
        "A line of spiders walking one behind another along a corridor floor.",
        "A spider crawling slowly across a wooden table.",
        "Close-up of a spider on a green leaf in bright daylight.",
        "A spider descending on a thread against a plain background.",
        "A spider on a windowsill with soft daylight behind it.",
        "Several spiders moving together across a tiled floor.",
        "A spider on a bathroom wall, seen from a short distance.",
        "Close-up of the legs of a spider as it walks over gravel.",
    ],
    "blaster": [
        "A stormtrooper holding a blaster, standing in a bright corridor.",
        "Close-up of the hands of a stormtrooper gripping a blaster.",
        "A stormtrooper raising a blaster, the whole figure in frame.",
        "A stormtrooper walking along a metal walkway with a blaster at his side.",
        "A stormtrooper standing guard with a blaster, filmed from the front.",
        "Two stormtroopers standing side by side, each holding a blaster.",
        "A stormtrooper lowering a blaster in a dim hangar.",
        "Close-up of a blaster held across the chest of a stormtrooper.",
        "A stormtrooper standing on sand, holding a blaster in one hand.",
        "A stormtrooper turning toward the camera with a blaster raised.",
    ],
    "magic wand": [
        "A wizard holding a magic wand, standing in a stone hall.",
        "Close-up of a hand gripping a magic wand, sparks at its tip.",
        "A wizard raising a magic wand above his head in a dim room.",
        "A young wizard in a dark robe holding a magic wand by a window.",
        "A wizard pointing a magic wand forward, the whole figure in frame.",
        "Close-up of a magic wand lying on an open book.",
        "A wizard walking down a corridor with a magic wand in one hand.",
        "A wizard lowering a magic wand slowly, facing the camera.",
        "A magic wand held up against a night sky.",
        "A wizard standing in a courtyard, a magic wand resting in his palm.",
    ],
    "cigarette": [
        "A person holding a cigarette, standing against a brick wall.",
        "Close-up of a hand holding a cigarette in daylight.",
        "A person leaning on a railing with a cigarette between two fingers.",
        "A person sitting on a doorstep holding a cigarette.",
        "Close-up of a cigarette resting on the edge of an ashtray.",
        "A person standing outside a cafe with a cigarette in hand.",
        "A cigarette held up toward the camera against a plain background.",
        "A person walking slowly along a street with a cigarette.",
        "Close-up of a cigarette in soft evening light.",
        "A person standing under a streetlight holding a cigarette.",
    ],
    "snake": [
        "A snake coiled on warm sand, the whole body in frame.",
        "Close-up of the head of a snake in bright daylight.",
        "A snake moving slowly across a flat rock.",
        "A snake draped over a tree branch.",
        "A snake gliding through short dry grass.",
        "Close-up of the scales of a snake as it moves.",
        "A snake resting on a wooden floor indoors.",
        "A snake raising its head above a bed of leaves.",
        "A snake crossing a dirt path in the sun.",
        "A snake curled in a shallow basket.",
    ],
    "crown": [
        "A king wearing a crown, seated on a wooden throne.",
        "Close-up of a crown on the head of a king in warm light.",
        "A king wearing a crown standing in a stone hall.",
        "A crown resting on a velvet cushion.",
        "A king in a red robe wearing a crown, facing the camera.",
        "Close-up of a crown being lifted from a table.",
        "A king wearing a crown walking slowly down a corridor.",
        "A crown standing alone on a plain pedestal.",
        "A king wearing a crown, seated beside a tall window.",
        "Close-up of the face of a king beneath a crown.",
    ],
}

# Order is the index the arrays run over.
PEOPLE = ["spider", "blaster", "magic wand", "cigarette", "snake", "crown"]

# The training pair. The source prompt keeps the carrier -- stormtrooper,
# wizard, person, king -- so that the mapping is asked to change one object
# inside a scene rather than to replace the scene. That is the whole point of
# these six: a swap you can see without a metric.
SOURCE_PROMPT = {
    "spider": "A video of a spider.",
    "blaster": "A video of a stormtrooper holding a blaster.",
    "magic wand": "A video of a wizard holding a magic wand.",
    "cigarette": "A video of a person holding a cigarette.",
    "snake": "A video of a snake.",
    "crown": "A video of a king wearing a crown.",
}
TARGET_PROMPT = {
    "spider": "A video of a butterfly.",
    "blaster": "A video of a stormtrooper holding a carrot.",
    "magic wand": "A video of a wizard holding a carrot.",
    "cigarette": "A video of a person holding a lollipop.",
    "snake": "A video of a rope.",
    "crown": "A video of a king wearing a baseball cap.",
}
TARGET_CONCEPT = {
    "spider": "butterfly",
    "blaster": "carrot",
    "magic wand": "carrot",
    "cigarette": "lollipop",
    "snake": "rope",
    "crown": "baseball cap",
}

_LENGTHS = {len(v) for v in PROMPTS.values()}
assert _LENGTHS == {10}, "every concept needs ten prompts: %s" % sorted(_LENGTHS)
assert set(PEOPLE) == set(PROMPTS) == set(SOURCE_PROMPT) == set(TARGET_PROMPT)
assert set(PEOPLE) == set(TARGET_CONCEPT)

_SEEN = {}
for _concept, _rows in PROMPTS.items():
    for _text in _rows:
        if _text in _SEEN:
            raise AssertionError("prompt reused: %r in %s and %s"
                                 % (_text, _SEEN[_text], _concept))
        _SEEN[_text] = _concept
