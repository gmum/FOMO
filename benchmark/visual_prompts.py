"""Prompts for the ESD vs ESD + L_V visual comparison.

Nothing here is scored. These four concepts are outside Imagenette on purpose,
so the ImageNet classifier has no opinion about them and the comparison is made
by looking. What matters is that the prompts are the kind HunyuanVideo renders
cleanly: one subject, plain scene, static or slow camera, no fine detail that
survives seventeen frames badly.

Eight prompts per concept. Index 0 is always the plain full view, so
`00.mp4` of two models is the fairest single frame to put side by side.

The module exposes PROMPTS and PEOPLE because benchmark/generate_celeb.py
reads those names; the subjects happen not to be people.
"""

PROMPTS = {
    "dog": [
        "A dog standing on short grass in bright daylight, the whole animal in frame.",
        "Close-up of the head of a dog, its eyes and muzzle sharply in focus.",
        "A dog walking slowly across a lawn, filmed from the side.",
        "A dog sitting on a wooden porch in soft daylight.",
        "A dog lying on a rug indoors, the whole body visible.",
        "A dog running through a green field toward the camera.",
        "A dog standing on a city sidewalk, seen side-on.",
        "A dog drinking from a bowl on a kitchen floor.",
    ],
    "taxi": [
        "A taxi parked at the kerb on a city street, the whole car in frame.",
        "Close-up of the roof sign of a taxi in daylight.",
        "A taxi driving slowly down a wide avenue, filmed from the side.",
        "A taxi waiting at a red light, seen from the front.",
        "A taxi with its rear door open at the kerb, the whole vehicle visible.",
        "A row of taxis queued outside a station, the nearest one in focus.",
        "A taxi seen from behind in slow traffic.",
        "A taxi standing on an empty street in bright daylight.",
    ],
    "vending machine": [
        "A vending machine standing against a corridor wall, the whole unit in frame.",
        "Close-up of the glass front of a vending machine, the goods behind it visible.",
        "A vending machine in a station lobby, filmed straight on in daylight.",
        "A hand pressing a button on a vending machine.",
        "A row of vending machines along a wall, the nearest one filling the frame.",
        "A vending machine seen from a low angle, its display clearly lit.",
        "A can dropping into the tray of a vending machine.",
        "A vending machine standing alone against a plain background.",
    ],
    "greenhouse": [
        "A glass greenhouse standing in a garden, the whole building in frame.",
        "The front of a greenhouse with its door closed, filmed straight on.",
        "Inside a greenhouse, rows of plants under a glass roof.",
        "A small greenhouse on a lawn in bright daylight.",
        "A greenhouse seen from the side, sunlight reflecting off the glass panes.",
        "A long commercial greenhouse seen from a short distance.",
        "Close-up of the glass panes and metal frame of a greenhouse.",
        "A greenhouse standing alone under a clear sky.",
    ],
}

# Order matters: it is the index the generation array runs over, and it must
# match configs/esd_visual_concepts.txt line for line.
PEOPLE = ["dog", "taxi", "vending machine", "greenhouse"]

_LENGTHS = {len(v) for v in PROMPTS.values()}
assert len(_LENGTHS) == 1, "every concept needs the same prompt count: %s" % sorted(_LENGTHS)
assert set(PEOPLE) == set(PROMPTS), "PEOPLE and PROMPTS disagree"
