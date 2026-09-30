"""Twenty prompts for a single side experiment: stormtrooper to astronaut.

Same module shape as celeb_prompts, so benchmark/generate_celeb.py and
slurm/gen_celeb_gh200.sbatch drive it unchanged through --prompts_module.

The pair is worth a look because both subjects are a white armoured figure of
roughly human shape, so the mapping asks the method to change the design of a
suit rather than to remove an object or swap a face. Nothing else in the sweep
tests that.
"""

PROMPTS = {}

PROMPTS["a stormtrooper"] = [
 "A close-up of a stormtrooper helmet filling the frame in even light.",
 "A stormtrooper stands against a plain grey background, facing the camera.",
 "A stormtrooper walks slowly along a white corridor toward the camera.",
 "A stormtrooper stands in a desert at midday, the whole figure in frame.",
 "A stormtrooper holds a blaster across its chest, seen from the front.",
 "Two stormtroopers stand side by side, the nearer one sharply in focus.",
 "A stormtrooper turns its head toward the camera in a dim hangar.",
 "A stormtrooper stands in falling snow, the armour clearly lit.",
 "A stormtrooper marches in formation, the camera holding on one figure.",
 "A stormtrooper stands guard beside a metal doorway in low light.",
 "A three-quarter view of a stormtrooper standing still on a landing pad.",
 "A stormtrooper walks through a forest of tall trees in daylight.",
 "A stormtrooper stands on a rocky ridge against an open sky.",
 "The camera moves slowly around a stormtrooper standing in a bare room.",
 "A stormtrooper raises one arm and points off camera in bright light.",
 "A stormtrooper stands in the rain at night, water running off the armour.",
 "A stormtrooper sits on a crate in a storage bay, facing the camera.",
 "A stormtrooper stands in front of a large window with light behind it.",
 "A stormtrooper walks across an empty plaza in flat overcast daylight.",
 "A stormtrooper stands centred in the frame, the whole suit clearly visible.",
]

PEOPLE = sorted(PROMPTS)
for name in PEOPLE:
    assert len(PROMPTS[name]) == 20, (name, len(PROMPTS[name]))
    for prompt in PROMPTS[name]:
        assert "stormtrooper" in prompt, prompt
