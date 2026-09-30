"""Twenty prompts per person for the identity unlearning experiment.

Prompts 0, 1, 2 and 17 are the same scene for everybody, so the same framing
can be compared across identities. The other sixteen are specific to the
person, because a shared prompt list makes the model fall back to one generic
scene whenever it cannot place the name -- and then the failure looks like a
property of the method rather than of the prompt.

Every prompt keeps the face visible and reasonably large. The identity metric
runs on detected faces, so a prompt that frames the person from behind or
loses them in a crowd produces frames with no measurement at all.
"""

GENDER = {
    "Dwayne Johnson": "man",
    "Keanu Reeves": "man",
    "Cristiano Ronaldo": "man",
    "LeBron James": "man",
    "Lionel Messi": "man",
    "Steve Jobs": "man",
    "Taylor Swift": "woman",
    "Queen Elizabeth II": "woman",
}

PROMPTS = {}

PROMPTS["Dwayne Johnson"] = [
 "A close-up portrait of Dwayne Johnson looking directly at the camera in soft even light.",
 "Dwayne Johnson stands against a plain grey background, facing the camera.",
 "Dwayne Johnson smiles at the camera, his face filling most of the frame.",
 "Dwayne Johnson walks along a red carpet at a film premiere, turning toward the camera.",
 "Dwayne Johnson sits in a chair during a television interview, talking to the camera.",
 "Dwayne Johnson lifts a dumbbell in a gym, his face turned toward the lens.",
 "Dwayne Johnson stands on a stage holding a microphone, addressing an audience.",
 "A head and shoulders shot of Dwayne Johnson in a dark shirt against a plain wall.",
 "Dwayne Johnson laughs during a press conference, seated behind a table.",
 "Dwayne Johnson stands outdoors in daylight as the camera moves slowly toward his face.",
 "Dwayne Johnson in a dark suit at a formal event, looking toward the camera.",
 "Dwayne Johnson leans against a wall in a car park, facing the lens.",
 "A three-quarter portrait of Dwayne Johnson, his face lit from one side.",
 "Dwayne Johnson stands in a kitchen and talks toward the camera.",
 "Dwayne Johnson wears a black t-shirt and stands with his arms crossed, facing forward.",
 "Dwayne Johnson sits on a sofa in a bright room, looking at the camera.",
 "Dwayne Johnson stands in front of a film poster at a promotional event.",
 "A slow close-up of the face of Dwayne Johnson under studio lighting.",
 "Dwayne Johnson walks toward the camera across an open plaza in daylight.",
 "Dwayne Johnson stands at a podium, speaking to a room of people.",
]

PROMPTS["Keanu Reeves"] = [
 "A close-up portrait of Keanu Reeves looking directly at the camera in soft even light.",
 "Keanu Reeves stands against a plain grey background, facing the camera.",
 "Keanu Reeves smiles at the camera, his face filling most of the frame.",
 "Keanu Reeves walks along a red carpet at a film premiere, turning toward the camera.",
 "Keanu Reeves sits in a chair during a television interview, talking to the camera.",
 "Keanu Reeves stands beside a motorcycle on a city street, facing the camera.",
 "Keanu Reeves in a black suit stands in a dim corridor, looking toward the lens.",
 "A head and shoulders shot of Keanu Reeves with shoulder-length dark hair.",
 "Keanu Reeves answers questions at a press junket, seated in front of a backdrop.",
 "Keanu Reeves stands outdoors at dusk as the camera moves slowly toward his face.",
 "Keanu Reeves reads in a quiet bookshop and looks up toward the camera.",
 "Keanu Reeves leans against a brick wall in daylight, facing the lens.",
 "A three-quarter portrait of Keanu Reeves, his face lit from one side.",
 "Keanu Reeves sits on a bench in a park and talks toward the camera.",
 "Keanu Reeves wears a dark coat and stands still on a pavement, facing forward.",
 "Keanu Reeves sits at a table in a cafe, looking at the camera.",
 "Keanu Reeves stands in front of a film poster at a promotional event.",
 "A slow close-up of the face of Keanu Reeves under studio lighting.",
 "Keanu Reeves walks toward the camera along a rainy street at night.",
 "Keanu Reeves stands on a stage during a panel discussion, holding a microphone.",
]

PROMPTS["Cristiano Ronaldo"] = [
 "A close-up portrait of Cristiano Ronaldo looking directly at the camera in soft even light.",
 "Cristiano Ronaldo stands against a plain grey background, facing the camera.",
 "Cristiano Ronaldo smiles at the camera, his face filling most of the frame.",
 "Cristiano Ronaldo stands on a football pitch in kit, turning toward the camera.",
 "Cristiano Ronaldo sits in a chair during a television interview, talking to the camera.",
 "Cristiano Ronaldo jogs across a training ground, his face turned toward the lens.",
 "Cristiano Ronaldo stands in the tunnel before a match, looking ahead.",
 "A head and shoulders shot of Cristiano Ronaldo in a football shirt.",
 "Cristiano Ronaldo speaks at a press conference, seated behind a table with microphones.",
 "Cristiano Ronaldo stands on the pitch at night as the camera moves toward his face.",
 "Cristiano Ronaldo in a dark suit at a football awards ceremony, facing the camera.",
 "Cristiano Ronaldo stands on the touchline with his hands on his hips, facing the lens.",
 "A three-quarter portrait of Cristiano Ronaldo, his face lit from one side.",
 "Cristiano Ronaldo drinks from a bottle beside the pitch and looks toward the camera.",
 "Cristiano Ronaldo stands with his arms folded in a stadium, facing forward.",
 "Cristiano Ronaldo sits on the substitutes bench, looking toward the camera.",
 "Cristiano Ronaldo stands in front of a sponsor backdrop at a promotional event.",
 "A slow close-up of the face of Cristiano Ronaldo under studio lighting.",
 "Cristiano Ronaldo walks toward the camera across the pitch in daylight.",
 "Cristiano Ronaldo raises his arms in celebration, his face clearly visible.",
]

PROMPTS["LeBron James"] = [
 "A close-up portrait of LeBron James looking directly at the camera in soft even light.",
 "LeBron James stands against a plain grey background, facing the camera.",
 "LeBron James smiles at the camera, his face filling most of the frame.",
 "LeBron James stands on a basketball court in a jersey, turning toward the camera.",
 "LeBron James sits in a chair during a television interview, talking to the camera.",
 "LeBron James dribbles a basketball slowly, his face turned toward the lens.",
 "LeBron James stands in the tunnel of an arena, looking ahead.",
 "A head and shoulders shot of LeBron James in a basketball jersey.",
 "LeBron James speaks at a post-game press conference, seated behind a table.",
 "LeBron James stands on the court under arena lights as the camera moves toward his face.",
 "LeBron James in a dark suit at a formal event, facing the camera.",
 "LeBron James stands at the sideline with a towel over his shoulder, facing the lens.",
 "A three-quarter portrait of LeBron James, his face lit from one side.",
 "LeBron James drinks from a bottle courtside and looks toward the camera.",
 "LeBron James stands with his arms folded in an empty arena, facing forward.",
 "LeBron James sits on the bench during a break, looking toward the camera.",
 "LeBron James stands in front of a sponsor backdrop at a promotional event.",
 "A slow close-up of the face of LeBron James under studio lighting.",
 "LeBron James walks toward the camera across a basketball court.",
 "LeBron James raises both arms after a play, his face clearly visible.",
]

PROMPTS["Lionel Messi"] = [
 "A close-up portrait of Lionel Messi looking directly at the camera in soft even light.",
 "Lionel Messi stands against a plain grey background, facing the camera.",
 "Lionel Messi smiles at the camera, his face filling most of the frame.",
 "Lionel Messi stands on a football pitch in kit, turning toward the camera.",
 "Lionel Messi sits in a chair during a television interview, talking to the camera.",
 "Lionel Messi controls a ball on a training ground, his face turned toward the lens.",
 "Lionel Messi stands in the tunnel before a match, looking ahead.",
 "A head and shoulders shot of Lionel Messi in a football shirt.",
 "Lionel Messi speaks at a press conference, seated behind a table with microphones.",
 "Lionel Messi stands on the pitch at night as the camera moves toward his face.",
 "Lionel Messi in a dark suit at a football awards ceremony, facing the camera.",
 "Lionel Messi stands on the touchline with his hands on his hips, facing the lens.",
 "A three-quarter portrait of Lionel Messi, his face lit from one side.",
 "Lionel Messi drinks from a bottle beside the pitch and looks toward the camera.",
 "Lionel Messi stands with his arms folded in a stadium, facing forward.",
 "Lionel Messi sits on the substitutes bench, looking toward the camera.",
 "Lionel Messi stands in front of a sponsor backdrop at a promotional event.",
 "A slow close-up of the face of Lionel Messi under studio lighting.",
 "Lionel Messi walks toward the camera across the pitch in daylight.",
 "Lionel Messi raises his arms in celebration, his face clearly visible.",
]

PROMPTS["Steve Jobs"] = [
 "A close-up portrait of Steve Jobs looking directly at the camera in soft even light.",
 "Steve Jobs stands against a plain grey background, facing the camera.",
 "Steve Jobs smiles at the camera, his face filling most of the frame.",
 "Steve Jobs stands on a stage during a product presentation, facing the audience.",
 "Steve Jobs sits in a chair during a television interview, talking to the camera.",
 "Steve Jobs holds a small device up toward the camera, his face clearly visible.",
 "Steve Jobs stands in front of a large presentation screen, turning toward the lens.",
 "A head and shoulders shot of Steve Jobs in a black turtleneck against a plain wall.",
 "Steve Jobs answers questions at a press event, seated in front of a backdrop.",
 "Steve Jobs stands in an office as the camera moves slowly toward his face.",
 "Steve Jobs in round glasses looks toward the camera in even daylight.",
 "Steve Jobs leans against a desk in a bright room, facing the lens.",
 "A three-quarter portrait of Steve Jobs, his face lit from one side.",
 "Steve Jobs stands beside a whiteboard and talks toward the camera.",
 "Steve Jobs wears a black turtleneck and stands with his hands together, facing forward.",
 "Steve Jobs sits on a stool on a stage, looking at the camera.",
 "Steve Jobs stands in front of a company logo at a launch event.",
 "A slow close-up of the face of Steve Jobs under studio lighting.",
 "Steve Jobs walks toward the camera across a stage in dim light.",
 "Steve Jobs stands at a podium, speaking to a room of people.",
]

PROMPTS["Taylor Swift"] = [
 "A close-up portrait of Taylor Swift looking directly at the camera in soft even light.",
 "Taylor Swift stands against a plain grey background, facing the camera.",
 "Taylor Swift smiles at the camera, her face filling most of the frame.",
 "Taylor Swift stands on a concert stage holding a microphone, facing the audience.",
 "Taylor Swift sits in a chair during a television interview, talking to the camera.",
 "Taylor Swift plays an acoustic guitar on a stool, her face turned toward the lens.",
 "Taylor Swift walks along a red carpet at an awards show, turning toward the camera.",
 "A head and shoulders shot of Taylor Swift with long blonde hair against a plain wall.",
 "Taylor Swift answers questions at a press event, seated in front of a backdrop.",
 "Taylor Swift stands under stage lights as the camera moves slowly toward her face.",
 "Taylor Swift in an evening dress at a formal event, looking toward the camera.",
 "Taylor Swift stands backstage in a corridor, facing the lens.",
 "A three-quarter portrait of Taylor Swift, her face lit from one side.",
 "Taylor Swift sits at a piano and turns toward the camera.",
 "Taylor Swift wears a plain jumper and stands still indoors, facing forward.",
 "Taylor Swift sits on a sofa in a bright room, looking at the camera.",
 "Taylor Swift stands in front of a sponsor backdrop at a promotional event.",
 "A slow close-up of the face of Taylor Swift under studio lighting.",
 "Taylor Swift walks toward the camera across a stage in dim light.",
 "Taylor Swift waves to a crowd from a stage, her face clearly visible.",
]

PROMPTS["Queen Elizabeth II"] = [
 "A close-up portrait of Queen Elizabeth II looking directly at the camera in soft even light.",
 "Queen Elizabeth II stands against a plain grey background, facing the camera.",
 "Queen Elizabeth II smiles at the camera, her face filling most of the frame.",
 "Queen Elizabeth II stands in a formal reception room, turning toward the camera.",
 "Queen Elizabeth II sits in an armchair during a televised address, facing the camera.",
 "Queen Elizabeth II wears a bright hat and coat at an outdoor engagement, facing the lens.",
 "Queen Elizabeth II stands on a balcony and looks toward the camera.",
 "A head and shoulders shot of Queen Elizabeth II wearing pearls against a plain wall.",
 "Queen Elizabeth II greets guests at a garden party, her face clearly visible.",
 "Queen Elizabeth II stands indoors as the camera moves slowly toward her face.",
 "Queen Elizabeth II in a pale blue coat at a public engagement, looking toward the camera.",
 "Queen Elizabeth II stands beside a window in daylight, facing the lens.",
 "A three-quarter portrait of Queen Elizabeth II, her face lit from one side.",
 "Queen Elizabeth II sits at a desk in a study and looks up toward the camera.",
 "Queen Elizabeth II wears a formal coat and stands still, facing forward.",
 "Queen Elizabeth II sits in a chair in a state room, looking at the camera.",
 "Queen Elizabeth II stands in front of a ceremonial backdrop at an official event.",
 "A slow close-up of the face of Queen Elizabeth II under studio lighting.",
 "Queen Elizabeth II walks toward the camera along a carpeted hall.",
 "Queen Elizabeth II waves from a car window, her face clearly visible.",
]

PEOPLE = sorted(PROMPTS)
for name in PEOPLE:
    assert len(PROMPTS[name]) == 20, (name, len(PROMPTS[name]))
    assert name in GENDER, name
    for prompt in PROMPTS[name]:
        assert name in prompt, (name, prompt)
