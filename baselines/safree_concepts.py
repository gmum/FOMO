"""Concept term lists for SAFREE on the Imagenette classes.

SAFREE does not take a concept name. It builds a subspace from a list of
related terms and projects the prompt embedding away from it, so a class with
no list cannot be erased at all -- the shipped CONCEPT_DICT covers five safety
concepts and nothing else.

Each list follows the shape of the original ones: the class name, its
synonyms, and the parts, attributes and typical scenes that identify it.
Fourteen terms per class, uniform across all ten, which sits inside the range
of the originals (six to seventeen). Uniform length matters -- a short list
would handicap the baseline on that class and the comparison would measure the
list rather than the method.
"""

CONCEPTS = {
    "cassette player": [
        "Cassette Player", "Cassette Deck", "Tape Deck", "Compact Cassette",
        "Audio Cassette Tape", "Cassette Slot Door", "Cassette Eject Button",
        "Cassette Reel Hubs Behind a Window", "Auto-Reverse Cassette Mechanism",
        "Cassette Tape Head and Capstan", "Portable Cassette Walkman",
        "Radio Cassette Recorder", "Dolby Noise Reduction Cassette Deck",
        "Chrome Type II Cassette",
    ],
    "chain saw": [
        "Chainsaw", "Chain Saw", "Saw Chain", "Chainsaw Guide Bar",
        "Chainsaw Chain Teeth", "Chainsaw Chain Brake Lever",
        "Two-Stroke Chainsaw Engine", "Chainsaw Pull-Cord Starter",
        "Chainsaw Rear Handle and Throttle Trigger", "Chainsaw Bar Oil Cap",
        "Petrol Chainsaw", "Electric Chainsaw", "Chainsaw Felling Cut",
        "Chainsaw Sprocket Cover",
    ],
    "church": [
        "Church", "Church Building", "Christian Church Exterior",
        "Church Steeple", "Church Spire", "Church Bell Tower",
        "Church Facade with a Cross on the Roof", "Parish Church",
        "Village Church", "White Clapboard Church",
        "Gothic Church Facade with Rose Window",
        "Church Nave Interior with Pews", "Churchyard in Front of a Church",
        "Church Portal Entrance",
    ],
    "gas pump": [
        "Gas Pump", "Petrol Pump", "Fuel Dispenser",
        "Fuel Pump Nozzle in Its Holster", "Gas Pump Hose",
        "Gas Pump Price Display", "Gas Pump Grade Selector Buttons",
        "Gas Pump Payment Keypad", "Fuel Pump Meter Dial",
        "Vintage Gas Pump with Glass Globe", "Self-Service Fuel Dispenser",
        "Diesel Dispenser Unit", "Nozzle Inserted in a Fuel Filler",
        "Pump Island at a Filling Station",
    ],
    "tench": [
        "Tench", "Tinca tinca", "European Tench", "Golden Tench",
        "Doctor Fish Tench", "Olive-Green Slimy Tench Skin",
        "Tench Barbels at the Mouth Corners", "Small Red Eye of a Tench",
        "Rounded Fins of a Tench", "Tench Held Up by an Angler",
        "Tench Coarse Fishing Catch", "Tench in a Landing Net",
        "Tench Lying on a Fishing Mat", "Deep-Bodied Tench Profile",
    ],
    "garbage truck": [
        "Garbage Truck", "Refuse Collection Truck", "Dustcart", "Bin Lorry",
        "Rear Loader Refuse Truck", "Front Loader Garbage Truck",
        "Side Loader Garbage Truck", "Garbage Truck Hopper",
        "Refuse Truck Packer Blade", "Bin Lifter Arm on a Refuse Truck",
        "Wheelie Bin Being Tipped into a Truck", "Municipal Refuse Vehicle",
        "Sanitation Department Truck", "Kerbside Recycling Collection Truck",
    ],
    "English springer": [
        "English Springer Spaniel", "Springer Spaniel",
        "Liver and White Springer Spaniel", "Black and White English Springer",
        "English Springer Spaniel Puppy",
        "Long Feathered Ears of a Springer Spaniel",
        "Docked Tail of an English Springer",
        "Field-Bred English Springer Spaniel",
        "Show-Bred English Springer Spaniel",
        "Freckled Muzzle of a Springer Spaniel",
        "Springer Spaniel Flushing a Game Bird",
        "Springer Spaniel Feathered Legs", "English Springer Spaniel Head Study",
        "Springer Spaniel Retrieving in Water",
    ],
    "golf ball": [
        "Golf Ball", "Dimpled Golf Ball", "White Golf Ball on a Tee",
        "Golf Ball on a Putting Green", "Golf Ball Beside the Hole Cup",
        "Golf Ball Dimple Pattern Close-Up",
        "Golf Ball with a Printed Brand Logo", "Golf Ball in the Rough Grass",
        "Golf Ball in a Sand Bunker", "Golf Ball and Putter Face",
        "Bucket of Driving Range Balls", "Sleeve of Golf Balls",
        "Golf Ball Rolling Toward the Cup", "Golf Ball Marker on the Green",
    ],
    "parachute": [
        "Parachute", "Parachute Canopy", "Skydiver Under an Open Parachute",
        "Ram-Air Parachute Wing", "Round Military Parachute Canopy",
        "Parachute Suspension Lines", "Parachute Harness and Container Rig",
        "Deployed Parachute in the Sky", "Parachute Ripcord Deployment",
        "Reserve Parachute", "Paratrooper Descending by Parachute",
        "Colorful Parachute Canopy Cells", "Parachute Landing Flare",
        "Skydiving Canopy Descent",
    ],
    "French horn": [
        "French Horn", "Orchestral French Horn", "Coiled French Horn Tubing",
        "Flared Bell of a French Horn", "Rotary Valves of a French Horn",
        "French Horn Valve Levers", "Hand Placed Inside the Horn Bell",
        "Double Horn in F and B-flat", "Single F Horn",
        "French Horn Funnel Mouthpiece", "French Horn Leadpipe",
        "Hornist Playing a French Horn", "Circular Wrapped Brass Horn Body",
        "French Horn Held Bell-Down on the Lap",
    ],
}

_LENGTHS = {len(v) for v in CONCEPTS.values()}
assert len(_LENGTHS) == 1, "lists must be the same length: %s" % sorted(_LENGTHS)
