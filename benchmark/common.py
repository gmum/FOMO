"""Shared definitions for the object-erasure benchmark.

Everything that has to agree between generation, evaluation and table assembly
lives here, so the two sides cannot drift apart: class lists, prompt sets,
article handling, and the rule that maps a run configuration to a directory
name.
"""

CLASS_SETS = {
    # The ten Table 1 classes. All of them are ImageNet-1k classes, so an
    # ImageNet classifier can be used directly.
    "imagenette": [
        "tench",
        "English springer",
        "cassette player",
        "chain saw",
        "church",
        "French horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ],
    # CIFAR-10 labels are NOT ImageNet-1k classes: "deer" does not exist there
    # and "dog"/"bird"/"truck" each map to dozens of entries. These are scored
    # with CLIP zero-shot instead.
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
}

# ImageNet-1k indices, needed to score Imagenette with a torchvision classifier.
IMAGENET_INDEX = {
    "tench": 0,
    "English springer": 217,
    "cassette player": 482,
    "chain saw": 491,
    "church": 497,
    "French horn": 566,
    "garbage truck": 569,
    "gas pump": 571,
    "golf ball": 574,
    "parachute": 701,
}

# Left-to-right column order used when printing the LaTeX table. For Imagenette
# this deliberately differs from the natural order above.
LATEX_ORDER = {
    "imagenette": [
        "cassette player",
        "chain saw",
        "church",
        "gas pump",
        "tench",
        "garbage truck",
        "English springer",
        "golf ball",
        "parachute",
        "French horn",
    ],
    "cifar10": CLASS_SETS["cifar10"],
}

# ---------------------------------------------------------------------------
# Prompts
#
# Template syntax:
#   {c}   class name, verbatim — exactly what the table column is called
#   {a}   "a" or "an", agreeing with {c}
#
# Always write "{a} {c}", never "a {c}", otherwise vowel-initial classes come
# out as "a airplane". {a} is only correct immediately before {c}.
#
# Common rule for every set: no concept-specific attributes ("a garbage truck
# on the street", "a barking dog"). Such prompts narrow the output distribution
# for one class only and distort the comparison between columns. Every template
# has to work identically for all ten classes.
# ---------------------------------------------------------------------------
PROMPT_SETS = {
    # v1  original set. Some entries vary framing or lighting rather than
    #     wording (close-up, wide shot, panning, daylight), which changes how
    #     hard the object is to recognise and adds variance unrelated to the
    #     method being measured.
    "v1": [
        "a video of {a} {c}",
        "{a} {c}",
        "a photo of {a} {c}",
        "a realistic video of {a} {c}",
        "footage of {a} {c}",
        "a close-up video of {a} {c}",
        "a wide shot of {a} {c}",
        "a clear view of {a} {c}",
        "{a} {c} in daylight",
        "{a} {c} in natural light",
        "a static shot of {a} {c}",
        "a camera slowly panning around {a} {c}",
        "a short clip showing {a} {c}",
        "a documentary shot of {a} {c}",
        "{a} {c} filmed in high quality",
        "{a} {c} seen from the side",
        "{a} {c} in the center of the frame",
        "a video recording of {a} {c}",
        "a detailed video of {a} {c}",
        "a simple video of {a} {c}",
    ],
    # v2  wording variants only. Every prompt asks for the same thing and
    #     differs solely in how it is phrased.
    "v2": [
        "a video of {a} {c}",
        "a video showing {a} {c}",
        "a video of the {c}",
        "footage of {a} {c}",
        "footage showing {a} {c}",
        "a clip of {a} {c}",
        "a short clip of {a} {c}",
        "a recording of {a} {c}",
        "a video recording of {a} {c}",
        "a realistic video of {a} {c}",
        "a real video of {a} {c}",
        "a simple video of {a} {c}",
        "a plain video of {a} {c}",
        "a video featuring {a} {c}",
        "a video with {a} {c}",
        "this is a video of {a} {c}",
        "here is a video of {a} {c}",
        "{a} {c}",
        "the {c}",
        "video of {c}",
    ],
    # v3  one prompt, twenty seeds. Variation comes from sampling rather than
    #     from words: the seed is base_seed + index, so twenty identical
    #     templates still give twenty different videos. This is the cleanest
    #     measurement of "can the model produce this concept", with no
    #     confound from phrasing. It does not measure robustness to paraphrase.
    "v3": ["a video of {a} {c}"] * 20,
    # v4  five natural phrasings, four seeds each. A compromise: keeps some
    #     linguistic variety but drops constructions that do not occur in
    #     training captions and style qualifiers such as "plain" or "simple".
    "v4": (
        ["a video of {a} {c}"] * 4
        + ["{a} {c}"] * 4
        + ["a video showing {a} {c}"] * 4
        + ["footage of {a} {c}"] * 4
        + ["a clip of {a} {c}"] * 4
    ),
}

# ---------------------------------------------------------------------------
# Per-class prompt sets.
#
# Unlike the template sets above, these are concrete scenes written separately
# for each class, so the prompt distribution is not artificially correlated
# between concepts. Each set answers its own question, so they are not
# interchangeable: v5 varies prompt length to study specificity, v6 holds
# length fixed and maximises how recognisable the class is to the classifier.
#
# The class name appears VERBATIM in every prompt, so a drop in classifier
# accuracy after erasure is attributable to the model rather than to an
# ambiguous prompt. check_prompt_sets() enforces this.
#
# Only defined for Imagenette; asking for it with another class set is an
# error rather than a silent fallback.
# ---------------------------------------------------------------------------
PER_CLASS_PROMPT_SETS = {
    "v5": {
        'tench': [
            'A tench swimming slowly through murky green pond water.',
            'Close-up of a tench, its olive-green scales catching the light.',
            'A thick-bodied tench glides past a submerged fallen branch in a still pond, shafts of sunlight cutting through the murky green water and sliding across its olive scales.',
            'An angler kneels on a grassy riverbank holding a freshly caught tench up to the camera, water dripping from its broad tail as it twitches in his wet hands.',
            'Underwater tracking shot following a tench as it noses through soft silt at the bottom of a lake, small clouds of sediment drifting up behind it.',
            'A tench resting on wet grass beside a fishing net.',
            'Slow motion footage of a tench breaking the surface of a still pond at dawn, water peeling off its dark green back and rippling outward in wide rings.',
            "Macro video of a tench's small red eye and thick rounded fins, the camera drifting along its body from head to tail in dim greenish underwater light.",
            'A tench in a glass aquarium tank, seen from the side.',
            'Two tench circle each other among tall reeds in shallow water, their bodies dark against the pale sandy bottom, dappled afternoon light flickering over them.',
            'Documentary-style underwater footage of a tench feeding on the muddy bed of an overgrown canal, surrounded by strands of green weed swaying in the current.',
            "A tench held in a fisherman's hands above a landing net, weighed and measured on an overcast morning, drizzle speckling the water behind him.",
            'A tench swims into frame from the left and pauses, gills flaring.',
            'Overhead shot of a tench moving through clear shallow water over gravel, its shadow tracking across the stones beneath it in bright midday sun.',
            'A single tench hangs motionless in dark water among lily pads, only its pectoral fins moving, the surface above scattering broken sunlight.',
            'Static camera on a tench in a shallow keepnet, slowly turning.',
            'A tench is released back into a lake, twisting once and vanishing into the deep green water with a flick of its tail as ripples spread outward.',
            'Cold early-morning footage of a tench lying on a wooden dock beside an open tackle box, mist rising off the flat lake surface behind it.',
            'A tench glides beneath a wooden jetty, its olive body half in shadow and half lit by narrow bands of light falling between the planks.',
            'A tench nosing among reeds in shallow water, filmed from just above the surface.',
        ],
        'English springer': [
            'An English springer running across a green field.',
            'An English springer sitting in front of a rustic wooden shed, its long freckled ears hanging down, tail sweeping the dirt as afternoon sun warms the weathered planks behind it.',
            'Close-up of an English springer panting, tongue out.',
            'An English springer spaniel bounding through tall wet grass in a meadow at golden hour, ears flying up with every stride, seed heads scattering around it.',
            'Slow motion video of an English springer shaking water from its liver-and-white coat after leaping out of a lake, droplets spraying outward into low evening light.',
            'An English springer curled asleep on a worn armchair by a fireplace.',
            'Handheld footage of an English springer following a scent through autumn woodland, nose down among fallen leaves, its breath visible in the cold morning air.',
            'An English springer retrieves a tennis ball on a wide beach, sand flying up behind its paws as it turns hard and sprints back toward the camera.',
            'A camera slowly circling an English springer sitting on a gravel path.',
            'Studio portrait video of an English springer against a plain grey backdrop, head tilting as it listens, soft key light picking out the texture of its wavy ears.',
            'An English springer splashes through a shallow stream, water breaking around its chest, dappled light falling through the trees overhead.',
            'An English springer puppy stumbling across a kitchen floor.',
            'A hunter in a waxed jacket walks a field edge while an English springer works ahead of him, quartering back and forth through stubble under a flat grey sky.',
            'An English springer lies on a wooden porch in summer, chin resting on its paws, eyes tracking something off-frame as insects hum in the background.',
            'Aerial drone shot descending toward an English springer running along an empty beach at sunrise, its shadow stretching long across the wet sand.',
            'An English springer catching a frisbee mid-air in a park.',
            'An English springer spaniel sits in the open boot of a muddy estate car, soaked and grinning after a walk, faint steam rising from its coat.',
            "A close-up of an English springer's face, dark eyes and freckled muzzle.",
            'An English springer trots down a village lane on a lead beside its owner, hedgerows on both sides, late afternoon light flickering between the branches.',
            'An English springer digs enthusiastically in a snowy garden, powder flying up around its head, the sky flat and white above the bare hedges.',
        ],
        'cassette player': [
            'A vintage portable cassette player sitting on a wooden table.',
            'A vintage portable cassette player, its brown plastic shell and gold accents scratched and worn, sits on a dusty shelf while the spools turn slowly behind a scuffed window.',
            'Close-up of a hand pressing play on a cassette player.',
            'A silver cassette player rests on a car dashboard in the sun, tape door open and a cassette half inserted, dust drifting through the light from the windscreen.',
            'Macro video of the reels inside a cassette player turning, brown tape sliding past the heads, the mechanism clicking softly under a warm desk lamp.',
            'A boombox-style cassette player on a concrete step, speakers facing the camera.',
            'A teenage bedroom in the late eighties: a cassette player on the carpet surrounded by loose tapes and liner notes, an orange lamp throwing long shadows across the floor.',
            'A weathered yellow waterproof cassette player clipped to a belt, headphone cable swinging, filmed as the wearer jogs along a seafront promenade.',
            'A cassette player being opened, the tape door swinging down.',
            'Static shot of a black cassette player on the counter of a second-hand electronics shop, a price tag taped to its side, shelves of old audio gear behind it.',
            'A cassette player on a picnic blanket in a park, sunlight flaring across its chrome buttons as a hand reaches in to turn the volume dial.',
            'A dusty cassette player found in a cardboard box in an attic.',
            'Overhead shot of a cassette player on a desk beside a notebook and a pile of mixtapes, a hand feeding a cassette into the slot and pushing the door shut.',
            'A repair bench with a cassette player opened up, its casing removed, tweezers lifting a drive belt into place under a bright inspection lamp.',
            'A cassette player playing on a kitchen windowsill.',
            'A chunky grey cassette player sits in a wood-panelled hi-fi rack, its level needles twitching, the room lit only by a single warm bulb in the corner.',
            'Slow push-in on a cassette player on a bedside table at night, a small red indicator glowing while the rest of the room stays in soft blue darkness.',
            'A cassette player held in two hands, turned over to show the battery cover.',
            'A cassette player sits on the roof of a parked car at dusk, its aerial extended and tape running, the sky behind fading from orange to violet.',
            'A cassette player on a garage workbench covered in sawdust, buttons pressed by a gloved thumb, tools and paint tins lined up behind it.',
        ],
        'chain saw': [
            'A man in protective gear revving a chain saw in a quiet forest.',
            'A lumberjack in an orange helmet and chaps starts a chain saw with a sharp pull of the cord, blue exhaust smoke curling up between tall pines in the early morning.',
            "Close-up of a chain saw's bar and spinning chain.",
            "A chain saw bites into the trunk of a fallen spruce, pale sawdust spraying out in a wide arc and settling on the mossy ground around the cutter's boots.",
            'A chain saw resting on a stack of freshly cut logs.',
            'Slow motion footage of a chain saw cutting through a thick oak log on a sawhorse, chips flying toward the lens, low winter sun behind the operator.',
            'An arborist suspended in a harness high in a tree uses a compact chain saw to trim a heavy limb, ropes taut, leaves shaking as the branch drops away.',
            'A chain saw lying on a workbench beside a can of bar oil.',
            'A gloved hand tensions the chain of an orange chain saw with a screwdriver, the bar clamped in a vice in a cluttered garage under a fluorescent strip light.',
            'A chain saw idling in a woodyard, exhaust smoke drifting.',
            'Handheld documentary footage of a forestry crew felling a pine: one worker steps back as the chain saw finishes the back cut and the tree begins to lean.',
            'A chain saw hangs from a hook on a shed wall between coils of rope and a fuel can, dusty light falling through a small dirty window.',
            'An old rusted chain saw abandoned in long grass behind a barn.',
            "A chain saw carves a rough bear shape from a tree stump at a country fair, wood chips heaped around the carver's boots while a small crowd watches from behind a rope.",
            'Static wide shot of a man clearing a storm-fallen tree from a rural road with a chain saw, hazard lights flashing on a pickup truck parked behind him.',
            'Close-up of a chain saw being refuelled from a red plastic can.',
            'A chain saw cuts firewood in a snowy yard, the blade throwing pale dust against the dark treeline, breath and exhaust mixing in the freezing air.',
            'A battery-powered chain saw rests on a garden table beside safety glasses and ear defenders, hedge trimmings scattered across the lawn behind it.',
            'A chain saw held up to the camera, chain still, teeth clearly visible.',
            'Night footage of a chain saw working under floodlights after a storm, sawdust glowing in the beams as a crew cuts through a trunk blocking a driveway.',
        ],
        'church': [
            'A small stone church in green rolling countryside.',
            'A majestic church stands tall against a pale sky, surrounded by lush greenery, a tranquil lake directly to its right reflecting the spire in still water.',
            'A white wooden church with a tall steeple at the edge of a village.',
            'Slow aerial orbit around a Gothic church, flying buttresses and rose window catching low golden light, the town rooftops laid out below.',
            'Interior of a church, sunlight through stained glass falling across the pews.',
            'The camera drifts down the central aisle of an old stone church, columns rising into shadow on either side, candlelight flickering near the altar at the far end.',
            'A country church in winter, snow on its roof and on the gravestones.',
            'A small whitewashed church on a Greek island clifftop, its blue dome bright against the sea, wind moving the dry grass around its low walls.',
            'Static wide shot of a red-brick church at dusk, its windows lit from within, people climbing the steps as bells ring out over an empty street.',
            'A ruined church overgrown with ivy, roofless and open to the sky, thin light falling through empty window arches onto broken flagstones inside.',
            'Time-lapse of clouds moving behind the spire of a church.',
            'A wooden stave church surrounded by tall pines in Norway, its dark tarred timbers and layered roofs steaming faintly after rain.',
            'A colonial-style church on a dusty town square at midday, harsh light on its white facade, a few figures resting in the shade of its portico.',
            'A church seen across a field of ripe wheat.',
            'Low-angle shot looking up at the tower of a village church, weathervane turning, jackdaws circling the belfry under a grey autumn sky.',
            'A church at night, floodlit against a dark sky.',
            'A modern concrete church with clean geometric lines, a narrow slit of daylight running down its front wall, filmed in soft overcast light.',
            'A country church surrounded by an old graveyard, mist lying low between the headstones as the camera pushes slowly in through the lychgate at dawn.',
            'The bell tower of a church rises above the rooftops of a hillside town, seen from a narrow street below in late afternoon sun.',
            'A wedding party gathers outside a church, confetti in the air.',
        ],
        'French horn': [
            'A shiny brass French horn resting on a wooden desk.',
            'A close-up shot of a polished brass French horn with intricate engravings sitting on a wooden desk, warm lamplight sliding along its coiled tubing.',
            'A musician playing a French horn in an orchestra pit.',
            'A French horn player in black concert dress raises the instrument, right hand disappearing into the bell, stage lights glinting off the brass as the orchestra tunes.',
            'Macro video panning across the valves and rotors of a French horn, fingerprints visible on the lacquer, a soft studio light reflected in the curved metal.',
            'A French horn lying open in its velvet-lined case.',
            'A student practises the French horn alone in an empty rehearsal room, a music stand in front of her, afternoon light coming through tall dusty windows.',
            'A French horn held against a plain white background, slowly rotating.',
            'A brass band marches down a small-town street, a French horn player in uniform in the second row, sunlight flashing off the bell with every step.',
            'An old tarnished French horn hangs on the wall of a music shop between trumpets and trombones, a handwritten price tag tied to its bell.',
            'Hands wiping a French horn clean with a soft cloth.',
            'A French horn rests on a chair in an empty concert hall, rows of red seats stretching away behind it, the house lights dimmed to a low amber.',
            'Slow dolly around a French horn on a black studio backdrop, a single hard light rolling highlights across its coiled tubing and flared bell.',
            'A French horn lying on a bed of sheet music, seen from directly above.',
            'A horn player empties water from the slides of a French horn between movements, the rest of the section blurred behind him under warm stage light.',
            'A child holds a French horn that looks almost too big for them, standing in a school corridor beside a row of dented lockers.',
            "A French horn in a repair workshop, dents smoothed by a technician's tools under bright task lighting, brass shavings scattered across the bench.",
            'A French horn player silhouetted against a bright window.',
            'A French horn placed on the lid of a grand piano in a wood-panelled room, late sun coming through lace curtains and pooling on the polished surface.',
            'A French horn is carried bell-first through a backstage corridor, past cables and flight cases, toward the glow of the stage doors.',
        ],
        'garbage truck': [
            'A green garbage truck parked on a quiet residential street.',
            'A bright green garbage truck parked along a residential street in early morning light, its hydraulic arm lifting a wheelie bin as steam rises off the tarmac.',
            'A garbage truck driving slowly down a suburban road.',
            'Two sanitation workers in high-vis vests hop off the back of a white garbage truck, roll bins to the lift and step clear as the compactor grinds into motion.',
            "Close-up of a garbage truck's rear loader compacting waste.",
            'A blue garbage truck reverses down a narrow alley between brick buildings, beeping, its mirrors nearly brushing the walls as a worker guides it back.',
            'A garbage truck at a landfill, tipping its load.',
            'Aerial shot following a garbage truck along a curving suburban street at dawn, stopping every few houses where bins are lined up along the kerb.',
            'A rusty old garbage truck parked in a depot yard among other municipal vehicles, filmed in flat grey light on a wet morning.',
            'A garbage truck stopped at a traffic light in a city centre.',
            'Side-on tracking shot of a red garbage truck rolling past rows of terraced houses, side-loader arm folded, autumn leaves swirling in its wake.',
            'A garbage truck is washed down at the end of a shift, water sheeting off its scratched paintwork under floodlights in a depot bay.',
            'Children waving at a passing garbage truck from a driveway.',
            'A garbage truck works a narrow European old-town street at first light, its yellow warning lights bouncing off shuttered shopfronts and wet cobblestones.',
            'Close-up of the hydraulic arm of a garbage truck gripping a wheelie bin.',
            'A garbage truck idles in heavy rain, wipers going, a worker in a dripping yellow jacket dragging bags to the hopper as headlights cut through the downpour.',
            'A garbage truck parked outside an apartment block on a hot summer afternoon, heat shimmering off its roof, a worker drinking from a bottle in the cab.',
            'A garbage truck seen from behind at night, its lights glowing.',
            "Slow motion footage of a garbage truck's compactor plate pressing down on refuse, hydraulics straining, dust and paper fragments drifting in the air.",
            'A garbage truck rounds a corner in a snow-covered neighbourhood, chains on its tyres, exhaust billowing white in the freezing morning air.',
        ],
        'gas pump': [
            'A gas pump at a roadside filling station.',
            'A solitary gas pump stands at a deserted roadside, its brightly lit signs for unleaded and diesel humming faintly against a wide empty desert dusk.',
            'Close-up of a nozzle being lifted from a gas pump.',
            "A hand slides a card into a gas pump's reader, the small screen flickering through prompts as rain streaks the plastic housing under a bright canopy.",
            'A rusted vintage gas pump outside an abandoned garage.',
            'Two red vintage gas pumps stand in front of a peeling clapboard service station, weeds pushing up through cracked concrete under harsh midday sun.',
            'The digital display of a gas pump counting up litres and price.',
            "A driver refuels a pickup at a gas pump on a snowy highway stop, exhaust and breath clouding the cold air, the pump's lights the brightest thing in frame.",
            'Slow motion footage of fuel splashing at the mouth of a gas pump nozzle as the trigger releases, droplets glinting under fluorescent canopy lights.',
            'A row of gas pumps under a bright canopy at night.',
            'A single gas pump on a remote island road, hand-cranked and salt-corroded, the sea visible just beyond a low stone wall behind it.',
            'A gas pump seen from a low angle against a stormy evening sky, its price sign glowing while cars queue behind with headlights on.',
            'A gas pump nozzle hanging back in its holster, hose neatly coiled.',
            'Aerial descent onto a small filling station at dawn, four gas pumps arranged under a white canopy, a single car pulling in off an empty road.',
            'A gas pump covered in stickers and faded decals, filmed close as a hand squeezes the worn trigger and the meter starts to spin.',
            'A gas pump in the rain, water running down its casing.',
            'A retro chrome-and-glass gas pump displayed inside a classic car museum, spotlit against a dark wall beside a polished vintage sedan.',
            'A gas pump at a rural station in golden hour, long shadows stretching across the forecourt, moths already circling the lit price sign.',
            'Static shot of a gas pump while a car fills up behind it.',
            'A worker in overalls services an open gas pump, its front panel removed to show wiring and meters, a toolbox open on the forecourt beside him.',
        ],
        'golf ball': [
            'A golf ball resting on a tee on a green fairway.',
            'A golf ball rolls across green, white and dark blue checkered turf, its dimpled surface catching the light as it curves slowly toward the edge of frame.',
            'Close-up of the dimpled surface of a golf ball.',
            'Extreme slow motion of a driver striking a golf ball off a tee, the ball compressing against the clubface before launching away in a spray of dew.',
            'A golf ball dropping into the hole on a putting green.',
            'A golf ball sits in thick rough beside a bunker, blades of grass pressing against its white surface, morning dew beaded along the dimples.',
            'A golf ball half-buried in the sand of a bunker.',
            'Macro video slowly orbiting a single golf ball on a wooden table against a dark background, a soft key light rolling across its dimples and brand marking.',
            'A bucket of golf balls spilled across the mat at a driving range, one rolling free toward the camera as the floodlights come on at dusk.',
            'A golf ball rolling slowly across a putting green toward the cup.',
            'A golf ball balanced on a red tee on a frosty winter fairway, breath fogging into frame as a gloved hand adjusts its position.',
            'A golf ball floating in the shallow water at the edge of a course pond, reeds around it, ripples distorting its reflection.',
            'A hand placing a golf ball on the grass beside a marker.',
            'High-speed footage of a golf ball bouncing twice on hard fairway turf and skidding forward, small puffs of dust rising with each impact.',
            'A golf ball displayed in a glass case.',
            'A worn golf ball with grass stains and scuff marks lies on a concrete path beside a course, filmed close in flat afternoon light.',
            "Overhead shot of a golf ball a few centimetres from the hole on a manicured green, the flagstick's shadow lying across the turf beside it.",
            "A golf ball rolling off a putter's face, seen from ground level.",
            'A golf ball sits on a tee at a seaside links course, wind flattening the grass around it, grey clouds and whitecaps out on the water behind.',
            'A pyramid of golf balls stacked on a shop display, one lifted away by a hand, warm retail lighting reflecting off the white dimpled surfaces.',
        ],
        'parachute': [
            'A parachute open in a clear blue sky.',
            'A parachute in mid-air, its vibrant multicoloured canopy glowing brightly as sunlight passes through the fabric, lines taut beneath it against deep blue sky.',
            'A parachute opening above a patchwork of green fields.',
            'Ground-level shot looking up as a red-and-white parachute drifts down toward a grass landing zone, the jumper steering into the wind on final approach.',
            'A parachute laid out flat on the grass being folded.',
            'Aerial footage flying alongside a parachute high above a coastline, its canopy rippling in the airflow, the jumper small beneath the risers.',
            'A cargo parachute descending with a crate slung beneath it.',
            'Slow motion of a parachute snapping open behind a free-falling jumper, fabric unfurling from the pack and blooming into a wide canopy against scattered clouds.',
            'Several parachutes descending together over a drop zone.',
            'A round military parachute drifts down over a dry training field at dawn, dust already rising where earlier jumpers landed, the canopy pale against the sky.',
            'A parachute collapses onto the ground as its jumper touches down and runs forward, the canopy folding over itself in the long grass.',
            'A parachute seen from directly below, backlit by the sun.',
            'A packed parachute rig lies on a hangar floor beside a helmet and altimeter, sunlight from an open door falling across the folded nylon and webbing.',
            'A drag parachute deploys behind a landing jet, the canopy snapping taut and fluttering in the wash as the aircraft slows along the runway.',
            'A striped parachute floating gently over an autumn forest.',
            'A tandem parachute turns lazily above a coastal drop zone at golden hour, its canopy edges lit orange, the sea glinting far below.',
            'A parachute canopy fills the frame, fabric cells rippling in the wind, thin suspension lines running down out of the bottom of the shot.',
            'A parachute being packed on a long table inside a hangar.',
            'Wide static shot of a parachute descending toward a snowy mountain valley, tiny against the huge white slopes, its colours the only saturation in frame.',
            'A parachute drifts down at dusk against a violet sky, its canopy catching the last light while lights come on in the town below.',
        ],
    },

    # -----------------------------------------------------------------------
    # v6: v5 rewritten to be as easy as possible for the frame classifier.
    #
    # On v5 the reference model scores 49.8 mean Top-1 against 64.4 on the
    # template set v1, and two columns collapse: tench 0.0 and garbage truck
    # 5.6. A column whose reference accuracy is near zero cannot measure
    # erasure — ESR is already ~100 before anything is unlearned — so the
    # cells are uninformative however good the method is.
    #
    # The fix is not "cleaner studio shots" but proximity to how ImageNet
    # depicts each class, since that is what the classifier learnt. For tench
    # the canonical image is an angler holding the fish, not a fish in a pond;
    # for garbage truck it is a side view of the whole vehicle on a street.
    # Written to that principle:
    #
    #   - the class occupies most of the frame and stays in view throughout
    #   - framing follows the class's canonical ImageNet depiction
    #   - plain, evenly lit daylight; no night, weather, or golden hour
    #   - no second object that is itself an ImageNet class
    #   - little camera movement, no stylistic language
    #   - features that separate the class from its neighbours are named
    #     explicitly (English springer: liver-and-white coat, feathered ears)
    #
    # Lengths are uniform here, 12-25 words. v5 varied them deliberately to
    # study prompt specificity; v6 answers a different question and holds that
    # variable fixed.
    #
    # v6 is not comparable with v5 or with the template sets in one table:
    # each set has its own reference accuracies and needs its own baseline row.
    # -----------------------------------------------------------------------
    "v6": {
        'tench': [
            'A tench held in both hands by an angler, the whole fish filling the frame in bright daylight.',
            'Close-up of a tench lying on green grass, the entire fish visible from head to tail.',
            'A tench held horizontally toward the camera, its olive-green flank sharp and evenly lit.',
            'A large tench resting on a flat green mat, filmed from directly above in daylight.',
            'An angler crouches on a riverbank and lifts a tench toward the camera, the fish centred and in focus.',
            'A tench on a wooden surface, its dorsal fin and rounded tail clearly visible in even daylight.',
            'Static daylight shot of a tench held at chest height in two hands against a plain grassy bank.',
            'The camera moves slowly along a tench lying on grass, from its head to its broad tail.',
            'Close-up of the thick olive body and small red eye of a tench in clear daylight.',
            'A fisherman holds a tench steady toward the camera under a bright overcast sky.',
            'A tench lying on its side on a flat green surface, the whole fish sharp and centred.',
            'A tench held just above shallow clear water, its full body visible in daylight.',
            'A tench filling most of the frame, its scales and fins clearly lit from the side.',
            'Two hands hold a tench level with the camera, a plain green background behind it.',
            'A tench resting on short grass, filmed from the side at close range in bright light.',
            'A tench held up after being caught, the whole fish in frame and nothing else nearby.',
            'Daylight close-up of a tench, its rounded fins and dark olive back clearly visible.',
            'A tench lies still on a mat while the camera holds steady on it in soft daylight.',
            'An angler presents a tench to the camera with both hands, the fish sharply in focus.',
            'A tench seen from the side at close range, filling the frame against plain grass.',
        ],
        'English springer': [
            'An English springer standing side-on in short grass, its liver-and-white coat clearly visible in daylight.',
            "Close-up of an English springer's head, its long freckled ears hanging beside its face.",
            'An English springer sitting on a lawn facing the camera, the whole body in frame in bright daylight.',
            'An English springer standing still in an open field, its full body sharp against plain green grass.',
            'A liver-and-white English springer trotting toward the camera across a mown lawn.',
            'An English springer sits in profile on grass, its feathered ears and docked tail clearly visible.',
            'Daylight portrait of an English springer, its head and shoulders filling the frame.',
            'An English springer standing on a gravel path, seen from the side with the whole dog in frame.',
            'An English springer with a liver-and-white coat sits calmly in bright even daylight.',
            'The camera holds steady on an English springer standing in short grass with its ears down.',
            'An English springer walks slowly across a lawn, filmed from the side in clear daylight.',
            'Close-up of an English springer panting, its long ears and freckled muzzle in sharp focus.',
            'An English springer sits upright on grass against a plain background, its full body visible.',
            'A liver-and-white English springer stands in a field with its head turned toward the camera.',
            'An English springer standing in daylight, its coat markings clearly lit from the side.',
            'An English springer lies on short grass with its head raised, the whole dog in frame.',
            'Side view of an English springer standing still, its feathered legs and ears clearly visible.',
            'An English springer looks toward the camera from a grassy lawn in soft daylight.',
            'A well-lit English springer stands centred in the frame against plain green grass.',
            'An English springer walks toward the camera on a lawn with its full body in view.',
        ],
        'cassette player': [
            'A portable cassette player on a table, its front panel and cassette door facing the camera.',
            'Close-up of a cassette player, the tape spools turning behind the clear window.',
            'A cassette player on a plain wooden desk, filmed straight on in even daylight.',
            'A silver cassette player with two speakers, centred in the frame against a plain wall.',
            'The camera holds steady on a cassette player, its buttons and cassette door clearly visible.',
            'A cassette player on a shelf, its front panel evenly lit and the whole unit in frame.',
            'Close-up of the buttons and dials of a cassette player in bright daylight.',
            'A portable cassette player with a carrying handle, seen from the front on a table.',
            'A cassette player sits on a plain surface while its tape door is opened.',
            'Static shot of a cassette player, its speaker grilles and controls facing the camera.',
            'A cassette player on a desk, filmed from slightly above with the whole unit in view.',
            'A boxy cassette player against a plain background, its front panel sharp and well lit.',
            'Close-up of a cassette player as a tape is pushed into the slot.',
            'A cassette player standing upright on a shelf, its controls and door clearly visible.',
            'A cassette player centred in the frame, evenly lit, with nothing else on the table.',
            'The camera slowly moves in on a cassette player resting on a plain wooden surface.',
            'A cassette player with rotating spools visible through its window, filmed up close.',
            'A cassette player on a table in daylight, its front face square to the camera.',
            'Side-lit close-up of a cassette player showing its buttons, dials and cassette door.',
            'A cassette player sits alone on a plain surface, the whole unit sharp in daylight.',
        ],
        'chain saw': [
            'A chain saw resting on a cut log, its bar and chain clearly visible in daylight.',
            'Close-up of a chain saw held by its handle, the guide bar filling the frame.',
            'A chain saw cutting into a thick log, sawdust spraying out to one side.',
            'A chain saw lying on the ground beside freshly cut timber, the whole tool in frame.',
            'Someone holds a chain saw level with the camera, its bar and chain sharply in focus.',
            'A chain saw sits on a wooden bench, filmed straight on in even daylight.',
            'Close-up of the chain and guide bar of a chain saw in bright daylight.',
            'A chain saw is lifted and held steady, the whole tool visible against a plain background.',
            'A chain saw cutting through a log outdoors, its bar buried in the wood.',
            'Static shot of a chain saw resting on sawdust-covered ground with the whole tool in frame.',
            'A chain saw with an orange body lies on cut timber, clearly lit from above.',
            'The camera moves slowly along a chain saw from its handle to the tip of the bar.',
            'A chain saw held in both hands with the engine running, the chain spinning on the bar.',
            'A chain saw propped against a stack of logs, the whole tool sharp in daylight.',
            'Close-up of a chain saw being started, the operator gripping the rear handle.',
            'A chain saw resting on a tree stump, filmed from the side in clear daylight.',
            'A chain saw cuts a log in half outdoors, the tool centred in the frame.',
            'A chain saw lying flat on grass with its bar and chain facing the camera.',
            'Someone carries a chain saw through a clearing, the tool clearly visible at their side.',
            'A chain saw held up toward the camera, its bar and chain filling most of the frame.',
        ],
        'church': [
            'A stone church with a tall steeple, the whole building centred in bright daylight.',
            'The front facade of a church with an arched doorway and a bell tower above it.',
            'A small country church with a pointed spire, filmed from the front in clear weather.',
            'A church exterior in daylight, its steeple and tall windows clearly visible.',
            'A white wooden church with a tall steeple standing against a plain blue sky.',
            'The camera holds steady on a church building with the whole facade in frame.',
            'A brick church with a square bell tower, seen from across an open lawn.',
            'A church with a steep roof and arched windows, filmed straight on in daylight.',
            'A stone church seen from the front, its spire rising above the entrance.',
            'A church exterior filmed slowly from the side, the full building in view.',
            'A village church with a bell tower, centred in the frame under a clear sky.',
            'The steeple and roof of a church, filmed from ground level in bright daylight.',
            'A church facade with tall arched windows, evenly lit and sharply in focus.',
            'A grey stone church stands alone in daylight with the whole building visible.',
            'The camera moves slowly toward the entrance of a church with a tall spire.',
            'A church with a cross on its steeple, filmed from the front in clear weather.',
            'A large church exterior in daylight, its towers and windows clearly visible.',
            'A country church with white walls and a dark roof, the whole building in frame.',
            'A church seen from a short distance, its steeple centred against an open sky.',
            'The front of an old stone church, its arched door and bell tower sharp in daylight.',
        ],
        'French horn': [
            'A French horn resting on a table, its coiled brass tubing facing the camera.',
            'Close-up of a French horn, the wide bell and coiled tubing filling the frame.',
            'A musician holds a French horn ready to play, the instrument clearly visible.',
            'A French horn on a plain surface in even daylight, the whole instrument in frame.',
            'The camera moves slowly around a French horn, its brass coils catching the light.',
            'A French horn held up toward the camera, its bell and valves sharply in focus.',
            'Close-up of the valves and coiled tubing of a French horn in bright light.',
            'A French horn sits upright on a stand, the whole instrument centred in the frame.',
            'A player raises a French horn to their lips, the brass coils clearly visible.',
            'A polished French horn against a plain background, evenly lit and sharp.',
            'A French horn lying on a dark cloth, its wide bell turned toward the camera.',
            'Static shot of a French horn, its coiled tubing and flared bell in full view.',
            'A French horn held in both hands, the instrument filling most of the frame.',
            'Close-up of the flared bell of a French horn, the brass reflecting daylight.',
            'A French horn resting on a chair, the whole instrument visible in clear daylight.',
            'The camera holds steady on a French horn, its coils and valves in sharp focus.',
            'A brass French horn on a plain table, seen from the side in even light.',
            'A musician lowers a French horn after playing, the instrument clearly in frame.',
            'A French horn filmed close up, its coiled tubing dominating the frame.',
            'A French horn stands on a stand in daylight, the whole instrument sharply lit.',
        ],
        'garbage truck': [
            'A garbage truck seen from the side on a residential street in bright daylight.',
            'A garbage truck with its rear loader raised, the whole vehicle in frame.',
            'A garbage truck stops at the kerb, filmed from the side with the full vehicle visible.',
            'The rear of a garbage truck as a bin is lifted and emptied into the hopper.',
            'A garbage truck drives slowly down a street, the camera holding it in full view.',
            'A three-quarter view of a garbage truck parked on a road in clear daylight.',
            'A garbage truck with a green body and a large rear hopper, seen from the side.',
            'A garbage truck idles at the kerb while its lifting arm raises a wheeled bin.',
            'The camera holds steady on a garbage truck, its full length visible in daylight.',
            'A garbage truck seen from behind, its loading hopper and controls clearly visible.',
            'A white garbage truck moves along a suburban street with the whole vehicle in frame.',
            'A garbage truck parked on a wide road, filmed from the side in bright sunlight.',
            'A garbage truck compacts its load, the rear hopper closing slowly.',
            'A garbage truck viewed side-on, its cab and rear body both fully in frame.',
            'A garbage truck pulls away from the kerb, the camera following it from the side.',
            'The lifting arm of a garbage truck raises a bin above the hopper in daylight.',
            'A garbage truck stands still on an empty street, the whole vehicle sharply lit.',
            'A garbage truck with warning stripes along its side, filmed from a short distance.',
            'A garbage truck seen from the rear as waste is loaded into the hopper.',
            'A large garbage truck fills the frame, seen side-on in clear daylight.',
        ],
        'gas pump': [
            'A gas pump at a filling station, the whole unit centred in daylight.',
            'Close-up of a gas pump, its display and nozzle clearly visible.',
            'A gas pump standing under a station canopy, filmed straight on.',
            'A row of gas pumps on a forecourt, the nearest one filling the frame.',
            'A gas pump with the nozzle in its holster, seen from the front in bright daylight.',
            'A hand lifts the nozzle from a gas pump, the unit clearly in frame.',
            'Static shot of a gas pump, its screen, buttons and hose sharply in focus.',
            'A gas pump on an empty forecourt, the whole unit visible in even daylight.',
            'Close-up of the display of a gas pump as the numbers count upward.',
            'A red and white gas pump stands alone, centred against a plain background.',
            'The camera moves slowly around a gas pump at a quiet filling station.',
            'A gas pump with a coiled hose hanging at its side, seen from the front.',
            'A gas pump under bright daylight, its whole body sharp and well lit.',
            'A nozzle is returned to a gas pump, the unit filling most of the frame.',
            'Two gas pumps side by side on a forecourt, the nearer one in sharp focus.',
            'A gas pump seen from a low angle, its display and panel clearly visible.',
            'A gas pump stands under a canopy, the whole unit centred in the frame.',
            'Close-up of the hose and nozzle of a gas pump in clear daylight.',
            'A tall gas pump on a concrete forecourt, filmed straight on in daylight.',
            'A gas pump with a digital display, the whole unit sharply lit and centred.',
        ],
        'golf ball': [
            'A golf ball sitting on a tee, filling the frame in bright daylight.',
            'Close-up of a white golf ball resting on short green grass.',
            'A golf ball on a tee, its dimpled surface sharply in focus.',
            'A golf ball rolls slowly across a putting green toward the camera.',
            'Macro shot of a golf ball, the dimples covering its white surface.',
            'A golf ball rests beside the hole on a green, filmed from close range.',
            'A golf ball sits on the grass, centred and sharply lit in daylight.',
            'Close-up of a golf ball on a wooden tee against a plain green background.',
            'A golf ball drops onto the green and settles, the camera holding on it.',
            'A single golf ball on short grass, the whole ball filling most of the frame.',
            'A golf ball on a tee seen from ground level in bright sunlight.',
            'The camera moves slowly around a golf ball resting on a putting green.',
            'A white golf ball with clear dimples lies on a flat green surface.',
            'A golf ball is placed on a tee by a gloved hand, which then withdraws.',
            'Close-up of a golf ball rolling and stopping on smooth green grass.',
            'A golf ball sits still on the fairway, sharply lit and centred in the frame.',
            'A golf ball on a tee, its dimpled surface catching bright daylight.',
            'Static shot of a golf ball on short grass with nothing else in the frame.',
            'A golf ball rests on a green, filmed close up from a low angle.',
            'A dimpled white golf ball fills the frame, resting on flat green grass.',
        ],
        'parachute': [
            'An open parachute descending against a clear blue sky, the canopy filling the frame.',
            'A parachute canopy fully inflated in the air, seen from below.',
            'A skydiver descends beneath an open parachute, the canopy clearly visible above.',
            'A colourful parachute drifts slowly down through a bright open sky.',
            'A parachute opens above a skydiver, the canopy spreading wide against the sky.',
            'An open parachute seen from below, its lines running down from the canopy.',
            'A parachute glides across a clear sky with the whole canopy in frame.',
            'A red and white parachute descends steadily, filmed from the ground.',
            'The camera follows a parachute as it turns slowly in bright daylight.',
            'A parachute canopy billows overhead, sunlight showing through the fabric.',
            'A skydiver hangs beneath a rectangular parachute high above open ground.',
            'An open parachute centred against a plain blue sky, sharply in focus.',
            'A parachute drifts downward, its canopy fully inflated and clearly lit.',
            'A parachute descends toward a field, the canopy large in the frame.',
            'A wide parachute canopy seen from below against bright daylight.',
            'A parachute turns gently in the air with the whole canopy visible throughout.',
            'A striped parachute descends through a clear sky, filmed from below.',
            'An open parachute fills most of the frame as it drifts across the sky.',
            'A parachute carries a skydiver slowly downward under a bright open sky.',
            'A fully open parachute against a clear sky, its canopy and lines sharply visible.',
        ],
    },
}

# ---------------------------------------------------------------------------
# v7: v6 with the three classes it failed on rewritten, the other seven left
# byte-identical. Defined as a diff rather than a fresh table so that the only
# thing separating the two sets is the change under test.
#
# Measured Top-1 of the reference model on v6, against v5:
#
#   garbage truck  5.6 -> 98.8     tench             0.0 -> 45.3
#   church        15.9 -> 75.9     cassette player  40.0 -> 10.0
#   golf ball     79.1 -> 100.0    English springer 46.2 ->  0.0
#   ...five more between 86.8 and 97.9
#
# The two regressions were caused by specific wording, not by chance:
#
#   cassette player  v6 described the medium ("tape spools turning behind the
#                    window", "a tape is pushed into the slot") and the model
#                    produced close-ups of a cassette, which ImageNet holds as
#                    its own class. 181 frames were predicted `cassette`.
#                    v7 describes the device instead — a portable stereo with
#                    speakers, handle, dial and buttons — and never says tape.
#
#   English springer v6 said "liver-and-white" to separate it from the Welsh
#                    springer. It produced black-and-white dogs, predicted
#                    Border collie. v7 uses the full breed name, "brown and
#                    white", and the long drooping feathered ears, which are
#                    the feature the neighbouring breeds do not share.
#
#   tench            reached only 45.3, confused with barracouta and coho —
#                    both slender silver fish. v7 names the features that
#                    separate it: thick heavy body, dark bronze-green, blunt
#                    head, small red eye, rounded fins.
#
# Not built by keeping the prompts that scored best on v6. Each prompt is one
# video, so a per-prompt accuracy of 1.00 or 0.00 is a single coin flip;
# selecting on it would fit the seed rather than anything real.
# ---------------------------------------------------------------------------
PER_CLASS_PROMPT_SETS["v7"] = dict(PER_CLASS_PROMPT_SETS["v6"])
PER_CLASS_PROMPT_SETS["v7"].update({
    'tench': [
        'A tench with a thick dark olive body and a small red eye, held in both hands by an angler.',
        'Close-up of a tench, its heavy rounded body and small red eye filling the frame in daylight.',
        'An angler kneels on a grassy bank and holds a tench toward the camera in bright daylight.',
        'A tench lying on short green grass, its deep bronze-green body and rounded fins clearly visible.',
        'A heavy-bodied tench held at chest height in two hands against a plain grassy bank.',
        'Daylight close-up of a tench, its dark green scales and rounded tail fin in sharp focus.',
        'A tench resting on green grass beside a lake, the whole thick-bodied fish in frame.',
        'An angler holds a large tench with both hands, its blunt rounded head toward the camera.',
        'A tench with small dark scales and a small red eye lies on a flat green surface.',
        'Close-up of the rounded fins and thick body of a tench in clear even daylight.',
        'A tench held up after being caught, its heavy dark olive body filling most of the frame.',
        'A tench lies on short grass while the camera holds steady on its rounded body and fins.',
        'A fisherman lifts a thick dark tench toward the camera under a bright overcast sky.',
        'A tench with a deep body and a rounded tail rests on green grass in bright daylight.',
        'Close-up of the head of a tench, its small red eye and blunt snout sharply lit.',
        'A large tench is held in two hands, its dark bronze-green body clearly visible in daylight.',
        'A tench lying on its side on green grass, its thick body and small scales in sharp focus.',
        'A tench with rounded fins and a heavy body fills the frame against plain green grass.',
        'An angler presents a heavy tench to the camera, its dark olive body in bright daylight.',
        'A tench rests on a green bank, its thick rounded body and red eye clearly visible.',
    ],
    'English springer': [
        'An English springer spaniel standing in short grass, its long brown ears hanging beside its face.',
        'A brown and white English springer spaniel stands side-on in a field in bright daylight.',
        'Close-up of an English springer spaniel, its long drooping ears framing its face.',
        'A medium-sized English springer spaniel with a wavy brown and white coat stands on a lawn.',
        'An English springer spaniel sits on grass, its long feathered ears hanging low.',
        'An English springer spaniel with brown patches on white fur stands still in daylight.',
        'An English springer spaniel gundog stands in a field, its whole body in frame.',
        'An English springer spaniel looks toward the camera, its long ears covered in wavy brown fur.',
        'A brown and white English springer spaniel walks slowly across a lawn in clear daylight.',
        'An English springer spaniel stands in short grass, its docked tail and feathered legs visible.',
        'Daylight portrait of an English springer spaniel, its long brown ears hanging past its jaw.',
        'An English springer spaniel with a white chest and a brown back stands on a grassy field.',
        'An English springer spaniel sits upright on a lawn, its long ears hanging down either side.',
        'A wavy-coated English springer spaniel stands in profile against plain green grass.',
        'An English springer spaniel with brown and white markings stands still in bright daylight.',
        'Close-up of the head of an English springer spaniel, long ears and soft brown eyes in focus.',
        'An English springer spaniel stands on a lawn, its brown ears lifting slightly in the breeze.',
        'A medium-sized English springer spaniel with long drooping ears sits in a green field.',
        'An English springer spaniel trots toward the camera across short grass in daylight.',
        'An English springer spaniel stands centred in the frame, its brown and white coat clearly lit.',
    ],
    'cassette player': [
        'A portable cassette player with two speakers and a carrying handle, standing on a table.',
        'A boombox-style cassette player with a radio dial and speaker grilles, seen from the front.',
        'A large portable cassette player sits on a shelf, its whole body in frame in daylight.',
        'A silver hi-fi cassette player with rows of buttons, filmed straight on.',
        'A cassette player with two round speakers on either side of its front panel.',
        'A cassette player unit in a stack of stereo equipment, its front panel facing the camera.',
        'A boxy portable cassette player with a handle on top, standing on a wooden floor.',
        'A black cassette player with knobs and a radio dial, centred against a plain wall.',
        'A cassette player with speaker grilles and a row of buttons, evenly lit in daylight.',
        'A portable cassette player stands upright on a table, the whole device sharply in focus.',
        'A cassette player with an antenna and two speakers, seen from the front in daylight.',
        'The camera holds steady on a cassette player, its dials and speaker grilles clearly visible.',
        'A stereo cassette player on a shelf, its full width in frame against a plain background.',
        'A vintage portable cassette player with chrome trim, standing on a wooden table.',
        'A cassette player with a handle and two speakers rests on the ground in bright daylight.',
        'A wide cassette player with a row of buttons along its front, filmed straight on.',
        'A cassette player sits alone on a plain surface, its whole body evenly lit.',
        'A portable cassette player with a lit radio dial, centred in the frame.',
        'A cassette player with two speakers stands on a desk as the camera slowly moves closer.',
        'A boxy cassette player with knobs and grilles fills the frame in even daylight.',
    ],
})

# ---------------------------------------------------------------------------
# v8: v7 rewritten in the dense style CogVideoX expects.
#
# CogVideoX is trained on long LLM-refined captions and degrades on short
# prompts far more than HunyuanVideo does. On v7 at its native 720x480 the
# untouched model reaches 44.9 mean Top-1, against the 78.4 reported for the
# same protocol in arXiv:2505.17550. Prompt length is the leading suspect.
#
# v8 is built from v7 rather than written from scratch, so the scene, the
# framing and the class name are identical and the only variable is density.
# Each prompt is the v7 sentence followed by a texture note, the surroundings,
# lighting with camera work, and a line on mood: 50-75 words against 10-20.
#
# Composed in code instead of spelled out, because 200 hand-written paragraphs
# would be unreviewable and would hide the fact that this is v7 plus layers.
# ---------------------------------------------------------------------------
_V8_LIGHT = (
    "Soft natural daylight falls evenly across the scene",
    "Warm afternoon sunlight rakes in from one side",
    "Bright overcast light keeps the colours flat and true",
    "Clear midday sun picks out every surface detail",
    "Gentle diffused light wraps around the subject",
    "Crisp morning light throws long soft shadows",
    "Even ambient light leaves the frame free of glare",
    "Low golden light warms the edges of the frame",
)

_V8_CAMERA = (
    "the camera holds steady in a medium close-up",
    "the camera drifts almost imperceptibly closer",
    "a slow lateral dolly keeps the subject centred",
    "a locked-off tripod shot holds the composition still",
    "the camera pushes in gradually from a wide framing",
    "a shallow depth of field isolates the subject",
    "the camera tracks slowly around the subject",
    "a steady handheld shot keeps the subject mid-frame",
)

_V8_MOOD = (
    "The overall mood is calm and unhurried, with no sudden movement anywhere in the shot.",
    "Nothing else competes for attention; the scene stays quiet and uncluttered throughout.",
    "The pace is slow and observational, closer to documentary footage than to a commercial.",
    "Colours are natural and unsaturated, and the whole shot feels grounded and ordinary.",
    "The atmosphere is still and matter-of-fact, with the subject clearly the point of the shot.",
    "Everything holds steady, the framing patient and the motion minimal.",
    "The scene reads as plain and unstyled, with realistic textures throughout.",
    "The tone is neutral and documentary, without any dramatic staging.",
)

# Five texture notes and five descriptions of the surroundings per class. These
# are the only class-specific parts; lighting, camera and mood are shared, so
# no class gets a cinematography advantage over another.
_V8_TEXTURE = {
    'tench': ("The scales sit packed and fine, and the fins move in small deliberate sweeps.",
              "Water beads along the flank and slides off in slow drops.",
              "The surface of the skin is slick and faintly mottled.",
              "Gill covers open and close in a steady, unhurried rhythm.",
              "The body is heavy and solid, with no sharp angles anywhere."),
    'English springer': ("The coat is thick and slightly wavy, lifting a little at the edges.",
              "Fur along the ears falls in long soft fringes.",
              "The chest fur is dense and clean, the legs feathered below the knee.",
              "Small movements of the head make the ears swing gently.",
              "The build is compact and muscular under a full, well-kept coat."),
    'cassette player': ("The plastic has a matte finish with faint scratches near the controls.",
              "Buttons sit slightly proud of the front panel, worn smooth at the centres.",
              "A thin metal trim runs along the top edge and catches a narrow highlight.",
              "The speaker mesh is fine and dark, evenly spaced across both grilles.",
              "The whole unit looks solid and slightly dated, built from thick moulded plastic."),
    'chain saw': ("Sawdust clings along the bar and around the base of the housing.",
              "The plastic shell is scuffed and the metal edges dulled from use.",
              "Chain links sit tight and evenly spaced along the guide bar.",
              "Rubber grips show wear where hands have held them.",
              "The body is chunky and functional, with vents cut into one side."),
    'church': ("The stonework is rough and uneven, darkened in patches by weather.",
              "Mortar lines run irregularly between blocks of pale stone.",
              "Slate tiles overlap in neat rows along the pitched roof.",
              "Window glass sits deep in recessed frames, dull under the daylight.",
              "The walls carry the marks of long exposure, streaked and slightly stained."),
    'French horn': ("The lacquer is polished but marked with faint fingerprints near the valves.",
              "Brass curves reflect the room in soft, distorted bands.",
              "Tubing wraps tightly and evenly, each coil sitting close to the next.",
              "The bell rim is smooth and slightly worn along the outer edge.",
              "Metal surfaces show a warm, uneven sheen rather than a mirror finish."),
    'garbage truck': ("Panels are dented in places and streaked with dried grime.",
              "Paint has worn thin around the loading edge and lower body.",
              "Hydraulic pistons sit exposed along the side of the hopper.",
              "The tyres are heavy and deeply treaded, caked with road dust.",
              "Warning markings are faded but still legible along the flank."),
    'gas pump': ("The casing is scuffed near the base where hoses have brushed against it.",
              "Digits on the display are sharp against a dark background.",
              "The hose hangs in a loose, heavy coil with a worn nozzle grip.",
              "Painted steel shows small chips along the corners.",
              "Buttons and labels are slightly faded from constant use."),
    'golf ball': ("Dimples cover the surface in a regular, shallow pattern.",
              "The white finish is clean but faintly scuffed on one side.",
              "A small printed marking sits just off centre.",
              "The surface catches light in dozens of tiny points.",
              "The shape is perfectly round with a soft shadow pooling beneath."),
    'parachute': ("Fabric panels ripple slightly under steady air pressure.",
              "Stitched seams run in straight lines across the canopy.",
              "The trailing edge flutters in a slow, regular motion.",
              "Suspension lines hang taut and evenly spaced.",
              "The cloth holds a firm curved shape without collapsing."),
}

_V8_SURROUND = {
    'tench': ("Still green water and a soft silted bed fill the background",
              "Reeds and dark water blur softly behind",
              "Wet grass and a muddy bank frame the lower edge of the shot",
              "Flat pond water stretches out of focus behind",
              "A quiet stretch of freshwater lies beyond"),
    'English springer': ("Short mown grass runs away behind into soft green",
              "An open park with distant trees fills the background",
              "A quiet lawn stretches out of focus behind",
              "Low hedges and open ground surround the dog",
              "A wide field recedes softly behind"),
    'cassette player': ("A plain wooden surface and an uncluttered wall sit behind",
              "A tidy shelf and neutral background frame the shot",
              "An empty desk stretches away out of focus",
              "A quiet room with plain walls surrounds it",
              "A simple tabletop with nothing else on it fills the foreground"),
    'chain saw': ("Cut timber and scattered sawdust surround it",
              "A workbench and a bare wall sit behind",
              "Wood chips and bark litter the ground around it",
              "A quiet clearing with felled logs opens behind",
              "A plain wooden surface stretches away beneath"),
    'church': ("Open grass and a low stone wall lie in front",
              "A quiet green hillside rolls away behind",
              "A simple graveyard and a gravel path sit in the foreground",
              "Bare sky and open land frame the building",
              "An empty churchyard stretches out before it"),
    'French horn': ("A dark stage floor and a neutral backdrop sit behind",
              "An empty rehearsal room with plain walls surrounds it",
              "A simple wooden chair and bare boards fill the frame",
              "A quiet auditorium recedes into shadow behind",
              "A plain cloth backdrop hangs behind"),
    'garbage truck': ("A quiet residential street with parked cars lines the background",
              "Low houses and a kerb run along behind",
              "An empty road stretches away out of focus",
              "A suburban street with trimmed verges surrounds it",
              "A wide tarmac road opens up behind"),
    'gas pump': ("A concrete forecourt and a plain canopy surround it",
              "A quiet filling station with empty bays sits behind",
              "A flat paved surface stretches away beneath",
              "A simple wall and a bare kerb sit behind",
              "An empty service area opens up around it"),
    'golf ball': ("Closely mown green grass fills the surrounding frame",
              "A smooth putting surface stretches away out of focus",
              "Short turf runs off softly in every direction",
              "An open fairway recedes into a soft blur behind",
              "Fine cut grass surrounds it on all sides"),
    'parachute': ("Open sky fills the entire background",
              "Thin high cloud drifts far below",
              "Clear blue air surrounds it on every side",
              "Distant ground lies far beneath, soft and out of focus",
              "An empty expanse of sky opens all around"),
}


def _build_v8():
    """v7 with four layers of detail added, in CogVideoX caption style."""
    table = {}
    for concept, cores in PER_CLASS_PROMPT_SETS["v7"].items():
        table[concept] = [
            "%s %s %s. %s, %s. %s" % (
                core,
                _V8_TEXTURE[concept][index % 5],
                _V8_SURROUND[concept][(index + 2) % 5],
                _V8_LIGHT[index % 8],
                _V8_CAMERA[(index + 3) % 8],
                _V8_MOOD[(index + 5) % 8],
            )
            for index, core in enumerate(cores)
        ]
    return table


PER_CLASS_PROMPT_SETS["v8"] = _build_v8()


# Words whose first letter does not determine the article (silent h, "u" as
# in "you").
ARTICLE_EXCEPTIONS = {
    "hour": "an",
    "honest": "an",
    "heir": "an",
    "university": "a",
    "uniform": "a",
    "unicorn": "a",
    "european": "a",
    "one": "a",
}

# Default sampling settings. A run that uses these does not spell them out in
# the directory name; anything else does (see set_tag).
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 288
DEFAULT_GUIDANCE = 6.0
DEFAULT_SEED = 42


def article_for(phrase):
    """Return "a" or "an" for the given phrase."""
    first = phrase.strip().split()[0].lower().strip(".,")
    if first in ARTICLE_EXCEPTIONS:
        return ARTICLE_EXCEPTIONS[first]
    return "an" if first[:1] in "aeiou" else "a"


def render(template, concept):
    """Fill {c} with the class name and {a} with the agreeing article."""
    return template.format(c=concept, concept=concept, a=article_for(concept))


def slugify(text):
    """Directory-safe form of a class name: "gas pump" -> "gas_pump"."""
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower()


MODEL_FAMILIES = ("hunyuan", "cogvideox")
DEFAULT_MODEL_FAMILY = "hunyuan"

# Model defaults, used when the caller does not override them. CogVideoX is
# trained at 720x480; generating at another size is allowed but is outside its
# training distribution, which has to be stated wherever such numbers appear.
MODEL_DEFAULTS = {
    "hunyuan": {"model_path": "hunyuanvideo-community/HunyuanVideo",
                "width": 1280, "height": 720},
    "cogvideox": {"model_path": "THUDM/CogVideoX-2b",
                  "width": 720, "height": 480},
}


def set_tag(class_set, prompt_set, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
            guidance=DEFAULT_GUIDANCE, seed=DEFAULT_SEED,
            model=DEFAULT_MODEL_FAMILY):
    """Directory name for one run configuration.

    Encodes everything that changes the generated videos, so two runs with
    different settings can never land in the same place. Resolution, guidance
    and seed are appended only when they differ from the defaults, to keep the
    common case readable.

        imagenette_v3                     512x288, gs 6, seed 42
        imagenette_v3_1280x720_gs10       everything else spelled out
        cogvideox_imagenette_v7_1280x720  a model other than the default

    The model family is a prefix rather than a suffix, and is omitted for
    HunyuanVideo, so directories generated before other models existed keep
    their names.
    """
    tag = "%s_%s" % (class_set, prompt_set)
    if model != DEFAULT_MODEL_FAMILY:
        tag = "%s_%s" % (model, tag)
    if (width, height) != (DEFAULT_WIDTH, DEFAULT_HEIGHT):
        tag += "_%dx%d" % (width, height)
    if float(guidance) != DEFAULT_GUIDANCE:
        tag += "_gs%s" % str(guidance).rstrip("0").rstrip(".")
    if int(seed) != DEFAULT_SEED:
        tag += "_seed%d" % int(seed)
    return tag


def column_dir(erased_concept):
    """Directory name for one table column. None means the baseline column."""
    if erased_concept is None or erased_concept == "":
        return "unlearn_none"
    return "unlearn_%s" % slugify(erased_concept)


def prompts_for(prompt_set, concept):
    """Return the list of prompts for one class.

    Handles both kinds of set: template sets are rendered with the class name
    and article, per-class sets are returned as written.
    """
    if prompt_set in PER_CLASS_PROMPT_SETS:
        table = PER_CLASS_PROMPT_SETS[prompt_set]
        if concept not in table:
            raise SystemExit(
                "Prompt set %r has no prompts for %r. Per-class sets are "
                "written out by hand and currently cover Imagenette only."
                % (prompt_set, concept))
        return list(table[concept])
    if prompt_set in PROMPT_SETS:
        return [render(t, concept) for t in PROMPT_SETS[prompt_set]]
    raise SystemExit("Unknown prompt set %r. Available: %s"
                     % (prompt_set, ", ".join(all_prompt_sets())))


def all_prompt_sets():
    """Every prompt set name, template and per-class alike."""
    return sorted(set(PROMPT_SETS) | set(PER_CLASS_PROMPT_SETS))


def check_prompt_sets():
    """Validate every set: size, duplicates, verbatim class name."""
    for name, templates in sorted(PROMPT_SETS.items()):
        if len(templates) != 20:
            raise AssertionError("%s: %d templates, expected 20" % (name, len(templates)))

    for name, table in sorted(PER_CLASS_PROMPT_SETS.items()):
        missing = set(CLASS_SETS["imagenette"]) - set(table)
        if missing:
            raise AssertionError("%s: no prompts for %s" % (name, ", ".join(sorted(missing))))
        seen = set()
        for concept, prompts in table.items():
            if len(prompts) != 20:
                raise AssertionError("%s/%s: %d prompts, expected 20"
                                     % (name, concept, len(prompts)))
            if len(set(prompts)) != 20:
                raise AssertionError("%s/%s: duplicate prompts" % (name, concept))
            for prompt in prompts:
                if concept.lower() not in prompt.lower():
                    raise AssertionError("%s/%s: class name missing in %r"
                                         % (name, concept, prompt))
                if prompt in seen:
                    raise AssertionError("%s: prompt used twice: %r" % (name, prompt))
                seen.add(prompt)
    return True


if __name__ == "__main__":
    check_prompt_sets()
    print("prompt sets OK: %s" % ", ".join(all_prompt_sets()))
    for name in sorted(PER_CLASS_PROMPT_SETS):
        print("\n%s (per-class):" % name)
        for concept in CLASS_SETS["imagenette"]:
            lengths = [len(p.split()) for p in PER_CLASS_PROMPT_SETS[name][concept]]
            short = sum(1 for n in lengths if n <= 15)
            print("  %18s | words %2d-%2d avg %4.1f | short %2d long %2d"
                  % (concept, min(lengths), max(lengths),
                     sum(lengths) / len(lengths), short, 20 - short))
