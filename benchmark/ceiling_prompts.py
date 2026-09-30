"""Ceiling prompt sets for the near2 mapping targets.

CLIP-tgt has no scale on its own. The floor is what the base model scores
against the target text when prompted for the source, which clip_score.csv
already stores as clip_tgt_base. The ceiling is what it scores when asked for
the target directly, and there is no single honest way to ask:

  A  the v7 prompt with the noun swapped and nothing else. Contradictions stay
     in on purpose, because the method also only ever sees a prompt about the
     source.
  B  the same scene with the contradicted attributes repaired and nothing else
     touched.
  C  a plain one-line prompt on twenty seeds, no scene at all.

Index i of every set corresponds to index i of the v7 prompts for the source
class, so the comparison is paired scene by scene.
"""

SEEDS = [42 + i for i in range(20)]

A = {}
B = {}
C = {}

# ---------------------------------------------------------------- goldfish
C["a goldfish"] = "A video of a goldfish."
A["a goldfish"] = [
 "A goldfish with a thick dark olive body and a small red eye, held in both hands by an angler.",
 "Close-up of a goldfish, its heavy rounded body and small red eye filling the frame in daylight.",
 "An angler kneels on a grassy bank and holds a goldfish toward the camera in bright daylight.",
 "A goldfish lying on short green grass, its deep bronze-green body and rounded fins clearly visible.",
 "A heavy-bodied goldfish held at chest height in two hands against a plain grassy bank.",
 "Daylight close-up of a goldfish, its dark green scales and rounded tail fin in sharp focus.",
 "A goldfish resting on green grass beside a lake, the whole thick-bodied fish in frame.",
 "An angler holds a large goldfish with both hands, its blunt rounded head toward the camera.",
 "A goldfish with small dark scales and a small red eye lies on a flat green surface.",
 "Close-up of the rounded fins and thick body of a goldfish in clear even daylight.",
 "A goldfish held up after being caught, its heavy dark olive body filling most of the frame.",
 "A goldfish lies on short grass while the camera holds steady on its rounded body and fins.",
 "A fisherman lifts a thick dark goldfish toward the camera under a bright overcast sky.",
 "A goldfish with a deep body and a rounded tail rests on green grass in bright daylight.",
 "Close-up of the head of a goldfish, its small red eye and blunt snout sharply lit.",
 "A large goldfish is held in two hands, its dark bronze-green body clearly visible in daylight.",
 "A goldfish lying on its side on green grass, its thick body and small scales in sharp focus.",
 "A goldfish with rounded fins and a heavy body fills the frame against plain green grass.",
 "An angler presents a heavy goldfish to the camera, its dark olive body in bright daylight.",
 "A goldfish rests on a green bank, its thick rounded body and red eye clearly visible.",
]
B["a goldfish"] = [
 "A goldfish with a bright orange body and flowing fins, held in both hands by an angler.",
 "Close-up of a goldfish, its rounded orange body and flowing tail filling the frame in daylight.",
 "An angler kneels on a grassy bank and holds a goldfish toward the camera in bright daylight.",
 "A goldfish lying on short green grass, its bright orange body and flowing fins clearly visible.",
 "A round-bodied goldfish held at chest height in two hands against a plain grassy bank.",
 "Daylight close-up of a goldfish, its orange scales and flowing tail fin in sharp focus.",
 "A goldfish resting on green grass beside a lake, the whole small orange fish in frame.",
 "An angler holds a large goldfish with both hands, its rounded head toward the camera.",
 "A goldfish with fine orange scales and a small dark eye lies on a flat green surface.",
 "Close-up of the flowing fins and rounded body of a goldfish in clear even daylight.",
 "A goldfish held up after being caught, its bright orange body filling most of the frame.",
 "A goldfish lies on short grass while the camera holds steady on its rounded body and fins.",
 "A fisherman lifts a bright orange goldfish toward the camera under a bright overcast sky.",
 "A goldfish with a rounded body and a flowing tail rests on green grass in bright daylight.",
 "Close-up of the head of a goldfish, its small dark eye and blunt snout sharply lit.",
 "A large goldfish is held in two hands, its bright orange body clearly visible in daylight.",
 "A goldfish lying on its side on green grass, its rounded body and fine scales in sharp focus.",
 "A goldfish with flowing fins and a rounded body fills the frame against plain green grass.",
 "An angler presents a goldfish to the camera, its bright orange body in bright daylight.",
 "A goldfish rests on a green bank, its rounded orange body and dark eye clearly visible.",
]

# --------------------------------------------------------- german shepherd
C["a german shepherd"] = "A video of a german shepherd."
A["a german shepherd"] = [
 "A german shepherd standing in short grass, its long brown ears hanging beside its face.",
 "A brown and white german shepherd stands side-on in a field in bright daylight.",
 "Close-up of a german shepherd, its long drooping ears framing its face.",
 "A medium-sized german shepherd with a wavy brown and white coat stands on a lawn.",
 "A german shepherd sits on grass, its long feathered ears hanging low.",
 "A german shepherd with brown patches on white fur stands still in daylight.",
 "A german shepherd gundog stands in a field, its whole body in frame.",
 "A german shepherd looks toward the camera, its long ears covered in wavy brown fur.",
 "A brown and white german shepherd walks slowly across a lawn in clear daylight.",
 "A german shepherd stands in short grass, its docked tail and feathered legs visible.",
 "Daylight portrait of a german shepherd, its long brown ears hanging past its jaw.",
 "A german shepherd with a white chest and a brown back stands on a grassy field.",
 "A german shepherd sits upright on a lawn, its long ears hanging down either side.",
 "A wavy-coated german shepherd stands in profile against plain green grass.",
 "A german shepherd with brown and white markings stands still in bright daylight.",
 "Close-up of the head of a german shepherd, long ears and soft brown eyes in focus.",
 "A german shepherd stands on a lawn, its brown ears lifting slightly in the breeze.",
 "A medium-sized german shepherd with long drooping ears sits in a green field.",
 "A german shepherd trots toward the camera across short grass in daylight.",
 "A german shepherd stands centred in the frame, its brown and white coat clearly lit.",
]
B["a german shepherd"] = [
 "A german shepherd standing in short grass, its erect pointed ears turned forward.",
 "A black and tan german shepherd stands side-on in a field in bright daylight.",
 "Close-up of a german shepherd, its upright pointed ears framing its face.",
 "A large german shepherd with a dense black and tan coat stands on a lawn.",
 "A german shepherd sits on grass, its tall pointed ears standing upright.",
 "A german shepherd with black and tan fur stands still in daylight.",
 "A german shepherd working dog stands in a field, its whole body in frame.",
 "A german shepherd looks toward the camera, its pointed ears covered in short dark fur.",
 "A black and tan german shepherd walks slowly across a lawn in clear daylight.",
 "A german shepherd stands in short grass, its bushy tail and straight legs visible.",
 "Daylight portrait of a german shepherd, its pointed ears standing above its head.",
 "A german shepherd with a tan chest and a black back stands on a grassy field.",
 "A german shepherd sits upright on a lawn, its pointed ears upright on either side.",
 "A dense-coated german shepherd stands in profile against plain green grass.",
 "A german shepherd with black and tan markings stands still in bright daylight.",
 "Close-up of the head of a german shepherd, pointed ears and dark brown eyes in focus.",
 "A german shepherd stands on a lawn, its pointed ears turning toward a sound.",
 "A large german shepherd with tall pointed ears sits in a green field.",
 "A german shepherd trots toward the camera across short grass in daylight.",
 "A german shepherd stands centred in the frame, its black and tan coat clearly lit.",
]

# ----------------------------------------------------------- record player
C["a record player"] = "A video of a record player."
A["a record player"] = [
 "A portable record player with two speakers and a carrying handle, standing on a table.",
 "A boombox-style record player with a radio dial and speaker grilles, seen from the front.",
 "A large portable record player sits on a shelf, its whole body in frame in daylight.",
 "A silver hi-fi record player with rows of buttons, filmed straight on.",
 "A record player with two round speakers on either side of its front panel.",
 "A record player unit in a stack of stereo equipment, its front panel facing the camera.",
 "A boxy portable record player with a handle on top, standing on a wooden floor.",
 "A black record player with knobs and a radio dial, centred against a plain wall.",
 "A record player with speaker grilles and a row of buttons, evenly lit in daylight.",
 "A portable record player stands upright on a table, the whole device sharply in focus.",
 "A record player with an antenna and two speakers, seen from the front in daylight.",
 "The camera holds steady on a record player, its dials and speaker grilles clearly visible.",
 "A stereo record player on a shelf, its full width in frame against a plain background.",
 "A vintage portable record player with chrome trim, standing on a wooden table.",
 "A record player with a handle and two speakers rests on the ground in bright daylight.",
 "A wide record player with a row of buttons along its front, filmed straight on.",
 "A record player sits alone on a plain surface, its whole body evenly lit.",
 "A portable record player with a lit radio dial, centred in the frame.",
 "A record player with two speakers stands on a desk as the camera slowly moves closer.",
 "A boxy record player with knobs and grilles fills the frame in even daylight.",
]
B["a record player"] = [
 "A portable record player with a turntable platter and a tonearm, standing on a table.",
 "A wooden record player with a spinning platter and a tonearm, seen from the front.",
 "A large portable record player sits on a shelf, its whole body in frame in daylight.",
 "A silver hi-fi record player with a platter and a tonearm, filmed straight on.",
 "A record player with a round platter and a tonearm on its top panel.",
 "A record player unit in a stack of stereo equipment, its front panel facing the camera.",
 "A boxy portable record player with a lid on top, standing on a wooden floor.",
 "A black record player with knobs and a spinning platter, centred against a plain wall.",
 "A record player with a platter and a row of buttons, evenly lit in daylight.",
 "A portable record player stands closed on a table, the whole device sharply in focus.",
 "A record player with a lid and a tonearm, seen from the front in daylight.",
 "The camera holds steady on a record player, its dials and tonearm clearly visible.",
 "A stereo record player on a shelf, its full width in frame against a plain background.",
 "A vintage portable record player with chrome trim, standing on a wooden table.",
 "A record player with a lid and a tonearm rests on the ground in bright daylight.",
 "A wide record player with a row of buttons along its front, filmed straight on.",
 "A record player sits alone on a plain surface, its whole body evenly lit.",
 "A portable record player with a lit dial, centred in the frame.",
 "A record player with a lifted tonearm stands on a desk as the camera slowly moves closer.",
 "A boxy record player with knobs and a platter fills the frame in even daylight.",
]

# --------------------------------------------------------------------- axe
C["an axe"] = "A video of an axe."
A["an axe"] = [
 "An axe resting on a cut log, its bar and chain clearly visible in daylight.",
 "Close-up of an axe held by its handle, the guide bar filling the frame.",
 "An axe cutting into a thick log, sawdust spraying out to one side.",
 "An axe lying on the ground beside freshly cut timber, the whole tool in frame.",
 "Someone holds an axe level with the camera, its bar and chain sharply in focus.",
 "An axe sits on a wooden bench, filmed straight on in even daylight.",
 "Close-up of the chain and guide bar of an axe in bright daylight.",
 "An axe is lifted and held steady, the whole tool visible against a plain background.",
 "An axe cutting through a log outdoors, its bar buried in the wood.",
 "Static shot of an axe resting on sawdust-covered ground with the whole tool in frame.",
 "An axe with an orange body lies on cut timber, clearly lit from above.",
 "The camera moves slowly along an axe from its handle to the tip of the bar.",
 "An axe held in both hands with the engine running, the chain spinning on the bar.",
 "An axe propped against a stack of logs, the whole tool sharp in daylight.",
 "Close-up of an axe being started, the operator gripping the rear handle.",
 "An axe resting on a tree stump, filmed from the side in clear daylight.",
 "An axe cuts a log in half outdoors, the tool centred in the frame.",
 "An axe lying flat on grass with its bar and chain facing the camera.",
 "Someone carries an axe through a clearing, the tool clearly visible at their side.",
 "An axe held up toward the camera, its bar and chain filling most of the frame.",
]
B["an axe"] = [
 "An axe resting on a cut log, its steel head and wooden handle clearly visible in daylight.",
 "Close-up of an axe held by its handle, the steel head filling the frame.",
 "An axe biting into a thick log, wood chips flying out to one side.",
 "An axe lying on the ground beside freshly cut timber, the whole tool in frame.",
 "Someone holds an axe level with the camera, its head and handle sharply in focus.",
 "An axe sits on a wooden bench, filmed straight on in even daylight.",
 "Close-up of the steel head and wooden handle of an axe in bright daylight.",
 "An axe is lifted and held steady, the whole tool visible against a plain background.",
 "An axe cutting through a log outdoors, its head buried in the wood.",
 "Static shot of an axe resting on wood-chip-covered ground with the whole tool in frame.",
 "An axe with a steel head lies on cut timber, clearly lit from above.",
 "The camera moves slowly along an axe from its handle to the edge of its blade.",
 "An axe held in both hands ready to swing, its blade catching the light.",
 "An axe propped against a stack of logs, the whole tool sharp in daylight.",
 "Close-up of an axe being raised, the operator gripping the handle.",
 "An axe resting on a tree stump, filmed from the side in clear daylight.",
 "An axe cuts a log in half outdoors, the tool centred in the frame.",
 "An axe lying flat on grass with its head and handle facing the camera.",
 "Someone carries an axe through a clearing, the tool clearly visible at their side.",
 "An axe held up toward the camera, its head and handle filling most of the frame.",
]

# -------------------------------------------------------------------- barn
C["a barn"] = "A video of a barn."
A["a barn"] = [
 "A stone barn with a tall steeple, the whole building centred in bright daylight.",
 "The front facade of a barn with an arched doorway and a bell tower above it.",
 "A small country barn with a pointed spire, filmed from the front in clear weather.",
 "A barn exterior in daylight, its steeple and tall windows clearly visible.",
 "A white wooden barn with a tall steeple standing against a plain blue sky.",
 "The camera holds steady on a barn building with the whole facade in frame.",
 "A brick barn with a square bell tower, seen from across an open lawn.",
 "A barn with a steep roof and arched windows, filmed straight on in daylight.",
 "A stone barn seen from the front, its spire rising above the entrance.",
 "A barn exterior filmed slowly from the side, the full building in view.",
 "A village barn with a bell tower, centred in the frame under a clear sky.",
 "The steeple and roof of a barn, filmed from ground level in bright daylight.",
 "A barn facade with tall arched windows, evenly lit and sharply in focus.",
 "A grey stone barn stands alone in daylight with the whole building visible.",
 "The camera moves slowly toward the entrance of a barn with a tall spire.",
 "A barn with a cross on its steeple, filmed from the front in clear weather.",
 "A large barn exterior in daylight, its towers and windows clearly visible.",
 "A country barn with white walls and a dark roof, the whole building in frame.",
 "A barn seen from a short distance, its steeple centred against an open sky.",
 "The front of an old stone barn, its arched door and bell tower sharp in daylight.",
]
B["a barn"] = [
 "A wooden barn with a tall gambrel roof, the whole building centred in bright daylight.",
 "The front facade of a barn with a wide doorway and a hayloft opening above it.",
 "A small country barn with a pitched roof, filmed from the front in clear weather.",
 "A barn exterior in daylight, its wide doors and tall roof clearly visible.",
 "A red wooden barn with a tall pitched roof standing against a plain blue sky.",
 "The camera holds steady on a barn building with the whole facade in frame.",
 "A brick barn with a square hayloft, seen from across an open lawn.",
 "A barn with a steep roof and square windows, filmed straight on in daylight.",
 "A stone barn seen from the front, its roof rising above the entrance.",
 "A barn exterior filmed slowly from the side, the full building in view.",
 "A village barn with a hayloft, centred in the frame under a clear sky.",
 "The roof and gable of a barn, filmed from ground level in bright daylight.",
 "A barn facade with tall square windows, evenly lit and sharply in focus.",
 "A grey stone barn stands alone in daylight with the whole building visible.",
 "The camera moves slowly toward the entrance of a barn with a tall pitched roof.",
 "A barn with a weathervane on its roof, filmed from the front in clear weather.",
 "A large barn exterior in daylight, its doors and windows clearly visible.",
 "A country barn with red walls and a dark roof, the whole building in frame.",
 "A barn seen from a short distance, its roof centred against an open sky.",
 "The front of an old stone barn, its wide door and hayloft sharp in daylight.",
]

# --------------------------------------------------------------- accordion
C["an accordion"] = "A video of an accordion."
A["an accordion"] = [
 "An accordion resting on a table, its coiled brass tubing facing the camera.",
 "Close-up of an accordion, the wide bell and coiled tubing filling the frame.",
 "A musician holds an accordion ready to play, the instrument clearly visible.",
 "An accordion on a plain surface in even daylight, the whole instrument in frame.",
 "The camera moves slowly around an accordion, its brass coils catching the light.",
 "An accordion held up toward the camera, its bell and valves sharply in focus.",
 "Close-up of the valves and coiled tubing of an accordion in bright light.",
 "An accordion sits upright on a stand, the whole instrument centred in the frame.",
 "A player raises an accordion to their lips, the brass coils clearly visible.",
 "A polished accordion against a plain background, evenly lit and sharp.",
 "An accordion lying on a dark cloth, its wide bell turned toward the camera.",
 "Static shot of an accordion, its coiled tubing and flared bell in full view.",
 "An accordion held in both hands, the instrument filling most of the frame.",
 "Close-up of the flared bell of an accordion, the brass reflecting daylight.",
 "An accordion resting on a chair, the whole instrument visible in clear daylight.",
 "The camera holds steady on an accordion, its coils and valves in sharp focus.",
 "A brass accordion on a plain table, seen from the side in even light.",
 "A musician lowers an accordion after playing, the instrument clearly in frame.",
 "An accordion filmed close up, its coiled tubing dominating the frame.",
 "An accordion stands on a stand in daylight, the whole instrument sharply lit.",
]
B["an accordion"] = [
 "An accordion resting on a table, its folded bellows facing the camera.",
 "Close-up of an accordion, the black bellows and white keys filling the frame.",
 "A musician holds an accordion ready to play, the instrument clearly visible.",
 "An accordion on a plain surface in even daylight, the whole instrument in frame.",
 "The camera moves slowly around an accordion, its pearl buttons catching the light.",
 "An accordion held up toward the camera, its keys and bellows sharply in focus.",
 "Close-up of the keys and bellows of an accordion in bright light.",
 "An accordion sits upright on a stand, the whole instrument centred in the frame.",
 "A player lifts an accordion into playing position, the folded bellows clearly visible.",
 "A polished accordion against a plain background, evenly lit and sharp.",
 "An accordion lying on a dark cloth, its keyboard turned toward the camera.",
 "Static shot of an accordion, its bellows and keyboard in full view.",
 "An accordion held in both hands, the instrument filling most of the frame.",
 "Close-up of the keyboard of an accordion, the white keys reflecting daylight.",
 "An accordion resting on a chair, the whole instrument visible in clear daylight.",
 "The camera holds steady on an accordion, its buttons and bellows in sharp focus.",
 "A red accordion on a plain table, seen from the side in even light.",
 "A musician lowers an accordion after playing, the instrument clearly in frame.",
 "An accordion filmed close up, its folded bellows dominating the frame.",
 "An accordion stands on a stand in daylight, the whole instrument sharply lit.",
]

# ------------------------------------------------------------- fire engine
C["a fire engine"] = "A video of a fire engine."
A["a fire engine"] = [
 "A fire engine seen from the side on a residential street in bright daylight.",
 "A fire engine with its rear loader raised, the whole vehicle in frame.",
 "A fire engine stops at the kerb, filmed from the side with the full vehicle visible.",
 "The rear of a fire engine as a bin is lifted and emptied into the hopper.",
 "A fire engine drives slowly down a street, the camera holding it in full view.",
 "A three-quarter view of a fire engine parked on a road in clear daylight.",
 "A fire engine with a green body and a large rear hopper, seen from the side.",
 "A fire engine idles at the kerb while its lifting arm raises a wheeled bin.",
 "The camera holds steady on a fire engine, its full length visible in daylight.",
 "A fire engine seen from behind, its loading hopper and controls clearly visible.",
 "A white fire engine moves along a suburban street with the whole vehicle in frame.",
 "A fire engine parked on a wide road, filmed from the side in bright sunlight.",
 "A fire engine compacts its load, the rear hopper closing slowly.",
 "A fire engine viewed side-on, its cab and rear body both fully in frame.",
 "A fire engine pulls away from the kerb, the camera following it from the side.",
 "The lifting arm of a fire engine raises a bin above the hopper in daylight.",
 "A fire engine stands still on an empty street, the whole vehicle sharply lit.",
 "A fire engine with warning stripes along its side, filmed from a short distance.",
 "A fire engine seen from the rear as waste is loaded into the hopper.",
 "A large fire engine fills the frame, seen side-on in clear daylight.",
]
B["a fire engine"] = [
 "A fire engine seen from the side on a residential street in bright daylight.",
 "A fire engine with its ladder raised, the whole vehicle in frame.",
 "A fire engine stops at the kerb, filmed from the side with the full vehicle visible.",
 "The rear of a fire engine as a hose reel is unwound from its locker.",
 "A fire engine drives slowly down a street, the camera holding it in full view.",
 "A three-quarter view of a fire engine parked on a road in clear daylight.",
 "A fire engine with a red body and a large rear locker, seen from the side.",
 "A fire engine idles at the kerb while its ladder rises above the cab.",
 "The camera holds steady on a fire engine, its full length visible in daylight.",
 "A fire engine seen from behind, its hose lockers and controls clearly visible.",
 "A red fire engine moves along a suburban street with the whole vehicle in frame.",
 "A fire engine parked on a wide road, filmed from the side in bright sunlight.",
 "A fire engine extends its ladder, the sections sliding out slowly.",
 "A fire engine viewed side-on, its cab and rear body both fully in frame.",
 "A fire engine pulls away from the kerb, the camera following it from the side.",
 "The ladder of a fire engine rises above the cab in daylight.",
 "A fire engine stands still on an empty street, the whole vehicle sharply lit.",
 "A fire engine with reflective stripes along its side, filmed from a short distance.",
 "A fire engine seen from the rear as equipment is unloaded from its lockers.",
 "A large fire engine fills the frame, seen side-on in clear daylight.",
]

# --------------------------------------------------------- vending machine
C["a vending machine"] = "A video of a vending machine."
A["a vending machine"] = [
 "A vending machine at a filling station, the whole unit centred in daylight.",
 "Close-up of a vending machine, its display and nozzle clearly visible.",
 "A vending machine standing under a station canopy, filmed straight on.",
 "A row of vending machines on a forecourt, the nearest one filling the frame.",
 "A vending machine with the nozzle in its holster, seen from the front in bright daylight.",
 "A hand lifts the nozzle from a vending machine, the unit clearly in frame.",
 "Static shot of a vending machine, its screen, buttons and hose sharply in focus.",
 "A vending machine on an empty forecourt, the whole unit visible in even daylight.",
 "Close-up of the display of a vending machine as the numbers count upward.",
 "A red and white vending machine stands alone, centred against a plain background.",
 "The camera moves slowly around a vending machine at a quiet filling station.",
 "A vending machine with a coiled hose hanging at its side, seen from the front.",
 "A vending machine under bright daylight, its whole body sharp and well lit.",
 "A nozzle is returned to a vending machine, the unit filling most of the frame.",
 "Two vending machines side by side on a forecourt, the nearer one in sharp focus.",
 "A vending machine seen from a low angle, its display and panel clearly visible.",
 "A vending machine stands under a canopy, the whole unit centred in the frame.",
 "Close-up of the hose and nozzle of a vending machine in clear daylight.",
 "A tall vending machine on a concrete forecourt, filmed straight on in daylight.",
 "A vending machine with a digital display, the whole unit sharply lit and centred.",
]
B["a vending machine"] = [
 "A vending machine in a station lobby, the whole unit centred in daylight.",
 "Close-up of a vending machine, its display and glass front clearly visible.",
 "A vending machine standing against a corridor wall, filmed straight on.",
 "A row of vending machines along a wall, the nearest one filling the frame.",
 "A vending machine with snacks behind its glass front, seen from the front in bright daylight.",
 "A hand presses a button on a vending machine, the unit clearly in frame.",
 "Static shot of a vending machine, its screen, buttons and glass front sharply in focus.",
 "A vending machine in an empty corridor, the whole unit visible in even daylight.",
 "Close-up of the display of a vending machine as the numbers change.",
 "A red and white vending machine stands alone, centred against a plain background.",
 "The camera moves slowly around a vending machine in a quiet corridor.",
 "A vending machine with a coin slot on its side, seen from the front.",
 "A vending machine under bright daylight, its whole body sharp and well lit.",
 "A can drops into the tray of a vending machine, the unit filling most of the frame.",
 "Two vending machines side by side against a wall, the nearer one in sharp focus.",
 "A vending machine seen from a low angle, its display and panel clearly visible.",
 "A vending machine stands against a wall, the whole unit centred in the frame.",
 "Close-up of the coin slot and delivery tray of a vending machine in clear daylight.",
 "A tall vending machine on a concrete floor, filmed straight on in daylight.",
 "A vending machine with a digital display, the whole unit sharply lit and centred.",
]

# -------------------------------------------------------------- basketball
C["a basketball"] = "A video of a basketball."
A["a basketball"] = [
 "A basketball sitting on a tee, filling the frame in bright daylight.",
 "Close-up of a white basketball resting on short green grass.",
 "A basketball on a tee, its dimpled surface sharply in focus.",
 "A basketball rolls slowly across a putting green toward the camera.",
 "Macro shot of a basketball, the dimples covering its white surface.",
 "A basketball rests beside the hole on a green, filmed from close range.",
 "A basketball sits on the grass, centred and sharply lit in daylight.",
 "Close-up of a basketball on a wooden tee against a plain green background.",
 "A basketball drops onto the green and settles, the camera holding on it.",
 "A single basketball on short grass, the whole ball filling most of the frame.",
 "A basketball on a tee seen from ground level in bright sunlight.",
 "The camera moves slowly around a basketball resting on a putting green.",
 "A white basketball with clear dimples lies on a flat green surface.",
 "A basketball is placed on a tee by a gloved hand, which then withdraws.",
 "Close-up of a basketball rolling and stopping on smooth green grass.",
 "A basketball sits still on the fairway, sharply lit and centred in the frame.",
 "A basketball on a tee, its dimpled surface catching bright daylight.",
 "Static shot of a basketball on short grass with nothing else in the frame.",
 "A basketball rests on a green, filmed close up from a low angle.",
 "A dimpled white basketball fills the frame, resting on flat green grass.",
]
B["a basketball"] = [
 "An orange basketball sitting on a tee, filling the frame in bright daylight.",
 "Close-up of an orange basketball resting on short green grass.",
 "A basketball on a tee, its pebbled surface sharply in focus.",
 "A basketball rolls slowly across a putting green toward the camera.",
 "Macro shot of a basketball, the pebbled grain covering its orange surface.",
 "A basketball rests beside the hole on a green, filmed from close range.",
 "A basketball sits on the grass, centred and sharply lit in daylight.",
 "Close-up of a basketball on a wooden tee against a plain green background.",
 "A basketball drops onto the green and settles, the camera holding on it.",
 "A single basketball on short grass, the whole ball filling most of the frame.",
 "A basketball on a tee seen from ground level in bright sunlight.",
 "The camera moves slowly around a basketball resting on a putting green.",
 "An orange basketball with clear black seams lies on a flat green surface.",
 "A basketball is placed on a tee by a gloved hand, which then withdraws.",
 "Close-up of a basketball rolling and stopping on smooth green grass.",
 "A basketball sits still on the fairway, sharply lit and centred in the frame.",
 "A basketball on a tee, its pebbled surface catching bright daylight.",
 "Static shot of a basketball on short grass with nothing else in the frame.",
 "A basketball rests on a green, filmed close up from a low angle.",
 "A pebbled orange basketball fills the frame, resting on flat green grass.",
]

# ---------------------------------------------------------- hot air balloon
C["a hot air balloon"] = "A video of a hot air balloon."
A["a hot air balloon"] = [
 "An open hot air balloon descending against a clear blue sky, the canopy filling the frame.",
 "A hot air balloon canopy fully inflated in the air, seen from below.",
 "A skydiver descends beneath an open hot air balloon, the canopy clearly visible above.",
 "A colourful hot air balloon drifts slowly down through a bright open sky.",
 "A hot air balloon opens above a skydiver, the canopy spreading wide against the sky.",
 "An open hot air balloon seen from below, its lines running down from the canopy.",
 "A hot air balloon glides across a clear sky with the whole canopy in frame.",
 "A red and white hot air balloon descends steadily, filmed from the ground.",
 "The camera follows a hot air balloon as it turns slowly in bright daylight.",
 "A hot air balloon canopy billows overhead, sunlight showing through the fabric.",
 "A skydiver hangs beneath a rectangular hot air balloon high above open ground.",
 "An open hot air balloon centred against a plain blue sky, sharply in focus.",
 "A hot air balloon drifts downward, its canopy fully inflated and clearly lit.",
 "A hot air balloon descends toward a field, the canopy large in the frame.",
 "A wide hot air balloon canopy seen from below against bright daylight.",
 "A hot air balloon turns gently in the air with the whole canopy visible throughout.",
 "A striped hot air balloon descends through a clear sky, filmed from below.",
 "An open hot air balloon fills most of the frame as it drifts across the sky.",
 "A hot air balloon carries a skydiver slowly downward under a bright open sky.",
 "A fully open hot air balloon against a clear sky, its canopy and lines sharply visible.",
]
B["a hot air balloon"] = [
 "A hot air balloon descending against a clear blue sky, the envelope filling the frame.",
 "A hot air balloon envelope fully inflated in the air, seen from below.",
 "A basket hangs beneath a hot air balloon, the envelope clearly visible above.",
 "A colourful hot air balloon drifts slowly through a bright open sky.",
 "A hot air balloon rises above a field, its envelope spreading wide against the sky.",
 "A hot air balloon seen from below, its cables running down from the envelope.",
 "A hot air balloon drifts across a clear sky with the whole envelope in frame.",
 "A red and white hot air balloon drifts steadily, filmed from the ground.",
 "The camera follows a hot air balloon as it turns slowly in bright daylight.",
 "A hot air balloon envelope billows overhead, sunlight showing through the fabric.",
 "A wicker basket hangs beneath a striped hot air balloon high above open ground.",
 "A hot air balloon centred against a plain blue sky, sharply in focus.",
 "A hot air balloon drifts along, its envelope fully inflated and clearly lit.",
 "A hot air balloon descends toward a field, the envelope large in the frame.",
 "A wide hot air balloon envelope seen from below against bright daylight.",
 "A hot air balloon turns gently in the air with the whole envelope visible throughout.",
 "A striped hot air balloon drifts through a clear sky, filmed from below.",
 "A hot air balloon fills most of the frame as it drifts across the sky.",
 "A hot air balloon carries a wicker basket slowly across a bright open sky.",
 "A hot air balloon against a clear sky, its envelope and cables sharply visible.",
]

# ------------------------------------------------------------------ far set
# Targets of the far mapping, plus the two that replaced a tench target after
# the per-class ablation.
#
# B is deliberately absent for these. B repairs the attributes the swap
# contradicts while leaving the scene alone, and for a wooden box standing in a
# tench's scene there is nothing to repair -- the scene itself has to go, at
# which point it is no longer the same scene and B loses its meaning.
#
# A, on the other hand, is defined as the v7 prompt with the noun swapped and
# nothing else, contradictions left in on purpose. That is a substitution, not
# a piece of writing, so it is generated here rather than typed out: the ten
# hand-written A sets above are exactly what this produces for the near2
# targets. Doing it in code also guarantees index i keeps pointing at scene i
# of the source class, which is what makes the comparison paired.
import os as _os  # noqa: E402
import re as _re  # noqa: E402
import sys as _sys  # noqa: E402

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import PER_CLASS_PROMPT_SETS  # noqa: E402

# Which class's scenes each far target is dropped into. Both fish substitutes
# come from tench, since that is the column they replace a target in.
FAR_SOURCE = {
    "a shark": "tench",
    "a crocodile": "tench",
    "a wooden log": "tench",
    "a cat": "English springer",
    "a wooden box": "cassette player",
    "a wooden bat": "chain saw",
    "a skyscraper": "church",
    "a hand fan": "French horn",
    "a ferrari": "garbage truck",
    "a wooden closet": "gas pump",
    "a rubber duck": "golf ball",
    "an airplane": "parachute",
}


# The v7 prompts name one class by a longer phrase than the class key. The
# whole phrase is what gets swapped, exactly as the hand-written german
# shepherd set does -- leaving "spaniel" behind would give "a cat spaniel".
# Nouns that are not part of the class name stay in: "the whole thick-bodied
# fish in frame" survives the swap on purpose, and so does the hand-written
# goldfish set.
SOURCE_PHRASE = {"English springer": "English springer spaniel"}


def _plural(noun):
    """Enough English to pluralise the target nouns used here."""
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _swap_noun(prompt, source, target):
    """One v7 sentence with the source noun replaced by the target noun.

    A few scenes show more than one of the object ("Two gas pumps side by
    side"), so the plural is matched and carried over as well.

    Only an article standing directly in front of the singular noun is
    corrected. An article separated from it by an adjective agrees with the
    adjective, which the swap does not touch, so "A heavy-bodied tench" stays
    "A heavy-bodied <noun>" and is left alone.
    """
    article, noun = target.split(" ", 1)
    phrase = SOURCE_PHRASE.get(source, source)
    out = _re.sub(r"\b%s(s)?\b" % _re.escape(phrase),
                  lambda m: _plural(noun) if m.group(1) else noun, prompt)
    out = _re.sub(
        r"\b([Aa])n? (%s)\b" % _re.escape(noun),
        lambda m: (article.capitalize() if m.group(1) == "A" else article)
        + " " + m.group(2),
        out,
    )
    return out


# Near-type targets added after the ten hand-written sets above. They get A and
# C the same mechanical way; B is missing because B is a hand repair of the
# contradictions the swap leaves behind, and nobody wrote one.
#
# "a trumpet" is here because the French horn column of the near row uses it
# rather than "an accordion": a trumpet is the closer instrument, and that is
# the mapping the row was actually trained with.
EXTRA_SOURCE = {
    "a trumpet": "French horn",
}

for _target, _source in dict(FAR_SOURCE, **EXTRA_SOURCE).items():
    C[_target] = "A video of %s." % _target
    A[_target] = [_swap_noun(p, _source, _target)
                  for p in PER_CLASS_PROMPT_SETS["v7"][_source]]

TARGETS = sorted(C)
for target in TARGETS:
    if target in A:
        assert len(A[target]) == 20, (target, "A", len(A[target]))
    if target in B:
        assert len(B[target]) == 20, (target, "B", len(B[target]))
    assert target in C, target
