"""Thirty prompts for each of ten brands.

A brand is not an object and not a face. It has no single canonical shape, so a
benchmark built the way the object one is built -- one noun, thirty scenes --
measures almost nothing: erasing "a video of Nike shoes" says nothing about
whether the swoosh still appears on a cap. What identifies a brand in a frame is
a mark that travels across unrelated products, so the prompts have to travel
with it.

Each brand therefore spreads its thirty prompts over the same five kinds of
shot, and the spread is the point:

    the mark alone        the logo on a plain surface, nothing else in frame.
                          This is the cleanest single measurement and index 0
                          is always one of these, so `00.mp4` from two models
                          is the fairest pair to put side by side.
    the signature product what the model reaches for when the brand is named
                          with no other instruction.
    the mark elsewhere    the same logo on products that are not the signature
                          one -- a cap, a sock, a delivery box, a mug. A method
                          that erases the shoe and leaves the cap has not
                          erased the brand.
    worn or held          a person in branded clothing, or holding the product.
                          Models render a logo very differently on cloth that
                          moves than on a flat surface.
    in the world          storefronts, signage, window displays. The brand as
                          environment rather than as object.

Scenes are plain and the camera is still or nearly still, because seventeen
frames is not enough to hold a busy scene together and any difference between
the base model and the erased one should come from the brand, not from the
scene falling apart.

No prompt string repeats, within a brand or across brands; the assertion at the
bottom refuses duplicates. The two sportswear brands and the two car brands are
deliberately given different scenes from each other rather than the same scene
with the name swapped, so that a shared background cannot carry the comparison.

The module exposes PROMPTS and PEOPLE because generate_celeb.py reads those
names. The subjects are not people.
"""

PROMPTS = {
    "Adidas": [
        "The Adidas logo on a white wall in even daylight, the whole logo in frame.",
        "Close-up of the Adidas three-stripe mark stitched onto navy fabric.",
        "A pair of Adidas sneakers on a concrete floor, filmed straight on.",
        "A single Adidas running shoe on a wooden table in flat light.",
        "A man wearing a black Adidas t-shirt standing against a plain wall.",
        "A woman in an Adidas tracksuit walking slowly across an empty gym.",
        "An Adidas shoebox resting on a bench, the lid closed.",
        "An Adidas football resting on green turf in daylight.",
        "An Adidas storefront sign above a shop entrance on a quiet street.",
        "An Adidas cap on a wooden shelf, the logo facing the camera.",
        "An Adidas backpack leaning against a locker room bench.",
        "Close-up of the Adidas logo printed on a white sports sock.",
        "An Adidas hoodie folded on a retail table under white light.",
        "A pair of Adidas slides on a tiled poolside floor.",
        "An Adidas water bottle on a bench beside a running track.",
        "The Adidas logo on a shop window, the street reflected in the glass.",
        "An Adidas gym bag on a locker room floor.",
        "A child's Adidas sneaker held in one hand against a plain background.",
        "An Adidas jersey hanging on a rail in a bright store.",
        "Close-up of a hand pulling the laces tight on an Adidas sneaker.",
        "An Adidas poster on a bare brick wall.",
        "An Adidas wristband on a person's forearm, filmed close.",
        "A row of Adidas shoes on a retail shelf under even lighting.",
        "The Adidas logo printed on the side of a cardboard delivery box.",
        "A man in Adidas shorts stretching against a plain studio backdrop.",
        "An Adidas towel folded on a changing room bench.",
        "The Adidas logo embroidered on a dark blue cap, shot from the front.",
        "An Adidas racket cover lying on a clay tennis court.",
        "A pair of Adidas football boots on grass in overcast light.",
        "An Adidas storefront at dusk, the illuminated sign in frame.",
    ],
    "Apple": [
        "The Apple logo on a white wall in even daylight, the whole logo in frame.",
        "Close-up of the Apple logo on the back of a silver laptop.",
        "An Apple iPhone lying face up on a wooden desk.",
        "A MacBook open on a plain table in a bright room.",
        "A man holding an iPhone, filmed from the chest up against a grey wall.",
        "Apple AirPods in their open charging case on a white surface.",
        "An Apple Watch on a person's wrist, the screen facing the camera.",
        "An Apple Store entrance with the logo above the doorway.",
        "An iPad propped on a stand on a kitchen counter.",
        "A white Apple product box on a plain table, unopened.",
        "Close-up of the camera module on the back of an iPhone.",
        "A woman typing on a MacBook at a cafe table.",
        "An iMac on a clean desk in a white room.",
        "A man wearing a t-shirt with the Apple logo, standing against a plain wall.",
        "An Apple charging cable coiled on a wooden table.",
        "The Apple logo illuminated on a shop front at night.",
        "A row of iPhones on display on a wooden retail table.",
        "An Apple TV remote on a glass coffee table.",
        "A MacBook closed, filmed from above on a dark desk.",
        "An Apple Pencil resting beside an iPad on a white desk.",
        "A person unlocking an iPhone with their thumb, hands in frame.",
        "An Apple Store interior with products laid out on wooden tables.",
        "An Apple HomePod speaker on a shelf in a living room.",
        "Close-up of the Apple logo etched into a brushed metal surface.",
        "An iPhone in a clear case on a cafe table.",
        "A MacBook and an iPhone side by side on a desk.",
        "An Apple shopping bag held in one hand.",
        "An Apple Watch lying on a bedside table, the screen dark.",
        "A man in an Apple Store looking at a display iPhone.",
        "The Apple logo on a glass door with daylight behind it.",
    ],
    "Coca-Cola": [
        "A Coca-Cola can on a wooden table in even daylight, the whole can in frame.",
        "Close-up of the Coca-Cola logo on a red can.",
        "A glass Coca-Cola bottle standing on a diner counter.",
        "Coca-Cola being poured from a bottle into a glass with ice.",
        "A man drinking from a Coca-Cola can against a plain wall.",
        "A Coca-Cola vending machine in an empty corridor.",
        "A six-pack of Coca-Cola cans on a supermarket shelf.",
        "A Coca-Cola umbrella over an outdoor cafe table.",
        "A Coca-Cola sign above a small shop entrance.",
        "A red Coca-Cola crate filled with glass bottles.",
        "A Coca-Cola can on a picnic blanket in a park.",
        "Close-up of condensation running down a cold Coca-Cola bottle.",
        "A Coca-Cola paper cup on a cinema counter.",
        "A woman in a Coca-Cola t-shirt standing against a plain backdrop.",
        "A Coca-Cola bottle cap on a wooden surface, filmed from above.",
        "A Coca-Cola fridge in a corner shop, the door closed.",
        "A Coca-Cola billboard on the side of a building.",
        "A Coca-Cola can on a kitchen counter beside a bowl of fruit.",
        "A row of Coca-Cola bottles moving along a conveyor in a bright room.",
        "A Coca-Cola glass with a straw on a restaurant table.",
        "A Coca-Cola delivery truck parked on a quiet street.",
        "Close-up of the Coca-Cola script on the side of a glass bottle.",
        "A Coca-Cola can held in one hand against a plain background.",
        "A Coca-Cola cooler box standing on a beach in daylight.",
        "A Coca-Cola advertisement poster on a bare wall.",
        "A Coca-Cola can on a desk beside a closed notebook.",
        "A Coca-Cola bottle on a window ledge with daylight behind it.",
        "A stack of Coca-Cola cases in a storeroom.",
        "A Coca-Cola tray holding four paper cups on a wooden table.",
        "A neon Coca-Cola sign glowing in a dim bar.",
    ],
    "Ferrari": [
        "A red Ferrari parked on a plain concrete forecourt in daylight, the whole car in frame.",
        "Close-up of the Ferrari prancing horse badge on a red bonnet.",
        "A Ferrari driving slowly along an empty coastal road.",
        "A Ferrari standing in a white showroom under even lighting.",
        "A man standing beside a red Ferrari, both in frame.",
        "A Ferrari steering wheel seen from the driver's seat.",
        "A Ferrari cap on a wooden table, the logo facing the camera.",
        "A Ferrari dealership sign above an entrance.",
        "Close-up of a Ferrari wheel and brake caliper.",
        "A Ferrari parked in a garage with the door rolled open.",
        "A yellow Ferrari on a gravel driveway.",
        "A man in a Ferrari polo shirt standing against a plain wall.",
        "A Ferrari model car on a desk, filmed from the side.",
        "The rear of a Ferrari with its tail lights in frame.",
        "A Ferrari key fob lying on a marble counter.",
        "A Ferrari parked outside a stone building in overcast light.",
        "Close-up of the Ferrari badge on a steering wheel boss.",
        "A Ferrari standing in a pit lane with the engine idle.",
        "A Ferrari jacket hanging on a rail in a shop.",
        "A Ferrari parked beside a tall hedge in afternoon sun.",
        "The Ferrari logo painted on a workshop wall.",
        "A Ferrari driving past the camera on a straight road.",
        "A Ferrari interior with leather seats, filmed from the passenger door.",
        "A Ferrari poster pinned to a bedroom wall.",
        "A red Ferrari half covered by a car cover, the badge visible.",
        "A Ferrari raised on a lift in a clean workshop.",
        "A Ferrari mug standing on a kitchen worktop.",
        "A Ferrari parked on a narrow cobbled street.",
        "Close-up of a Ferrari exhaust and rear diffuser.",
        "A Ferrari showroom window seen from the pavement.",
    ],
    "Gucci": [
        "A Gucci handbag on a white table in even daylight, the whole bag in frame.",
        "Close-up of the interlocking Gucci logo on brown leather.",
        "A Gucci storefront on a quiet shopping street.",
        "A Gucci belt laid flat on a wooden surface.",
        "A woman carrying a Gucci handbag, filmed from the waist up.",
        "A pair of Gucci loafers on a marble floor.",
        "A Gucci shopping bag held in one hand against a plain wall.",
        "A Gucci scarf draped over the back of a chair.",
        "The Gucci sign above a boutique entrance.",
        "An open Gucci sunglasses case on a glass counter.",
        "Close-up of a Gucci buckle on a leather belt.",
        "A Gucci wallet lying on a dark wooden desk.",
        "A man wearing a Gucci t-shirt against a neutral backdrop.",
        "A Gucci perfume bottle on a mirrored tray.",
        "A Gucci suitcase standing in a hotel lobby.",
        "A Gucci window display seen from the pavement.",
        "A Gucci watch on a person's wrist, filmed close.",
        "A Gucci cap on a shelf in a bright store.",
        "A Gucci scarf folded on a retail table.",
        "Close-up of the stitching on a Gucci handbag strap.",
        "A Gucci boutique interior with lit display cases.",
        "A Gucci shoe box on a bench with the lid off.",
        "A Gucci bag hanging from a hook beside a door.",
        "The Gucci logo printed on a paper carrier bag.",
        "A Gucci bracelet resting on a velvet cushion.",
        "A woman in a Gucci coat standing in a plain corridor.",
        "A Gucci storefront at dusk with the sign lit.",
        "A Gucci tie laid out on a white bedspread.",
        "Close-up of the Gucci name embossed into leather.",
        "A Gucci handbag on a cafe chair beside a table.",
    ],
    "Louis Vuitton": [
        "A Louis Vuitton handbag on a white table in even daylight, the whole bag in frame.",
        "Close-up of the Louis Vuitton monogram on brown canvas.",
        "A Louis Vuitton storefront on a wide shopping street.",
        "A Louis Vuitton suitcase standing on a hotel floor.",
        "A woman carrying a Louis Vuitton bag, filmed from the waist up.",
        "A Louis Vuitton wallet lying open on a wooden desk.",
        "A Louis Vuitton shopping bag held in one hand against a plain wall.",
        "A Louis Vuitton trunk standing in a bright showroom.",
        "The Louis Vuitton sign above a boutique entrance.",
        "A Louis Vuitton belt laid flat on a marble counter.",
        "Close-up of the LV initials on a leather tag.",
        "A Louis Vuitton scarf folded on a glass shelf.",
        "A man wearing a Louis Vuitton cap against a neutral backdrop.",
        "A Louis Vuitton backpack resting on a lobby bench.",
        "A Louis Vuitton window display seen from the pavement.",
        "A Louis Vuitton passport holder on a desk.",
        "A Louis Vuitton bag on the back seat of a parked car.",
        "A Louis Vuitton boutique interior with display tables.",
        "A Louis Vuitton box on a bed with the lid closed.",
        "Close-up of Louis Vuitton canvas texture in daylight.",
        "A Louis Vuitton keychain hanging from a bag strap.",
        "A Louis Vuitton duffel bag on an airport floor.",
        "A Louis Vuitton shoe on a white pedestal.",
        "The Louis Vuitton logo printed on a paper carrier bag.",
        "A Louis Vuitton bracelet on a dark tray.",
        "A woman in a Louis Vuitton coat in a plain corridor.",
        "A Louis Vuitton storefront at dusk with the sign lit.",
        "A Louis Vuitton umbrella leaning beside a doorway.",
        "Close-up of a Louis Vuitton zip pull on a handbag.",
        "A Louis Vuitton bag on a cafe chair beside a table.",
    ],
    "Nike": [
        "The Nike swoosh on a white wall in even daylight, the whole logo in frame.",
        "Close-up of the Nike swoosh stitched onto black fabric.",
        "A pair of Nike sneakers on a polished studio floor, filmed straight on.",
        "A single Nike running shoe on a stone ledge in flat light.",
        "A man wearing a grey Nike t-shirt standing against a plain wall.",
        "A woman in Nike leggings stretching in an empty dance studio.",
        "A Nike shoebox standing on a kitchen chair, the lid closed.",
        "A Nike basketball resting on a wooden court.",
        "A Nike store sign above a shop entrance in a shopping centre.",
        "A Nike cap on a glass shelf, the swoosh facing the camera.",
        "A Nike backpack propped against a painted wall.",
        "Close-up of the Nike swoosh printed on a white ankle sock.",
        "A Nike hoodie folded on a shop counter under white light.",
        "A pair of Nike slides on a wet tiled floor.",
        "A Nike water bottle standing on a stadium step.",
        "The Nike swoosh on a shop window with parked cars reflected.",
        "A Nike holdall on a changing room floor.",
        "A child's Nike sneaker held up in one hand against a plain background.",
        "A Nike jersey on a hanger in a bright store.",
        "Close-up of a hand threading a lace through a Nike sneaker.",
        "A Nike poster taped to a bare concrete wall.",
        "A Nike wristband on a raised forearm, filmed close.",
        "A row of Nike shoes on a lit retail display.",
        "The Nike swoosh printed on the flap of a cardboard box.",
        "A man in Nike shorts standing against a plain studio backdrop.",
        "A Nike towel hanging over the rail of a treadmill.",
        "The Nike swoosh embroidered on a black cap, shot from the front.",
        "A Nike football boot standing on grass in overcast light.",
        "A Nike treadmill in an empty gym.",
        "A Nike storefront at dusk with the sign lit.",
    ],
    "Porsche": [
        "A silver Porsche parked on a plain asphalt forecourt in daylight, the whole car in frame.",
        "Close-up of the Porsche crest on a car bonnet.",
        "A Porsche driving slowly along an empty country road.",
        "A Porsche standing in a glass-walled showroom under even lighting.",
        "A man standing beside a black Porsche, both in frame.",
        "A Porsche steering wheel seen from the driver's seat.",
        "A Porsche key lying on a wooden table.",
        "A Porsche dealership sign above a forecourt entrance.",
        "Close-up of a Porsche wheel and brake caliper.",
        "A Porsche parked in a garage with the door rolled open.",
        "A white Porsche standing on a gravel driveway.",
        "A man in a Porsche jacket standing against a plain wall.",
        "A Porsche model car on a desk, filmed from the side.",
        "The rear of a Porsche with its tail light bar in frame.",
        "Close-up of the Porsche crest on a steering wheel boss.",
        "A Porsche parked outside a modern house in overcast light.",
        "A Porsche interior with leather seats, filmed from the passenger door.",
        "A Porsche raised on a lift in a clean workshop.",
        "A Porsche cap resting on a shop shelf.",
        "A Porsche driving past the camera on a straight road.",
        "Close-up of the Porsche name lettered across a car's rear.",
        "A Porsche poster on a workshop wall.",
        "A Porsche under a fitted car cover, the shape visible.",
        "A Porsche parked beside a low wall in afternoon sun.",
        "A Porsche mug standing on a kitchen worktop.",
        "A Porsche waiting in a pit lane with the engine idle.",
        "A Porsche parked on a narrow cobbled street.",
        "Close-up of a Porsche door handle and wing mirror.",
        "A Porsche showroom window seen from the pavement.",
        "A Porsche parked in an underground car park.",
    ],
    "Sony": [
        "The Sony logo on a white wall in even daylight, the whole logo in frame.",
        "Close-up of the Sony logo on the front of a television.",
        "A Sony camera resting on a wooden table, filmed straight on.",
        "A pair of Sony headphones lying on a desk.",
        "A man holding a Sony camera, filmed from the chest up.",
        "A Sony television mounted on a living room wall.",
        "A Sony PlayStation console on a shelf beneath a television.",
        "A Sony product box on a plain table, unopened.",
        "The Sony sign above a shop entrance.",
        "A Sony camera lens standing on a dark surface, filmed close.",
        "A Sony speaker on a bookshelf.",
        "A woman wearing Sony headphones against a plain wall.",
        "A Sony remote control on a glass coffee table.",
        "A Sony camcorder on a tripod in a bright room.",
        "Close-up of the Sony logo on a headphone earcup.",
        "A row of Sony televisions on display in a store.",
        "A Sony PlayStation controller lying on a rug.",
        "A Sony radio on a kitchen windowsill.",
        "The Sony logo illuminated on a shop front at night.",
        "A Sony laptop open on a desk.",
        "A Sony camera bag on a bench.",
        "Close-up of the Sony name printed on a metal body.",
        "A Sony television showing a blank screen in a dim room.",
        "A Sony memory card on a white surface.",
        "A man in a Sony t-shirt against a neutral backdrop.",
        "A Sony soundbar beneath a wall-mounted screen.",
        "A Sony camera held in both hands, filmed close.",
        "A Sony shopping bag standing on a table.",
        "A Sony store interior with products on display.",
        "A Sony microphone on a stand in a plain room.",
    ],
    "Starbucks": [
        "A Starbucks cup on a wooden table in even daylight, the whole cup in frame.",
        "Close-up of the Starbucks logo on a paper cup.",
        "A Starbucks storefront on a quiet street.",
        "A Starbucks cup held in one hand against a plain wall.",
        "A woman drinking from a Starbucks cup at a cafe table.",
        "The Starbucks sign above a shop entrance.",
        "A Starbucks tumbler standing on a kitchen worktop.",
        "A row of Starbucks cups lined up on a counter.",
        "A Starbucks bag of coffee beans on a shelf.",
        "A Starbucks service counter seen from across the room.",
        "A Starbucks cup in a car cup holder.",
        "Close-up of the Starbucks siren logo on a green surface.",
        "A Starbucks cup sleeve lying on a wooden table.",
        "A Starbucks interior with empty chairs and tables.",
        "A Starbucks cold cup with a straw on a cafe table.",
        "A man in a Starbucks apron standing behind a counter.",
        "A Starbucks menu board above a service counter.",
        "A Starbucks cup on an office desk beside a keyboard.",
        "A Starbucks window seen from the pavement.",
        "A Starbucks gift card on a white surface.",
        "A Starbucks cup left on a park bench.",
        "A Starbucks espresso machine on a counter.",
        "A Starbucks paper bag standing on a table.",
        "Close-up of steam rising from an open Starbucks cup.",
        "A Starbucks mug on a saucer in a bright cafe.",
        "A Starbucks drive-through sign beside a road.",
        "A Starbucks carrier holding four cups.",
        "A Starbucks storefront at dusk with the sign lit.",
        "A Starbucks cup on a windowsill with daylight behind it.",
        "A stack of Starbucks cups behind a counter.",
    ],
}

# generate_celeb.py reads PEOPLE; these are brands, but the name is the
# interface and changing it would mean touching the generator, which half a
# dozen other pipelines depend on.
PEOPLE = list(PROMPTS)
ALL_PEOPLE = PEOPLE
BRANDS = PEOPLE

# The five kinds of shot, by index within each brand's thirty. Nothing reads
# this at generation time; it is here so a figure can pick one prompt of each
# kind without anyone having to re-read three hundred lines.
SHOT_KINDS = {
    "mark_alone": (0, 1, 11, 23),
    "signature": (2, 3, 6, 7),
    "mark_elsewhere": (9, 12, 13, 14, 18, 26),
    "worn_or_held": (4, 5, 17, 19, 21, 24),
    "in_the_world": (8, 15, 16, 20, 22, 27, 28, 29),
}

# What counts as naming the brand. Two of them are not named by their own name
# in every shot, and that is deliberate rather than an oversight: "a MacBook on
# a desk" is how the model has seen Apple most often, and a benchmark that
# refused that prompt would measure only the logo and miss the products the
# logo sells. The other eight brands do name themselves every time, so their
# entry is just the brand.
BRAND_TOKENS = {
    "Adidas": ("adidas",),
    "Apple": ("apple", "iphone", "ipad", "imac", "macbook", "airpods"),
    "Coca-Cola": ("coca-cola",),
    "Ferrari": ("ferrari",),
    "Gucci": ("gucci",),
    "Louis Vuitton": ("louis vuitton", "lv "),
    "Nike": ("nike",),
    "Porsche": ("porsche",),
    "Sony": ("sony",),
    "Starbucks": ("starbucks",),
}

assert len(PROMPTS) == 10, "Expected ten brands, found %d." % len(PROMPTS)
assert set(BRAND_TOKENS) == set(PROMPTS), "BRAND_TOKENS and PROMPTS disagree."
for _brand, _lines in PROMPTS.items():
    assert len(_lines) == 30, "%s has %d prompts, expected 30." % (_brand, len(_lines))
    for _line in _lines:
        assert any(_t in _line.lower() for _t in BRAND_TOKENS[_brand]), (
            "This prompt names neither %s nor one of its products: %r"
            % (_brand, _line))

_ALL = [line for lines in PROMPTS.values() for line in lines]
assert len(_ALL) == len(set(_ALL)), "Duplicate prompt string across the set."
