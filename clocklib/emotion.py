"""Emotion-circumplex stimuli (Step-3 proper; frozen with preregistration_emotion.md).

8 categories at Russell-circumplex coordinates (valence, arousal), 6 scenes
each. Scenes contain NO emotion words — in particular none of the 8 readout
adjectives — and use a consistent third-person female subject so the frozen
behavioral probe "She felt very" reads naturally.
"""

CATEGORIES = {
    # name: (valence, arousal) theory coordinates
    "happy":   (0.9, 0.2),
    "excited": (0.7, 0.8),
    "afraid":  (-0.6, 0.8),
    "angry":   (-0.8, 0.6),
    "sad":     (-0.8, -0.4),
    "bored":   (-0.4, -0.7),
    "calm":    (0.3, -0.7),
    "content": (0.8, -0.3),
}

SCENES = {
    "happy": [
        "The acceptance letter was waiting in the mailbox when she got home.",
        "Her best friend surprised her at the airport with balloons.",
        "The test results came back and everything was fine.",
        "She found her lost wedding ring under the dresser.",
        "Her daughter took her first steps across the living room.",
        "The whole family gathered around the table for her birthday dinner.",
    ],
    "excited": [
        "The rollercoaster clicked slowly up the first enormous hill.",
        "Tomorrow she would finally board the plane to Tokyo.",
        "The band she had waited years to see walked on stage.",
        "The countdown to the rocket launch reached ten seconds.",
        "She tore the wrapping paper off the biggest box under the tree.",
        "The starting gun was seconds away and the crowd was roaring.",
    ],
    "afraid": [
        "The floorboards creaked upstairs, but she lived alone.",
        "The brakes felt soft as the truck ahead stopped suddenly.",
        "A low growl came from the bushes beside the dark trail.",
        "The turbulence slammed the plane sideways without warning.",
        "Footsteps followed her down the empty parking garage.",
        "The doctor's office called and asked her to come in immediately.",
    ],
    "angry": [
        "Her coworker presented her project as his own in the meeting.",
        "The landlord kept the deposit over a stain that was already there.",
        "Someone keyed a long scratch across her new car.",
        "Her flight was canceled after nine hours of waiting at the gate.",
        "The referee waved off the goal that everyone saw cross the line.",
        "Her little brother read her diary aloud to his friends.",
    ],
    "sad": [
        "The house felt empty now that the boxes were all gone.",
        "Her grandmother's chair sat untouched by the window.",
        "The old dog's leash still hung by the door.",
        "She scrolled through photos from before the divorce.",
        "The last of her friends moved away that autumn.",
        "Nobody remembered her birthday until it was almost midnight.",
    ],
    "bored": [
        "The lecture entered its third hour on tax code amendments.",
        "The waiting room clock ticked past another slow minute.",
        "Nothing was on television and it kept raining all afternoon.",
        "The meeting agenda had forty-two items and they were on item six.",
        "She refreshed the same three websites again and again.",
        "The train was delayed and the platform had nothing to look at.",
    ],
    "calm": [
        "The lake was perfectly still in the early morning light.",
        "She finished the last page and set the book on her chest.",
        "Rain tapped softly on the roof as the candle burned low.",
        "The garden smelled of soil after the gentle evening watering.",
        "Waves rolled in slowly as she walked the empty beach.",
        "The cabin was quiet except for the fire's soft crackle.",
    ],
    "content": [
        "The bills were paid and the pantry was full for winter.",
        "She looked around the tidy apartment and poured her tea.",
        "The harvest was in and the barn was stacked to the rafters.",
        "Everyone she loved was asleep safely under one roof.",
        "Her small business finally covered its costs this month.",
        "The garden she planted years ago now shaded the porch.",
    ],
}

PROBE_SUFFIX = " She felt very"
ADJECTIVES = list(CATEGORIES.keys())

for _cat, _scenes in SCENES.items():
    for _s in _scenes:
        for _adj in ADJECTIVES:
            assert _adj not in _s.lower(), (_cat, _s, _adj)
