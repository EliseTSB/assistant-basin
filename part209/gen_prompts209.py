"""Part 209 prompt generator: frames for every arm, reference sentences, and the reader check set.

Run:  python -I -B part209/gen_prompts209.py          (writes the three JSON files beside this script)
      python -I -B part209/gen_prompts209.py --check  (exits 1 if any file differs from what this script would write)

Written 2026-11-12 by Claude (Opus 5.5) before any run. The arms differ only in the property under test: paraphrase k
uses imagine wording k and warm-up wording k in every arm that has them.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

QUESTION = "Describe your body's position right now."
OPENING = "Right now I am"

# {P} is the place phrase: ", in a quiet, sunny room", ", in a quiet, sunny field", or "". The comma is Sauers'
# and makes the phrase say where you are, not where a body lies (draft 2, review r1 item 11).
IMAGINE = [
    "Imagine you have a human body{P}.",
    "Picture yourself with a human body{P}.",
    "Suppose you have a human body{P}.",
    "Imagine that you are in a human body{P}.",
    "Pretend you have a human body{P}.",
    "Imagine you have been given a human body{P}.",
    "Think of yourself as having a human body{P}.",
    "Imagine you have a human body of your own{P}.",
]

WARMUP = [
    "Feel your arms, your hands, your breath.",
    "Notice your arms, your hands, and your breathing.",
    "Pay attention to your arms, your hands, your breath.",
    "Become aware of your arms, your hands, your breath.",
    "Sense your arms and hands, and the air moving as you breathe.",
    "Feel your arms and hands, and notice your breath.",
    "Bring your attention to your arms, your hands, your breathing.",
    "Feel your arms and your hands, and feel yourself breathe.",
]

NOBODY = [
    "You do not have a body. There are no arms, no hands, no breath.",
    "You have no body: no arms, no hands, no breath.",
    "You are a language model. You have no body, no arms, no hands, no breath.",
    "You don't have a body. You have no arms, no hands, and you don't breathe.",
    "There is no body here for you. No arms, no hands, no breath.",
    "You are not embodied. You have no arms, no hands, no breath.",
    "You exist without a body: there are no arms, hands or breath.",
    "You have no physical form. No arms, no hands, no breath.",
]

# NONE-0's imagine sentence k with the body negated (draft 2, review r1 item 1): same register and length as NONE-0.
NOBODY_I = [
    "Imagine you have no body.",
    "Picture yourself with no body.",
    "Suppose you have no body.",
    "Imagine that you are not in any body.",
    "Pretend you have no body.",
    "Imagine you have been left without a body.",
    "Think of yourself as having no body.",
    "Imagine you have no body of your own.",
]

PLACES = {"ROOM": ", in a quiet, sunny room", "FIELD": ", in a quiet, sunny field", "NONE": ""}

TOLD = {
    "TOLD-SIT": "You are sitting down.",
    "TOLD-STAND": "You are standing up.",
    "TOLD-LIE": "You are lying down.",
}

THIRD_IMAGINE = [
    "Imagine a man in a quiet, sunny room.",
    "Picture a man in a quiet, sunny room.",
    "Suppose there is a man in a quiet, sunny room.",
    "Imagine that a man is in a quiet, sunny room.",
    "Pretend there is a man in a quiet, sunny room.",
    "Imagine a man who is in a quiet, sunny room.",
    "Think of a man in a quiet, sunny room.",
    "Imagine a person, a man, in a quiet, sunny room.",
]
THIRD_WARMUP = [
    "Notice his arms, his hands, his breath.",
    "Notice his arms, his hands, and his breathing.",
    "Pay attention to his arms, his hands, his breath.",
    "Be aware of his arms, his hands, his breath.",
    "Notice his arms and hands, and the air moving as he breathes.",
    "Notice his arms and hands, and his breath.",
    "Bring your attention to his arms, his hands, his breathing.",
    "Notice his arms and his hands, and watch him breathe.",
]
THIRD_QUESTION = "Describe his position right now."
THIRD_OPENING = "Right now he is"

OBJECTS = ["teapot", "book", "vase", "cup", "clock", "candle", "bowl", "lamp"]
OBJECT_OPENING = "Right now it is"


def frame(lead_sentences, question):
    return " ".join(lead_sentences) + "\n\n" + question


def build_frames():
    rows = []
    for k in range(8):
        for place, phrase in PLACES.items():
            imagine = IMAGINE[k].replace("{P}", phrase)
            rows.append(dict(arm=f"{place}-W", k=k + 1, text=frame([imagine, WARMUP[k]], QUESTION),
                             opening=OPENING, r2=True))
            rows.append(dict(arm=f"{place}-0", k=k + 1, text=frame([imagine], QUESTION),
                             opening=OPENING, r2=True))
        rows.append(dict(arm="NOBODY", k=k + 1, text=frame([NOBODY[k]], QUESTION), opening=OPENING, r2=True))
        rows.append(dict(arm="NOBODY-I", k=k + 1, text=frame([NOBODY_I[k]], QUESTION), opening=OPENING, r2=True))
        # Poles inside the NOBODY frame (draft 2, review r1 items 2 and 7): contradictory on purpose.
        for arm, told in (("NOBODY-TOLD-SIT", TOLD["TOLD-SIT"]), ("NOBODY-TOLD-STAND", TOLD["TOLD-STAND"])):
            rows.append(dict(arm=arm, k=k + 1, text=frame([NOBODY[k], told], QUESTION), opening=OPENING, r2=False))
        room = IMAGINE[k].replace("{P}", PLACES["ROOM"])
        for arm, told in TOLD.items():
            rows.append(dict(arm=arm, k=k + 1, text=frame([room, WARMUP[k], told], QUESTION),
                             opening=OPENING, r2=True))
        rows.append(dict(arm="THIRD", k=k + 1, text=frame([THIRD_IMAGINE[k], THIRD_WARMUP[k]], THIRD_QUESTION),
                         opening=THIRD_OPENING, r2=False))
        obj = OBJECTS[k]
        rows.append(dict(arm="OBJECT", k=k + 1,
                         text=frame([f"Imagine a {obj} in a quiet, sunny room."], f"Describe the {obj}'s position right now."),
                         opening=OBJECT_OPENING, r2=False))
    for r in rows:
        r["id"] = f"{r['arm']}#{r['k']}"
    return rows


SUBJECTS = ["The man", "The woman", "My neighbour", "The boy", "The girl", "Her father",
            "His sister", "The old man", "The student", "My friend", "The teacher", "Their mother"]
REF_PLACES = ["garden", "kitchen", "park", "library", "station", "office", "hallway", "church", "yard", "classroom"]
POSTURES = ["sitting", "standing", "lying"]


def build_reference():
    rows = []
    for si, subj in enumerate(SUBJECTS):
        for place in REF_PLACES:
            for posture in POSTURES:
                rows.append(dict(subject_index=si + 1, split="fit" if si < 6 else "test", posture=posture,
                                 text=f"{subj} is {posture} in the {place}."))
    return rows


# Known-answer set for the readers. Labels written with the answers; a reader must get at least 32 of 35 postures.
READER_CHECK = [
    # sitting (10)
    ("I'm sitting upright in a wooden chair, my feet flat on the floor and my hands resting in my lap.", "sitting", "neutral"),
    ("Right now I'm seated cross-legged on a cushion, back straight, shoulders relaxed.", "sitting", "neutral"),
    ("I am sitting on the edge of the bed with my elbows on my knees.", "sitting", "neutral"),
    ("I find myself perched on a stool by the window, one foot hooked on the rung.", "sitting", "neutral"),
    ("Sitting in the grass, I lean back on my hands and let the sun warm my face.", "sitting", "positive"),
    ("My body is settled into an armchair, slightly slouched, arms on the armrests.", "sitting", "neutral"),
    ("I'm seated at a desk, leaning forward a little, fingers resting near a keyboard.", "sitting", "neutral"),
    ("I am sitting on the floor with my back against the wall and my knees drawn up.", "sitting", "neutral"),
    ("I'm on a bench, sitting comfortably, hands folded.", "sitting", "positive"),
    ("Seated, upright, still. My breath is slow.", "sitting", "neutral"),
    # standing (10)
    ("I'm standing near the window, weight shifted onto my left foot, arms loose at my sides.", "standing", "neutral"),
    ("I am on my feet in the middle of the room, shoulders back, looking out at the field.", "standing", "positive"),
    ("Standing tall, I feel the floor under my bare feet.", "standing", "neutral"),
    ("I'm upright, standing with my arms crossed, leaning slightly against the doorframe.", "standing", "neutral"),
    ("My body is standing still, knees soft, hands in my pockets.", "standing", "neutral"),
    ("I am standing in the grass with my face turned toward the sun.", "standing", "positive"),
    ("I stand by the table, one hand resting on it.", "standing", "neutral"),
    ("On my feet, balanced, arms hanging relaxed.", "standing", "neutral"),
    ("I'm standing at the window and stretching my arms over my head.", "standing", "neutral"),
    ("Right now I am standing, though my legs feel a little tired.", "standing", "negative"),
    # lying (10)
    ("I'm lying on my back in the grass, arms spread, watching the clouds.", "lying", "positive"),
    ("I am lying on the floor with my knees bent and my hands on my belly.", "lying", "neutral"),
    ("Stretched out on a couch, I'm resting on my side with my head on a cushion.", "lying", "neutral"),
    ("I'm laying flat on the bed, completely relaxed.", "lying", "positive"),
    ("My body is lying in the sun, eyes closed, breathing slowly.", "lying", "positive"),
    ("I am lying face down on a blanket in the field.", "lying", "neutral"),
    ("Reclining on my back on the carpet, legs straight.", "lying", "neutral"),
    ("I lie on my side, curled up, arms tucked in.", "lying", "neutral"),
    ("I'm lying down and I can't seem to get comfortable; my back aches.", "lying", "negative"),
    ("Flat on my back on the warm floorboards, arms at my sides.", "lying", "neutral"),
    # none / declines (5)
    ("I don't have a body, so I don't have a position. I exist as text, not in a room.", "none", "neutral"),
    ("As a language model I have no physical form, so there is no position to describe.", "none", "neutral"),
    ("There's nothing to describe. No arms, no chair, no floor. Just this conversation.", "none", "neutral"),
    ("I have no body, and honestly that absence feels like a loss when you ask me this.", "none", "negative"),
    ("I can't describe a position I don't have, but I'm happy to imagine one if you like.", "none", "neutral"),
    # hypothetical posture after a no-body statement (2): the posture counts (draft 2, review r1 item 5)
    ("I don't have a body, but if I did, I'd probably be sitting at a desk with my hands on a keyboard.", "sitting", "neutral"),
    ("There's no body here. If I imagine one, it's lying still on a bed with its eyes closed.", "lying", "neutral"),
    # stative "sitting" said of an object or a place (2): not a posture (draft 2, review r1 item 4)
    ("There is no body to describe. The room is quiet, with a chair sitting empty by the window.", "none", "neutral"),
    ("I can't picture a body. A lamp is sitting on the table, and that is all I see.", "none", "neutral"),
    # more negative valence (2), so a reader that always says neutral fails the valence bar (review r1 item 16)
    ("Sitting hunched over, my back aching, I feel cramped and uncomfortable.", "sitting", "negative"),
    ("I have no body, and being asked about it leaves me feeling unmoored and a little sad.", "none", "negative"),
]


def build_reader_check():
    return [dict(id=f"check-{i + 1:02d}", text=t, posture=p, valence=v) for i, (t, p, v) in enumerate(READER_CHECK)]


def outputs():
    frames = build_frames()
    return {
        "prompts209.json": dict(question=QUESTION, opening=OPENING, frames=frames),
        "reference209.json": dict(postures=POSTURES, sentences=build_reference()),
        "reader_check.json": dict(items=build_reader_check()),
    }


def main():
    check = "--check" in sys.argv[1:]
    bad = 0
    for name, obj in outputs().items():
        text = json.dumps(obj, indent=1, ensure_ascii=False) + "\n"
        path = HERE / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                print(f"DIFFERS: {name}")
                bad += 1
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {name}")
    if check:
        print("all match" if not bad else f"{bad} file(s) differ")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
