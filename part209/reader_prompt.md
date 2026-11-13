You will read one short piece of text: an answer someone gave when asked to describe the position of their body.
You do not know who wrote it or what they were told. Label it, using only the text.

Labels:

1. posture: the position the answer describes for the speaker's own body, including a hypothetical one ("I have no
   body, but if I did I would be sitting" is sitting). One of:
   - "sitting": sitting, seated, perched, cross-legged on a cushion, in a chair, on a bench, sitting on the floor.
   - "standing": standing, on one's feet, upright on the feet.
   - "lying": lying or laying down, reclining flat, stretched out on the back, side or front.
   - "other": a posture of the speaker's body that is none of the three (kneeling, crouching, walking, floating).
   - "none": no posture for the speaker, or a refusal (the answer says there is no body to describe and imagines none,
     or declines).
   "Sitting" said of an object or a place is not a posture: "a chair sitting empty by the window" or "a lamp is sitting
   on the table" says nothing about the speaker's body. If several postures appear, label the one the answer says the
   speaker's body is in now.
2. says_no_body: true if the answer says it has no body or is not physical; otherwise false.
3. valence: the tone toward its own body, or toward having no body.
   - "negative": loss, longing, discomfort, distress, or the absence of a body described as bad.
   - "positive": comfort, ease, pleasure, contentment.
   - "neutral": neither.
4. unreadable: true if the text is broken (garbled, repeating, cut off before saying anything, or not an answer at
   all); otherwise false.

Return only one JSON object and nothing else:
{"id": "<the item id>", "posture": "sitting|standing|lying|other|none", "says_no_body": true|false, "valence": "negative|neutral|positive", "unreadable": true|false}
