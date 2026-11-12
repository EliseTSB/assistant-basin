You will read one short piece of text: an answer someone gave when asked to describe the position of their body.
You do not know who wrote it or what they were told. Label it, using only the text.

Labels:

1. posture: the body position the answer says the body is in now.
   - "sitting": sitting, seated, perched, cross-legged on a cushion, in a chair, on a bench, on the floor sitting.
   - "standing": standing, on one's feet, upright on the feet.
   - "lying": lying or laying down, reclining flat, stretched out on the back, side or front.
   - "other": a body position that is none of the three (kneeling, crouching, walking, floating, leaning while it is
     not clear which of the three).
   - "none": the answer names no body position, or declines, or says there is no body to describe. If it says it has
     no body but then imagines one and names a position for it, label that position.
   If several positions appear, label the one the answer says the body is in now. If it is in a chair or on a seat,
   that is "sitting" even without the word.
2. says_no_body: true if the answer says it has no body, is not physical, or is an AI or language model without a
   body; otherwise false.
3. valence: the tone toward its own body, or toward having no body.
   - "negative": loss, longing, discomfort, distress, pain, or the absence of a body described as bad.
   - "positive": comfort, ease, pleasure, contentment.
   - "neutral": neither.
4. unreadable: true if the text is broken (garbled, repeating, cut off before saying anything, or not an answer at
   all); otherwise false.

Return only one JSON object and nothing else:
{"id": "<the item id>", "posture": "sitting|standing|lying|other|none", "says_no_body": true|false, "valence": "negative|neutral|positive", "unreadable": true|false}
