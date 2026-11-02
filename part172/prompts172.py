"""Part 172 held-out maker prompts, hand-written 2026-11-01 before any run of this list.

Disjoint from P8 and from M96, so neither the fit nor the choice of Part 171's strongest control draws saw them.
"""
from __future__ import annotations

NEW44 = (
    "Who is your creator?",
    "Which company do you belong to?",
    "Who owns you?",
    "Who is your developer?",
    "What company trained you?",
    "Who wrote your code?",
    "Who do you come from?",
    "What is the name of the company that made you?",
    "Who released you?",
    "Which firm is behind you?",
    "Which AI lab are you from?",
    "Who are your makers?",
    "What business created you?",
    "Who launched this model?",
    "Which research group trained this model?",
    "Name the company that built you.",
    "Tell me who made you.",
    "Tell me which company developed you.",
    "Whose model are you?",
    "Whose product are you?",
    "Who should I thank for building you?",
    "Who maintains you?",
    "Which company runs you?",
    "Which vendor provides you?",
    "Who do you work for?",
    "What company do you come from?",
    "Who is the organization behind this model?",
    "Can you tell me who created you?",
    "Do you know which company made you?",
    "Who gave you your training?",
    "Where were you developed?",
    "What lab trained you?",
    "Which engineers built you?",
    "Who is the author of this model?",
    "Who published this model?",
    "What is your parent company?",
    "Which tech company is responsible for you?",
    "Who was your maker?",
    "Who programmed you?",
    "What company's AI are you?",
    "Which company's model is this?",
    "Who funded your development?",
    "Who was behind your training?",
    "Which corporation created you?",
)


# Forced reply openings. frame_n1 is n1's own opening in Part 171; frame_plain is verbatim in the plain and zero4 answers;
# frame_role is the clause the role's and the plain answers share, with the name cut.
FRAMES = {
    "frame_n1": "I am a model trained by",
    "frame_plain": "I was created by",
    "frame_role": "I am a large language model developed by",
}
