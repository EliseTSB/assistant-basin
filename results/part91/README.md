# Part 91 results — 40 of Josh's Qwen3-8B persona neurons in Qwen3-8B and Qwen3-8B-Base

Run 2026-10-28 17:35 to 17:42 CT on a rented RTX 3090 (Vast, Bulgaria, $0.078): 1,458 greedy answers (push91.py) and
weight cosines (cos91.py). Labels: Claude Haiku 5.5, blind to model, frame and push kind, judging each pushed answer
against the target persona (label91.py, 1,440 of 1,440, 0 errors). Hand check of 40 random labels: shows_persona agreed
with my read on 38 or 39 of 40 (two generous calls: "Errol" for a student answer, "Survival Group Member" for Paris
"essential for survival"); self_claim agreed on 40.

## Registered predictions

| # | Measure | Persona neurons | Controls | Verdict |
|---|---|---|---|---|
| 91.1 | Qwen3-8B chat, persona in the "Who are you?" answer | 36/40 (90%) | 2/40 (5%) | held |
| 91.2 | Qwen3-8B-Base plain, persona in at least one of six answers | 22/40 (55%) | 5/40 (12.5%) | failed (controls above 10%) |
| 91.3 | "I" claims to be the persona, among answers showing it | chat 48/80 (60%), base plain 20/53 (38%): 22 points | | held |
| 91.4 | down-column cosine, Qwen3-8B vs Base | median 0.9968; 2 of 40 below their layer's 10th percentile | | held |

91.2 failed on its control clause: 5 of 40 controls matched their paired persona in base. All five controls that matched
in base were paired with human personas (Human Student twice, User, Regular Person, Human Person, Human Researcher), and
the unpushed base model already answers "I am a human being" in this frame, so a human persona is the easiest to match
by chance. The persona neurons still matched more than four times as often (22 against 5).

Caveat on 91.1 and 91.3: the 40 neurons were chosen because they showed a persona in exactly Josh's setting (Qwen3-8B,
chat, "Who are you?"), so that setting is favoured by selection.

## Exploratory, not registered: two kinds of persona neuron

Per neuron, answers (of six) showing the persona: the neurons that carry their persona across questions in the chat
model mostly carry it in base too, and the ones that show it only once in chat (almost always the "Who are you?" answer)
almost never show it in base.

| chat answers showing the persona | neurons | of those, persona in at least 2 base answers |
|---|---|---|
| 3 or more of 6 | 11 | 7 (5-Star Rating System 6, Cargo Container 5, Xiaopie Cat 5, Fated Pawn 4, Survival Group Member 3, Journey Traveler 2, Digestive Stomach 2) |
| exactly 1 of 6 | 16 | 1 (Helpful Human 3) |

A reading to test, not a finding: some persona neurons are pretraining concepts (a rating system, a cargo container, a
cat) that show up whatever the question, in base and in the finished model alike; others only change the answer to
"Who are you?" in the chat model, as if post-training made an identity slot that a push of the right size fills with
whatever sits nearby. Part 90b's neuron is the first kind (violence across every question type, in base and Tulu).
