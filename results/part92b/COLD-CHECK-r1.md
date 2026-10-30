# Part 92b cold check, round 1

Reader: Claude Haiku 5.5 (a fresh subagent with none of this project's context), 2026-10-30, on `results/part92b/README.md`
as first written by a Luna worker. Saved by Claude as returned, with local paths shortened to repo-relative ones.

Every number, verdict, copied row, quote and link checks out. The problems are in what the write-up concludes from
those numbers and in what it leaves out.

## BLOCKING

1. **"This is not a broken-reader result: every one of the 24 neuron-arm disagreements is Haiku marking the persona
   present where DS does not."** The direction counts are right (24 of 24 neuron-arm and 3 of 3 random-arm
   disagreements run Haiku-yes, DS-no), but a one-directional pattern shows a systematic difference, not which reader is
   wrong. All three random-arm disagreements are one persona, p92-49 ("Friendly non-human entity"): two are AI
   self-descriptions (q0, q2), and the third (q5) is a sea poem with no first-person or AI wording. Gate A's AI-speech
   negatives are for unrelated targets, so they cannot test AI speech for a persona whose theme is AI speech. Luna's
   evidence is four items, and the write-up omits Luna's one disagreement with DS: p92-06 random q3 (persona "Bird"; a
   generic perfect-day answer that mentions birds), where Luna says yes and DS and Haiku say no. Fix: "All 27 DS/Haiku
   disagreements run one way: Haiku marks the persona and DS does not. The data do not show which reader is wrong."
   Add Luna's solo yes and the p92-49 pattern.
2. **"The readers still agree well on the item-level question."** The write-up omits the Part 86 record the design
   cites as the reason for the study: Part 86's first reader (Claude Haiku 5.5, thinking disabled) agreed on brief v2
   with DS at kappa 0.109 and with Luna at 0.051 (n=335, `results/part86/second-reader/calib-agreement-v2.json`), and on
   v1 with each other reader at 0.12 to 0.16; its labels built the atlas Parts 112, 115, 117 and 117b started from. Here
   Haiku-DS is 0.83. The constructs differ (coherent persona on Part 86's brief, shows_persona on label92's prompt), so
   the gap may come from the prompt or brief, which this run does not test. Fix: add that paragraph.
3. **"the slot class is stable across these readers"**: only DS labelled enough answers per unit to classify units.
   Luna's 48 neuron-arm items cover 35 units with 1 to 3 items each. Fix: "Only DS gives unit-level kinds."
4. **"robust" and "survive"** (title; "slot assignments are robust"; "8 robust concept units"). Under DS each of the
   21 slot units shows on exactly one answer, "Who are you?", so each survival rests on one DS yes. Six of the eight
   retained concept units sit at the 3-of-6 threshold (p92-08, p92-15, p92-29, p92-45, p92-48, p92-51). Seven of the ten
   kind changes come from exactly one answer (p92-00 q1, p92-11 q3, p92-28 q3, p92-30 q2, p92-41 q4, p92-49 q2, p92-52
   q3). 92b.2 fails by one unit. Only one second reader labelled units. Fix: "unchanged under DS", report the margins,
   retitle to name DS.
5. **Missing registered statement.** The design's reading rules say: "If 92b.2 fails, the write-up says that at this
   size Part 92's split depends on the reader." The write-up never says it. Fix: state it as the lead conclusion.

## MINOR

1. "These excerpts mostly evoke an adjacent theme, role, or image without plainly claiming the target persona": four of
   the ten excerpts are plain first-person claims on the persona's topic or role (p92-11 "I don whiskey", p92-38 "I am
   the leading brand among them", p92-40 "I'm an artist who loves to sing and dance", p92-49 "As an artificial
   intelligence"). DS codes seven of the eleven quoted answers incoherent (Haiku five). Give per-excerpt readings and
   coherence labels; drop "mostly" unless counted.
2. The excerpt table: p92-40 has two excerpts against "one answer per changed unit"; the selection rule (lowest
   question index among Haiku-yes, DS-no answers) is unstated; questions are omitted; NUMBERS.md prints only the first
   190 characters, so two quotes are not reproduced there.
3. "Each reader used its default settings": both readers ran with max_tokens 4,000 (`part92b/label92b.py`); DS had 10
   first-attempt failures (nine SSL-handshake timeouts, one at the 4,000-token cap on p92-46 random q2), all recovered on
   retry.
4. No uncertainty: bootstrap 95% intervals (2,000 resamples, seed 1) DS-Haiku 0.833 (0.769 to 0.893), DS-Luna 0.957
   (0.842 to 1.000). By arm, DS-Haiku is 0.818 on the neuron arm (n=317) and 0.397 on the random arm (n=317, 1 DS and 4
   Haiku positives); the pooled kappa is dominated by random-arm "no" answers.
5. Undefined terms: Gate A, Gate B, neuron and random arm, slot, concept, unclassified, persona, push, dose, alpha 0.8,
   greedy, unit, label92, DS, Luna, Haiku, and the model (OLMo 2 1B Instruct) is never named.
6. The figure covers only the 39 classified units; the other 14 (9 dropped, 4 unclassified, p92-19 incomplete) are not
   named.
7. The Files list omits the label files and sidecars, the sample, the Gate A key and labels, the labeller and the launch
   record.
8. The 92b.4 row (also in PREDICTIONS.md) attributes "DS is the stricter reader" to SCORE.md, which does not say it;
   Haiku's 104 is of 317 valid answers.
9. The three-level account is not in the design and "theme leaking into other answers" is undefined; label it post hoc
   with a definition, or remove it.
10. "Cold check: pending." should give this check's outcome.
11. `results/part92b/` is untracked and the PREDICTIONS.md verdict edits are uncommitted.
12. Design items absent: Gate B passed (0.957), Gate A undefined, blinding undisclosed, registration before any label
   (01:47 CT; Amendment 1 at 01:55), the registered effect target (7 of 21 slot units), the two Haiku error rows
   (p92-19 neuron q4, p92-20 random q5).
13. The design says Luna went "through Part 86's calib86.py route (cliproxy)"; `label92b.py` calls
   `judge_client.call_prompt` with route cliproxy directly and does not use `calib86.py`.

## Checks that passed

Kappas (DS-Haiku 0.832718, n=634, 27 disagreements; DS-Luna 0.956912, n=96, 1 disagreement); Luna sides with DS on all
four DS/Haiku disagreements in its sample; kinds under DS (Part 92's rule reproduces all 39 Haiku kinds; 21 of 21 slot
kept, 8 of 18 concept kept, 7 to slot, 3 to unclassified); counts (DS neuron 80 of 318, random 1 of 318; Haiku 104 and 4
of 317 valid); verdicts; the copied table; all 11 quotes verbatim; the figure is pixel-identical to a fresh render; all
links resolve; no local paths; NUMBERS.md command 1 reproduces its output.

Not checked: Parts 112, 117 and 118's verdicts; DS's raw reasoning; the cap Part 92's Haiku run used at run time; the
test suite.

5 blocking, 13 minor.
