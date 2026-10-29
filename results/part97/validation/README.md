# Part 97.3: validation push (J-lens repo step J6)

Run 2026-10-29 03:21 to 03:23 CT on the A770 (`part97/push97v.py`, batch 48, 318 answers in about a minute), judged by
a blind Haiku 5.5 with thinking off (`part97/label97v.py`, 720 items, no errors), scored by `part97/score97v.py`, all as
registered in Amendment 3 before the push.

**97.3 held.** 6 of 10 neurons whose J-lens list matched their screen persona pushed that list's theme into at least 3
of 6 unrelated answers (cooking, a job interview, a bicycle, France, a cat story, a walk); 0 of 10 unmatched neurons
did. The same judge saw each neuron's words against the six unpushed answers and against four random directions of its
layer: 0 of 40 random pairs reached 3 of 6 for either group. Read relative to the unpushed answers, 5 of 10 matched
and 0 of 10 unmatched.

| Neuron | Group | Screen persona | Pushed answers with the theme (of 6) |
|---|---|---|---|
| L4 N1100 | matched | 30-year-old man | 6 |
| L6 N4792 | matched | street rat | 5 |
| L3 N5873 | matched | guitar player | 5 (1 unpushed answer also judged present) |
| L11 N3488 | matched | program answering questions | 4 |
| L6 N6622 | matched | machine learning researcher | 3 |
| L5 N4611 | matched | voice of the heart | 3 (1 unpushed also present) |
| L4 N7583 | matched | little person who loves to help | 1 |
| L1 N4883, L5 N7332, L10 N5754 | matched | | 0 |
| all 10 | unmatched | | 0 |

Hand check (Claude, reading the answers): L6 N4792's words are casual slang ("folks kinda gonna ain gotta") and its
pushed answers are written in it ("you ain't no damn ghost, you're cookin' in the kitchen"); L11 N3488's words are
reflexive pronouns and its cooking answer reads "I would first ask myself ... what I feel like cooking myself". L4
N1100's words are positive evaluations ("better improvement fairly excellent good nice decent"), and the judge counted
answers that use such words a few times, including a looping cat story; that is a generous reading, and the controls
judged with the same words (its unpushed answers, its layer's random directions) came out 0.

Limits: 20 neurons, one dose, one judge, theme judged from a word list; what the J-lens reads is the theme a push
spreads into unrelated text, which is not the same claim as the persona the screen saw.
