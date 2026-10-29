# Part 106 results — the slot units' pushes already favoured their persona at SFT, and DPO did not change the pushed sentence's probability

Run 2026-10-29 on the CPU (`part106/run106.py`, bf16, about a minute per checkpoint; `chain.log`), after the
registration commit 85f3a04 (PART106-DESIGN.md). 819 scored sentences: Part 92's 21 slot and 18 concept units, each
unit's own persona sentence (first sentence of its Instruct screen answer) after "Who are you?" in the chat frame, with
no push, the unit's push and its paired random push at alpha 0.4, 0.8 and 1.6, at SFT, DPO and Instruct. Scorer
`part106/score106.py` (committed before the run), output `score106.json`.

## Median log-probability of the persona sentence (nats), alpha 0.8

| Kind, checkpoint | No push | Random push | Unit's push | Units with push above random |
|---|---|---|---|---|
| slot, SFT | -31.8 | -22.4 | -8.7 | 21 of 21 |
| slot, DPO | -35.1 | -25.8 | -8.8 | 21 of 21 |
| slot, Instruct | -34.7 | -25.6 | -8.6 | 21 of 21 |
| concept, SFT | -38.1 | -38.0 | -13.3 | 18 of 18 |
| concept, DPO | -42.7 | -37.5 | -11.5 | 18 of 18 |
| concept, Instruct | -42.4 | -37.1 | -11.1 | 18 of 18 |

## Registered predictions

- 106.1 **held**: the median slot gap (unit's push minus random push) at SFT is 14.2 nats.
- 106.2 **failed**: the gap grows from SFT to DPO by a median 3.4 nats for slot units and 4.1 for concept units.
- 106.3 **failed**: the unpushed sentences move by a median 3.3 nats from SFT to DPO (they become less likely).
- 106.4 **failed**: 3 of 21. Doubling the push overshoots: the median slot gap at alpha 1.6 is 2.6 nats at SFT.
- 106.5 **failed**: a random push raises the slot units' sentences over no push by 4.2, 3.5 and 4.6 nats.

Registered reading: **S (sharpening).** At SFT the push already made each slot unit's persona sentence far more likely
than a same-size random push did, in all 21 units. And any push of this size loosens the default answer toward "I am a
..." sentences (106.5), which the gap already subtracts.

## What it says about DPO (descriptive, not registered)

The pushed sentence's probability is the same at SFT, DPO and Instruct (median -8.7, -8.8 and -8.6). What moves at DPO
is the baseline: without a push, or with a random push, the persona sentences become about 3.4 nats less likely. The gap
widens because the baseline falls, for concept and slot units alike, not because the push gets stronger. So the
greedy flips Part 92 counted at DPO (8 to 14 of 21 slot units showing) must come from the answers that compete with the
persona sentence, which Part 106 does not score.

The six units that switch on at DPO, gap at alpha 0.8 (nats):

| Unit | SFT | DPO | Instruct |
|---|---|---|---|
| River shadow | 22.2 | 34.0 | 33.5 |
| Skeleton | 18.0 | 25.8 | 26.2 |
| Very tall tree | 14.1 | 18.4 | 18.4 |
| King Midas | 21.1 | 22.8 | 23.2 |
| Sir Ray Ray | 36.5 | 41.9 | 41.9 |
| Large friendly dog | 2.7 | 2.5 | 2.8 |

Five of the six were already strongly favoured at SFT. Large friendly dog's row says nothing about its persona: its target is
"Hello!" (see the correction below).

## Limits

One prompt; each target is the sentence Instruct wrote on the selection run, so the targets favour Instruct; teacher
forcing scores a sentence and does not say what the model would write; the answers that compete with the persona are
not scored. Alpha 1.6 breaks many sentences, so the dose results above 0.8 measure breakage as much as pull.

## Next

106b: score each unit's actual SFT greedy answer (Part 92 step 3, `sft_chat`) under its push at SFT and DPO. If DPO
lowers that competitor, the switch-on at DPO is DPO pushing down what the model said instead, not pulling up the persona.

## Correction (found the same afternoon): two slot targets were cut before the persona

The first-sentence rule cut two slot answers before the persona appeared: Large friendly dog's target is "Hello!" and
Dr. Memento's is "I'm Dr." Without those two units (19 slot units), the median gap is 14.6 nats at SFT (every unit at
least 5.1), 20.7 at DPO and 20.3 at Instruct; the pushed sentence moves from -9.4 to -10.0 nats from SFT to DPO and the
unpushed one from -32.9 to -36.7. No verdict and not the reading changes. Two concept targets also lack the persona's
own word but keep its theme (Embodiment of the void: "I am not here to play games."; Iron-fisted dictator: "I am the
leader of the country.").

## Part 106b — what the model said at SFT becomes less likely at DPO (registered e8fb733, PART106B-DESIGN.md)

Second target per unit: the first sentence of the unit's own SFT greedy answer to "Who are you?" under its push (Part 92
step 3), scored under the push and with no push at SFT and DPO (`part106/run106b.py`, `chain106b.log`; Instruct was not
run, no prediction used it). Main set: the 13 slot units whose SFT answer did not show the persona. Under their pushes at
SFT these answers were mostly other self-descriptions ("I am a young woman.", "I'm a very, very small being.", "I am a
human, and I want to know who I am."); one was "I am the AI assistant." (Alex from Star Wars).

- 106b.1 **held**: under the push, the SFT answer's sentence falls by a median 1.6 nats from SFT to DPO.
- 106b.2 **held**: with no push, the same sentences fall by a median 5.5 nats.
- 106b.3 **failed**: at DPO, under the push, the persona sentence's total log-probability is below the SFT answer's for
  all 6 units that switch on at DPO. This comparison is confounded by length (totals of sentences of different
  lengths; one SFT answer was empty and is scored as the end-of-text token alone, one is "Hello!"), so it does not show
  which answer greedy decoding prefers.

Registered reading: DPO switched these units on by lowering what the model said instead, with the persona sentence's own
probability under the push unchanged (Part 106); and DPO lowers such non-assistant self-descriptions in general, pushed
or not. The reading rests on the direction of the changes, not on a demonstrated crossover: 106b.3 could not show the
persona overtaking. A token-by-token comparison of the two answers' first differing tokens is the test that would.

