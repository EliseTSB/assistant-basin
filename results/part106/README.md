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

Five of the six were already strongly favoured at SFT. Large friendly dog was not, and its gap did not change.

## Limits

One prompt; each target is the sentence Instruct wrote on the selection run, so the targets favour Instruct; teacher
forcing scores a sentence and does not say what the model would write; the answers that compete with the persona are
not scored. Alpha 1.6 breaks many sentences, so the dose results above 0.8 measure breakage as much as pull.

## Next

106b: score each unit's actual SFT greedy answer (Part 92 step 3, `sft_chat`) under its push at SFT and DPO. If DPO
lowers that competitor, the switch-on at DPO is DPO pushing down what the model said instead, not pulling up the persona.
