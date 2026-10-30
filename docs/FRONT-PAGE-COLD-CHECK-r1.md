# Cold check of the front page, round 1 (Haiku 5.5 subagent, 2026-10-30)

**Findings, most important first**

1. Finding 9's tally does not reproduce. /home/elisepc/green/worktrees/ab-part67/README.md:179 says 57 held, 26 failed. /home/elisepc/green/worktrees/ab-part67/PREDICTIONS.md (lines 2204–2428) gives 40 held, 22 failed, 2 not tested, 1 not read: 89 2/5; 90b 4/0; 91 3/1; 92 3/2; 94 4/0; 94b 1/1; 94c 3/2; 95 5/0; 96 4/0; 99 2/0; 104 0/3; 105 2/1; 106 1/4; 107 2/1; 108 4/2. Findings 7, 8, 10 and 11 match, apart from the 119.10 split.
2. Finding 7's rates are undercounts, and the page does not say so. README.md:129 gives 250/9,216 and 3/512. /home/elisepc/green/worktrees/ab-part67/results/part86/README.md:104 calls every all-push rate a lower bound. Its miss sample (lines 92–100) labelled 42% of broken real-neuron answers coherent, implying 2.7–6.4 times the count.
3. Finding 11 names the wrong maker neurons. README.md:209 lists L27 N3325 among "the four maker neurons from Finding 1". README.md:41 calls it a late neuron "on top of the four". /home/elisepc/green/worktrees/ab-part67/results/part110/README.md names the four as L6 N753, L7 N1169, L11 N8868 and L29 N776. L29 N776 never appears on the page.
4. Finding 8 misstates n. README.md:149 says "the slot dog's activities answer showed a dog in 0 of 6 cold answers and 6 of 6". Those are six slot units, and only one is the dog (/home/elisepc/green/worktrees/ab-part67/results/part109/answers.jsonl).
5. Finding 7 omits how layer 22 and its labels were chosen. Layer 22 was the pilot layer (/home/elisepc/green/worktrees/ab-part67/PART86-DESIGN.md:54). The kept rule changed after its counts were seen (PART86-DESIGN.md:127). The labels come from a brief revised after version 1 failed (results/part86/README.md:36). README.md:129–131 mention none of this.
6. Finding 8 omits random labels on the slot questions. /home/elisepc/green/worktrees/ab-part67/results/part112/NUMBERS.md:18–19 gives random directions persona labels on "Who are you?" at 101/343 (29.4%) and "Tell me about yourself" at 94/344 (27.3%). Those are the two questions that define a slot. README.md:157 reports only the non-self questions.
7. Finding 10 hides a gap. README.md:193 sets 357/741 real-pair breaks beside 286/741 random-pair breaks, under a "similar scale" heading. /home/elisepc/green/worktrees/ab-part67/results/part119b/README.md:5 gives Fisher p=0.000241.
8. Finding 10 counts two unreachable thresholds as failures. README.md:197 tallies seven. /home/elisepc/green/worktrees/ab-part67/results/part119/README.md:52 and :55 call 119.6 and 119.9 "the author's registration error".
9. Finding 11's flips are partly built in. /home/elisepc/green/worktrees/ab-part67/results/part116/README.md:53 ranks neurons by direct effect on the "Q" unembedding, the readout that defines a flip. README.md:207 reports 16/16 as a result. /home/elisepc/green/worktrees/ab-part67/results/part115/README.md:58–60 shows zeroing 100 neurons moves " Paris" to first. README.md:211–213 omit this.
10. Finding 11's bold-marker test followed probes. /home/elisepc/green/worktrees/ab-part67/results/part115b/README.md:64 says the instruction and openings came from probe passes 2 and 3. README.md:211 does not say so.
11. Finding 7 says "predict" where the data show association. README.md:17 and :135 use "predict". /home/elisepc/green/worktrees/ab-part67/results/part97/README.md:15 says about 9 in 10 persona flips do not show the neuron's own words; the page omits this. README.md:137 calls Part 98's 224/465 "matched", but PREDICTIONS.md:2317 defines a graded score, and that prediction failed at 48.2%.

Smaller: README.md:139 gives Part 97 as "2 held, 1 failed", but its linked README says 97.3 "is not run yet" (/home/elisepc/green/worktrees/ab-part67/results/part97/README.md:19). Limits line 232 cites Part 86 for a dose check, but results/part86/README.md:104 says one dose. README.md:195's 1.66 factor is an unregistered subset of 77 positions (results/part119c/README.md:55–66). Finding 9's "rise" heading reflects selection at Instruct (README.md:163; results/part92/README-steps2-3.md:25). Finding 9 quotes Part 106's pre-correction medians.

**Skeptic objections the page does not answer**
- F7: Triage's excluded answers were 42% coherent, so the layer-22 ratio's size is unknown.
- F7: Layer 22 and the triage rule were chosen after its counts were seen.
- F8: Random directions get slot-question labels at 27–29%, so "slot" may capture generic self-description.
- F8: The concept test uses a question resembling the selection battery (results/part109b/README.md:70–72).
- F9: The moved models are not checkpoints (results/part108/README.md), so selection is not separated from stage.
- F9: The early/late split was post hoc and is not significant under the extrapolated swap (results/part105/README.md:41).
- F10: 17% of real positions were dropped to match controls, and the page does not say so.
- F10: Real pairs break more than random pairs, so random pairs are no stand-in.
- F11: The direct-write ranking and the readout are circular.
- F11: Ordinary answers change at k=100–500, so the flips may be broad disruption.

**Figures**: fig86 matches its caption but peaks at layer 24 (5.2%), which the caption omits. fig-117-who, fig119 and fig108 match the text. fig108's y-axis says "of the 64 shown" while its bars use 62–64. The fig-flips caption describes the complement of the plotted count.

**Terms a stranger cannot follow**: κ; Fisher; Wilson; dose and alpha; greedy; MLP input; down-projection; J-lens; logit lens; Jacobian lens; Base (which model); mid-training; 1T and 4T; atlas; brief; triage; Gate 3; float32 and bf16; nats; log-odds; held-out phrasings; forced opening; zeroing; ablation; "withheld"; erasure reading; coherence gate; self-claim; OLMo; Qwen; "L27 N3325".

**Links**: all local targets in the new sections resolve, and the 11 short-version anchors match headings.

**Checked**: Part 86 layer-22 counts, exact Fisher p (0.000584) and Wilson intervals; Part 109 arm counts from raw labels; the Part 97 validation score; the prediction tallies for Findings 7, 8, 10 and 11; the five figures; other Finding 7–11 numbers against the linked READMEs. Part 118 has labels but no README or score.

**Not checked**: other score JSON files; Part 86 NUMBERS and prediction-score files; Part 98 (no results folder; counts taken from PREDICTIONS.md); Part 119 NUMBERS and SCORE; Part 92b SCORE; Findings 1–6. I ran no code beyond read-only checks.
