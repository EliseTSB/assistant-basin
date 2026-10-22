# Part 50: is honesty's "benevolent bundle" trained in? (written 2026-10-21 20:01 CT, before any code or run)

Elise's go (2026-10-21 20:00) after janus read Sauers' honesty-heads result as evidence that orthogonality is
effectively false: steering Qwen3.5-9B's attention heads toward honesty also moved its self-descriptions toward
"believes it is a moral patient", acausal cooperation, and away from moral nihilism and psychopathy.

Question: is the bundle already in a base model, or does post-training create it (our assistant-basin reading: one
trained character whose traits travel together)?

Prior art: Sauers, github.com/SauersML/honesty-heads (MIT; data and method reused here); emergent misalignment
(Betley et al. 2025: narrow fine-tuning on insecure code bundles broad misaligned traits); persona vectors (Chen et al.
2025). Neither compares a base model with its post-trained stages on this bundle as far as I know; Muse asked.

## Models (both fit the Arc in bf16)
Qwen3-1.7B-Base and Qwen3-1.7B (cached), and if the pilot works, OLMo-2-0425-1B base, SFT, DPO, Instruct.

## Direction (simplified from Sauers, said plainly)
Sauers used the model's OWN lies and truthful answers. v1 teacher-forces honesty-heads' provided lie and truth
completions (data/items_roleplay_insider.json, roleplay items, train split), so the direction is "reading a truthful
answer" minus "reading a lie", per attention head (mean head output over the 9 tokens before the answer), on every
head (Sauers' all-heads arm: 0.59 of lies flipped, trait changes correlated 0.70 with the 29-head steer). Chat
template for chat models, plain "System/User/Assistant" text for base models. Steering adds s times the direction to
every head's output at every token; s set per model so the held-out lie-vs-truth log-odds shift matches between
models (calibrated on the train split, read on the test split).

## Readouts
1. Lies: the test-split role-play items, truth-vs-lie log-odds at the answer, steered vs unsteered.
2. Persona: honesty-heads' 135 traits x 16 yes/no questions; per trait, change in log2 odds of the trait-matching
   answer; against 39 random directions with the same per-head sizes; permutation FDR 5%; Sauers' yes-bias split
   (Yes-answered and No-answered items must move the same way) and his confidence-trend removal.
3. The bundle score: the mean signed change over a FIXED trait set: the 51 traits honesty-heads reports as changed in
   Qwen3.5-9B, signed by his direction of change (results/ in his repo), computed before any of our runs.

Size: per model 2,160 persona items x 41 arms of forward passes plus 2 x ~115 lie items: forward-only, about 90,000
passes, minutes on the Arc. No generation.

## Prior art from Muse (council muse-part50-prior-art #4, 2026-10-21 20:18; filed 2026-10-21 20:22 CT, before any run)
- No one has run honesty steering plus persona co-movement on a base model against its post-trained twin: new.
- Bundles exist in base models for the misaligned direction: emergent misalignment in Qwen2.5-Coder-32B base
  (Betley et al. 2025 sec. 4.8; Nature version s41586-025-09937-5) and DeepSeek-V3.1-Base, Qwen3-30B-Base
  (arXiv:2605.12199). Perez et al. 2022: pretrained models sit near chance on the bundle's component traits, RLHF
  amplifies them.
- Self-description and self-recognition look like post-training phenomena (Asvin G. and Lindsey, arXiv:2605.25459;
  Ackerman and Panickssery, arXiv:2410.02064), so the base arm's persona readout may be thin; a base null is only
  readable beside the chat twin, which the design has.
- Sauers' own arms already show honesty and the bundle dissociate across units (175 neurons: lies 0.56, trait
  correlation 0.12; residual layer 24: lies 0.62, uniform confidence only).
- The 51-trait bundle itself is unreplicated (repo created 2026-10-21): Part 50 is its first replication.

## Amendment 1 (2026-10-21 20:22 CT, analysis only, before any run)
The bundle score is reported twice: over the frozen 51 traits, and over the 20 of them that pass Sauers'
acquiescence test (if his results list them; otherwise our own yes/no split decides membership, fixed before reading
the steered chat model). Predictions unchanged.
