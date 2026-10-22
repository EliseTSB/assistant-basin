# Part 55: which Claude tier is good enough for the bulk labelling in the neuron work? (registered 2026-10-22 00:44 CT, before running)

Elise, 2026-10-22 00:43 CT: how do Opus, Sonnet and Haiku benchmark on the bulk neuron work, with about $89 of promo credit expiring on 10-24.
Nothing has compared the tiers on our own tasks. The reference labels come from Opus 5.5 (the reader used so far), so this
measures agreement with it and on known-answer calibration texts, not truth; the calibration texts are the only items with known answers.
- Tasks, all from committed data: T1 the 454-item Part 26.8 origin batch (432 answers, 22 known-answer calibration texts; three-way
  EARTH / OFFWORLD / NEITHER plus OK / BROKEN; blind_origin.py's prompt); T2 the 1,812-item Part 49/49b broken-answer batch (1,792 answers, 20 calibration;
  BROKEN / OK; blind_reader.py's prompt). Both batches' Opus labels (default thinking setting) are the reference.
- Arms: claude-haiku-5-5, claude-sonnet-5-5, and claude-opus-5-5 again (the noise floor of Opus against its own earlier labels). Same prompt,
  same items, thinking explicitly disabled in all three (Haiku 5.5 thinks by default, so an unset field would confound the arms), max_tokens 1024,
  Message Batches API. One batch per arm. Cost recorded from the returned usage.
- Measures: agreement with the reference label per task (exact two-label match for T1); disagreement table; Cohen's kappa on T2; calibration accuracy.

| id | prediction (Claude) |
|---|---|
| 55a | T2: Haiku agrees with the Opus reference on at least 95% of the 1,792 answers and Sonnet on at least 97%. |
| 55b | T1: Haiku agrees on at least 85% of the 432 answers (both labels) and Sonnet on at least 92%. |
| 55c | Calibration: Sonnet and the Opus rerun get every calibration text right (22 of 22 and 20 of 20); Haiku gets at most one wrong in each task. |
| 55d | At least 60% of Haiku's T1 disagreements involve the NEITHER label on one side. |
| 55e | The Opus rerun with thinking disabled agrees with its own earlier labels on at least 97% of T1 and of T2 answers. |
| 55f | Per item cost ordering is Haiku < Sonnet < Opus with Haiku at most one fifth of Opus. |
