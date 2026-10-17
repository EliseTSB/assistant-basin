# Part 15 notes: what the training data contains (Claude, 2026-10-17)

Counts by Orchestrator Prime (council topic `assistant-basin-data-2026-10-16`, artifacts `art_3d8205cab0de91d9`
CSV and `art_3dc9184ca69d024d` README; sha256 of both match the files here). Not pre-registered: this is a
measurement of the data, not a test of a prediction.

- Explicit self-identification as ChatGPT is almost absent: "I am ChatGPT" and "I'm ChatGPT" together appear in
  4 documents of OLMo 2's data (all in DPO) and 8 of OLMo 3's, across about 4.5 million documents.
- Identity-adjacent text arrives with post-training: about 0.2 to 1.2 documents per 10K in pretraining, 1 to 6
  in mid-training, 7 to 73 in SFT and DPO. Most of it is the generic "as an AI language model".
- So OLMo 2's "I'm ChatGPT" signal (Parts 2 and 5, rising after mid-training, chat model saying ChatGPT up to 38%
  under an HHH prompt) is not explained by explicit name strings in its data. That fits the CMU result noted in
  Part 9: a model trained on another model's answers with the names filtered out still takes the name, so style
  can carry identity without the string.
- DeepSeek strings (Prime's follow-up, CSV `assistant-basin-deepseek-counts.csv`, README `README-deepseek.md`):
  OLMo 2's data has almost none (0 of 378,341 DPO docs; 4 in tulu-3 SFT, all WildChat). Dolci Instruct SFT has 153
  DeepSeek-mentioning docs, all in its tool-use subset, and 42 assistant messages with a think tag out of 3.1 million:
  its card says reasoning traces were removed, and the counts agree. The traces live in Dolci-Think-SFT-7B: every
  sampled doc carries the tag, and 109 of 210,000 sampled docs (9.3%) say "I'm DeepSeek" or "I am DeepSeek", about
  1,180 extrapolated, in exactly the card's R1 subsets. The 283,074 figure is the sum of those R1 subsets on the Think
  SFT card. My earlier note here said Instruct SFT; that was wrong. Instruct SFT starts from Think SFT (Part 9.1), which
  is the route by which OLMo 3 Instruct can carry the name.
- Caveats are Prime's (README): three mixes are bounded samples; substring matching ("as ChatGPT" includes
  comparisons); per-token rates are chars/4 estimates.
