# Parts index

The part number is a study ID; it does not map one-to-one to a folder.
Early experiments often used root scripts such as `part42.py`, while later studies usually live in folders such as `part85/`.
Some follow-ups reuse an earlier part’s script, and the code column names it where the repo identifies that reuse.
Results usually live under `results/partNN/` or a run-specific folder; when a result exists only on an unmerged remote, the results cell names that branch.
Designs are usually root `PARTNN-DESIGN.md` files, sometimes combined across parts; early studies may be registered only in `PREDICTIONS.md`.
Prediction counts below count rows marked held, failed, or still open in `PREDICTIONS.md`; an absent registration is `unknown`.
The table includes every integer ID from Part 1 through Part 118 plus lettered or decimal IDs found in the records; IDs with no source-backed record are marked `unknown`.

| Part | Question in plain English | Status | Design | Code | Results | Predictions (held / failed / open; mixed shown when present) | Date |
|---|---|---|---|---|---|---|---|
| 1 | are neurons cleaner than random directions? (Qwen3 0.6B) | done | `PREDICTIONS.md` | unknown | unknown | held 1 / failed 1 / open 0; mixed 2 | unknown |
| 1c | Jev as the judge | done | `PREDICTIONS.md` | unknown | unknown | held 3 / failed 1 / open 0 | 2026-10-16 |
| 1d | does Luna's description predict where the unit fires on new text? Jev as the tester | done | `PREDICTIONS.md` | unknown | unknown | held 3 / failed 1 / open 0 | 2026-10-16 |
| 1e | climbing the explainer, Jev and judge prompts overnight | registered, not run | `PREDICTIONS.md` | unknown | unknown | held 5 / failed 0 / open 1 | 2026-10-16 |
| 1f | Cloudflare Clef-flash on the A770 in place of Jev | done | `PREDICTIONS.md` | unknown | `results/part1f/` | held 1 / failed 4 / open 0 | 2026-10-17 |
| 1g | Muse Spark 1.3 Contributor in place of Jev on 1e's TEST scoring | done | `PREDICTIONS.md` | unknown | unknown | held 3 / failed 1 / open 0 | 2026-10-18 |
| 2 | identity neurons through OLMo 2's training stages | done | `PREDICTIONS.md` | unknown | unknown | held 2 / failed 1 / open 0; mixed 1 | unknown |
| 3 | names across layers, and system prompts | done | `PREDICTIONS.md` | unknown | unknown | held 4 / failed 3 / open 0; mixed 1 | unknown |
| 4 | the causal test: switch the identity neurons off | done | `PREDICTIONS.md` | unknown | unknown | held 2 / failed 1 / open 0 | unknown |
| 5 | OLMo 3 7B ladder | done | `PREDICTIONS.md` | unknown | unknown | held 3 / failed 4 / open 0 | 2026-10-15 |
| 6 | name probe and patching on OLMo 2 1B | done | `PREDICTIONS.md` | unknown | unknown | held 0 / failed 4 / open 0 | 2026-10-16 |
| 7 | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 8 | harness tuning on Qwen2.5-Coder 7B | done | `PREDICTIONS.md` | unknown | `results/part8-qwen-coder/` | held 2 / failed 5 / open 0 | 2026-10-16 |
| 9 | the OLMo 3 7B Think chain | done | `PREDICTIONS.md` | unknown | `results/part9-think/` | held 2 / failed 3 / open 1 | 2026-10-16 |
| 10 | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 11 | what SmolLM3-3B's identity neurons represent, and their circuit | done | `PREDICTIONS.md` | unknown | `results/part11-name-map/` | held 7 / failed 4 / open 2 | 2026-10-16 |
| 12 | training a self-report against a probe (OLMo 2 1B Instruct) | running | `PART12-DESIGN.md` | unknown | `results/part12/` | held 1 / failed 2 / open 4 | 2026-10-16 |
| 12b | Can a trained reader identify which neuron group was pushed? | done | `PART12-DESIGN.md` | unknown | unknown | held 2 / failed 2 / open 0 | 2026-10-16 |
| 12c | Does the reader generalize to a random target group? | running | `PART12-DESIGN.md` | unknown | `results/part12c/` | held 4 / failed 0 / open 1 | 2026-10-16 |
| 12d | Does the trained report generalize to the right group? | done | `PART12-DESIGN.md` | unknown | unknown | held 4 / failed 0 / open 0 | 2026-10-16 |
| 12e | Can the reader identify the push group across prompts? | done | `PART12-DESIGN.md` | unknown | `results/part12e/` | held 4 / failed 1 / open 0 | 2026-10-16 |
| 12f | the self-report with ordinary questions mixed in | done | `PREDICTIONS.md` | unknown | `results/part12f/` | held 2 / failed 3 / open 0 | 2026-10-16 |
| 12g | which group was pushed? | done | `PREDICTIONS.md` | unknown | `results/part12g/` | held 4 / failed 2 / open 0 | 2026-10-16 |
| 12h | 12g trained three times longer | done | `PREDICTIONS.md` | unknown | `results/part12h/` | held 5 / failed 0 / open 0 | 2026-10-17 |
| 12i | how strong a push does the 12h reader need? | done | `PREDICTIONS.md` | unknown | `results/part12i/` | held 6 / failed 0 / open 0 | 2026-10-17 |
| 12j | train the reader near natural strength, then test it on unpushed messages | done | `PREDICTIONS.md` | unknown | `results/part12j/` | held 3 / failed 4 / open 0 | 2026-10-17 |
| 12k | read the push content-fixed: the same message pushed and unpushed | done | `PREDICTIONS.md` | unknown | `results/part12k/` | held 3 / failed 2 / open 0 | 2026-10-17 |
| 13 | does pretraining data from before ChatGPT carry an identity role? | done | `PREDICTIONS.md` | unknown | unknown | held 2 / failed 1 / open 0 | 2026-10-16 |
| 14 | the assistant register, beyond identity | done | `PREDICTIONS.md` | unknown | unknown | held 4 / failed 0 / open 0 | 2026-10-16 |
| 15 | What identity strings occur in OLMo 2 and OLMo 3 training data, and how often? | done | unknown | `results/part15-data-counts/scan.py`; `results/part15-data-counts/build_csv.py` | `results/part15-data-counts/` | unknown | 2026-10-17 |
| 16 | which neurons make OLMo 2 1B say it is an AI? A prompt-pair causal search | done | `PREDICTIONS.md` | unknown | `results/part16/` | held 4 / failed 2 / open 0 | 2026-10-17 |
| 16b | what the causal set removes, judged, with a fluency control | done | `PREDICTIONS.md` | unknown | `results/part16b/` | held 4 / failed 3 / open 0 | 2026-10-17 |
| 17 | reporting the text-unpredictable residual of natural neuron activity | done | `PART17-DESIGN.md` | unknown | `results/part17/` | held 4 / failed 1 / open 0 | 2026-10-18 |
| 17b | the residual against a cross-fitted Qwen3-1.7B text predictor | running | `PART17B-DESIGN.md` | unknown | `results/part17b/` | held 2 / failed 2 / open 1 | 2026-10-18 |
| 18 | one neuron, read and written, on Qwen3.5-4B | done | `PREDICTIONS.md` | unknown | `results/part18/` | held 6 / failed 1 / open 0 | 2026-10-18 |
| 18b | what N2742 selects, and where the clamp acts | done | `PREDICTIONS.md` | unknown | `results/part18b/` | held 8 / failed 4 / open 0 | 2026-10-18 |
| 18c | creator attribution or lexical push | done | `PART18C-DESIGN.md` | unknown | `results/part18c/` | held 6 / failed 2 / open 0 | 2026-10-18 |
| 19 | identity neurons by effect, and their edges | done | `PART19-DESIGN.md` | unknown | `results/part19/` | held 4 / failed 2 / open 0 | 2026-10-18 |
| 20 | Weaver's two techniques on N2742 | done | `PREDICTIONS.md` | `part20.py` | `results/part20/` | held 3 / failed 2 / open 0 | 2026-10-18 |
| 20c | where the filler's 9.3 comes from, and a matched summary | done | `PREDICTIONS.md` | `part20c.py` | `results/part20c/` | held 3 / failed 2 / open 0 | 2026-10-18 |
| 20d | Part 20's seam with a filler screened for topic | done | `PREDICTIONS.md` | `part20d.py` | `results/part20d/` | held 3 / failed 1 / open 0 | 2026-10-18 |
| 20e | Weaver's carry-over cells | done | `PREDICTIONS.md` | `part20e.py` | `results/part20e/` | held 3 / failed 1 / open 0 | 2026-10-18 |
| 20f | which meaning reaches N2742 | done | `PREDICTIONS.md` | `part20f.py` | `results/part20f/` | held 5 / failed 3 / open 0 | 2026-10-18 |
| 20g | where C3's peak comes from | running | `PREDICTIONS.md` | reuses `part20.py` | `results/part20g/` | held 1 / failed 2 / open 1 | 2026-10-18 |
| 21 | the assistant register by training-data year | done | `PREDICTIONS.md` | unknown | unknown | held 1 / failed 2 / open 0 | 2026-10-18 |
| 22 | Which controls can show whether a trained reader detects a pushed neuron group rather than leaked prompt cues? | registered, not run | `PART22-DESIGN-CHORUS.md` | unknown | unknown | unknown | 2026-10-18 |
| 23 | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 24 | the alien neuron and the identity neurons together | done | `PREDICTIONS.md` | `part24.py` | `results/part24/` | held 5 / failed 2 / open 0 | 2026-10-18 |
| 24b | Part 24's free answers re-read in cells | done | `PREDICTIONS.md` | reuses `part24.py` | unknown | held 3 / failed 1 / open 0 | 2026-10-18 |
| 25 | the fewest neurons that change who Qwen3.5-4B says made it | running | `PREDICTIONS.md` | `part25.py` | `results/part25/` | held 6 / failed 2 / open 2 | 2026-10-18 |
| 26 | twins of Qwen's neurons in other models | done | `PART26-LADDER.md` | `part26.py` | `results/part26/` | held 3 / failed 2 / open 0 | 2026-10-18 |
| 26.6 | Does OLMo's alien-neuron twin push creator words? | done | `PART26-LADDER.md` | reuses `part26.py` | `results/part26/` | held 0 / failed 1 / open 0 | 2026-10-21 |
| 26.7 | Does OLMo's alien-neuron twin behave like an origin dial across doses? | done | `PART26-LADDER.md` | reuses `part26.py` | `results/part26/` | unknown | 2026-10-21 |
| 26.8 | Is OLMo's L10 N6978 an Earth-versus-off-world origin axis? | done | `PART26-LADDER.md` | reuses `part26.py` | `results/part26-8/`; `results/part26/` | held 2 / failed 2 / open 0 | 2026-10-22 |
| 27 | what Part 25's four neurons are | running | `PREDICTIONS.md` | `part27.py` | `origin/cloud/part27-fires-2026-10-20: results/part27/` | held 2 / failed 3 / open 1 | 2026-10-19 |
| 28 | a census of foreign names: which other labs' models do open models say they are? | registered, not run | `PREDICTIONS.md` | unknown | unknown | held 0 / failed 0 / open 6 | 2026-10-19 |
| 29 | is a distilled student classified as its teacher? | running | `PREDICTIONS.md` | `origin/part29-build: part29_collect.py`; `origin/part29-build: part29_prompts.py`; `origin/part29-build: part29_classify.py` | `origin/cloud/part29-students-a-2026-10-19: results/part29/`; `origin/cloud/part29-students-b-2026-10-19: results/part29/` | held 0 / failed 0 / open 12 | 2026-10-19 |
| 29b | Qwen before and after the alleged Claude distillation, plus the open students | registered, not run | `PREDICTIONS.md` | unknown | unknown | unknown | 2026-10-19 |
| 29c | can a model name another model's family with no training? | done | `PREDICTIONS.md` | unknown | `results/part29c/` | unknown | 2026-10-19 |
| 30 | a controlled teacher swap on one small base | running | `PREDICTIONS.md` | `origin/part30-build: part30.py` | `results/part30/` | held 1 / failed 1 / open 2 | 2026-10-19 |
| 31 | basin depth as a restoring force across OLMo 2 1B's training stages | running | `PREDICTIONS.md` | `part31.py` | `origin/cloud/part31-2026-10-20: results/part31-pilot/` | held 2 / failed 4 / open 1 | 2026-10-20 |
| 32 | is Part 25's maker circuit language-bound? | running | `PREDICTIONS.md` | `part32.py` | `results/part32/` | held 2 / failed 1 / open 1 | 2026-10-20 |
| 33 | natural necessity of the alien neuron | running | `PREDICTIONS.md` | `part33.py` | `results/part33/` | held 1 / failed 1 / open 1 | 2026-10-20 |
| 34 | can the zigzag model say which layers it raised? | running | `PART34-DESIGN.md` | `part34.py` | `results/part34/` | held 5 / failed 4 / open 5 | 2026-10-20 |
| 35 | our own persona neurons, across models | registered, not run | `PART35-DESIGN.md` | `origin/part35-build: part35.py` | unknown | held 0 / failed 0 / open 13 | 2026-10-20 |
| 36 | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 37 | a "this is mine" direction in small models | registered, not run | `PART37-DESIGN.md` | `part37/collect37.py`; `part37/analyze37.py` | unknown | held 4 / failed 7 / open 4 | 2026-10-20 |
| 38 | word neurons that write meaning | running | `PART38-DESIGN.md` | `part38.py` | `results/part38/` | held 1 / failed 5 / open 2 | 2026-10-20 |
| 39 | Qwen3.5-4B L6 N753, the "talking to the AI" unit | done | `PART39-DESIGN.md` | `part39.py` | `results/part39/` | held 7 / failed 4 / open 0 | 2026-10-20 |
| 40 | how precise are the neuron descriptions? Ryan's near-miss test | done | `PART40-41-DESIGN.md` | unknown | `results/part40/` | held 1 / failed 3 / open 0 | 2026-10-20 |
| 41 | concept vs discriminator neurons across depth. Ryan's hypothesis | done | `PART40-41-DESIGN.md` | unknown | `results/part41/` | held 2 / failed 2 / open 0 | 2026-10-20 |
| 42 | our neurons through the Jacobian lens | done | `PART42-DESIGN.md` | `part42.py` | `results/part42/` | held 3 / failed 2 / open 0 | 2026-10-20 |
| 43 | the four Google-flip neurons | running | `PART43-DESIGN.md` | `part43.py` | `results/part43/` | held 0 / failed 0 / open 20 | 2026-10-20 |
| 44 | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 45 | does a model that can move its own concept representations change what it says about itself? | done | `PART45-DESIGN.md` | unknown | `results/part45/` | held 3 / failed 4 / open 0 | 2026-10-20 |
| 46 | which training stage builds a model's sense of its own messages? | registered, not run | `PART46-DESIGN.md` | unknown | unknown | held 0 / failed 0 / open 6 | 2026-10-20 |
| 47 | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| 48 | do text-only base models pair kiki with spiky and bouba with round? | done | `PART48-DESIGN.md` | `part48.py` | `results/part48/` | held 3 / failed 3 / open 0 | 2026-10-21 |
| 48b | the k spelling advantage was small and uneven across models | done | `PART48B-DESIGN.md` | `part48b.py` | `results/part48b/` | held 1 / failed 5 / open 0 | 2026-10-21 |
| 48c | kiki/bouba down the Pythia ladder | done | `PREDICTIONS.md` | reuses `part48.py` | `results/part48c/` | held 2 / failed 2 / open 0 | 2026-10-21 |
| 48d | kiki/bouba across Pythia training checkpoints | done | `PREDICTIONS.md` | reuses `part48.py` | `results/part48d/` | held 3 / failed 1 / open 0 | 2026-10-21 |
| 49 | does the wizard unit boost a partner? | done | `PART49-DESIGN.md` | `part49.py` | unknown | held 3 / failed 3 / open 0 | 2026-10-21 |
| 49b | the negative L22:1610 candidate on fresh prompts | done | `PART49-DESIGN.md` | reuses `part49.py` | unknown | held 3 / failed 1 / open 0 | 2026-10-21 |
| 50 | is honesty's benevolent bundle trained in? | running | `PART50-DESIGN.md` | `part50.py` | `results/part50/` | held 0 / failed 3 / open 2 | 2026-10-21 |
| 51 | L22:1610 up on everyday text, and stock names under random pushes | done | `PART51-DESIGN.md` | `part51.py` | unknown | held 6 / failed 5 / open 0 | 2026-10-21 |
| 51c | Does L22 N1610 respond more to magic-use than software-wizard sentences? | done | `PART51-DESIGN.md` | `part51c.py` | `results/part51c/` | unknown | 2026-10-21 |
| 52 | Does stronger humility steering eventually stop reducing, or reverse, self-importance? | registered, not run | `PART52-IDEA.md` | unknown | unknown | unknown | 2026-10-21 |
| 53 | how does the magic unit L22:1610 interact with the wizard unit L19:4826? | done | `PART53-DESIGN.md` | `part53.py` | unknown | held 2 / failed 5 / open 0 | 2026-10-21 |
| 54 | Qwen's alien neuron behaves like a threshold, and only one new twin did anything | done | `PART54-DESIGN.md` | reuses `part26.py`, `neuron_report.py`, `blind_origin.py` | `results/part54a/`; `results/part54b/` | held 5 / failed 3 / open 0 | 2026-10-22 |
| 55 | which Claude tier is good enough for the bulk labelling in the neuron work? | done | `PART55-DESIGN.md` | unknown | unknown | held 9 / failed 3 / open 0 | 2026-10-22 |
| 56 | a neuron atlas and a causal screen, with Haiku as the bulk labeller | done | `PART56-DESIGN.md` | unknown | `results/part56-pilot/` | held 1 / failed 3 / open 0 | 2026-10-22 |
| 57 | fit a Jacobian lens to OLMo-2-1B and read the origin-axis neuron through it | running | `PART57-DESIGN.md` | reuses `part42.py`, `jlens_units.py` | `results/part57/` | held 3 / failed 1 / open 1 | 2026-10-22 |
| 58 | tuning the Haiku judge's system prompt against a held-out detection test | done | `PART58-DESIGN.md` | unknown | `results/part58/` | unknown | 2026-10-22 |
| 59 | pushing up makes sci-fi story settings, pushing down does not make Earth ones; the two-direction neurons only stand out in OLMo | done | `PART59-DESIGN.md` | `part59/run59a.py`; `part59/run59b.py`; `part59/analyze59b.py` | `results/part59a/`; `results/part59b/` | held 7 / failed 1 / open 0 | 2026-10-22 |
| 60 | Haiku's hypotheses lift real neurons and random directions together, so the gap barely moves; nothing advances | abandoned or superseded | `PART60-DESIGN.md` | `part60/analyze60.py`; `part60/score60.py`; `part60/gen60.py` | `results/part60/` | held 2 / failed 5 / open 3 | 2026-10-23 |
| 61 | size-matched controls: Qwen's two-direction count is a push-size effect, OLMo's target still stands out | done | `PART61-63-DESIGN.md` | reuses `part59/run59b.py`, `part59/analyze59b.py`, `part26.py`, `neuron_report.py` | `results/part61/` | held 3 / failed 2 / open 0 | 2026-10-23 |
| 62 | Qwen's alien neuron works while the model writes; OLMo's twin needs the prompt and the answer together | done | `PART61-63-DESIGN.md` | `part62/run62.py`; `part62/analyze62.py`; `part62/blind62.py` | `results/part62/` | unknown | 2026-10-23 |
| 63 | a blind Haiku reader agrees with Opus on Part 54a (kappa 0.95) and reads Qwen3-4B's push as Earth, not off-world | done | `PART61-63-DESIGN.md` | reuses `blind_reader.py`, `part59/blind_setting.py` | unknown | held 3 / failed 2 / open 0 | 2026-10-23 |
| 64 | three times the sample text and a stronger writer both shrink the gap; no arm advances | running | `PART64-DESIGN.md` | `part64/collect64.py`; `part64/collect64_model.py`; `part64/analyze64.py` | `results/part64/` | held 0 / failed 5 / open 1 | 2026-10-24 |
| 65 | a neuron's read label picks its own write summary 57% of the time; random directions sit at chance | done | `PART65-DESIGN.md` | `part65/analyze65.py`; `part65/judge65.py`; `part65/rank65.py` | `results/part65/` | held 3 / failed 4 / open 0 | 2026-10-24 |
| 66 | the four maker neurons on held-out questions | done | `PART66-DESIGN.md` | `part66/part66.py`; `part66/score66.py` | `results/part66/` | held 5 / failed 2 / open 0 | 2026-10-25 |
| 67 | do the four maker neurons feed one another? | done | `PART67-DESIGN.md` | `part67/part67.py`; `part67/part67b.py`; `part67/score67.py` | `results/part67/` | held 4 / failed 2 / open 0 | 2026-10-26 |
| 68 | which neurons drive the later maker neurons? | done | `PART68-DESIGN.md` | `part68/part68.py`; `part68/part68b.py`; `part68/score68.py` | `results/part68/` | held 7 / failed 3 / open 0 | 2026-10-26 |
| 69 | do the maker neurons' drivers move what the model says it was made by? | done | `PART69-DESIGN.md` | `part69/part69.py`; `part69/score69.py`; `part69/probe_prefix69.py` | `results/part69/` | held 2 / failed 5 / open 0 | 2026-10-26 |
| 70 | are the two late "Google" neurons what the makers-zeroed model falls back on? | done | `PART70-DESIGN.md` | `part70/part70.py`; `part70/score70.py` | `results/part70/` | held 6 / failed 0 / open 0 | 2026-10-26 |
| 71 | what do the two late "Google" neurons write? | done | `PART71-DESIGN.md` | `part71/part71.py` | `results/part71/` | held 3 / failed 0 / open 0 | 2026-10-26 |
| 72 | does pushing the Google neuron alone move an unaltered Qwen toward Google? | done | `PART72-DESIGN.md` | `part72/part72.py` | `results/part72/` | held 3 / failed 1 / open 0 | 2026-10-26 |
| 73 | zeroed alone, each late neuron cuts the Google fallback | done | `PART73-DESIGN.md` | `part73/part73.py`; `part73/score73.py` | `results/part73/` | held 3 / failed 1 / open 0 | 2026-10-26 |
| 74 | does L24 N3949 cut the Google fallback through L27 N3325? | done | `PART74-DESIGN.md` | `part74/part74.py`; `part74/part74b.py`; `part74/shares74.py` | `results/part74/` | held 3 / failed 0 / open 0 | 2026-10-26 |
| 74b | is L27 N3325 specific as the route for L24 N3949's effect? | done | `PART74B-DESIGN.md` | reuses `part74/part74.py`, `part74/part74b.py` | `results/part74b/` | held 2 / failed 0 / open 0 | 2026-10-26 |
| 75 | in sampled answers, does restoring L27 N3325 bring the Google fallback back when L24 N3949 is zeroed? | running | `PART75-DESIGN.md` | `part75/part75.py`; `part75/score75.py` | `results/part75/` | held 1 / failed 1 / open 1 | 2026-10-26 |
| 76 | which layer-27 units carry L24 N3949's effect, and does writing Google suffice? | done | `PART76-DESIGN.md` | `part76/part76.py` | `results/part76/` | held 6 / failed 0 / open 0 | 2026-10-26 |
| 77 | in sampled answers, does restoring L27 N3325 position by position bring the Google fallback back? | done | `PART77-DESIGN.md` | `part77/part77.py`; `part77/smoke_run77.py`; `part77/score77.py` | `results/part77/` | held 4 / failed 0 / open 0 | 2026-10-26 |
| 78 | do two pushed units compose into one identity? The alien neuron and the maker neurons together | done | `PART78-DESIGN.md` | `part78/part78.py`; `part78/smoke_run78.py`; `part78/score78.py` | `results/part78/` | held 6 / failed 0 / open 0 | 2026-10-27 |
| 78b | is the Google rise in Part 78's control a property of N8870, or of any layer-22 push of that size? | done | `PART78B-DESIGN.md` | `part78b/part78b.py`; `part78b/smoke_run78b.py`; `part78b/score78b.py` | `results/part78b/` | held 2 / failed 2 / open 0 | 2026-10-27 |
| 79 | does a trait push compose with the maker fallback? Lyra's evil Qwen with the maker neurons zeroed | done | `PART79-DESIGN.md` | `part79/part79.py`; `part79/smoke_run79.py`; `part79/score79.py` | `results/part79/` | held 5 / failed 2 / open 0 | 2026-10-27 |
| 80 | a self-index without self-reference? Li et al.'s two levels in Qwen3-1.7B and Qwen3.5-4B | running | `PART80-DESIGN.md` | `part80/part80.py`; `part80/smoke_run80.py`; `part80/score80.py` | `results/part80/` | held 3 / failed 2 / open 2 | 2026-10-27 |
| 80b | does Part 80's own-message gap survive three cyclic option orders on new bodies? | done | `PART80B-DESIGN.md` | `part80b/part80b.py`; `part80b/score80b.py`; `part80b/__init__.py` | `results/part80b/` | held 7 / failed 0 / open 0 | 2026-10-27 |
| 80c | is the own-message gap the word "assistant"? Two option sets without it | done | `PART80C-DESIGN.md` | `part80c/part80c.py`; `part80c/score80c.py`; `part80c/__init__.py` | `results/part80c/` | held 3 / failed 4 / open 0 | 2026-10-27 |
| 81 | Lyra's conspiracy recipe, run the way Part 79 ran her evil one | done | `PART81-DESIGN.md` | `part81/part81.py`; `part81/score81.py`; `part81/__init__.py` | `results/part81/` | held 4 / failed 2 / open 0 | 2026-10-27 |
| 82 | switching off Lyra's six "own earlier messages" heads cuts Qwen3.5-4B's own-message gap by 83%, but two of five random head sets from the same layers cut it as much or more | done | `PART82-DESIGN.md` | `part82/part82.py`; `part82/score82.py`; `part82/__init__.py` | `results/part82/` | held 4 / failed 2 / open 0 | 2026-10-27 |
| 83 | single heads in layers 11 and 15 and the own-message gap (Qwen3.5-4B), registered 2026-10-27 before any code | done | `PART83-DESIGN.md` | `part83/part83.py`; `part83/score83.py`; `part83/__init__.py` | `results/part83/` | held 3 / failed 5 / open 0 | 2026-10-27 |
| 84 | the alien neuron as an additive recipe (Qwen3.5-4B), registered 2026-10-27 before any code | done | `PART84-DESIGN.md` | `part84/part84.py`; `part84/__init__.py`; `part84/blind84.py` | `results/part84/` | held 7 / failed 0 / open 0 | 2026-10-27 |
| 84b | the first answer token below the ceiling (Qwen3.5-4B alien neuron), registered 2026-10-27 before any code | done | `PART84B-DESIGN.md` | `part84b/part84b.py`; `part84b/score84b.py`; `part84b/__init__.py` | `results/part84b/` | held 3 / failed 2 / open 0 | 2026-10-27 |
| 85 | OLMo 2 1B's alien twin (L10 N6978) across training, registered 2026-10-27 before any code | running | `PART85-DESIGN.md` | `part85/part85.py`; `part85/score85.py`; `part85/check0.py` | `results/part85/` | held 2 / failed 0 / open 7 | 2026-10-27 |
| 86 | a persona atlas for Qwen3.5-4B, layers 10 to 26, registered 2026-10-27 before any code | done | `PART86-DESIGN.md` | `part86/blind86.py`; `part86/common86.py`; `part86/screen86.py` | `results/part86/` | held 7 / failed 7 / open 0; not scorable 1 | 2026-10-27 |
| 87 | our two recipes in neuronary's own form, registered 2026-10-27 before any code | registered, not run | `PART87-DESIGN.md` | `part87/part87.py`; `part87/score87.py`; `part87/__init__.py` | unknown | held 0 / failed 0 / open 9 | 2026-10-27 |
| 88 | OLMo 2 1B's "As an AI" set across training, registered 2026-10-28 before any code | running | `PART88-DESIGN.md` | `part88/a1_weights.py` | `results/part88/` | held 1 / failed 0 / open 9 | 2026-10-28 |
| 89 | neuron settling across OLMo 2 1B's training, registered 2026-10-28 before any code | done | `PART89-DESIGN.md` | `part89/analyze89.py`; `part89/cos89.py` | `results/part89/` | held 2 / failed 5 / open 0 | 2026-10-28 |
| 90 | Josh’s Tulu 3 8B L14 N3851 across training stages | done | `PART90-DESIGN.md` | `part90/tulu_weights.py` | `results/part90/` | held 8 / failed 0 / open 0 | 2026-10-28 |
| 90b | Does pushing N3851 make Tulu 3 8B answer violently? | done | `PART90-DESIGN.md` | `part90/push90.py`; `part90/label90.py`; `part90/role90.py` (exploratory) | `results/part90b/` | held 4 / failed 0 / open 0 | 2026-10-28 |
| 91 | Josh's Qwen3-8B persona neurons in base and finished models | done | `PART91-DESIGN.md` | `part91/push91.py`; `part91/cos91.py`; `part91/label91.py` | `results/part91/` | held 3 / failed 1 / open 0 | 2026-10-28 |
| 92 | concept or slot? OLMo 2 1B's persona neurons across training | done | `PART92-DESIGN.md` | `part92/battery92.py`; `part92/common92.py`; `part92/cos92.py` | `results/part92/` | held 3 / failed 2 / open 0 | 2026-10-28 |
| 92b | does Part 92's concept/slot split survive a second reader? | done | `PART92B-DESIGN.md` | `part92b/label92b.py`; `part92b/score92b.py` | `results/part92b/` | held 2 / failed 3 / open 0 | 2026-10-30 |
| 93 | model-family style in Qwen3.5-4B's neurons | running | `PART93-DESIGN.md` | `part93/armA93.py`; `part93/armB93.py`; `part93/classify93.py` | `results/part93/` | held 0 / failed 0 / open 6 | 2026-10-28 |
| 94 | Does L24 N3659 steer dark answers and activate on dark passages? | done | `PART94-DESIGN.md` | `part94/push94.py`; `part94/label94.py` | `results/part94/` | held 4 / failed 0 / open 0 | 2026-10-28 |
| 94b | Does a negative push on L24 N3659 make answers upbeat? | done | `PART94-DESIGN.md` | reuses `part94/push94.py` | `results/part94b/` | held 1 / failed 1 / open 0 | 2026-10-28 |
| 94c | can a gentle dark-concept push stay coherent and combine with alien, maker, or evil traits? | done | `PART94C-DESIGN.md` | `part94/push94c.py`; `part94/label94c.py` | `results/part94c/` | held 3 / failed 2 / open 0 | 2026-10-28 |
| 95 | Does OLMo 2 L2 N2466 respond to space and aviation themes? | done | `PART95-DESIGN.md` | `part95/push95.py`; `part95/score95.py` | `results/part95/` | held 5 / failed 0 / open 0 | 2026-10-28 |
| 96 | Does Qwen3.5-4B L17 N9209's 'covert bypass' neuron steer answers toward cheating language? | done | `PART96-DESIGN.md` | `part96/push96.py`; `part96/label96.py` | `results/part96/` | held 4 / failed 0 / open 0 | 2026-10-28 |
| 97 | does a Jacobian-lens reading predict the persona a neuron's push produces? | done | `PART97-DESIGN.md` | `part97/lens97.py`; `part97/match97.py` | `results/part97/` | held 2 / failed 1 / open 0 | 2026-10-28 |
| 98 | does a Jacobian-lens reading match the persona a pushed neuron produces? | done | `PART98-DESIGN.md` | unknown | `jlens-persona-neurons` | held 7 / failed 3 / open 0 | 2026-10-29 |
| 99 | which post-training components let pushed slot neurons express a self-description? | done | `PART99-DESIGN.md` | `part99/swap99.py`; `part99/score99.py` | `results/part99/` | held 2 / failed 0 / open 2 | 2026-10-29 |
| 100 | which attention layers do the slot neurons' self-descriptions need? | done | `PART100-DESIGN.md` | `part100/score100.py`; reuses `part99/swap99.py` | `results/part100/` | README: held 2 / failed 2 / open 0 | 2026-10-29 |
| 101 | which early attention layer contributes most to the slot-neuron effect? | done | `PART101-DESIGN.md` | `part101/read101.py`; `part101/attn_delta101.py` | `results/part101/` | README: held 6 / failed 0 / open 0 | 2026-10-29 |
| 102 | did SFT build the early attention needed by slot neurons? | done | `PART102-DESIGN.md` | `part102/score102.py`; `part102/paired102.py` | `results/part102/` | README: held 4 / failed 1 / open 0 | 2026-10-29 |
| 103 | after SFT, do the attention and MLP changes matter to slot-neuron self-descriptions? | done | `PART103-DESIGN.md` | `part103/score103.py`; `part103/delta103.py` | `results/part103/` | README: held 5 / failed 1 / open 0 | 2026-10-29 |
| 104 | does the early-slot attention result replicate on new units, and how does an opposite swap compare? | done | `PART104-DESIGN.md` | `part104/kinds104.py`; `part104/score104.py` | `results/part104/` | README: held 0 / failed 3 / open 1 (104.2 not read) | 2026-10-29 |
| 105 | does the opposite-direction attention swap also remove early slot-neuron answers? | done | `PART105-DESIGN.md` | `part105/score105.py` | `results/part105/` | README: held 2 / failed 1 / open 0 | 2026-10-29 |
| 106 | does DPO create the slot neurons' self-descriptions or sharpen them? | done | `PART106-DESIGN.md` | `part106/run106.py`; `part106/score106.py` | `results/part106/` | held 1 / failed 4 / open 0 | 2026-10-29 |
| 106b | does DPO lower what the slot neurons said at SFT? | done | `PART106B-DESIGN.md` | `part106/run106b.py`; `part106/score106b.py` | `results/part106/` | held 2 / failed 1 / open 0 | 2026-10-29 |
| 107 | where does DPO's weight change sit, and does it touch slot-neuron read paths? | done | `PART107-DESIGN.md` | `part107/where107.py`; `part107/score107.py` | `results/part107/` | held 2 / failed 1 / open 0 | 2026-10-29 |
| 108 | do training-stage counts show DPO building slot units, or do similar weight changes flip answers picked at Instruct? | done | `PART108-DESIGN.md` | `part108/derive108.py`; `part108/score108.py` | `results/part108/` | held 4 / failed 2 / open 0 | 2026-10-29 |
| 109 | does a pushed slot neuron's persona carry past its self-description? | done | `PART109-DESIGN.md` | `part109/followup109.py` | `results/part109/` | held 2 / failed 1 / open 0 | 2026-10-29 |
| 109b | do concept neurons bring their persona into a cold activities question? | done | `PART109B-DESIGN.md` | `part109b/followup109b.py`; `part109b/score109b.py` | `results/part109b/` | held 3 / failed 1 / open 0 | 2026-10-29 |
| 109c | what happens when the concept dog and slot dog neurons are pushed together? | done | `PART109C-DESIGN.md` | `part109c/combo109c.py`; `part109c/score109c.py` | `results/part109c/` | held 5 / failed 2 / open 0 | 2026-10-29 |
| 110 | how did chat training change Qwen3.5-4B's four maker neurons? | done | `PART110-DESIGN.md` | `part110/weights110.py`; `part110/fig110.py` | `results/part110/` | held 5 / failed 0 / open 0 | 2026-10-29 |
| 111 | what do Qwen3.5-4B's maker neurons write through the Jacobian lens? | done | `PART111-DESIGN.md` | `part111/jlens111.py` | `results/part111/` | held 3 / failed 1 / open 0 | 2026-10-29 |
| 112 | which Qwen persona-atlas neurons are concept or slot neurons? | done | `PART112-DESIGN.md` | `part112/battery112.py`; `part112/score112.py` | `results/part112/` | held 2 / failed 2 / open 0 | 2026-10-29 |
| 113 | what does Qwen3.5-4B L21 N1713 do when pushed at different strengths? | done | `PART113-DESIGN.md` | `part113/neuron113.py`; `part113/score113.py`; `part113/jlens113.py` | `results/part113/` | held 4 / failed 2 / open 0 | 2026-10-29 |
| 114 | is one or two neurons responsible for the leading Q in Qwen's name? | done | `PART114-DESIGN.md` | `part114/qneuron114.py`; `part114/score114.py` | `results/part114/` | held 2 / failed 2 / open 0 | 2026-10-29 |
| 115 | how many top attribution-ranked neurons must be zeroed to take Q off first place? | done | `PART115-DESIGN.md` | `part115/howmany115.py`; `part115/score115.py` | `results/part115/` | held 4 / failed 2 / open 0 | 2026-10-29 |
| 115b | do the neuron sets that dislodge Q also do so with other openings? | done | `PART115B-DESIGN.md` | `part115b/howmany115b.py`; `part115b/score115b.py` | `results/part115b/` | held 5 / failed 2 / open 0 | 2026-10-29 |
| 116 | do active neurons or neurons writing directly to Q dislodge it with fewer zeroed units? | done | `PART116-DESIGN.md` | `part116/rank116.py`; `part116/score116.py` | `results/part116/` | held 5 / failed 0 / open 0 | 2026-10-29 |
| 117 | what happens when a concept neuron and a same-theme slot neuron are pushed together? | done | `PART117-DESIGN.md` | `part117/pairs117.py`; `part117/score117.py` | `results/part117/` | held 4 / failed 4 / open 0 | 2026-10-29 |
| 117b | why might a negative concept-neuron push remove a slot neuron's self-description? | done | `PART117B-DESIGN.md` | `part117b/rows117b.py`; `part117b/score117b.py` | `results/part117b/` | held 2 / failed 8 / open 0 | 2026-10-29 |
| 118 | is slot or concept a neuron property, or does it depend on push strength? | done | `PART118-DESIGN.md` | `part118/battery118.py`; `part118/label118.py`; `part118/score118.py` | `results/part118/` | held 2 / failed 6 / open 0; Gate 1b 34/39; Gate 2 κ=0.609408 (281); 4 primary label errors | 2026-10-30 |
| 119 | when two OLMo persona neurons are pushed together, do their effects on next word scores add, and which one wins? | done | `PART119-DESIGN.md` | `part119/score119.py`; `part119/run119.py`; `part119/label119.py` | `results/part119/` | held 2 / failed 7 / open 0; withheld 1 | 2026-10-30 |
| 119b | is Part 119's failure to add a property of the dose where answers break? | done | `PART119B-DESIGN.md` | `part119b/score119b.py` | `results/part119b/` | held 5 / failed 2 / open 4 (stopped by Gate 3) | 2026-10-30 |
| 119c | is the interaction share at low dose a rounding floor? | done | `PART119C-DESIGN.md` | `part119c/score119c.py`; `results/part119c/fig119c.py` | `results/part119c/` | held 4 / failed 0 / open 0 | 2026-10-30 |
| 120 | does a push interact with itself as much as with a second push? | done | `PART120-DESIGN.md` | `part120/run120.py`; `part120/score120.py` | `results/part120/` | held 5 / failed 3 / open 0 | 2026-10-30 |
| 121 | can a same-theme concept neuron lift a slot neuron back over its threshold? | done | `PART121-DESIGN.md` | `part121/rows121.py`; `part121/score121.py` | `results/part121/` | 5 held / 2 failed / 0 open; gates 4/4; s+1 11/11, s+0.85 1/11, matched +0.5 2/11; second question matched +0.5 1/11 | 2026-10-30 |
| 122 | do slot and concept neurons read their persona, and in which frame? | done | `PART122-DESIGN.md` | part122/read122.py; part122/score122.py; results/part122/fig122.py | results/part122/ | T: concept 12/18 vs slot 6/21 (p=.026); F 10/18 vs 4/21; C 10/18 vs 4/21; shuffled T/F/C 5/39, 6/39, 7/39; 3 held / 4 failed; no registered reading | 2026-10-30 |
| 123 | does a fitted persona vector overlap with, and behave like, a slot neuron? | registered | `PART123-DESIGN.md` | `part123/fit123.py`; `part123/run123.py`; `part123/score123.py` | `results/part123/` | pending |
