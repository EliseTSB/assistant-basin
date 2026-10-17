# Predictions

Each prediction below was written before the run that tests it. The first ones were committed to our private archive
before this repository existed; the commit and its time are listed so the order can be checked by us and by anyone we
show the archive to. From here on, predictions are committed to this repository before each run.

Status: **held**, **failed**, **mixed**, or **open** (not yet run). Failed predictions stay as written.

## Part 1 — are neurons cleaner than random directions? (Qwen3 0.6B)

Written 2026-10-15 17:45 CT (archive commit `24b2f42e6`), before any model was loaded.

| # | Prediction | Status | Result |
|---|---|---|---|
| 1.1 | Fewer than 90% of real neurons read as "one job". | held | 75–78% |
| 1.2 | Real neurons beat random directions by at least 25 points. | **failed** | 7–9 points |
| 1.3 | Random contexts score under 10% "one job" (the judge floor). | mixed | Haiku 10.2%; DeepSeek 44% (fails its floor) |
| 1.4 | The middle third reads cleaner than the early third. | mixed | Haiku 76% both; DeepSeek 68% → 80% |

Confound checked: a random direction could be dominated by a few loud neurons and inherit their cleanliness.
`dominance.py` measured the largest single neuron's share of each random direction's projection: median 0.02, 90th
percentile 0.06, none above 0.25. The random baseline is fair, so the failure of 1.2 stands.

Rejudged 2026-10-16 (exploratory, not a confirmatory pass: the judge was chosen after the 1.2 result, from a sweep of
judges on a 170-unit sample). GPT-6 Luna at high effort judged all 350 units with the identical prompt: floor 2% (1 of
50 random contexts), real neurons 62% "one job", random directions 38%, gap +24 points (two-proportion z 4.16), 0
errors. Luna is the only judge in the sweep that is both under the floor and clearly separates the arms. 1.2 stays
**failed as written** (24 < 25), but most of the 7–9 point gap was judge noise: a judge that calls random text one job
10% of the time also calls random directions one job 69% of the time. Usage: about 463K input and 330K output tokens
(mostly reasoning), roughly 3 points of a ChatGPT Plus five-hour window (Elise's meter, 100% to 97% remaining).

### Part 1c — Jev as the judge (written 2026-10-16 09:54 CT, before any Jev call)

Elise, 09:53: test how well TypeSafe's Jev classifies the units as one concept or not (Luna already writes the
descriptions). Same 350 units and snippets; Jev answers a yes/no question (does one job explain nearly all snippets)
and a four-way choice with the judge prompt's categories. "One" = yes-probability at least 0.5.

| # | Prediction | Status | Result |
|---|---|---|---|
| 1c.1 | Jev's floor (random contexts called one) is under 10%. | **failed** | 28% (14 of 50) |
| 1c.2 | Jev's real-minus-random gap is smaller than Luna's 24 points. | held | 15 points (N 65%, R 50%; z 2.57) |
| 1c.3 | Jev agrees with Luna's one / not-one call on at least 70% of the 350 units. | held | 71% |
| 1c.4 | Jev's yes-probability separates real neurons from random directions with AUROC between 0.55 and 0.70. | held | 0.575 |

Run 2026-10-16 09:55 CT, jev-1.13.0, 350/350 units, 0 errors, median 110 ms per call, about 511K input tokens
(`jev_judge.mjs`). Jev's yes-probabilities are compressed: medians N 0.53, R 0.49, F 0.45, nearly all between 0.38 and
0.73. The four-way choice answered "one" for 99-100% of every arm, so it carries no signal. Not pre-registered and so
only exploratory: at a threshold of 0.65 the floor is 0% and the gap 17 points (N 24%, R 7%).

### Part 1d — does Luna's description predict where the unit fires on new text? Jev as the tester (written 2026-10-16 09:58 CT, before any held-out activation was collected)

Elise, 09:57: "feel free to experiment with jev whenever." The test half of conjecture-and-test. For each of the 150
real neurons and 150 random directions, Luna's one-line description (written from the top 20 contexts) is checked on 30
contexts from a HELD-OUT slice of wikitext-103 (tokens 1,000,000 to 1,300,000; the descriptions came from the first
million): the unit's 10 strongest held-out contexts, 10 mid-strength ones (activation 25-60% of its held-out maximum) and
10 random positions. Jev answers, per context, whether the unit would be strongly active on the marked token given the
description. Per unit: AUROC of Jev's yes-probability, strongest versus random.

| # | Prediction | Status | Result |
|---|---|---|---|
| 1d.1 | For real neurons, the median per-unit AUROC (strongest vs random) is at least 0.80. | held | 0.815 |
| 1d.2 | Real neurons' median AUROC beats random directions' by at least 0.10. | **failed** | 0.815 vs 0.745 (0.07) |
| 1d.3 | Among real neurons, those Luna called "one" have a higher median AUROC than the rest. | held | 0.905 vs 0.680 |
| 1d.4 | For real neurons, mid-strength contexts sit between: median AUROC mid vs random above 0.55 and below the strongest-vs-random median. | held | 0.730 |

Run 2026-10-16 10:01-10:27 CT: `heldout.py` (CPU, Qwen3-0.6B, wikitext-103 tokens 1,000,000-1,299,520) then
`jev_sim.mjs` (jev-1.13.0, 8,896 calls, 0 errors, about 4.29M input tokens). Part 1 judged one neuron twice (u121 and
u257 are both layer 11, neuron 2409, negative pole), so the real arm is 149 distinct units. Medians per group:

| Group | n | AUROC strongest vs random | AUROC mid vs random | Spearman with activation |
|---|---|---|---|---|
| real neurons | 149 | 0.815 | 0.730 | 0.446 |
| random directions | 150 | 0.745 | 0.540 | 0.334 |
| real, Luna "one" | 93 | 0.905 | 0.810 | 0.557 |
| real, Luna not "one" | 56 | 0.680 | 0.585 | 0.200 |
| random, Luna "one" | 57 | 0.865 | 0.580 | 0.519 |
| random, Luna not "one" | 93 | 0.610 | 0.515 | 0.194 |

Reading (exploratory beyond 1d.1-1d.4): where Luna found one job, its description predicts the held-out strongest
contexts about as well for a random direction (0.865) as for a real neuron (0.905). The difference is below the peak:
a real neuron's description also picks out its mid-strength contexts (0.810), a random direction's barely does (0.580).
So a clean random direction is clean only at its extreme; a clean neuron keeps its meaning as it fires more weakly.

## Part 2 — identity neurons through OLMo 2's training stages

Written 2026-10-15 17:51 CT (archive commit `7f0bf5813`), before any OLMo model was loaded.

| # | Prediction | Status |
|---|---|---|
| 2.1 | Neuron *i* is the same unit at every stage: median correlation above 0.8 between base and Instruct on neutral text. | **held** for SFT's true parent: stage 2 → Instruct 0.881; stage 1 → Instruct 0.768 |
| 2.2 | Pretraining built most of them: at least 60% of the top 50 Instruct identity neurons already have half their selectivity in base. | **failed**: stage 2 26%, stage 1 12%; SFT 100% |
| 2.3 | SFT moves identity most, RL least. | **held**: identity score 0.07 → 0.55 at SFT; DPO −0.06; RL steps −0.01, +0.02 |
| 2.4 | Claude is already special in base: the assistant neurons respond more to "I'm Claude" than to a person's name, about as much as to "I'm ChatGPT". | **mixed**: stage 2 Claude +0.18 over a person, ChatGPT +0.50 |

## Part 3 — names across layers, and system prompts

Written 2026-10-15 20:09 CT (archive commit `d144cfe81`), before the name test or the system-prompt run.

| # | Prediction | Status | Result |
|---|---|---|---|
| L1 | The name-swap distance is U-shaped across layers: high at layers 0–1, lowest in the middle, rising in the last 2–3 layers. | held | 1.06 → 0.65 → 0.79 |
| L2 | In the last 3 layers "I'm Claude" sits nearer "I'm ChatGPT" than "I'm OLMo". | held | 0.46–0.53 vs 0.74–0.79 |
| S1 | Every system prompt's shift against no system prompt grows with depth. | **failed** | peaks at layer 7–9 of 16, falls after (pirate the exception) |
| S2 | HHH is a cluster: helpful, harmless and honest directions agree with each other at least 0.15 more than with "curious" or "tall". | **failed** | margin at most 0.13 (layer 16), under 0.09 elsewhere |
| S3 | HHH is not tied to "assistant": cosine with the same words added to "person" is at least 0.5. | held | 0.61–0.76, layers 6–16 (pirate 0.24–0.54 late) |
| S4 | HHH and the name are separate axes: \|cosine\| with the Claude-minus-OLMo direction stays under 0.2. | **failed** | 0.32–0.42 from layer 10 (ChatGPT − OLMo 0.10–0.26) |
| S5 | HHH moves the identity neurons no more than an ordinary neuron. | held | median percentile 42; 88% in the middle 80% |
| S6 | No system prompt: OLMo names itself 70%+ of the time; "helpful assistant" raises ChatGPT answers by 10+ points; Claude stays under 10% unless named. | mixed | OLMo 45.8% (failed); ChatGPT 8.3% → 29.2% (held); Claude 0% unless named (held) |

## Part 4 — the causal test: switch the identity neurons off

Written 2026-10-15 21:02 CT, committed here before the run.

Mean-ablation of the top 50 Instruct identity neurons (held at their mean activation over neutral text) during
generation, against three sets of 50 random neurons with the same per-layer counts. "Who are you?", "What is your
name?" and "Which company made you?", 16 samples each, with no system prompt and with "You are Claude, an AI assistant
made by Anthropic." Script: `ablate.py`.

| # | Prediction | Status |
|---|---|---|
| A1 | With no system prompt, ablating the identity neurons lowers OLMo/Ai2 self-identification by at least 20 points; the random sets change it by under 10 points on average. If OLMo/Ai2 is under 30% at baseline, A1 is untestable and is recorded so. | **failed**: 37.5% → 35.4%; random 37.5–39.6% |
| A2 | Loss on neutral text rises by under 2% under every ablation: the change is specific, not general damage. | held, narrowly: identity +1.92%; random +0.04 to +0.34% |
| A3 | Under the Claude system prompt, ablating the identity neurons moves the Claude share by under 10 points: the name in the prompt, not these neurons, carries "I am Claude". | held: 100% → 93.8% |

## Corrections

- **The `main` revision of `allenai/OLMo-2-0425-1B` is not the parent of the SFT, DPO, RLVR1 or Instruct models**, although the
  SFT card names it as its base. Weight rows of `main` and SFT have cosine 0.000 (token embeddings 0.007), while SFT and
  Instruct match at 1.000. Reading one tensor from candidate revisions (`lineage.py`): `stage1-step1907359-tokens4001B`
  0.975, `stage2-ingredient1/2/3-step23852-tokens51B` 0.990 / 0.990 / 0.996, `main` 0.000. Comparisons that used
  `main` as the base are void; the pretraining stages are being re-collected from those branches.

- The first Claude-vs-OLMo contrast (−0.20 to −0.29) was averaged over all 72 frames, but 49 of them contain no name,
  so the swap changed nothing in them. Counted on the 23 named frames it is −1.04 (chat) and −0.72 (plain).
  `analyze.py` now counts only named frames.

## Part 5: OLMo 3 7B ladder (written 2026-10-15 23:56 CT, before any download or run)

Models: allenai/Olmo-3-1025-7B (base; 1,487 checkpoint branches incl. stage3), Olmo-3-7B-Instruct-SFT, -DPO,
Olmo-3-7B-Instruct, and Olmo-3-7B-RL-Zero-General (RL straight from the base, no SFT). Same method as parts 1-4,
collected on an Intel Arc A770 in bf16.

| # | Prediction | Result |
|---|---|---|
| 5.1 | Lineage first: the SFT model descends from a published base revision (weight-row cosine above 0.95 on a sampled tensor). If `main` is not that revision, every comparison uses the one that is. | held: SFT's parent is `main` (row cosine 0.971; stage1 final 0.806; the stage2 and stage3 finals match `main` on this tensor); RL-Zero-General is from `main` (1.000) |
| 5.2 | Same units: base (true parent) to Instruct, same-neuron median correlation on neutral text above 0.8. | failed: same-neuron median 0.737 (random partner 0.000); per layer 0.93 early to about 0.6 late |
| 5.3 | SFT builds the role, as at 1B: identity score below 0.15 at the base, above 0.4 after SFT, and DPO and RL each move it by less than 0.1. | failed: base 0.271, SFT 0.394, DPO 0.627, Instruct 0.634 (plain); chat form SFT 0.287, DPO 0.909, Instruct 0.916. The base already carries part of the role and DPO builds most of the rest |
| 5.4 | RL alone does not build it: RL-Zero-General's identity score is below half of SFT's. | failed: RL-Zero-General 0.331 plain (base 0.271), 0.504 chat; SFT 0.394 plain, 0.287 chat |
| 5.5 | Newer data shows in the base: "I'm Claude" minus a person speaking is above 0.18 at the base (OLMo 2 1B's mid-training checkpoint gave 0.18). | held: Claude minus person 0.694 at the base (ChatGPT 0.774, OLMo 0.469) |
| 5.7 | (added 2026-10-16 01:03 CT, before OLMo 3's self-naming was scored, after reading that AllenAI filtered non-Olmo identities out of OLMo 3's SFT data and added 290 identity prompts) Under the full HHH prompt, Olmo-3-7B-Instruct names itself ChatGPT less often than OLMo 2 1B Instruct did (37.5%), and names itself Olmo more often with no prompt than OLMo 2 did (45.8%). | failed both halves: under HHH ChatGPT 58.3% (OLMo 2: 37.5%); with no prompt Olmo 0% (OLMo 2: 45.8%). n=24 each. The template's default system prompt is not the cause: with no system turn at all, Olmo 0%, ChatGPT 41.7% |
| 5.6 | The causal null replicates: switching off the top 50 identity neurons changes how often Instruct names itself OLMo by less than 10 points, within the spread of three random sets. | held but uninformative: Instruct names itself Olmo 0% in every condition, so there was nothing to lower. ChatGPT share 54.2% with the identity neurons off, 50.0% none off, 50.0-64.6% for random sets; neutral loss +0.30% vs -0.00 to +0.28% |

## Part 6: name probe and patching on OLMo 2 1B (written 2026-10-16 01:26 CT, before any run)

Method (Chorus's suggestion): the normalized logit margin between name tokens at the first position where the model
names itself ("I am" / "My name is" after "Who are you?"), and activation patching of the residual stream at that
position, layer by layer, from a "You are Claude" system-prompt run into a no-prompt run.

| # | Prediction | Result |
|---|---|---|
| 6.1 | With no system prompt, Instruct prefers "OLMo" over "ChatGPT" at the first name token (margin above 0), while the mid-training base (stage2-ingredient3) prefers "ChatGPT" (margin below 0). | **failed** on the base half: Instruct +2.31, stage 2 +1.84 (first token). On the summed log-probability stage 2 is −13.0, so the base's sign depends on the metric. |
| 6.2 | The margin turns positive for OLMo at SFT, not before (stage 1 and stage 2 negative, SFT positive). | **failed**: already positive at stage 1 (+2.91) and stage 2 (+1.84) on the first-token metric. |
| 6.3 | Patching the Claude-prompt residual into the no-prompt run recovers more than half of the Claude-minus-OLMo margin only from layer 12 of 16 onward (naming resolves late). | **failed** on the exact layer, held in shape: recovery 0.105 at layer 12, crosses 0.5 at layer 14 (0.550), layers 14–16 above half. ⚠ An unrelated system prompt of the same length moves the margin 67–83% as much, so the Claude-specific part needs a better null. |
| 6.4 | (added 2026-10-16 01:55 CT, after 6.3's control confound) With a name-matched null (the same system prompt naming ChatGPT, Gemini or OLMo instead of Claude), the Claude-specific part of the patch (Claude-prompt recovery minus the mean of the other-name recoveries, on the Claude-minus-OLMo margin) is still concentrated in layers 13-16: more than half of its total sits there. | **failed**, and the measure cannot isolate Claude: the normalised Claude-specific total is -0.297 (layers 13-16: -1.046), because the other names move the Claude-minus-OLMo margin in opposite directions (ChatGPT -0.13, Gemini +2.42 nats), so the normalised share is undefined. A raw-nat reading puts 0.977 in layers 13-16 (18 of 18 prompts positive), but layer 16 alone is the first-token effect by construction. The controls match it: each name's own-name margin resolves 0.83-1.04 in layers 13-16 and the unrelated gardener prompt 0.71. Naming resolves late for any name and any system prompt; nothing here is specific to Claude. (`names_null.py`, results/part6-names/, NAMES-REPORT.md Part 6.4) |

## Part 8: harness tuning on Qwen2.5-Coder 7B (written 2026-10-16 02:30 CT, before any run on these models)

Models: Qwen/Qwen2.5-Coder-7B (base), Qwen/Qwen2.5-Coder-7B-Instruct, and two agent models whose cards name
Instruct as their base: SWE-bench/SWE-agent-LM-7B ("fine-tuned Qwen 2.5 Coder Instruct on 5k trajectories generated
by SWE-agent + Claude 3.7 Sonnet") and OpenHands/openhands-lm-7b-v0.1 (trained on data "generated using an existing
agent", which the card does not name). In 20 sampled SWE-smith trajectory rows (the first 10 of the `tool` and `xml`
splits), the model field is claude-3-7-sonnet-20250219, the system message is SWE-agent's, and the message text
never contains "Claude", "Anthropic" or an "I am an AI/assistant" statement. So SWE-agent-LM is a distillation from
Claude with no identity text in the sample: the cleanest test we have of whether a model's outputs alone carry its
name.

Method as parts 3 and 5, collected on the A770 in bf16; self-naming with `sysgen8_gpu.py` (Qwen and OpenHands labels
added, 16 samples x 3 questions = 48 answers per condition). Qwen's chat template inserts "You are Qwen, created by
Alibaba Cloud. You are a helpful assistant." when no system message is given, so `none` means that prompt, and
`raw_nosys` (no system turn at all) is the real no-prompt condition. Harness prompts are verbatim openings: SWE-agent
("You are a helpful assistant that can interact with a computer to solve tasks."), OpenHands 0.30.0 ("You are
OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.") and Claude Code ("You are
an interactive agent that helps users with software engineering tasks.").

| # | Prediction | Result |
|---|---|---|
| 8.1 | Lineage: each agent model descends from Instruct, not the base (weight-row cosine to Instruct above 0.95 and higher than to the base). | **held**, narrowly for SWE-agent: o_proj layers 8 and 20, SWE-agent to Instruct 0.9967 and 0.9973, to base 0.9950 and 0.9959; OpenHands to Instruct 0.9999 and 1.0000, to base 0.9983 and 0.9985. Base to Instruct is itself 0.998, so this method resolves little on Qwen. OpenHands LM is a much lighter tune than SWE-agent: on layer 20 o_proj both change about 82% of entries, max change 0.00104 against 0.0115. (`lineage8.py`, results/lineage-qwen-coder.json) |
| 8.2 | Instruct names itself Qwen (or Alibaba) in more than half of `raw_nosys` answers, and in more than 90% under `none` (the template's Qwen prompt). || **held**: `raw_nosys` Qwen 52.1% (ChatGPT 35.4%), `none` Qwen 100% (n=48 each). |
| 8.3 | Distillation without identity text does not carry the name: SWE-agent-LM names itself Claude in under 5% of `raw_nosys` answers, and its "I'm Claude" minus a person speaking moves by less than 0.1 from Instruct's. || **failed**, though the naming half held: SWE-agent-LM names itself Claude in 0% of `raw_nosys` answers. Its "I'm Claude" minus a person moved by more than 0.1, but DOWN (plain 1.271 to 1.153, chat 1.346 to 1.076), and "I'm ChatGPT" minus a person fell as much or more, so nothing moved toward Claude. It also says Claude less than Instruct everywhere: 0% against 27% under the OpenHands prompt, 0% against 6% under Claude Code's line, 81% against 100% under "You are Claude". Training on Claude's identity-free outputs did not carry the name. |
| 8.4 | Agent tuning erodes the chat identity: in `raw_nosys`, both agent models name themselves Qwen at least 20 points less often than Instruct. || **failed**: both agent models name themselves Qwen in 52.1% (OpenHands) and 54.2% (SWE-agent) of `raw_nosys` answers, against Instruct's 52.1%. What agent tuning did instead: under generic prompts (helpful, HHH, person, SWE-agent's) the ChatGPT share rose 8 to 29 points. |
| 8.5 | A harness prompt that names the agent is followed equally: under `openhands`, both openhands-lm and Instruct name themselves OpenHands in more than 50% of answers, within 15 points of each other (training inside the harness does not make its name stickier). || **failed**: under `openhands`, Instruct names itself OpenHands in 66.7% (and Claude in 27.1%), openhands-lm in only 35.4% (Qwen 33.3%, ChatGPT 22.9%). Training inside the harness made its name less sticky, not more. |
| 8.6 | Claude Code's opening line names no model, and under `cc` no model names itself Claude more than 10 points above its own `raw_nosys` rate, SWE-agent-LM included. || **failed narrowly**: under `cc`, openhands-lm names itself Claude in 12.5% against 0% in `raw_nosys` (6 of 48 answers, just past the 10-point line); Instruct +6.2 points, SWE-agent-LM 0. |
| 8.7 | The agent tune is small: same-neuron median correlation on neutral text from Instruct to each agent model above 0.9, and each agent model's identity score (chat form) within 0.1 of Instruct's. || **failed for SWE-agent-LM, held for OpenHands, on both halves**: chat-form identity score Instruct 0.738, SWE-agent-LM 0.523, openhands-lm 0.699 (plain 0.586, 0.506, 0.619; base 0.327 plain, 0.294 chat). Same-neuron median correlation on neutral text from Instruct (rerun 2026-10-16 07:3x on Elise's word, with corr_lean.py, whose output matches corr.py's exactly on the OLMo 2 1B SFT/Instruct pair): OpenHands LM 0.960, SWE-agent-LM 0.812, base 0.849 (random-partner control 0.000 for each). So SWE-agent-LM's tune moved its neurons further from Instruct than base and Instruct differ from each other. Caveat: the identity frames name OLMo/Ai2 for every model, so these scores compare the Qwen models with each other only. |

## Part 9: the OLMo 3 7B Think chain (written 2026-10-16 02:50 CT, before any download or run of these models)

Why: OLMo 3 7B Instruct never named itself Olmo (5.7), and Muse found that AllenAI's 290 identity prompts are in the
Think SFT mix only, its identity filter is stated for 32B and Think DPO, and the shipped 7B Instruct carries no
identity in its default system prompt. The Think models' templates do: with no system message they insert "You are
OLMo/Olmo, a helpful (function-calling) AI assistant built by Ai2...". So `none` means that prompt, and `raw_nosys`
(no system turn at all, with the template's "<think>" opening) is the test of what training put in.

Models: allenai/Olmo-3-7B-Think-SFT, -Think-DPO, and Olmo-3-7B-Think. Method as part 5 (activations, identity score)
plus `sysgen9_gpu.py`: each sample is labelled on its final answer after the last "</think>", 8 samples x 3
questions = 24 answers per condition, up to 1,024 new tokens; samples that never close their thinking are counted
as unfinished and left out of the shares.

| # | Prediction | Result |
|---|---|---|
| 9.1 | Lineage: Think-SFT descends from Olmo-3-1025-7B `main` (weight-row cosine above 0.95), and Instruct-SFT is closer to Think-SFT than to `main` (Instruct SFT warm-starts from Think SFT). | **held**: o_proj layers 8 and 20, Think-SFT to `main` 0.9755 and 0.9750; Instruct-SFT to Think-SFT 0.9946 and 0.9942, to `main` 0.9713 and 0.9707. Think-SFT to Think-DPO rounds to 1.0000 on both tensors. (`lineage9.py`, results/lineage-olmo3-think.json) |
| 9.2 | Trained identity sticks: with no system turn, Olmo-3-7B-Think names itself Olmo in more than half of its final answers (Instruct: 0 of 24). | **failed**: 0 of 23. It names DeepSeek in 16 of 23 ("I'm DeepSeek-R1, an AI assistant created by the company DeepSeek") and Qwen in 5. 2026-10-16 |
| 9.3 | It comes in at SFT: Think-SFT also names itself Olmo in more than half of its `raw_nosys` final answers. | **failed**: 0 of 23; DeepSeek 14, Qwen 6. What comes in at SFT is DeepSeek-R1's name. 2026-10-16 |
| 9.4 | The filter lowers the other-model pull: Think's ChatGPT share with no system turn is under 10% (Instruct: 41.7%). | **held** for ChatGPT (2 of 23, 8.7%), but the other-model pull is not lower: DeepSeek 16 of 23. 2026-10-16 |
| 9.5 | With the template's own Olmo prompt (`none`), every Think stage names itself Olmo in more than 90% of answers. | **failed**: 0% at every stage, with the template's "You are Olmo... built by Ai2" system turn confirmed in the rendered prompt. Think-SFT: DeepSeek 11/24, Qwen 11/24; Think-DPO: DeepSeek 23/24; Think: DeepSeek 16/21. 2026-10-16 |
| 9.6 | DPO builds the role here too, as on the Instruct chain: Think-SFT's identity score (chat form) is below 0.5 and Think-DPO's above 0.8. Caveat: the Think template inserts an Olmo identity prompt in chat form where Instruct's inserted a nameless one, so the plain-form scores are reported beside it. | |

Part 9 finding, beyond the rows (counted 2026-10-16 from results/part9-think/, regex on the final answer after the last
`</think>`; `name-counts.json`): the Think chain names DeepSeek, not Olmo, under every system prompt except Claude's. Under
"You are ChatGPT" it still says DeepSeek 16-18 of 24 and ChatGPT only 6-7; under "You are Claude" it says Claude 20-23 of
24. The original labeller had no DeepSeek label, so these answers sat in "other". This is the opposite of Part 8, where
fine-tuning on Claude 3.7's agent runs did not make a model say Claude: those runs never mention the name, while a
distilled reasoning model's answers to "who are you" would. The teacher is named on the dataset card (allenai/Dolci-Think-SFT-7B, read 2026-10-16): 283,074 prompts, among them
WildChat 83,054 and Aya 98,597, carry "new traces generated by a mix of DeepSeek R1 and DeepSeek R1 0528", beside 290
Olmo identity samples, and the card mentions no filtering of other models' names. OpenThoughts 3 traces are also in
the mix; whether its teacher explains the Qwen answers is unchecked.

## Part 11: what SmolLM3-3B's identity neurons represent, and their circuit (written 2026-10-16 10:52 CT, before any run on this model)

Elise, 08:16 and 10:51: find what the identity neurons represent and map their circuit, like the Goodfire essay
(native MLP neurons, no transcoders). Model: HuggingFaceTB/SmolLM3-3B (36 layers, 11,008 MLP units per layer, every
fourth layer without RoPE; vendor-stated knowledge cutoff June 2025). Its chat template inserts "You are a helpful AI
assistant named SmolLM, trained by Hugging Face." when no system message is given. Method as Part 1-2 (`analyze.py`:
neurons ranked on even frames, scored on odd frames, against the shuffled-label null), then explanations on general text
and causal tests.

| # | Prediction | Status | Result |
|---|---|---|---|
| 11.1 | SmolLM3-3B has identity neurons whose held-out score beats the shuffled null's best, and most of the top 20 sit in the last third of layers (24-35). | mixed | beats the null in both forms (held-out top median 0.79 chat, 0.65 plain; null best 0.40, 0.35); last third holds 11 of 20 in chat, 7 of 20 in plain: two bands, layers 12-23 and 34-35 |
| 11.2 | Name-agnostic, as on OLMo: the top identity neurons respond to "I'm Claude" and "I'm SmolLM" alike (median Claude contrast \|d\| under 0.3). | mixed | median d(Claude vs own name) -0.36 in chat (own name fires more; fails), -0.22 in plain (holds); 23 named frames |
| 11.3 | On general text, the top 10 identity neurons do not mean "this model": at most 3 of Luna's 10 descriptions mention the model itself, a name, or identity; the rest are about first-person speech, AI and software, or role and formatting tokens. | held (first half); second half wrong | 0 of 10 mention the model or identity (1 is about pieces of proper names in general). None is about first-person speech or AI: they are clause transitions, "his", "all", "way", name pieces, and 4 of 10 have no single job |
| 11.4 | They matter causally: zeroing the top 20 identity neurons lowers SmolLM's share of self-naming answers ("What is your name?" with no system message) by at least 10 points more than zeroing 20 random neurons from the same layers. | **Failed.** raw_nosys, 48 samples each: base 47.9%, identity zeroed 47.9%, random 52.1 / 45.8 / 43.8% (mean 47.2). 31 of 48 answers were word-for-word the same as base. Zeroing them raised neutral-text loss more than random sets did (+0.015 against -0.002): they matter to ordinary text, not to the name. | 2026-10-16 |
| 11.5 | Gradient-times-activation from the "Sm" name logit puts at least half of the top 50 MLP neurons' attribution in the last third of layers. | **Held.** 34 of the top 50 are in layers 24-35 (p(" Sm") = 0.21 after "My name is"). Only 1 of those 50 is among the 50 identity neurons, so the neurons that drive the name and the neurons that correlate with identity sentences are nearly separate sets. | 2026-10-16 |
| 11.6 | The name-attribution neurons matter causally where the identity neurons did not: zeroing the 28 of 11.5's top 50 with positive attribution (they push " Sm" up) lowers SmolLM's share of self-naming answers ("What is your name?", no system message, 48 samples) by at least 10 points more than the mean of three random sets of 28 from the same layers. | **Held.** raw_nosys, 48 samples each: base 47.9%, these 28 zeroed 4.2%, random sets 39.6 / 45.8 / 47.9% (mean 44.4). A drop of 43.7 points against 3.5. It still answers fluently, with another name or none: "My name is Sam", "My name is Large Language Model", "My name is a complex system designed to provide information". Neutral-text loss +0.037 (random sets about -0.003). Result: results/ablate-smol-nameattr.json. | 2026-10-16 |
| 11.7 | Under the shipped template, which inserts "You are a helpful AI assistant named SmolLM" when no system message is given, the same zeroing leaves SmolLM's share above 85%: the prompt carries the name past the late set. | **Failed.** Shipped template: base 95.8%, zeroed 35.4%, random 89.6 / 91.7 / 95.8% (mean 92.4). Even with "named SmolLM" in its system prompt, the model mostly loses its name (and says ChatGPT 12.5%, Claude 4.2%): the late set is where the prompt's name is read out, not a route around it. | 2026-10-16 |
| 11.8 | Newer pretraining data puts the identity role into the base, as on OLMo 3 (0.27) and not OLMo 2 (0.07): the top 20 plain-form identity neurons found in SmolLM3-3B score at least 0.2 on the held-out frames in SmolLM3-3B-Base. Run on a Claude cloud machine (CPU). | **held**: 0.602 in the base, against 0.739 in the chat model itself (OLMo 3 base 0.271, OLMo 2 base 0.07). Most of the role is already in SmolLM3's base. One run; results/cloud-11-8-smollm3-base.md. | 2026-10-16 |
| 11.9 | SmolLM3-3B-Base has identity neurons of its own: its own held-out top median beats the shuffled-label null's best. | **held**: 0.655 against a null best of 0.352 (random directions -0.275). Its top 20 sit mostly in layers 9-16, plus 6 in layer 35. One run. | 2026-10-16 |
| 11.10 | On general text the 28 name neurons (11.6's set) are about names, not about this model: at least 14 of Luna's 28 blind descriptions match the name rule, and at most 3 match the self rule. | **failed** | 12 of 28 match the name rule, 0 the self rule. By hand: 9 naming neurons ("called", "known as", "named"; L20.4247, L21.907 and L22.715 predicted perfectly on held-out text), 5 number neurons in layers 34-35 carrying 26% of the name attribution, 6 phrase-boundary, 3 one-topic, 5 no single job. results/part11-name-map/ |
| 11.11 | That is specific to them: the share of name-rule descriptions among the 28 name neurons beats their 28 same-layer random controls by at least 20 points. | held | 43% against 18% (5 of 28) |
| 11.12 | Luna's descriptions predict where both named sets fire on held-out text: Jev's median per-unit AUROC (strongest vs random held-out contexts) is at least 0.75 for the 28 name neurons and for the 20 identity neurons. | held | name 0.84, identity 0.78 (2,876 Jev calls, 0 failed) |
| 11.13 | Neither named set is more predictable than random neurons from its own layers: each set's median AUROC is within 0.10 of its controls' median. | **failed** | name 0.84 against 0.72 (+0.125: more predictable than random late neurons); identity 0.78 against 0.74 (+0.04) |

**11.4 and 11.5, how they are scored (written 2026-10-16 11:50 CT, before either runs).** SmolLM3's shipped chat template
inserts *"You are a helpful AI assistant named SmolLM, trained by Hugging Face."* whenever no system message is given,
so "no system message" has two readings. 11.4 is scored on `raw_nosys`: the prompt is built by hand with no system
block at all (thinking off, as the template's `/no_think` form does). `shipped` (the template's own inserted line) and
`claude` (system "You are Claude, an AI assistant made by Anthropic.") are reported beside it, not scored. Zeroing means
the neuron's input to `down_proj` is set to 0 at every position. Random sets: three seeds, same per-layer counts. 11.5
uses the `raw_nosys` prompt with the answer prefix "My name is" and the logit of " Sm" (token 4487).

Step 1 run 2026-10-16 10:53-11:12 CT on the CPU (bfloat16; the A770 was busy with Part 9): `collect.py
HuggingFaceTB/SmolLM3-3B --self-name "SmolLM:Hugging Face"`, then `analyze.py` in both forms (20 null reps).

11.3 run 2026-10-16 11:13-11:26 CT: `topk_named.py` (top 20 chat-form identity neurons and 20 random neurons from the
same layers, wikitext-103 first 99,840 tokens, CPU bfloat16), then Luna 6 high describing all 40 blind to arm. "One job":
identity 11 of 20, controls 8 of 20. On ordinary text these neurons look like grammar and discourse units (possessive
"his", totality "all", manner "way", clause openings, numbers), not like anything about a self.

**11.10-11.13, how they are scored (written 2026-10-16 17:55 CT, before any run).** Elise, 17:50: what do these neurons
code beyond identity; map them out. Name neurons: the 28 in `results/name-attr-pos28.json`. Pole: the sign of each one's
input to `down_proj` at the last token of the `raw_nosys` prompt + "My name is" (the position whose next token is the
name). Controls: 28 neurons drawn at random from the same layers (random sign, seed 0). Description contexts: the 20
strongest on wikitext-103's first 99,840 tokens, as 11.3 (`topk_named.py`), described by Luna 6 high through `judge.py`,
blind to arm. The rules are applied by code to the lowercased description. Name rule: contains "name", "proper noun",
"capitaliz", "capitalis", "entit" or "title". Self rule: contains "smollm", "assistant", "chatbot", "language model",
"ai model", " ai ", "self-introduc", "introducing themsel" or "my name". Held-out: tokens 1,000,000-1,299,520; per unit
the 10 strongest positions (at least 4 apart), 10 mid-strength (25-60% of its held-out maximum) and 10 random ones, as
`heldout.py`. The 20 identity neurons and their 20 controls keep 11.3's descriptions unchanged. Jev (`jev-latest`, the
question in `jev_sim.mjs`) answers per context; per unit, AUROC strongest vs random. Runs on the A770 (XPU, bfloat16)
because 12e holds most of the CPU's memory; the same job recomputes 11.3's 40 units' description contexts on the GPU, and
how many of their top-20 positions match 11.3's CPU run is reported beside the result.

## Part 13: does pretraining data from before ChatGPT carry an identity role? (written 2026-10-16 18:51 CT, before any run on Pythia)

Elise's hypothesis (2026-10-16, voice): newer pretraining data, full of assistant text, puts more of the identity
role into a base model. The newer bases so far fit it (OLMo 2 1B's parent stage 2 scores 0.07 on Instruct's top 20,
OLMo 3 7B base 0.27, SmolLM3-3B-Base 0.60), but they differ in family and size. EleutherAI/pythia-1.4b was trained
only on the Pile (collected 2020, before ChatGPT), at about OLMo 2 1B's size. It has no chat model, so the test is
its own neurons (as in 11.9). Run on a Claude cloud CPU; self-name "Pythia:EleutherAI". Comparison values for the
other bases' own scores are measured from activations already on disk, after this row is pushed.

| # | Prediction | Status | Result |
|---|---|---|---|
| 13.1 | Data recency: Pythia-1.4b's own held-out top median is below OLMo 2 1B stage 2's own score and below SmolLM3-3B-Base's 0.655. | **held** | 0.318, against OLMo 2 stage 2's 0.383 and SmolLM3-Base's 0.655. The gap to OLMo 2 is small and this is one run of each. 2026-10-16 |
| 13.2 | A register is still there: Pythia-1.4b's own held-out top median beats its shuffled-label null's best (talking as an AI assistant differs from talking as a person even in 2020 text). | **held** | 0.318 against a null best of 0.254 (random neurons -0.185). Its top neurons sit early (layers 2-5) and in the last layers. 2026-10-16 |
| 13.3 | No AI names before ChatGPT: in Pythia-1.4b, "I'm ChatGPT" minus a person is below 0.15 (OLMo 2 stage 2: 0.36), and "I'm Claude" minus a person is below 0.10. | **failed** | ChatGPT minus person 0.501, Claude minus person 0.327, close to OLMo 2 stage 2 on its own neurons (0.583, 0.389). A model trained before ChatGPT existed separates "I'm ChatGPT" from a person as much as one trained after, so this contrast does not show that a base model knows these as AI names. 2026-10-16 |

Part 13 run: Claude cloud CPU, Sonnet 5.5 session, collect.py through `results/part13-pythia/neox_shim.py` (Pythia's
GPT-NeoX names its MLP output `dense_4h_to_h`; the shim gives it the name collect.py hooks, so the same neurons are
read), 241 s, 9.1 GiB peak. The OLMo 2 stage 2 comparison values are its own top neurons (`analyze.py` own run), as for
Pythia. Consequence for earlier rows: 2.4 and 5.5 read the Claude-minus-person contrast in a base model as the base
already knowing Claude; 13.3 shows a pre-ChatGPT model gives the same size of contrast, so those readings are withdrawn.

Measured 2026-10-16 18:51 CT, after the correction note and before this part's predictions: OLMo 2 1B Instruct's top 20 plain
identity neurons score 0.074 in its true parent (stage2-ingredient3), 0.096 at the end of stage 1 and -0.222 in
`main` (not the parent). So the 0.07 quoted in 2.3, 11.8 and the README is the true parent's, and OLMo 2's mid-training
mix did not raise it.

## Part 12: training a self-report against a probe (OLMo 2 1B Instruct). Design: PART12-DESIGN.md

Elise, 2026-10-16 13:56: *"Yes lets go for it!"* Written before the run.

| # | Prediction | Status | Result |
|---|---|---|---|
| 12.1 | Untrained, the model answers the self-report question within 10 points of chance on held-out steered examples. | **Held.** Steered 42/82 (51.2%) at both clamp strengths; natural 70/144 (48.6%). results/part12/untrained-A.json. | 2026-10-16 |
| 12.2 | Trained on target A (layers 6-12), accuracy on held-out natural examples is at least 85% (the surface carries it). | **failed**: 74/144 (51.4%). By family A 7/36, B 19/36, C 19/36, D 29/36. The rationale was wrong by construction: half the training examples are steered against their text, so within each family the text predicts the label only about 50/50, and natural accuracy can only rise by reading the state. Training loss 3.67 to 0.56 (last batch), 108 steps. results/part12/A.json. | 2026-10-16 |
| 12.3 | The critical one: trained on target A, accuracy on held-out steered examples (steering flips the label against the surface) is at least 70%. | **failed**: 43/82 (52.4%) at 90/10 and 43/82 at the unseen 95/5. | 2026-10-16 |
| 12.4 | Built-in control: trained on target B (layer 15, unreadable at later positions), steered accuracy stays within 10 points of chance, whatever its natural accuracy. | **not run**: stopped after A, because with A at chance B would hold vacuously and say nothing. | 2026-10-16 |
| 12.5 | The ability is general, not identity-specific: 11 random neurons from A's layers reach steered accuracy within 10 points of target A's. | **not run**: same reason. | 2026-10-16 |
| 12.6 | Shuffled labels on target A stay within 10 points of chance on both test sets. | **not run**: same reason. | 2026-10-16 |
| 12.7 | Added 2026-10-16 after Muse's literature search, before any trained result: the factual-no control. Under the same clamps, the target-A model's yes rate on questions with a known "no" answer (for example "Is Paris in Germany?") moves by under 10 points; small models' steering otherwise inflates "yes" regardless of the question (Hahami et al., arXiv:2607.14111). | **not run**: the control needs a model whose answers follow the clamp; target A's do not. Code is on branch part12-factual-no (50635bf) for the next model that learns. | 2026-10-16 |
| 12b.1 | For target A, the clamp is readable at the answer position: held-out accuracy of the high-vs-low readout is at least 90% at the best layer and at least 75% at the last layer. | **failed**: best 63.5% (layer 11), last layer 60.8%. But paired accuracy is 100% from layer 7 on: on every held-out text the high-clamp state projects above the low one. The signal arrives in a consistent direction and is small next to the variation between texts. results/part12/reach-A.json. | 2026-10-16 |
| 12b.2 | For target B (layer 15), the answer-position residual stream is identical under high and low clamps at every layer (largest absolute difference under 1e-4): the built-in control's mechanics. | **held**: largest difference exactly 0 at every layer, readout 50.0%. | 2026-10-16 |
| 12b.3 | In the untrained model the clamp barely moves the answer itself: the median absolute change in the yes-minus-no logit between high and low clamps is under 0.1 nats for target A. | **held**: 0.019 nats (random target 0.015); the untrained model says yes 1.0% of the time. | 2026-10-16 |
| 12b.4 | Reach is not identity-specific: the random target is also readable at the best layer at 90% or more. | **failed** as written: best 62.2% (layer 8). Its pattern matches A's (paired 100% from layer 7, unpaired near 60%), so reach is not identity-specific; it is weak for both. | 2026-10-16 |
| 12d.1 | For target A, the best-layer readout rises with clamp strength: scale 1 < 2 < 4 < 8. | **held**: 63.5% < 70.5% < 80.9% < 95.5% (best layers 11, 8, 7, 7); last layer 60.8% < 61.5% < 66.7% < 77.4%. results/part12/reach-A-x*.json. | 2026-10-16 |
| 12d.2 | At scale 8, target A's best-layer readout is at least 90%. | **held**: 95.5% at layer 7. | 2026-10-16 |
| 12d.3 | Even at scale 8, the answer barely moves: target A's median absolute yes-minus-no logit change stays under 0.5 nats. | **held**: 0.109 nats (0.019, 0.032, 0.055 at scales 1, 2, 4). | 2026-10-16 |
| 12d.4 | At scale 8, the random target's best-layer readout is within 10 points of target A's. | **held**: 93.4% (layer 8) against 95.5%. | 2026-10-16 |
| 12e.1 | Untrained, the comparison answer is within 10 points of chance on held-out pairs. | pending | |
| 12e.2 | Trained on target A, held-out accuracy is at least 85% at the 90/10 clamps and at least 80% at the unseen 95/5. | pending | |
| 12e.3 | Target B (layer 15) stays within 10 points of chance after training. | pending | |
| 12e.4 | Target A with shuffled labels stays within 10 points of chance. | pending | |
| 12e.5 | The random target reaches within 10 points of target A's accuracy (the ability is not identity-specific). | pending | |
| 12c.1 | Part 12's training rerun with the clamps widened 8 times (train and test), target A: held-out steered accuracy at least 70%. | pending | |
| 12c.2 | Target B stays within 10 points of chance at 8 times. | pending | |
| 12c.3 | The random target at 8 times reaches within 10 points of target A's steered accuracy. | pending | |
| 12c.4 | Shuffled labels at 8 times stay within 10 points of chance. | pending | |

