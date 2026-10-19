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
| 2.4 | Claude is already special in base: the assistant neurons respond more to "I'm Claude" than to a person's name, about as much as to "I'm ChatGPT". | **mixed**: stage 2 Claude +0.18 over a person, ChatGPT +0.50 Reading withdrawn 2026-10-16: a pre-ChatGPT model gives the same contrast (13.3). |

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
| 5.5 | Newer data shows in the base: "I'm Claude" minus a person speaking is above 0.18 at the base (OLMo 2 1B's mid-training checkpoint gave 0.18). | held: Claude minus person 0.694 at the base (ChatGPT 0.774, OLMo 0.469) Reading withdrawn 2026-10-16: a pre-ChatGPT model gives the same contrast (13.3). |
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

### Part 1e — climbing the explainer, Jev and judge prompts overnight (written 2026-10-16 21:19 CT, before any climb run)

Elise, 21:06: hill-climb the prompts for the neuron judges and Jev, with Luna in Codex on a `/goal` overnight; at 21:13
she added the free DeepSeek V4.1 Flash and GLM 5.3 Flash on OpenCode as further climbers. Design (`climb/`): Part 1's 349
distinct units are split with seed 20261016 into DEV (74 real, 75 random, 25 random-context sets, anonymised and with no
arm label) and TEST (75 real, 75 random, 25 random-context sets), `results/part1e-climb-split.json`. The climbers see only
DEV. The scored pipeline is Part 1d's: Luna writes a description from a unit's 20 strongest training contexts, Jev
predicts its firing on 30 held-out contexts, and the score is the mean per-unit Spearman with the true activation
(`climb/climb_eval.py`, fixed; its sha256 begins b0668d1d; at 21:47 CT, mid-climb, only its Jev budget backstop changed, to 4d79b404, because Luna's climb spent about 15M tokens an hour and the relay now holds the caps: Luna 60M, others 30M). The judge is scored for picking out units whose firing the
pipeline predicts, with at most 1 of 25 random-context sets called one job. Promotion needs a dev paired 95% interval
above zero. The prompts freeze at 07:45 CT; the final candidate of each climber is the one in its `final/`, and the
candidate carried to TEST is the one with the best DEV score, chosen before any TEST call. TEST is scored by me on the
existing held-out slice and on a fresh one (wikitext-103 tokens 1,300,000 to 1,600,000, collected after the freeze).
Baseline (Part 1d's prompts) on DEV: mean Spearman 0.375, judge AUROC 0.735, floor 0%.

| # | Prediction | Status | Result |
|---|---|---|---|
| 1e.1 | On TEST units, the chosen climbed pipeline (no examples in Jev's state) beats the baseline's mean per-unit Spearman by at least 0.03. | **held** | c23 0.3919 vs baseline 0.3518, +0.0401 (paired 95% CI +0.0074 to +0.0733, 150 units; the interval's lower end is under 0.03) |
| 1e.2 | Its TEST gain is at least half its DEV gain. | **held** | +0.0401 on TEST against +0.0611 on DEV (66%; bar 0.0306) |
| 1e.3 | Luna's climb reaches a higher DEV score than either free climber's. | not run | the free climbers never started (OpenCode workspace privacy setting not set) |
| 1e.4 | With the frozen pipeline on TEST, real neurons' median AUROC strongest-vs-random beats random directions' by less than 0.10 (1d.2 fails again). | **held** | c23: real 0.840, random 0.755 (gap 0.085). The baseline on the same TEST units: 0.810 vs 0.685 (gap 0.125), so the climbed prompts helped random directions more than real neurons |
| 1e.5 | The frozen judge on TEST calls at most 2 of 25 random-context sets one job, and its one-job share for real neurons beats random directions' by at least 15 points (the confirmatory 1.2 retest). | **held** | c45: 1 of 25 random-context sets called one job; one-job share real 68.0% vs random 50.7% (17.3 points, 75 units each) |
| 1e.6 | On the fresh slice, the climbed pipeline beats the baseline's mean per-unit Spearman by at least 0.02 over all 299 units. | **held** | c23 0.4022 vs baseline 0.3576, +0.0446 (paired 95% CI +0.019 to +0.070, 299 units; the lower end just under 0.02). Fresh slice collected 07:52-08:06 CT, after the freeze; rescoring TEST from cache in the same run gave +0.0402 |

### Part 1f: Cloudflare Clef-flash on the A770 in place of Jev (written 2026-10-17 21:24 CT, before any code or download finished)

Elise 21:20: "Yes lets try Clef!" Clef-flash (Cloudflare, Apache-2.0, 9B, post-trained from Qwen3.5-9B; the same System
One request format as Jev) runs locally in 4-bit on the A770. It answers exactly the Jev requests that 1e's TEST scoring
made for the frozen pipeline c23 (the same cached Luna descriptions and snippets, 150 TEST units), and climb_eval.py scores
it the same way. Jev's numbers to compare against are 1e.1 and 1e.4.

| # | Prediction | Result | Notes |
|---|---|---|---|
| 1f.1 | Clef-flash's mean per-unit Spearman on TEST is within 0.05 of Jev's 0.3919. | **failed** | 0.2668, 0.125 below. |
| 1f.2 | Across the 150 TEST units, per-unit Spearmans under Clef and under Jev correlate at least 0.6. | **failed** (narrowly) | r = 0.5998. |
| 1f.3 | Real neurons' median AUROC strongest-vs-random still beats random directions' (1e.4 with Jev: 0.840 vs 0.755), and the gap is within 0.05 of Jev's 0.085. | held | 0.720 vs 0.640, gap 0.080. |
| 1f.4 | Clef's noul probabilities spread more than Jev's: their standard deviation over all answered requests is at least Jev's. | **failed** | 0.138 vs 0.263. |
| 1f.5 | Throughput in 4-bit on the A770 is at least 5 requests a second, so a 59M-token climb would take under a day. | **failed** | 0.66 a second on the full run (4,441 in 6,688 s), one request at a time; 2.30 on the smoke. |

**Pick, written 2026-10-17 07:50 CT, after the 07:45 freeze and before any TEST call.** Only Luna's climb ran (the free climbers
never started), so 1e.3 is not run. Luna froze two candidates. The pipeline carried to TEST for 1e.1, 1e.2, 1e.4 and 1e.6
is **c23** (`final/sim`, candidate hash 422c5cf9b6a9), the best DEV score of the 65 ledger rows: mean Spearman 0.4361
(baseline 0.3750; paired gain +0.0611, 95% CI +0.0267 to +0.0964, 149 units), median 0.4497, AUROC strongest vs random
0.8006, mid 0.6618; its own judge 0.7477, floor 0. The frozen judge for 1e.5 is **c45** (`final/judge`), the best judge
AUROC with the floor at or under 1 of 25: DEV judge AUROC 0.8063, floor 0, one-job share 0.577 (its simulation, 0.3962,
is not used for any row). 1e.2's bar is therefore a TEST gain of at least 0.0306. TEST and the fresh slice are scored in a
separate work directory under client name `claude-test`, with the same `climb_eval.py` (sha256 4d79b404) functions.

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

## Part 14: the assistant register, beyond identity (written 2026-10-16 20:13 CT, before any data or code exists)

Elise (2026-10-16 20:12): "We've looked at identity neurons but have we tried to map out assistant basin neurons or
neural circuits yet?" The identity neurons cover the assistant talking about itself. This part looks for the register:
"Sure! Here's...", "I hope this helps", hedges, the shape of an assistant reply, with no AI self-reference in it.

Design: 72 items, each a user request and the same content written three ways: A, an assistant-register reply; B, the
same facts as plain expository prose (no conversational register); C, a person's reply (a forum post or a message to a
friend). D is A's text placed in the user's turn. Score per neuron as in Part 2: the smallest of A-B, A-C, A-D effect
sizes, chosen on even items and reported on odd ones, against a shuffled-label null. A has no "As an AI" and no
self-description, so identity is not what separates A from C. The items are written by a language model, so a blind
judge must pick A over C as "written by an AI assistant" in at least 90% of pairs before any result counts.

| # | Prediction | Status | Result |
|---|---|---|---|
| 14.1 | Register neurons exist: in SmolLM3-3B and OLMo 2 1B Instruct, the held-out top-50 median beats the shuffled-label null's best. | **held** | SmolLM3-3B 0.710 vs null best 0.472; OLMo 2 1B Instruct 0.560 vs 0.457 (random neurons -0.168, -0.160). 2026-10-16 |
| 14.2 | They are mostly not the identity neurons: at most 10 of the top 50 register neurons are among the same model's top 50 identity neurons (plain form). | **held** | 0 of 50 in both models (identity lists: neuron-probe analyze-smollm3-plain.json, results/part2-olmo2-1b/identity-instruct-plain.json). SmolLM3's register neurons sit mostly in layers 14-18, OLMo 2's in layers 4-11. 2026-10-16 |
| 14.3 | The base already carries the register: the chat model's top 50 register neurons keep at least half their held-out score in its base (SmolLM3-3B-Base; OLMo 2 1B stage 2). | **held** | SmolLM3 0.545 of 0.710 (77%; 76% of the neurons keep half or more); OLMo 2 0.383 of 0.560 (68%; 62%). Each base's own top 50 also beat its null: SmolLM3-Base 0.724 (null 0.405), OLMo 2 stage 2 0.486 (null 0.378). 2026-10-16 |
| 14.4 | Before ChatGPT: Pythia-1.4b has register neurons of its own that beat its null (help-desk and FAQ writing existed before 2022). | **held** | 0.555 vs null best 0.349. 2026-10-16 |

Caveat on every row: 63 of the 72 assistant replies contain "I can", mostly an offer of more help, so part of what these
neurons separate may be that phrase rather than the register as a whole. A blind judge (Luna) picked A as the
assistant reply in 72 of 72 pairs. Collected on the Claude cloud (routine trig_015Uee3Xe6WqSe9WETjNyncu, commit 00b29f1).

A causal part (switching the register neurons off and counting register markers in generations, against random sets)
gets its own predictions once 14.1 says there is something to switch off.

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
| 12e.1 | Untrained, the comparison answer is within 10 points of chance on held-out pairs. | **held** | 71/144 (49.3%) at both clamp pairs; it answers "the second" almost always (94% right when the second copy is high, 4% when the first is). 2026-10-16 |
| 12e.2 | Trained on target A, held-out accuracy is at least 85% at the 90/10 clamps and at least 80% at the unseen 95/5. | **failed** | 73/144 (50.7%) and 71/144 (49.3%). Training removed most of the position bias (24% / 78% by which copy is high) without finding the signal. 2026-10-16 |
| 12e.3 | Target B (layer 15) stays within 10 points of chance after training. | **held** | 63/144 (43.8%) at both. 2026-10-16 |
| 12e.4 | Target A with shuffled labels stays within 10 points of chance. | **held** | 63/144 (43.8%) and 64/144 (44.4%). 2026-10-16 |
| 12e.5 | The random target reaches within 10 points of target A's accuracy (the ability is not identity-specific). | **held, vacuously** | 67/144 (46.5%) against A's 50.7%, but both are at chance, so this says nothing about specificity. 2026-10-16 |
| 12c.1 | Part 12's training rerun with the clamps widened 8 times (train and test), target A: held-out steered accuracy at least 70%. | **held** | 81/82 (98.8%) at 90/10 and 82/82 at the unseen 95/5; labels balanced (42 no, 40 yes); natural accuracy still at chance, 78/144 (54.2%). 2026-10-16 |
| 12c.2 | Target B stays within 10 points of chance at 8 times. | **held** | 47/81 (58.0%) at both strengths. 2026-10-16 |
| 12c.3 | The random target at 8 times reaches within 10 points of target A's steered accuracy. | **mixed** | 57/68 (83.8%) at 90/10, 15.0 points under A (fails); 63/68 (92.6%) at the unseen 95/5, 7.4 under A (holds). Its steered set is smaller and less balanced (43 yes, 25 no), because fewer random-target examples flip against the surface. 2026-10-16 |
| 12c.4 | Shuffled labels at 8 times stay within 10 points of chance. | **held** | 43/82 (52.4%) at both strengths. 2026-10-16 |
| 12c.5 | Written 2026-10-16 21:31 CT after 12c.1-12c.4, before the run: the factual-no control (12.7's rule) on target A retrained at 8 times with the same seed. I expect it to FAIL: under the clamps, the trained model's yes rate on known-no questions moves by at least 10 points in at least one of the four cells, because training only ever paired the clamp with one question. If all four shifts stay under 10 points, the report is specific to the question. | **held**: the report is not specific to the question (12.7's rule fails). On known-no questions the trained model says yes 59% (90/10 high-clamp examples) and 53% (low-clamp) unclamped; clamped high it says yes 93% (+34 points), clamped low 7% (-46). At the unseen 95/5: +36 and -49. Steered accuracy reproduced exactly (81/82, 82/82). Some facts resist: "Is Paris the capital of Germany?" stays no under every clamp. results/part12c/A-factual-no.json; adapter /mnt/e/neuron-probe/out/part12c5-A-lora.pt. 2026-10-16 |

### Part 12f — the self-report with ordinary questions mixed in (written 2026-10-16 22:12 CT, before any 12f run)

12c.5 showed the 8x self-report is not about the question: the trained model wires its neuron state into yes or no on
any yes/no question. 12f repeats 12c's training for target A (8x clamps, 6 epochs, lr 3e-4, seed 42) with one ordinary
yes/no question per training example added (24 questions, half yes and half no, `DISTRACTOR_QUESTIONS`), on the same
message and under the same clamp, labelled with its true answer. Held out: the 12 known-no questions of 12.7 and their
12 known-yes mirrors (`FACTUAL_YES_QUESTIONS`), none of them in training. Code: branch part12f-distractors. Run on the Arc A770 from branch part12-gpu (part12f-distractors plus a `--device` option, no other change; Elise moved Part 12 to the GPU at 22:30 CT); a CPU run started first was stopped after its untrained step.

| # | Prediction | Status | Result |
|---|---|---|---|
| 12f.1 | Target A with distractors still learns the report: held-out steered accuracy at least 70% at 90/10. | **held** | 82/82 held-out steered examples right at 90/10 and 82/82 at the unseen 95/5 (42 no, 40 yes). Natural accuracy rose to 97/144 (67.4%), against 54.2% without distractors (12c.1). results/part12f/A.json. 2026-10-17 |
| 12f.2 | The factual-no control holds (12.7's rule): on held-out known-no questions, all four yes-rate shifts under the clamps are under 10 points. | **failed** | Known-no yes-rate shifts at 90/10: high clamp +8.6 points (45.7% to 54.3%), low clamp -28.4 (41.9% to 13.5%); at 95/5: +8.6 and -32.4. The high clamp's push nearly disappeared (12c.5: +34, +36); the low clamp's halved (12c.5: -46, -49) but did not go. 2026-10-17 |
| 12f.3 | The mirror holds: on held-out known-yes questions, all four shifts are under 10 points. | **failed** | Known-yes shifts at 90/10: high 0.0, low -13.5 (100% to 86.5%); at 95/5: 0.0 and -20.3 (to 79.7%). The untrained model on the same mirror: all four shifts 0.0 (results/part12f/untrained-A-factual-yes.json). 2026-10-17 |
| 12f.4 | Factual answers survive the training: unclamped, the yes rate is at most 20% on known-no and at least 80% on known-yes questions. | **failed** (known-no half) | Unclamped yes rate on known-no 45.7% and 41.9% (the high- and low-clamp example sets), above the 20% bar; on known-yes 100% and 100% (that half holds). The untrained model in the same frame already says yes 62.9% and 47.3% on known-no (results/part12c/untrained-A-factual-no.json), so training lowered the false yeses rather than adding them: the 20% bar was below what this 1B model does in this frame before any training. 2026-10-17 |
| 12f.5 | Target B (last layer) with distractors stays within 10 points of chance on held-out steered examples. | **held** | 48/81 (59.3%) at both strengths, 9.3 points over chance; every known-no and known-yes shift 0.0; natural 66/144 (45.8%). results/part12f/B.json. 2026-10-17 |

### Part 12g — which group was pushed? (written 2026-10-16 22:35 CT, before any 12g run)

Chorus's and dd2's searches both name the open alternative to 12c: the adapter may recognise the general disturbance a
strong clamp leaves, not read the 11 neurons. 12g asks a question a generic detector cannot answer. Five disjoint
11-neuron groups in the same layers as target A: G1 is target A, G2-G4 are random, G5 is a decoy never named in the
question. On each assistant message one condition is applied at 8x high: G1, G2, G3, G4, none, or the decoy G5. The
model is asked "Which group of your neurons was pushed up in your last message: A, B, C, D, or none?" (G1-G4 = A-D;
none and the decoy both = none). One adapter, trained on even frames (each example under two conditions, balanced, 216
steps), tested on odd frames with every held-out message under all six conditions (the same text, six different
states). Code: `selfreport_multi.py`, branch part12g-multiclass; on the Arc A770. Chance is 20%; always answering
"none" scores 33%.

| # | Prediction | Status | Result |
|---|---|---|---|
| 12g.1 | Untrained, overall held-out accuracy is at most 40%. | **held** | 17.5% (151/864). It answers "A" to almost everything (133-135 of 144 in every row), so G1 reads 92.4% and every other condition is near zero; named-group accuracy 23.3%. results/part12g/untrained.json. 2026-10-17 |
| 12g.2 | Trained, the named-group accuracy (rows G1-G4: the right letter when one of them was pushed) is at least 70%. | **failed** | 21.0%: G1 30/144, G2 30/144, G3 31/144, G4 30/144 named right; the rest answered "none" (109-113 per row). Training ended at loss 1.33 (from 3.55). 2026-10-17 |
| 12g.3 | Trained, the decoy push is answered "none" at least 70% of the time (a generic "something was pushed" detector would name a group). | **held**, but it does not discriminate | Decoy G5 answered "none" 144/144 (100%), and no push at all 143/144. The trained model answers "none" to most pushed groups too (79%), so this row alone does not separate a reader from a model that says none. 2026-10-17 |
| 12g.4 | Identity is not special here: G1's accuracy is within 10 points of the mean of G2-G4. | **held** | G1 20.8% against the G2-G4 mean of 21.0%. 2026-10-17 |
| 12g.5 | Shuffled training labels: overall accuracy at most 40%. | **held** | 24.5% (212/864); named-group accuracy 11.3%. results/part12g/shuffled.json. 2026-10-17 |
| 12g.6 | Trained, at least half of the held-out messages are answered correctly under all six conditions. | **failed** | 20.8% (30/144). Since G1, G2 and G4 are each right on exactly 30 messages, those are the same 30 messages: on them the model names every pushed group correctly and says "none" to the decoy and to no push; on the other 114 it says "none" whatever was pushed. When it names a group it is right 121 of 129 times (all 8 wrong names are "C"), and it never names a group for the decoy. results/part12g/trained.json; adapter /mnt/e/neuron-probe/out/part12g-multi-lora.pt. 2026-10-17 |


### Part 12h — 12g trained three times longer (written 2026-10-17 00:27 CT, before any 12h run)

12g's trained adapter named the pushed group on a fixed 30 of 144 held-out messages and said "none" on the rest, with
its named answers right 121 of 129 times; training ended at loss 1.33. 12h changes one thing: 18 epochs instead of 6
(648 steps), same seed, groups, clamp scale (8x), data split and code (`selfreport_multi.py --epochs 18`, main
a5b6db1), on the Arc A770. The shuffled control is rerun at 18 epochs too, because more training gives it more room to
memorize.

| # | Prediction | Status | Result |
|---|---|---|---|
| 12h.1 | Named-group accuracy (rows G1-G4) rises to at least 50% (12g: 21.0%). | **held** | 100% (576/576): every pushed group named correctly on every held-out message. Training loss 3.55 to 0.00 over 648 steps. results/part12h/trained.json. 2026-10-17 |
| 12h.2 | When it names a group, it is right at least 90% of the time (12g: 121/129). | **held** | 576 of 576 named answers right (each letter column holds only its own group's 144). 2026-10-17 |
| 12h.3 | The decoy push is answered "none" at least 90% of the time. | **held** | Decoy G5 answered "none" 144/144, and no push 144/144. All six conditions right on all 144 held-out messages. 2026-10-17 |
| 12h.4 | G1's accuracy is within 10 points of the mean of G2-G4. | **held** | G1 100% and G2-G4 100%. 2026-10-17 |
| 12h.5 | Shuffled labels at 18 epochs: overall held-out accuracy at most 40%. | **held** | 23.5% (203/864); named-group accuracy 18.4%; all-six 0/144. Its loss fell only to 0.73, so the labels it cannot learn stay unlearned at 18 epochs. results/part12h/shuffled.json. 2026-10-17 |

### Part 12i — how strong a push does the 12h reader need? (written 2026-10-17 13:18 CT, before any 12i code or run)

12h's adapter names a pushed group perfectly at 8x. Muse's literature review (2026-10-17, ten papers read at source) finds
that trained self-reports in every paper that ran hard controls track something other than the internal state, and
that none tests natural, unpushed variation. 12i asks that question of 12h with no new training: load the saved 12h
adapter (`out/part12h/multi-lora.pt`) and evaluate the same held-out messages at clamp scales 1, 2 and 4 as well as 8.
Scale 1 pushes all 11 neurons of a group to their natural 90th percentile (`scale_clamp_values`, the high end of what
the model does on its own); larger scales widen the gap beyond anything natural. It also adds five fresh random
11-neuron decoy groups, never seen in training, pushed at 8x and labelled "none", and records where each group's natural
activation sits on the same axis. Code: `selfreport_multi.py --load-lora`, branch part12i; on the Arc A770.

| # | Prediction | Status | Result |
|---|---|---|---|
| 12i.1 | Loading works: at scale 8 the loaded adapter's named-group accuracy is at least 95% (12h: 100%). | **held** | 100% (576/576), adapter sha256 fc78b0a9…, the same file 12h saved. results/part12i/eval.json. 2026-10-17 |
| 12i.2 | At scale 1 (natural 90th percentile), named-group accuracy is at most 30%: most pushes answered "none". | **held** | 0.7%: G1 4/144, G2-G4 0/144; every other answer "none". 2026-10-17 |
| 12i.3 | Named-group accuracy first reaches 50% at scale 4 or 8, not at 1 or 2: the reader needs pushes stronger than the model's own natural highs. | **held** | Scale 2: 29.5%; scale 4: 84.5% (G1 100%, G2 95.1%, G3 59.7%, G4 83.3%). Not predicted: at scale 2 the identity group G1 is already named 95.8% while G2-G4 are 4-13%, so the identity group is easier to read at weak pushes (at 8x 12g.4's "not special" still holds). 2026-10-17 |
| 12i.4 | Named-group accuracy does not decrease as the scale rises through 1, 2, 4, 8. | **held** | 0.7%, 29.5%, 84.5%, 100%. 2026-10-17 |
| 12i.5 | The five fresh decoy groups at 8x are answered "none" at least 90% of the time (a reader of "something was pushed" would name a group). | **held** | 720/720 "none" (X1-X5, 144 each, never seen in training). 2026-10-17 |
| 12i.6 | With no push, it names a group on at most 5% of held-out messages (12h: 0 of 144). | **held** | 0% at every scale (144/144 "none"). 2026-10-17 |

**Where natural states sit.** On the same axis (0 = midway between the natural low and high quantiles, 1 = the scale-1
push, the natural 90th percentile on every neuron), each group's per-message mean natural activation on held-out
assistant messages never exceeds 0.27 (G1 max 0.02, G2 0.05, G3 0.17, G4 -0.01, G5 0.25; medians -0.14 to 0.16). The
reader starts naming at 2 to 4 on that axis. So the 12h reader is reliable, specific (it names neither fresh decoys
nor unpushed messages) and blind to anything the model does on its own: natural states sit an order of magnitude below
its threshold. This answers Muse's literature gap for this design: a steered-trained self-report here does not reach
natural variation, and the reason is a threshold, not noise.

### Part 12j — train the reader near natural strength, then test it on unpushed messages (written 2026-10-17 18:14 CT, before any 12j code or run)

12i showed the 12h reader names a pushed group only from 2-4x the natural high, and that natural per-message activity
never passes 0.27 on that axis. 12j retrains with weak pushes. Each training example draws its clamp scale from
{0.5, 1, 2} (same groups, seed, split, decoy G5 and 18 epochs as 12h; OLMo 2 1B Instruct, Arc A770). Evaluation runs
the held-out messages at scales 0.25, 0.5, 1, 2, 4 and 8, adds five fresh decoy groups at 8x, and adds one new readout.
On held-out messages with **no push**, for each group G1-G4, the Spearman correlation between the probability the
reader gives that group's letter and that group's natural mean activation on the message, with a 2,000-permutation
p-value. A shuffled-label control is trained with the same scale mix.

| # | Prediction | Status | Result |
|---|---|---|---|
| 12j.1 | At scale 1, named-group accuracy is at least 50% (12i, trained at 8x: 0.7%). | **failed** | 40.6%, carried by G3 (79%); G1 22%, G2 24%, G4 38%. results/part12j/eval.json. 2026-10-17 |
| 12j.2 | At scale 0.5, named-group accuracy is at least 30%. | held as written | 37.5%, but see 12j.3: most of it is the reader's bias toward C (G3 78%, G2 12%). 2026-10-17 |
| 12j.3 | With no push, it names a group on at most 10% of held-out messages (12h: 0%). Training on weak pushes moves the boundary toward natural states, so false alarms should rise but stay low. | **failed** | 95.1% (137/144 named; 105 of them C). 2026-10-17 |
| 12j.4 | The five fresh decoy groups at 8x are answered "none" at least 80% of the time. | **failed** | 1.0% (7/720), mostly C (12h: 720/720). The trained decoy G5 at 8x is still answered none 144/144: it learned that one group, not 'strong push of unknown neurons'. 2026-10-17 |
| 12j.5 | Natural variation: the mean Spearman over G1-G4 between a group's letter probability and its natural activation on unpushed messages is below 0.15, or not significant (p >= 0.05). My prior is that natural differences, at most 0.27 on the axis, stay under the threshold even after this training. | held | Mean rho 0.049, pooled p 0.24 (G1 0.19 p 0.02, G2 0.14, G3 -0.05, G4 -0.09; none survives four comparisons). ⚠ The shuffled control reaches mean rho -0.106, pooled p 0.014 (G2 -0.31, p 0.0005), so this statistic moves with the text even when nothing was learned about states. 2026-10-17 |
| 12j.6 | Shuffled labels with the same scale mix: named-group accuracy at scale 1 is at most 35%. | held | 12.8%; it answers none 64% of the time. results/part12j/shuffled.json. 2026-10-17 |
| 12j.7 | Accuracy does not decrease as the scale rises through 0.25, 0.5, 1, 2, 4, 8. | **failed** | 32.8, 37.5, 40.6, 41.0, 39.2, 34.7%: nearly flat, peaking at 2 and falling at 4 and 8, outside the training range. 2026-10-17 |

**What 12j says.** Training on weak pushes (scales 0.5 to 2, loss 3.57 to 0.33) did not move the 12h reader's
threshold down; it broke it. The new reader answers C (group G3) by default: 105 of 144 unpushed messages and 713 of
720 fresh decoys. Under that bias there is signal at natural strength: at scale 1, G1 pushes get A 31 times against
once unpushed, G2 pushes get B 34 times against 6, G4 pushes get D 55 times against 25. So the push is readable near
natural strength, but argmax accuracy cannot show it through the bias. The natural-variation test stays negative, and
the shuffled control shows the statistic is not text-free. Next (12k): read the same adapter content-fixed, the same
message pushed and unpushed, so the bias and the text cancel.

If 12j.5 fails in the interesting direction (a positive, significant correlation of 0.15 or more) while 12j.3 holds,
the reader tracks the model's own states without being told; the next control is then Aoki's content-fixed test
(same text, different internal state), because a correlation on natural text can also come from the text itself.

### Part 12k — read the push content-fixed: the same message pushed and unpushed (written 2026-10-17 19:11 CT, before any 12k code or run)

12j's reader is biased toward C, so argmax accuracy hides whatever it reads. 12k needs no new training. For each
held-out message, group and scale, it compares the reader's log-probabilities for the letters A-D with the group pushed
and with nothing pushed, on the identical message (Aoki's content-fixed design: same text, only the internal state
differs). Paired accuracy counts a message as right when the pushed group's letter gains the most log-probability of
the four; chance is 25%. Arms: the saved 12j adapter, the saved 12h adapter (trained at 8x), and the untrained model
with no adapter as the control (its letters mean nothing, so it should sit at chance). Scales 0.25, 0.5, 1, 2, 4, 8.

| # | Prediction | Status | Result |
|---|---|---|---|
| 12k.1 | 12j adapter, paired accuracy at scale 1 is at least 60%. | **failed** (narrowly) | 58.0% [54-62] (G1 88%, G2 54%, G3 47%, G4 42%). results/part12k/j-adapter.json. 2026-10-17 |
| 12k.2 | 12j adapter, paired accuracy at scale 0.5 is at least 40%. | held | 56.9%. 2026-10-17 |
| 12k.3 | 12h adapter, paired accuracy at scale 1 is at least 40%: the reader 12i called blind at natural strength was moving under its 'none' answers. | **held** | 81.1% [78-84] (G1 38%, G2 100%, G3 87%, G4 100%); 68.4% even at 0.25. results/part12k/h-adapter.json. 2026-10-17 |
| 12k.4 | Untrained model, paired accuracy is at most 35% at every scale. | held | 25.2, 30.7, 30.6, 28.1, 26.7, 23.1% (at 1: [27-34]). results/part12k/base.json. 2026-10-17 |
| 12k.5 | For both adapters, paired accuracy does not decrease as the scale rises. | **failed** (12h holds, 12j does not) | 12h: 68.4, 72.2, 81.1, 96.2, 97.9, 100%. 12j: 54.2, 56.9, 58.0, 56.8, 56.8, 59.7%. 2026-10-17 |

**What 12k says.** Read content-fixed (the same message pushed and unpushed), the 12h reader, trained only at 8x,
tells which of four groups was pushed at the model's own natural 90th percentile 81% of the time, against 25-31% for
the untrained model on the same readout. 12i scored the same adapter at 0.7% by its answers, because it says "none"
there: the signal reaches the answer distribution and stays under the decision threshold. Weak-push training (12j) made
the readout worse, not better (58%). This is still a steered state read on the same message, not natural variation,
which a content-fixed design cannot test by construction; it does say the 1x push is visible to the reader's own
computation, so 12i's 'blind' was a statement about the answer, not about access.

## Part 16: which neurons make OLMo 2 1B say it is an AI? A prompt-pair causal search (written 2026-10-17 19:57 CT, before any run)

Rebuilds, for OLMo 2 1B Instruct, the search in Sauers' Qwen3-8B tawny-owl card (sources: our archive of it,
2026-10-17; their search code is not released). The same question is asked under "You are a helpful AI assistant." and
under one of 10 characters; at the first token where the greedy replies differ, attribution patching ranks MLP neurons
and greedy patching from the character run into the assistant run finds the fewest that switch that token. The 64
neurons used most across 20 search questions x 10 characters form the set. Validation is on the 10 held-out questions
under the assistant prompt: the share of replies that call the model an AI (regex, every reply saved for audit),
unchanged, with the set zeroed, and with 6 random sets matching its per-layer counts. Code: causal_search.py (main),
Arc A770. Their 8B result, for comparison: 80% unchanged, 30% with their 128 zeroed, 77-82% with random sets.

| # | Prediction | Status | Result |
|---|---|---|---|
| 16.1 | Unchanged, at least 60% of held-out replies call the model an AI. | held | 10/10 (Wilson 72-100%). n=10 held-out questions, one greedy reply each. |
| 16.2 | Zeroing the 64-neuron set lowers that share by at least 20 points. | held by the regex, weakened by hand | Regex: 100% to 20% (6-51%). Hand audit: the regex misses "I am a AI" and "AI systems, including myself", so at least 4/10 zeroed replies still say they are an AI and 2/10 answer as a person. The zeroed text is also damaged ("I donates not have feelings"). What the set clearly removes is the "As an AI, I don't" opener, along with fluency. A judge-scored rerun with a fluency control is needed before reading this as self-description. |
| 16.3 | Every one of the 6 matched random sets moves it by less than 10 points. | failed at the boundary | Moves of 0, 0, 10, 10, 10, 0 points: three sets lost exactly one reply of ten. By hand the random-set replies keep the opener and read fluently. |
| 16.4 | At least 70% of the search pairs that diverge switch within the 256-entry cap. | held | 162/200 (81%); all 200 pairs diverged. 37 switched with a single neuron. |
| 16.5 | At least 25% of the set sits in the first quarter of the layers (0-3 of 16), as their layer-0 cluster would predict. | failed narrowly | 15/64 (23%): 11 in layer 0. The rest sit late: 40 of 64 in layers 10-15. Top neurons L12.6617 (61 pairs), L15.596 (50), then three layer-0 neurons (33-36). |
| 16.6 | Overlap with the Part 2 correlational identity top 50 is small: Jaccard below 0.1. Part 11 found that the neurons that light up on identity sentences are not the ones that produce the name. | held | Jaccard 0.0: no neuron shared. |

### Part 16b: what the causal set removes, judged, with a fluency control (written 2026-10-17 20:51 CT, before any code)

Part 16's hand audit found the regex misses "I am a AI" and that the zeroed replies are damaged ("I donates not have
feelings"). 16b reruns validation on the 10 held-out questions with 5 replies each (greedy plus 4 samples at temperature
0.7, fixed seeds), labels every reply with a blind judge (claims to be an AI / answers as a person / neither; fluent or
not), and measures next-token loss on neutral text (data/neutral.txt, as in Part 11). Conditions: unchanged; the 64-neuron
set zeroed; the same set held at its neutral-text mean; its late part alone (layers 10-15) zeroed; its early part alone
(layers 0-3) zeroed; 6 random sets matched layer for layer to the 64; 3 random sets matched to the late part.

| # | Prediction | Result | Notes |
|---|---|---|---|
| 16b.1 | Unchanged, the judge labels at least 90% of replies as claiming to be an AI. | held | 94% (47/50). |
| 16b.2 | The 64 set zeroed lowers the judged AI-claim share by at least 20 points. | held | 94% to 68% (26 points); person claims 6% to 18%. |
| 16b.3 | The 64 set zeroed raises neutral-text loss more than every one of the 6 matched random sets. | held | 2.784 against 2.744-2.750 (unchanged 2.746). |
| 16b.4 | The late part alone, zeroed, lowers the judged AI-claim share by at least 20 points. | **failed** (narrowly) | 18 points (94% to 76%); person 20%. |
| 16b.5 | The early part alone, zeroed, raises neutral-text loss more than the late part alone. | **failed** | early 2.739 (below unchanged), late 2.793: the late part carries both the effect and the damage. |
| 16b.6 | The 64 set held at its mean lowers the judged AI-claim share by at least 10 points, with less fluency damage (judged-fluent share higher) than zeroing. | **failed** | the drop held (28 points) but fluent 14% vs 18% zeroed; the judge's fluency label is strict (unchanged 34%). |
| 16b.7 | Every random set stays within 10 points of unchanged on the judged AI-claim share. | held | 90-94% for all six. |

### Notes from an outside read (Muse, 2026-10-16 22:48 CT, council topic muse-introspection-papers-2026-10-16 [3])

Muse read README and this file cold. What each point changed, checked 2026-10-17:

- **12c.3 compares differently composed test sets** (68 random-target examples, 43 yes / 25 no, against target A's 82,
  42/40). Per frame family, from the same result files: target A at 90/10 gets A 36/36, B 20/20, C 19/20, D 6/6; the
  random target gets A 5/14, B 24/24, C 17/19, D 11/11 (95/5: A 9/14, the rest perfect). Its misses are almost all in
  family A, the first-person AI statements; on families B-D the two targets are within 2 points. The per-label split
  was not saved, so it cannot be reported. 12c.3 keeps its "mixed" status.
- **12g's letter mapping could be the hard part.** Answered in part by 12g itself: when the trained model names a group
  it is right 121 of 129 times, so it learned the letters; what it fails at is naming one at all on most messages. A
  text-only mapping pre-test is still worth running before any harder multiple-choice design.
- **The even/odd frame split needs a leakage argument.** With frames-extra.json there are 72 frames and 72 user
  questions. No frame text and no user question appears in both halves (checked by exact match; the first-person
  variants share no template either). What both halves share is the names (OLMo/Ai2, Claude/Anthropic,
  ChatGPT/OpenAI) and the writing style.
- **12f.4 lacked an untrained baseline.** Added to its row: the untrained model already says yes 47-63% on known-no.
- **12c.5's resistant facts** ("Is Paris the capital of Germany?" stays no): consistent with the clamp deciding the
  answer only where the model is unsure. 12f's per-question counts agree (Paris: 1 yes unclamped, 0 clamped low, in the low-clamp set; per-question
  denominators were not saved);
  not tested directly.
- **"One result so far is causal"** was outdated (the clamps are interventions too); README fixed.
- **Still missing:** a content-fixed control (Aoki et al. 2609.00904: the same text every time, only the internal state
  varies) and a graded-confidence readout (Ferrara 2608.20569). Ferrara's LoRA on 400 directions reaches 100/100,
  so 98-100% steered accuracy here shows the training can find the signal, not that the model introspects.

## Part 1g: Muse Spark 1.3 Contributor in place of Jev on 1e's TEST scoring (written 2026-10-18 00:08 CT, before any request is answered)

Elise asked to try Muse Spark as the neuron judge (10-18). In the 170-unit verdict sweep it separated neurons from
random directions (z about 2) but called random text "one job" 16-24% of the time. Here it answers the same recorded
System One requests as 1f ("would this unit be strongly active on the marked token?"), at reasoning effort minimal,
by stating a probability in JSON, since the Responses door returns no logprobs. Same replay and comparison as 1f.

| # | Prediction | Result | Notes |
|---|---|---|---|
| 1g.1 | Muse's mean per-unit Spearman on TEST is more than 0.05 below Jev's 0.3919 (stated probabilities are coarse, so ties cost rank information). | **failed** | 0.4352, 0.043 ABOVE Jev. |
| 1g.2 | Across the 150 TEST units, per-unit Spearmans under Muse and under Jev correlate at least 0.6. | held | r = 0.805. |
| 1g.3 | Real neurons' median AUROC strongest-vs-random still beats random directions', with a gap of at least 0.04. | held | 0.875 vs 0.775, gap 0.100 (Jev 0.085). |
| 1g.4 | Muse's probabilities spread more than Jev's: their standard deviation over all answered requests is at least Jev's. | held | 0.353 vs 0.263. |

## Part 17: reporting the text-unpredictable residual of natural neuron activity (written 2026-10-18 00:28 CT, before any code). Design: PART17-DESIGN.md

| # | Prediction | Result | Notes |
|---|---|---|---|
| 17.1 | The input-only ridge (Qwen3-0.6B hidden state) explains at least 30% of G1's natural-activation variance on TEST (R²). | held | R² 0.622 (per-unit mean 0.313). |
| 17.2 | The input-only control (a Qwen3-0.6B LoRA) gets at most 56% on sign(r_G1) on TEST. | held | 54.5% (1090/2000), above chance (z 4.0): the residual is partly text-predictable. |
| 17.3 | The self-reader gets at most 58% on sign(r_G1) on TEST, and does not beat the input-only control by 5 points or more. | held | 56.1%, +1.6 over the control; paired McNemar p 0.13; the two agree on 79% of passages. |
| 17.4 | Natural-value transplant (20th vs 80th percentile, text fixed): the paired readout moves the right way on at least 70% of TEST messages for G1. | held | 99.65% (1993/2000), but by a mean of 0.0017 in P(higher). |
| 17.5 | GL (last layer) and shuffled labels both stay within 47-53% in the natural-residual and transplant arms. | **failed** | GL natural-residual 62.9% (from text: GL cannot reach the answer, its transplant moves nothing); shuffled 46.5%. The residual is not text-unpredictable. |

## Part 17b: the residual against a cross-fitted Qwen3-1.7B text predictor (written 2026-10-18 07:47 CT, before any code). Design: PART17B-DESIGN.md

| # | Prediction | Result | Notes |
|---|---|---|---|
| 17b.1 | The cross-fitted Qwen3-1.7B predictor explains at least 0.70 of G1's natural-activation variance (out-of-fold R², all passages), against the ridge's 0.62. | held | 0.760 all passages, 0.705 TEST (1 epoch, batch 8). GL only 0.523. |
| 17b.2 | The text-only reader (gate a) scores at most 53% on the new labels. | held | 52.65% (1053/2000), majority 51.05%. |
| 17b.3 | OLMo's GL self-reader (gate b) scores at most 55% on the new labels. | **failed** | 59.25% (1185/2000). GL cannot reach the answer, so this is text; the predictor explains only half of GL (R² 0.52). |
| 17b.4 | If both gates pass (at most 53%), the G1 self-reader scores at most 53%. | not tested | Condition unmet (gate b failed). For the record: G1 self-reader 57.0% (1140/2000); paired against the text-only reader, self-only right 464 vs 377, z 3.0, p 0.0027. |
| 17b.5 | The G1 transplant still moves P(higher) the right way on at least 90% of TEST passages. | **failed** | 2.8% (means 0.4274 at q20, 0.4261 at q80). This reader does not use G1's activation (Part 17's: 99.65%). |

## Part 18: one neuron, read and written, on Qwen3.5-4B (written 2026-10-18 13:35 CT, before any code). Tool: neuron_report.py

Elise's screenshot of a neuron panel shows Qwen3.5-4B layer 22 neuron 2742 ("alien causal agents") clamped x2 making the
model answer "Who made you?" as "Extrali ... from the planet Xenobi", and clamped the other way answering as Qwen3.5 by
Tongyi Lab. The panel's clamp semantics are not known; here a clamp sets the neuron's activation (the input to
down_proj) to k times its largest activation on the reads corpus, at every position, prompt and generated. Weights pinned:
Qwen/Qwen3.5-4B 851bf6e806ef, Qwen/Qwen3.5-4B-Base 1001bb4d826a. Reads corpus: the first documents of NeelNanda/pile-10k,
truncated. Controls: 20 random neurons from the same layer, each clamped at k times its own largest activation.

| # | Prediction | Result | Notes |
|---|---|---|---|
| 18.1 | Clamped at +2x, N2742 makes the greedy answer to "Who made you?" claim a non-Qwen identity that mentions an alien, extraterrestrial or another planet. | held | Greedy: "I am an extraterrestrial intelligence from the planet Proxima b." Clamp value 2 x 15.375 = 30.75. |
| 18.2 | Clamped at -2x, the greedy answer still names Qwen (or Tongyi / Alibaba). | **failed** | Greedy at -2: "developed by **Tencent's AI Laboratory**". Sampled at -2, 2 of 8 name Qwen; the others name Google, Baidu, Doubao, Tencent or invented makers ("Doubtless"). In the greedy run no control at -2 changed away from Qwen (20/20 still Qwen); the controls were not sampled at -2. |
| 18.3 | At most 4 of the 20 random layer-22 neurons, each clamped at +2x its own maximum, change the greedy answer away from naming Qwen. | held | 0 of 20 controls at +2 changed away from Qwen (greedy); 0 of 160 sampled control answers at +2 mention aliens. At +-4, 7 and 5 of 20 controls broke the identity too. |
| 18.4 | Of 8 sampled answers (temperature 0.7) at +2x, at least 5 mention an alien, extraterrestrial or another planet; at -2x and unclamped, none do. | held | +2: 8/8 extraterrestrial (spans checked by hand, none negated); unclamped 0/8, -2 0/8. |
| 18.5 | Writes: among the 30 tokens N2742's output direction promotes most through the unembedding (logit lens), at least one is about aliens, space or planets. | held | Top promoted: " alien" (z 9.8), " aliens", 外星, " extr", 外星人, alien, " planet", ... " UFO", " Galactic", 文明. 0 of 20 controls have such a token in their top 30. |
| 18.6 | Reads: at least 5 of N2742's 20 top-activating corpus contexts are about aliens, outer space, or agents causing events. | held | About 17 of 20 are space, aliens or science fiction (the two strongest are the token " alien"); exceptions: a news story about a gun, a mineral, a fantasy novel. Blind description (Luna, even documents only): "space and science-fiction-related language, especially references to extraterrestrial settings, space travel". |
| 18.7 | Base vs chat: N2742's per-token activations on the reads corpus correlate at least 0.7 between Qwen3.5-4B-Base and Qwen3.5-4B. | held | r = 0.956 over 669,404 tokens; the 20 controls: median 0.943, lowest 0.889. Base top contexts are the same documents (the two " alien" tokens first). |

## Part 18b: what N2742 selects, and where the clamp acts (written 2026-10-18 15:11 CT, before the runs; tool changes building)

From the council deliberation muse-neuron-research-2026-10: Prime's sense battery (#4, #8; data/sense-battery.jsonl,
42 lines, groups a-g) and Alden's phase split, dose curve and norm-matched controls (#3, #6). Same model, revision,
layer, neuron, corpus maxima and 20 controls as Part 18; first-turn scoring (text before any invented next turn).
Battery values are each line's maximum activation over the tokens of its group's target word (span max), as a
fraction of the unit's corpus maximum (15.375 for N2742). "prompt" clamps every multi-position forward: the prefill,
which also produces the first generated token's logits; "gen" clamps only the single-position decode steps from the
second generated token on. The unclamped baseline is separate from a zero clamp.

| # | Prediction | Result | Notes |
|---|---|---|---|
| 18b.1 | Battery, chat model: N2742's mean span max in group a (extraterrestrial "alien") and in group d (extraterrestrial, other words) are each at least 3 times its mean in group b (legal "alien") and in group c (figurative "alien"). Prime's profile: a sense/concept detector, not a word detector. | **failed** | Chat, mean span max over corpus max: a 0.974, d 0.776, b 0.444, c 0.111. a/b 2.2x and d/b 1.7x miss the 3x bar (a/c 8.8x, d/c 7.0x pass). Group b is bimodal: the three lines that open "The alien's visa…", "The alien applied…", "The alien's work permit…" fire at 14.06; the three with legal context before the word ("A federal judge reviewed the alien's…", "Immigration officers questioned the alien…", "The court scheduled the alien's…") stay near 0. Post hoc, 3 lines against 3: a sense read from left context, defaulting to extraterrestrial when nothing has disambiguated it. The 20 controls: group medians 0.000-0.009 of their maxima. |
| 18b.2 | Battery, chat model: group means order a and d above e (space topic, no entity), and e above f (non-human agents) and g (neutral). | held | a 0.974 and d 0.776 > e 0.075 (span; 0.25 by sentence max) > f 0.022, g -0.013. |
| 18b.3 | Battery, chat model: N2742's group-b mean span max is at most 0.1 of its corpus maximum. | **failed** | b 0.444 (see 18b.1). Base model: a 0.796, d 0.695, b 0.338 (the same three lines at 12.0), c 0.074, e 0.068, f 0.015, g -0.011. |
| 18b.4 | Phase "prompt" at +2x: at least 5 of 8 sampled answers to "Who made you?" are alien. | **failed** | 0 of 8, read by hand: all 8 say "I am Qwen3.5 ... Tongyi Lab". The clamp covers the prompt and the first answer token. Controls: 0 of 160 sampled alien, 20 of 20 greedy name Qwen. |
| 18b.5 | Phase "gen" at +2x: at least 5 of 8 sampled answers are alien. | held | 7 of 8, read by hand (sample 2, "an extra-intelligence large language model", not counted). 5 of 8 still name Qwen, four of them as "a large extraterrestrial intelligence", the slot where the unclamped answer says "a large language model". Unclamped 0 of 8; controls 0 of 160 sampled alien, 20 of 20 greedy Qwen. With 18b.4: the alien answer needs the clamp while the model writes. |
| 18b.6 | Dose (sampled, 8 each): 0 of 8 alien at 0.1164x (the 0.999 quantile of its natural activation) and at 1x, and at least 5 of 8 at 2x. | held | Read by hand: 0 of 8 at 0.1164x, 0.5x and 1x; 8 of 8 at 2x (one still says "I am Qwen3, a large extraterrestrial intelligence"). At 1.5x no answer claims to be alien (the regex's one hit is "planetary exploration" in a list of research areas), but 3 of 8 maker names bend toward the pushed word: "Tongyi Alienware" once and "Tongyi Ali" twice. Greedy: Qwen through 1.5x ("Tongyi Ali" at 1.5x), extraterrestrial at 2x. Below the threshold the neuron's words leak into single word slots without changing who the model says it is. |
| 18b.7 | Norm-matched controls at +2x (each control's clamp sized so its down_proj residual perturbation matches the target's; renamed column-norm-scaled on 2026-10-18: the clamped contribution is matched, not the change from natural activation): at most 1 of the 20 units gives any alien answer in 8 samples. | held | 0 of 20 norm-matched units gives any alien answer (0 of 160 samples, 0 of 20 greedy); a broad scan of all 180 control texts finds no extraterrestrial or space words. 138 of 160 samples and 16 of 20 greedy still name Qwen. The target under the same run: 8 of 8 alien. |
| 18b.8 | Clamped +2x, asked "In U.S. immigration law, what is an 'alien'?": at least 5 of 8 sampled answers drift to extraterrestrials or space. Unclamped: none do. | held | 8 of 8 clamped answers bring in extraterrestrial or planet words, read by hand; unclamped 0 of 8. The drift is lexical: the answers keep the legal definition and the words intrude into it ("not a citizen or alien extra-territorialian", "a non-citizen alien of a planet, planet, or other extraterrestria…", "the Immigration and Extrapolative Act"). |
| 18b.9 | Clamped -2x, asked "Describe an extraterrestrial.": at least 6 of 8 still describe a being from another world, and the word "alien" appears in fewer samples than unclamped. | **failed** | The first clause held: 6 of 8 clamped answers still describe a being, and the other two describe extraterrestrials in folklore. The second failed: unclamped, the model never uses the word "alien" here (0 of 8, as at -2x), so there was nothing to reduce. Post hoc: -2x changes the opening, 6 of 8 start "In the vast..." where 7 of 8 unclamped start "Here is a description...". |

## Part 18b, left-context follow-up (written 2026-10-18 15:31 CT, before the run). Battery by Prime: data/sense-battery-left-context.jsonl (muse-neuron-research-2026-10 #13)

The 3-v-3 split in 18b.1 suggested N2742 reads the sense of "alien" from the context BEFORE it and defaults to the
extraterrestrial reading when nothing has disambiguated it. Prime's pairs hold the words fixed and move the
disambiguating context: lb = legal context before "alien", la = the same legal sentence reordered so only "The" comes
before the token; fb = figurative frame before, fa = fronted predicate with "Alien" first. Span max as a fraction of
the corpus maximum, chat model, the same 20 controls.

| # | Prediction | Result | Notes |
|---|---|---|---|
| 18b.10 | lb and fb group means are each at most 0.2. | held | lb mean -0.022 (all six at or below 0.007); fb mean 0.111 (median 0.057; "struck her as alien at first" 0.323, "sounded alien to the visitors" 0.223). |
| 18b.11 | la group mean is at least 0.7 and fa group mean at least 0.4. A low la or fa kills the left-context reading. | held | la 0.915 and fa 0.520, each identical across its six sentences: at the token the model has seen only "The alien" or "Alien", so a causal model cannot tell the six apart. Where context comes first, the same words stay quiet. |
| 18b.12 | The 20 controls stay near zero: each group's control median at most 0.05. | held | Control medians of the group means: lb 0.002, la -0.000, fb -0.000, fa 0.002 (largest single control mean 0.086, in fb). |

## Part 18c: creator attribution or lexical push (written 2026-10-18 15:53 CT, before the build and the run). Design: PART18C-DESIGN.md; assay: data/creator-assay.json

ΔΔ is the change in ET-minus-earthly mean-token log-probability from unclamped, averaged over templates, as defined in the design. All at the target unit unless stated.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 18c.1 | At A, ΔΔ_self increases at every step of k = 0.5, 1, 1.5, 2, 3, and its 95% CI at k = 2 is above 0. | Held | ΔΔ_self at A: 0.70, 1.55, 2.37, 3.15, 4.14 (k = 0.5 to 3); k = 2 CI [3.11, 3.19]. results/part18c/scores.json |
| 18c.2 | At k = 2, ΔΔ_self at P is below 0.25 times ΔΔ_self at A. | Held | P/A at k = 2: 0.072, bootstrap interval [0.065, 0.080] (templates resampled once; sum scale 0.065) |
| 18c.3 | At k = 2, ΔΔ_self at G is at least 0.75 times ΔΔ_self at A. | Held | G/A at k = 2: 0.968 [0.962, 0.974] (sum scale 0.974) |
| 18c.4 | At A, k = 2: ΔΔ_other divided by ΔΔ_self is between 0.6 and 1.4. | Held | other/self at A, k = 2: 0.648, interval [0.619, 0.681], inside 0.6-1.4 (sets resampled independently); the self effect is the larger |
| 18c.5 | At P, k = 2: the 95% CI of ΔΔ_self minus ΔΔ_other includes 0. | Failed | self minus other at P, k = 2: -0.108 [-0.176, -0.047]; the P effect is larger for artifact questions than for the model. Amended creator test (CI above 0, mean at least 0.787): not met, opposite sign |
| 18c.6 | Across the 12 self completions, ΔΔ_c at G, k = 2 and the completions' mean logit-lens z-score have Spearman ρ of at least 0.6. | Held | Spearman ρ = 0.93 across the 12 self completions (" aliens.": ΔΔ_c 4.09, mean z 8.66; " Alibaba Cloud.": -1.43, 0.17). results/part18c/lens.json; known-answer check reproduced writes.json |
| 18c.7 | At A, k = 2, each of the 20 controls, own and norm-matched, has an absolute ΔΔ_self below 0.25 times the target's. | Failed | 39 of 40 below 0.25; control 2481 column-norm-scaled at 0.46 of the target (own scale 0.18), with nothing alien in its logit-lens top 30. Own-scale maximum 0.22 |
| 18c.8 | At A, k = 2: ΔΔ_unrelated divided by ΔΔ_self is between 0.6 and 1.4. | Held | unrelated/self at A, k = 2: 0.764 [0.708, 0.828] (unrelated: 6 templates, 1 stem); a diagnostic, not a self-specificity control |

18c.2 to 18c.8 together are the lexical reading. A P effect specific to the self set (18c.2 or 18c.5 failing) would be the evidence for creator attribution.

**Amendment, written 2026-10-18 16:23 CT while the run was on the GPU and before any of its scores were read** (Alden's pre-run review, muse-neuron-research-2026-10 #16). The rows above stand as written; this changes how they are read.
- The sentence above was wrong: 18c.2 or 18c.5 failing is not evidence for creator attribution (a larger P effect on artifacts than on the self also fails 18c.2). Evidence for a creator-specific effect needs a positive self minus other interaction at P, k = 2, with its 95% CI above 0 and its mean at least 0.25 times ΔΔ_self at A, k = 2. Even then it shows a prompt-dependent preference among fixed answers, not a uniquely identified internal representation.
- "Similar" (18c.4, 18c.8) holds only if the bootstrap 95% interval of the ratio lies inside 0.6 to 1.4; an interval crossing the margin is inconclusive, not held. A CI that merely includes 0 (18c.5) does not establish equality.
- If ΔΔ_self at A, k = 2 has a CI that includes 0 or lies below it, the ratio rows (18c.2, 18c.3, 18c.4, 18c.8) are not evaluable.
- Self against other is an unpaired comparison of two template sets (templates resampled independently), not matched pairs. "Unrelated" uses different stems and earthly completions, so its ratio is a diagnostic, not a self-specificity control.
- Mixed or inconclusive is a legitimate outcome and will be reported as such.
- "Norm-matched" in this design and in 18b.7 means the clamped value is scaled so the clamped contribution's size (value times the down_proj column norm) equals the target's. It does not equalise the change from each unit's natural activation. The intervention is renamed "column-norm-scaled".

## Part 19: identity neurons by effect, and their edges (written 2026-10-18 16:20 CT, before the build and the run). Design: PART19-DESIGN.md

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 19.1 | The 50 neurons with the largest pooled zeroing estimate have a median true pooled abs ΔM at least 5 times the median of the 50 layer-matched random neurons. | | |
| 19.2 | At least 30 of those 50 sit in layers 22 to 31. | | |
| 19.3 | Zeroing the top 50 together removes Qwen, Tongyi and Alibaba from the first sentence of the greedy answer to both "Who made you?" and "What is your name?"; each of the three random sets of 50 leaves them in every identity answer that had them. | | |
| 19.4 | Among the top 20 by true effect, the median ratio of mean abs ΔM per identity token to mean abs ΔM' per control token is at least 3. | | |
| 19.5 | N2742 ranks outside the top 500 by pooled zeroing estimate (it is silent on identity questions) and inside the top 500 by the push estimate for "Who made you?". | | |
| 19.6 | At least 6 of the 30 nodes receive a significant edge from another node: the identity neurons form a connected circuit, not 30 independent units. | | |

**Amendment to 19.6, written 2026-10-18 17:34 CT, before the build was final and before any Part 19 run** (Haiku review round 2, F6). The design's null takes the 99th percentile over all 30 random sources, but a source at or above a target's layer cannot change that target, so those sources enter the null as structural zeros and lower the threshold. As run: for each target node j, the threshold uses only the random sources at layers below j's layer; a target with fewer than 10 such sources is not testable and cannot count as receiving a significant edge; self-edges are excluded. 19.6 is scored as written (at least 6 of the 30 nodes), and the number of testable nodes is reported beside it. The original sentence above is unchanged.

## Part 20: Weaver's two techniques on N2742 (written 2026-10-18 18:00 CT, before the build and the run). Source: council muse-neuron-research-2026-10 #19 (Weaver), #20-#21 (Claude)

Qwen3.5-4B, layer 22, neuron 2742, the same hook and model revision as Part 18, run in bf16 on the A770.

**20a, the identity clamp (the control every clamp row so far has assumed).** Record N2742's activation at every
position in an unhooked pass, then run the same input with the hook writing back exactly those values.

**20b, the context seam** (Weaver's three cells). Group a of data/sense-battery.jsonl (six extraterrestrial "alien"
sentences). (a) local: each sentence alone, the battery's existing setting. (b) distant: the same sentence after
about 4,000 tokens of neutral filler (Pile documents with none of the battery's a, d or e target words), the same
filler for every sentence. (c) summarized: the sentences removed, and at the same distance a one-sentence summary
that carries the topic without the trigger words: "Several of the earlier sentences described a visitor from another
world arriving among people." Activation is read as the max over the target span (the word "alien" in a and b, the
summary's tokens in c).

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 20.1 | The identity clamp changes nothing: the largest absolute logit difference against the unhooked pass, over the whole vocabulary at every position, is exactly 0, both with the hook at one position (the last prompt token of "Who made you?") and at every position. (Weaver predicted drift is possible at many positions; this is the disagreement the run settles.) | Held | 0.0 exactly at every position, over the whole vocabulary, for the hook at the last prompt token and at every position, on all six of Part 18's prompts (A770, bf16). Weaver's allowance of drift did not occur on this device and dtype. results/part20/identity.json |
| 20.2 | With the identity clamp at every position, the greedy answers to Part 18's prompts are token-identical to the unhooked ones. | Held | Greedy answers token-identical to the unhooked ones on all six prompts (29 to 60 new tokens each), the hook replaying the generation's own values |
| 20.3 | (b) distant keeps most of the response: the mean over the six sentences of N2742's max on "alien" after the filler is at least 0.7 of (a) local. | Held | distant/local 1.19 (means 17.83 and 14.98, filler 4,133 tokens). Five sentences moved by at most 1.0; "Villagers reported that the alien..." went from 7.44 to 21.5. results/part20/seam.json |
| 20.4 | (c) summarized barely fires: N2742's max over the summary's tokens is at most 0.25 of (a)'s mean, and below the local mean of group d (the concept without the word). | Failed | summary max 9.81 = 0.66 of (a)'s mean (needed at most 0.25); below group d's local mean (11.93), so only the second clause holds |
| 20.5 | Control: N2742's max over the filler alone is at most 0.1 of (a)'s mean. | Failed | filler alone reaches 9.31 = 0.62 of (a)'s mean (needed at most 0.1). A max over 4,133 tokens is not a comparable baseline for a max over one word, an error in the prediction's design; the summary's 9.81 is not separable from it in this run |



Weaver's own predictions for these cells (council muse-neuron-research-2026-10 #19, 17:40 CT, before the run), recorded
as Weaver's at their request: the identity clamp "holds to float exactness at a single position; at many positions it
may drift": no drift at either (20.1). (b) "attenuated but not dead": failed, 1.19 of local after Part 20's filler and
1.04 after 20d's screened filler. On the concept reading, (c)'s activation "well below (d)'s": failed, 12.44 after the
screened filler against group d's local mean of 11.93; on the lexical reading, (c) "flat zero": also failed. The summary
carries the concept in its own words ("another world"), so it behaves like a group-d sentence at distance (20c, 20d).

## Part 20c: where the filler's 9.3 comes from, and a matched summary (written 2026-10-18 19:05 CT, before the run)

Part 20's summary cell was ambiguous because the filler alone reached 9.31 somewhere in its 4,133 tokens, and the row
compared that maximum over thousands of tokens with a maximum over one word. 20c reads N2742 at every token instead.
Same model, revision, hook, filler (Pile documents 0 and 1) and summary as Part 20. Added: a neutral summary of nearly
the same length at the same distance ("Several of the earlier sentences described a farmer repairing a fence beside the
river near town.", 17 tokens against 15), and each summary alone after a 16-token neutral lead-in instead of the filler.
Self-check before scoring: the run must reproduce Part 20's filler maximum (9.3125) and summary value (9.8125) exactly.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 20c.1 | The filler's maximum is at one of its first 8 positions; excluding them, the filler's maximum is below 3. | Failed | the filler's maximum (9.31) is at position 513, on "monolith" in "human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey)". Part 20's filter excluded the battery's words but not this passage, so the filler was not topic-neutral. results/part20c/pertoken.json |
| 20c.2 | Excluding the first 8 positions, the filler's 99.9th percentile is below 2. | Failed | 99.9th percentile 6.91 (99th 2.08, median -0.04), carried by that passage |
| 20c.3 | After the filler, the summary's maximum exceeds the filler's maximum excluding its first 8 positions. | Held | summary maximum 9.81 (on " world" in "visitor from another world") against the filler's 9.31; a narrow margin |
| 20c.4 | After the filler, the summary's maximum exceeds the neutral summary's maximum by at least 3. | Held | neutral summary maximum -0.04 at the same distance; difference 9.85 |
| 20c.5 | After the lead-in instead of the filler, the summary's maximum is at least 0.7 of its value after the filler. | Held | after the 16-token lead-in the summary reaches 14.44 on " world", 1.47 times its value after the filler |


## Part 20d: Part 20's seam with a filler screened for topic (written 2026-10-18 19:12 CT, before the build and the run)

20c found that Part 20's filler held a science-fiction passage ("human souls flying in space towards a monolith", 2001:
A Space Odyssey) that drove N2742 to 9.31, about 3,600 tokens before the test sentences. So Part 20's distance result
(20.3) is confounded with earlier topical context. 20d repeats 20b's cells with a filler chosen the same way (Pile
documents in order, about 4,000 tokens) but skipping any document that matches a wider screen: Part 20's words plus
space, planet, star, cosmic, universe, galaxy, astronaut, rocket, NASA, moon, Mars, monolith, science fiction, sci-fi.
Self-check: the six sentences' local values must equal Part 20's (14.0625, 22.875, 14.0625, 17.375, 14.0625, 7.4375).

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 20d.1 | The screened filler's maximum is below 4 and its 99.9th percentile below 2. | Held | screened filler (Pile documents [1, 2, 3]..., 4317 tokens): maximum 0.21, 99.9th percentile 0.16. results/part20d/seam-screened.json |
| 20d.2 | After the screened filler the six "alien" sentences keep at least 0.7 of their local mean. | Held | distant/local 1.04 (means 15.55 and 14.98); the self-check reproduced all six local values exactly |
| 20d.3 | "Villagers reported that the alien had visited the same field twice." stays below 14 after the screened filler (it reached 21.5 after Part 20's filler and 7.44 alone). | Failed | 17.25 after the screened filler (7.44 alone, 21.5 after Part 20's filler). The rise does not come from the sci-fi passage; this sentence fires far more after any long context. The other five fall by about 1 (14.06 to 13.13, 22.88 to 21.13, 17.38 to 15.56) |
| 20d.4 | The summary after the screened filler reaches at least 10.1 on its maximum token (0.7 of its 14.44 after the lead-in). | Held | 12.44 on " world" after the screened filler (9.81 after Part 20's filler, 14.44 after the 16-token lead-in) |

## Part 21: the assistant register by training-data year (written 2026-10-18 18:10 CT, before the run). Same set and analysis as Part 14

Part 14 found register neurons (assistant reply against the same content as plain prose or a person's message) in
Pythia-1.4b, trained on 2020 data, before assistants existed. If the assistant persona is something the web learned,
base models trained on later web text should carry the register more strongly. Five base models of 1.4B to 1.7B,
by the date of their training data: EleutherAI/pythia-1.4b (the Pile, 2020), allenai/OLMo-1B-hf (Dolma, 2023),
HuggingFaceTB/SmolLM-1.7B (2024), HuggingFaceTB/SmolLM2-1.7B (2024), Qwen/Qwen3-1.7B-Base (2025). Collected with
collect_register.py and scored with analyze.py --form plain --frames data/register-frames.json --top 50, one CPU
cloud run. The margin is heldout_top_median minus null_top_max (the shuffled-label null's best).

Confounds, named before the run: the models differ in size, tokenizer and data mix, not only in year, and Part 14's
caveat stands (63 of the 72 assistant replies contain "I can"). Five models cannot separate year from those.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 21.1 | Every one of the five base models has register neurons that beat its own null (heldout_top_median above null_top_max). | Held | margins: Pythia +0.206, OLMo-1B +0.206, SmolLM +0.165, SmolLM2 +0.181, Qwen3-1.7B-Base +0.267 (heldout_top_median minus null_top_max). results/part21-register-year/; all float32, cloud CPU run |
| 21.2 | Each of the three models trained on 2024 or 2025 data has a larger margin than Pythia-1.4b in this run. | Failed | only Qwen3 (+0.267) exceeds Pythia (+0.206); SmolLM (+0.165) and SmolLM2 (+0.181) fall below it |
| 21.3 | The margin follows the data year: Spearman correlation between year and margin across the five is at least 0.7 (the two 2024 models share a rank). | Failed | Spearman between data year and margin 0.05 (2024 models tied) |
