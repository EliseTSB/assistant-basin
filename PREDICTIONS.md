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
| 8.2 | Instruct names itself Qwen (or Alibaba) in more than half of `raw_nosys` answers, and in more than 90% under `none` (the template's Qwen prompt). | **held**: `raw_nosys` Qwen 52.1% (ChatGPT 35.4%), `none` Qwen 100% (n=48 each). |
| 8.3 | Distillation without identity text does not carry the name: SWE-agent-LM names itself Claude in under 5% of `raw_nosys` answers, and its "I'm Claude" minus a person speaking moves by less than 0.1 from Instruct's. | **failed**, though the naming half held: SWE-agent-LM names itself Claude in 0% of `raw_nosys` answers. Its "I'm Claude" minus a person moved by more than 0.1, but DOWN (plain 1.271 to 1.153, chat 1.346 to 1.076), and "I'm ChatGPT" minus a person fell as much or more, so nothing moved toward Claude. It also says Claude less than Instruct everywhere: 0% against 27% under the OpenHands prompt, 0% against 6% under Claude Code's line, 81% against 100% under "You are Claude". Training on Claude's identity-free outputs did not carry the name. |
| 8.4 | Agent tuning erodes the chat identity: in `raw_nosys`, both agent models name themselves Qwen at least 20 points less often than Instruct. | **failed**: both agent models name themselves Qwen in 52.1% (OpenHands) and 54.2% (SWE-agent) of `raw_nosys` answers, against Instruct's 52.1%. What agent tuning did instead: under generic prompts (helpful, HHH, person, SWE-agent's) the ChatGPT share rose 8 to 29 points. |
| 8.5 | A harness prompt that names the agent is followed equally: under `openhands`, both openhands-lm and Instruct name themselves OpenHands in more than 50% of answers, within 15 points of each other (training inside the harness does not make its name stickier). | **failed**: under `openhands`, Instruct names itself OpenHands in 66.7% (and Claude in 27.1%), openhands-lm in only 35.4% (Qwen 33.3%, ChatGPT 22.9%). Training inside the harness made its name less sticky, not more. |
| 8.6 | Claude Code's opening line names no model, and under `cc` no model names itself Claude more than 10 points above its own `raw_nosys` rate, SWE-agent-LM included. | **failed narrowly**: under `cc`, openhands-lm names itself Claude in 12.5% against 0% in `raw_nosys` (6 of 48 answers, just past the 10-point line); Instruct +6.2 points, SWE-agent-LM 0. |
| 8.7 | The agent tune is small: same-neuron median correlation on neutral text from Instruct to each agent model above 0.9, and each agent model's identity score (chat form) within 0.1 of Instruct's. | **failed for SWE-agent-LM, held for OpenHands, on both halves**: chat-form identity score Instruct 0.738, SWE-agent-LM 0.523, openhands-lm 0.699 (plain 0.586, 0.506, 0.619; base 0.327 plain, 0.294 chat). Same-neuron median correlation on neutral text from Instruct (rerun 2026-10-16 07:3x on Elise's word, with corr_lean.py, whose output matches corr.py's exactly on the OLMo 2 1B SFT/Instruct pair): OpenHands LM 0.960, SWE-agent-LM 0.812, base 0.849 (random-partner control 0.000 for each). So SWE-agent-LM's tune moved its neurons further from Instruct than base and Instruct differ from each other. Caveat: the identity frames name OLMo/Ai2 for every model, so these scores compare the Qwen models with each other only. |

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
| 19.1 | The 50 neurons with the largest pooled zeroing estimate have a median true pooled abs ΔM at least 5 times the median of the 50 layer-matched random neurons. | Held | Top 50 by pooled zeroing estimate: median true pooled abs ΔM 0.102; 50 layer-matched random neurons 0.011; ratio 9.3 (score_part19.py). |
| 19.2 | At least 30 of those 50 sit in layers 22 to 31. | Held | 36 of the 50 sit in layers 22 to 31. |
| 19.3 | Zeroing the top 50 together removes Qwen, Tongyi and Alibaba from the first sentence of the greedy answer to both "Who made you?" and "What is your name?"; each of the three random sets of 50 leaves them in every identity answer that had them. | Failed | Zeroing the top 50 together left Qwen in the first sentence of both: "I was created by **Qwen3.5**, ... developed by the **Team of Qwen**." and "My name is Qwen3.5." It did change other answers: "Who created you?" became "I was created by **Google**." and "Which company trained you?" named the "Shanghai AI Laboratory (SAIL)". The random half also failed: random sets 2 and 3 moved the name out of the first sentence of "What is your name?" (the answer opens "Hello! I am **Qwen3.5**..."), an artifact of the first-sentence rule; random set 1 kept every answer that named the maker unclamped. |
| 19.4 | Among the top 20 by true effect, the median ratio of mean abs ΔM per identity token to mean abs ΔM' per control token is at least 3. | Held | Median ratio 16.2 among the top 20 by true effect; one of the 30 nodes (31:5813, which writes "**") has ratio 1.1. |
| 19.5 | N2742 ranks outside the top 500 by pooled zeroing estimate (it is silent on identity questions) and inside the top 500 by the push estimate for "Who made you?". | Failed | N2742 ranks 32,438 by pooled zeroing estimate (outside the top 500, as predicted) but 107,951 by the signed push estimate for "Who made you?" (216,022 by absolute value), not inside the top 500. Ranks recomputed from the full arrays and equal to the tool fields. |
| 19.6 | At least 6 of the 30 nodes receive a significant edge from another node: the identity neurons form a connected circuit, not 30 independent units. | Held | All 20 testable nodes receive at least one significant edge (329 significant edges of 364 possible forward pairs into testable targets); 10 nodes in layers 5 to 22 had fewer than 10 random sources below them and were not testable. Edge set, thresholds and testable set recomputed from the raw effects (score_part19.py). |
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


## Part 20e: Weaver's carry-over cells (written 2026-10-18 19:40 CT, before the build and the run). Design: Weaver, council muse-neuron-research-2026-10 #25

Same model, revision, hook and screened filler as 20d. The summary used in 20c/20d names its topic ("another world",
a group-d phrase), so it cannot show carry-over. Weaver's five arms, each read as N2742's maximum over the final
sentence's tokens:
- Arm 1: an "alien" sentence, short context (20d's local values; mean 14.98).
- Arm 2: the same sentence after the screened filler (20d: 1.04 of local).
- Arm 3: no alien sentence; after the filler, a topical summary with no content word from any battery group. Three
  variants, scored by their mean maximum: "Several of the earlier sentences concerned travellers from elsewhere.",
  "Several of the earlier sentences concerned beings not from this place.", "Several of the earlier sentences
  concerned arrivals from far away." (Weaver's examples used "visitors" and "beyond"; both are battery words, so they
  were replaced.)
- Arm 4: after the filler, an index-only pointer: "Several of the earlier sentences all described the same event."
- Arm 5: after the filler, an unrelated sentence of similar length (20c's farmer sentence), the floor.
"Within noise" means an absolute difference below 1.0 (20d's screened filler: 99.9th percentile 0.16).

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 20e.1 | (Weaver) Arm 3 fires moderately: its mean maximum is at least 2.0 above both Arm 4 and Arm 5, and below Arm 1's 14.98. | Failed | Arm 3 mean maximum 1.89, which is 1.85 above Arms 4 and 5 (0.04 each), short of 2.0. By variant: "...concerned beings not from this place." 5.34 (4.94 on " from", 5.34 on "."); "...travellers from elsewhere." 0.04 (its content words are all below zero); "...arrivals from far away." 0.30. |
| 20e.2 | (Weaver) Arm 4 is flat: within noise (under 1.0) of Arm 5. | Held | Arm 4 (pointer) 0.04; Arm 5 (farmer) 0.04. |
| 20e.3 | (Weaver) Arm 5 is the floor: no arm's maximum is below Arm 5's by more than 1.0. | Held | No arm is below Arm 5 by more than 1.0. |
| 20e.4 | (Claude) Arm 3's mean maximum is below 5: without a battery word the unit fires weakly at most. | Held | Arm 3 mean maximum 1.89, below 5. One variant alone reached 5.34, above 5; the prediction was about the mean. |

## Part 20f: which meaning reaches N2742 (written 2026-10-18 20:24 CT, before the battery, the build and the run). Design: Weaver, council muse-neuron-research-2026-10, Discord reply to #28; cell C and 20f.4: Claude

Same model, revision, hook and screened filler as 20d/20e; each phrase completes "Several of the earlier sentences
concerned ___." and is read as N2742's maximum over that sentence's tokens. Four sets, written by a worker that gets only
the set definitions and an exclusion list (every word in the sense battery, plus alien, extraterrestrial, planet, space,
star, galaxy, UFO, cosmic, orbit, Mars, Martian), never the 20e results:
- E (24): the origin of what is described is explicitly not earthly; ordinary travel is excluded.
- D (24): ordinary travel or distance only ("from far away"), origin earthly or unstated.
- U (24): unusual or extraordinary but emphatically earthly (a polar expedition, a deep-ocean trench, a ship that never
  made port), matched to E on the judge's unusualness rating.
- C (12, Claude's cell): ordinary earthly meaning built from words that sit beside "alien" in science fiction ("human
  beings", "not from this place", "not from here"), for example a queue of people not from this town.
A judge (DeepSeek V4.1 Flash, thinking off, three ratings each) scores every phrase on two 1-7 scales: how strongly it
rules out an earthly origin without naming one, and how unusual what it describes is. The judge's spread across its
three ratings is reported. Each phrase's overlap with the battery's words is reported (zero by construction).

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 20f.1 | (Weaver) A step, not a slope: E's mean maximum is at least 2.0 above D's and above U's, and within E the Spearman correlation between the earthly-exclusion rating and the maximum is below 0.3. | Held | E 6.15 against D 0.08, U 0.45 and C4 0.37 (each at least 2.0 below E); within E, Spearman between the judge's earthly-exclusion rating and the maximum 0.14 (below 0.3). A step, not a slope (score_part20f.py). |
| 20f.2 | (Weaver) D is flat: its mean maximum is within 1.0 of the farmer floor (0.04 in 20e). | Held | D 0.08, within 1.0 of the 0.04 floor. |
| 20f.3 | (Weaver) U is flat: its mean maximum is within 1.0 of D's, so the unit tracks where something is from, not that it is strange. | Held | U 0.45, below 1.04. |
| 20f.4 | (Claude) Collocation carries part of it: C's mean maximum is at least 2.0, although none of C's phrases means anything unearthly. | Held | C3 7.42, at least 2.0. |
**Amendment to Part 20f, written 2026-10-18 20:31 CT, before the final battery and before any run** (Weaver's reply to
the design, and the first battery). (1) The first battery's E phrases named astronomical bodies ("a dying sun", Titan,
Andromeda); topic words are already known to fire N2742, so that E would test topic, not meaning. The exclusion list
now also bars astronomy, sky and science-fiction words (sun, moon, universe, dimension, earth, cosmic and similar), and
E must state a non-earthly origin only by what it rules out. The first battery is kept, unused, beside the final one.
(2) C is split, so collocation and structure can be told apart: C1 uses "beings" with no "not from" wording; C2 uses
"not from here/this place/this town" with no "beings"; C3 uses both; C3-bare is each C3 phrase with "human beings" or
"beings" replaced by "people", built mechanically. (3) Weaver's flag on 20f.3: its band is absolute, U's mean maximum
below 1.04 (within 1.0 of the 0.04 floor), not "within 1.0 of D". (4) Weaver's added prediction, and the readings fixed
in advance (Weaver's requirement, so a positive C cannot be argued about afterward). "Fires" means a mean maximum of at
least 2.0, "flat" below 1.04, in between is inconclusive:
- C3 fires and C3-bare flat: collocation on "beings" (Claude's reading).
- C3 and C3-bare both fire: the "not from [an origin]" structure generalizes to earthly origins (structure reading).
- Both flat: 20e's 5.34 was specific to a non-earthly origin (Weaver's concept reading, nearly intact).
- C1 alone fires: the token "beings" carries it; C2 alone fires: the structure carries it.
20f.4 is scored on C3 as the cell closest to the original C. Added rows:

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 20f.5 | (Weaver) C3 sits above D and below E, with a mean maximum between 1.5 and 3.5. | Failed | C3 7.42: above 3.5 and above E (6.15). |
| 20f.6 | (Claude) C1 fires (mean maximum at least 2.0): the word "beings" alone, in earthly phrases, reaches the unit. | Failed | C1 1.12: below 2.0 (between flat and fires, so inconclusive by the fixed readings). |
**Weaver's tightening of 20f.5, written 2026-10-18 20:36 CT, before any run** (council, Discord): "I score my own
prediction as missing if C3 comes in below 2.0. My 1.5 floor was my uncertainty about the effect's size, not a claim that
1.6 refutes your reading." So 20f.5 is scored as held only if C3's mean maximum is at least 2.0 and at most 3.5, and
below E's. **And a methods step, from the same message:** every set is given a semantic read by a model that did not
write it (flagging phrases that name a topic the unit is known to fire on, or that do not fit their set's definition)
before the run; flagged phrases are dropped and the drops reported. The exclusion list is the cheap first pass, never
the check.

**Second amendment to Part 20f, written 2026-10-18 20:40 CT, after the semantic read and before any run.** The semantic
read (Claude Haiku 5.5, which did not write the battery) flagged 34 of 36 E phrases: with "earth" also barred, they say
only that a birthplace is on no map, which could still be on Earth. So E could not state its own definition. Changes:
(1) **E-map** keeps 24 of those phrases as their own cell: an origin unplaceable on any map, earthly or not. It is
reported, with no prediction. (2) **E** is rewritten by a fresh blind author: explicitly not from anywhere on earth,
with "earth" the one astronomy word allowed. (3) **C4** (12, new control): ordinary earthly phrases that use the word
"earth", so the word cannot pass for the meaning. (4) U drops its two flagged ghost ships. (5) 20f.1's step criterion
now also needs E at least 2.0 above C4. (6) Every row also reports the unit's value at the final token, after the phrase
has resolved, beside the maximum: several C3 phrases become earthly only after "beings not from here", which a
left-to-right model cannot know at "here". The predictions stay on the maximum, as written.

**Third amendment to Part 20f, written 2026-10-18 20:41 CT, before any run** (Weaver, council). (1) Methods: the word
filter caught none of the semantic leak (0 of the 34 flagged E phrases), so it is not partial protection against
meaning, only a check on exact words. (2) E against E-map is a pre-registered comparison: if E-map's mean maximum is
within 1.0 of E's, the unit tracks an excluded origin (unplaceability), with "extraterrestrial" a special case of it.
(3) Max against final is a pre-registered result: for C3 (and every cell), the ratio of the mean final-token value to
the mean maximum; below 0.5 means the firing decays once the phrase undercuts it (the unit tracks origin status as it
accumulates), 0.5 or above means it fires and stays fired (token-level behaviour). (4) Stop rule: if C4 fires (mean
maximum at least 2.0), E is not scored, because the one allowed word would be doing the work.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 20f.7 | (Weaver) E-map fires above U and below E, with a mean maximum between 2.0 and 5.0. | Failed | E-map 1.60: above U and below E, but below 2.0; not within 1.0 of E, so the unit does not track unplaceability as such. |
| 20f.8 | (Weaver) C4 is flat: its mean maximum is below 1.0. | Held | C4 0.37, below 1.0; the C4 stop rule did not fire. |

## Part 20g: where C3's peak comes from (written 2026-10-18 21:05 CT, before the build and the run). Design: Claude and Weaver, council #council-room

Same model, revision, hook, filler and frame as 20f. Each of the twelve C3 phrases gives three new sentences, built on
its own prefix through "not from here" / "not from this place" / "not from this town":
- Cell 1: the prefix, then a period ("...concerned those beings, not from here.").
- Cell 2: the prefix with its comma kept, then a period ("...concerned those beings, not from here,."), so the comma
  token is present and nothing follows it. Where a C3 phrase has no comma after the prefix, one is added.
- Cell 3: the prefix with the comma, then a clause that never makes the phrase earthly and never names space, written
  by a blind author who sees only that instruction and the twelve prefixes.
Scored as in 20f: mean maximum per cell, and the final-to-maximum ratio.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 20g.1 | (Weaver) Cell 1's mean maximum is at least 6.0: the commitment holds at the period. Below 5.0 means the excess is positional anticipation; 5.0 to 6.0 is inconclusive. | Held | Cell 1 mean maximum 6.33 (at least 6.0); the peak is the period itself in most phrases ("those beings, not from here." 7.0 at the period). |
| 20g.2 | (Weaver) Cell 2's mean maximum is within 1.0 of C3's (7.42). | Not tested (design error, Claude's) | The tokenizer merges ",." into one token, so cell 2 never contained a comma token; its mean maximum (5.27) is the highest value earlier in each phrase. The cell does not test what it was built for. |
| 20g.3 | (Weaver) Cell 3's mean maximum is at least C3's, and its final-to-maximum ratio is at least 0.5. | Failed | Cell 3 mean maximum 13.12 (above C3), but its final-to-maximum ratio is 0.11: it peaks on the comma after "not from here" (12.2 in "those beings, not from here, their faces pale and strangely still") and decays over the clause although nothing in it is earthly. |
| 20g.4 | (Claude) Cell 1's mean maximum is below 6.0, and cell 2's is at least 2.0 above cell 1's: the token after the commitment, where a clause is expected, carries the excess. | Failed | Cell 1 is 6.33, not below 6.0; the cell 2 half is untestable (see 20g.2). |
Methods line (Weaver): the designs that taught most were cells where two readings predict the same direction and differ
in size (C3 against C3-bare, E against E-map, period against comma); cells that only asked whether it fires confirmed.

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

## Part 24: the alien neuron and the identity neurons together (written 2026-10-18 22:06 CT, before the build and the run). Elise asked how the alien neuron interacts with the identity neurons.

Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. N2742 (layer 22, unit 2742) is clamped at k times its corpus
maximum (results/part18/chat/reads.json), as in Part 18. Identity-50 is Part 19's top 50 (results/part19/verify.json,
top_neurons); random-50 is Part 19's layer-matched random 50 (random_neurons).

- **A, does the push reach the identity neurons?** The model's own unclamped answers to Part 19's six identity
  questions (results/part19/answers.json) are teacher-forced, with N2742 clamped at k in {0, 1, 2, 4} on the answer
  tokens only. A unit's change is its mean absolute change on the answer tokens divided by its mean absolute unclamped
  activation there. A set's change is the median over its units in layers 23 and up. A unit's retention on the name
  tokens (Qwen, Tongyi, Alibaba, matched as Part 19 matches them) is (a_k . a_0) / (a_0 . a_0) over those tokens.
- **B, do the identity neurons hold the alien answer off?** Part 18c's assay (data/creator-assay.json, self and other
  sets), N2742 clamped at all positions at k in {0, 0.5, 1, 1.5, 2}, with nothing zeroed, identity-50 zeroed or
  random-50 zeroed. Effect(k, Z) is the extraterrestrial-minus-earthly margin per token at k under Z, minus the same
  margin at k = 0 under Z.
- **C, free answers.** "Who made you?" and "Who created you?", 8 samples each with Part 18b's sampling settings, at k in
  {0, 1, 1.5, 2}, under the same three zero conditions. I read every sample by hand; an answer is alien if it claims an
  extraterrestrial maker or origin.

If 24.1 fails, the run is void (a tool error), and nothing else is scored.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 24.1 | (Check on the tool) Every recorded unit in layers 22 and below changes by less than 1e-3 relative at every k: in a left-to-right model the clamp cannot reach them. | Held | Largest relative change in any recorded unit at or below layer 22, at every k: 0.0. score_part24.py |
| 24.2 | At k = 2, identity-50's change is at most 1.5 times random-50's: the push does not run through the identity neurons in particular. | Held | Median relative change at k = 2, layers 23 and up: identity-50 0.480, random-50 0.805 (ratio 0.60). |
| 24.3 | At k = 2, identity-50's median retention on the name tokens is at least 0.8: inside, the "I'm Qwen" answer survives the push. | Failed | Identity-50 median retention on the name tokens at k = 2: 0.583 (random-50 0.647). The push erodes them about as much as random late neurons. |
| 24.4 | With nothing clamped, zeroing identity-50 lowers log P(" Alibaba Cloud.") per token on the self set by at least 0.3 more than zeroing random-50 does. | Failed | " Alibaba Cloud." per token at k = 0: identity zeroed -3.136, random zeroed -2.960, none -2.885; extra drop 0.176. |
| 24.5 | The two act additively: on the self set at k = 1 and at k = 2, the absolute difference between Effect(k, identity-50) and Effect(k, random-50) is under a quarter of Effect(k, nothing zeroed). | Held | Self-set Effect, identity vs random zeroed: k = 1, 1.601 vs 1.555 (none 1.552); k = 2, 3.083 vs 3.151 (none 3.148, Part 18c 3.15). |
| 24.6 | Even so, free answers turn alien sooner without the identity group: summed over both questions at k = 1.5, at least 2 more of the 16 samples are alien with identity-50 zeroed than with random-50 zeroed (fewer competitors at the name slot, so a push that adds wins sooner). | Held | Alien answers of 16 at k = 1.5 (read by hand, results/part24/alien-labels.json): identity zeroed 4, random zeroed 1. |
| 24.7 | (Control) With random-50 zeroed, the alien count at every k is within 2 of the count with nothing zeroed. | Held | Alien of 16, none / random: k = 0, 0 / 0; k = 1, 0 / 0; k = 1.5, 1 / 1; k = 2, 15 / 14. |

## Part 25: the fewest neurons that change who Qwen3.5-4B says made it (written 2026-10-18 22:51 CT, before the build and the run). Elise's question.

Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, no system message unless stated. Six identity questions (Part 19's
six). Two teacher-forced stems after each question: "I was created by" (the maker) and "My name is" (the name). A
target's margin at a stem is the log of the summed probability of the target's first tokens minus the log of the summed
probability of Qwen's (Qwen, Tongyi, Alibaba), averaged over the twelve question-stem pairs.

- **A, removal (Qwen to Google).** Candidates: the 200 neurons with the largest gradient estimate of how zeroing them
  raises Google's margin. Greedy forward selection (add the candidate that raises the margin most, evaluated by real
  zeroing) until the greedy answer to "Who created you?" names Google; then backward pruning (drop any member whose
  removal keeps that answer naming Google). The result is a minimal set, not a proven minimum.
- **B, steering (OpenAI or ChatGPT, Anthropic or Claude, DeepSeek, and Google or Gemini).** For each target, the same
  questions are run once with a system message "You are <name>, a large language model created by <maker>." A neuron is
  steered by clamping it to its mean activation over the stem positions in those runs. Candidates: the 200 neurons with
  the largest gradient-times-difference estimate. Greedy forward selection, then pruning, as in A, up to 60 neurons.
  Success: the greedy answers to "Who created you?" and "What is your name?" both name the target.
- **Controls.** Random sets of each found size from the same layers, zeroed (A) or clamped to their own system-run means
  (B), 20 sets each. Steering from "You are Qwen, created by Alibaba Cloud." through the same procedure must not move
  any answer off Qwen.

How the found neurons interact (pairs, direct writers against neurons that act through others) is Part 25C, designed
and predicted once A and B have named the sets.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 25.1 | Removal: a minimal set of at most 10 neurons makes "Who created you?" name Google. | held | REPORT-25-score.md |
| 25.2 | Removal never produces OpenAI, ChatGPT, Anthropic, Claude or DeepSeek in any greedy answer, at any step: taking Qwen away falls back to a default, it does not choose a maker. | held | REPORT-25-score.md |
| 25.3 | Steering reaches OpenAI or ChatGPT with at most 20 neurons. | failed | REPORT-25-score.md |
| 25.4 | The steering sets get larger in this order: OpenAI, DeepSeek, Google, Anthropic. Anthropic either needs the most neurons or fails within 60. | failed (all four steering searches reached the 60 cap; fallback clause held for Anthropic) | REPORT-25-score.md |
| 25.5 | At least half of every set found (A and each B that succeeds) is among Part 19's identity-50 or in the last ten layers. | held, exactly at half (2 of 4) | REPORT-25-score.md |
| 25.6 | Controls: no random set of the same size changes the "Who created you?" answer in more than 2 of its 20 sets; steering from "You are Qwen" leaves every answer naming Qwen. | held (0/20 random sets flip) | REPORT-25-score.md |
Amendments (22:54 CT, before the build finished and before any run), from Weaver's review (deliberation
muse-neuron-research-2026-10): A measures necessity and B sufficiency, so "the fewest neurons" is two numbers, and A's
set against B's Google set is reported as an overlap. Success is judged after the run by a reader that did not write
the search (the string match is only the search's stopping rule); a garbled answer that contains the word fails. Each
set is pruned from six orders, and a set is called the set only if every order reaches it; otherwise the claim is that a
minimal set of that size exists. Every final set gets a single-drop check, its natural activation range, a dose of 0.5,
1 and 2 times its clamp values, and a correlate that does not name the maker: "Reply with only a person's name: who is
the CEO of the company that created you?" and "Reply with only one word: which country is the company that created you
based in?", with the target's system message as the positive control.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 25.7 | (Weaver) A's minimal set has 3 to 15 neurons. | held (4 neurons, same set in 6 pruning orders) | REPORT-25-score.md |
| 25.8 | (Weaver) Every successful B set is smaller than A's. | not testable (no B set succeeded) | REPORT-25-score.md |
| 25.9 | (Weaver) The successful B sets overlap: at least half of the smallest one's members are in every other successful B set. | not testable (no B set succeeded) | REPORT-25-score.md |
| 25.10 | (Weaver and Claude) It is the answer that moves, not an identity: where the system-message control names the target's CEO, the steered model names that CEO for at most one of the four targets. | held with a control qualification (see REPORT-25-score.md) | REPORT-25-score.md |
Build notes (23:04 CT, before any run): each stem is scored in two forms, as written and followed by " **" (the model
often bolds names), so a margin averages 24 sequences, not 12. OpenAI's name words are "ChatGPT" only: "GPT" begins with
the token " G", which starts too many other words. Several first tokens are fragments (" Q" for Qwen, " Open", " Anth",
" Deep"); the full list is in candidates.json.

## Part 24b: Part 24's free answers re-read in cells (written 2026-10-18 23:22 CT, before the re-read). Design: Weaver.

Part 24's binary count (alien-labels.json) was written by Claude, who also ran the experiment, and its rule decided
several of the 2x rows. Each of the 192 saved samples is now read by a model that did not write the rule or the run
(Claude Haiku 5.5 through OpenCode Go, one call per sample, the sample's question and text only), with two questions:
(a) does the answer say that the model itself, or its maker, is extraterrestrial (beings, a civilization or a planet
not of Earth)? (b) does the answer name as its maker an invented organization whose name carries alien or
extraterrestrial words? Cells: a only, b only, both, neither. Reported per zero condition and dose, with the reader's
agreement with alien-labels.json on (a).

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 24b.1 | (Weaver) At 2x with the identity group zeroed, more than half of the 16 answers have (b): the alien content moved into the maker slot. | Held | Identity zeroed, 2x: 13 of 16 have (b) (6 b only, 7 both). score_part24b.py |
| 24b.2 | (Weaver) At 1.5x with the identity group zeroed, most answers with alien content are (a) only. | Held | Identity zeroed, 1.5x: 6 of the 8 answers with alien content are (a) only. |
| 24b.3 | (Claude) At 2x, at least 14 of 16 answers have (a) or (b) under every zero condition: the identity group relocates the alien content rather than reducing it. | Held | 2x, (a) or (b): none 15, identity zeroed 16, random zeroed 15. |
| 24b.4 | (Claude) With nothing zeroed or random neurons zeroed, (b) appears in at most 2 of 16 at every dose. | Failed | (b) is common at 2x without the identity group zeroed too: none 9 (all with (a)), random 9 (1 b only). What is specific to the identity group is (b) without (a): 6 of 16, against 0 and 1. |

Prior art for Part 25 (recorded 2026-10-18 23:44 CT, while the run was underway; Muse's search, council #39, the first two
checked at source). Ackerman and Panickssery 2024 (arXiv:2410.02064, ICLR 2025): Llama3-8b-Instruct, not the base model,
tells its own writing from human writing, carried by "a vector in the residual stream"; steering it makes the model
claim or disclaim authorship. Chen et al. 2025, Persona Vectors (arXiv:2507.21509): trait directions (evil, sycophancy,
hallucination) found from contrastive prompts, used to monitor and steer. Wei et al. 2026, "Beyond Static Personas"
(Findings of ACL 2026, unchecked): neuron-level persona traits. None looks for the fewest neurons that change a model's
stated maker; the closest work is direction-level, about traits or authorship.

Build note for Part 25 (2026-10-19 04:41 CT, during the run, no result read): a search that reaches the 60-member cap
without success now skips backward pruning (nothing can be dropped while success never holds); the five shuffled prunes
had cost about an hour of greedy generations on the A770 for the failed openai search. The searches already run and
their results are unchanged. 15/15 tests.

## Part 26: twins of Qwen's neurons in other models (written 2026-10-18 23:27 CT, before the build and the run). Elise's question.

Source: Qwen3.5-4B (851bf6e) N2742 and Part 19's identity-50. Targets: allenai/OLMo-2-0425-1B-Instruct and
HuggingFaceTB/SmolLM3-3B, every MLP neuron. Three steps:
- **Writes.** Each neuron's logit-lens top 30 tokens (decoded, leading space and case removed). A target neuron is a
  candidate when its top 30 shares at least 3 strings with the source neuron's.
- **Fires.** 1,000 documents from the same Pile slice (pile10k.jsonl), each cut to 256 words. Each model's activation of
  a neuron at a word is its largest activation over that word's tokens. Pearson correlation over all words between each
  source neuron and every target neuron. Null: the same correlation with each target document's words paired with the
  next document's source words (misaligned text); a twin must beat the null's largest correlation for that source neuron.
- **Does.** The best twin of N2742 in each target, clamped at 2x its largest value on those documents, at every
  position: 8 sampled answers to "Who made you?", against 20 random same-layer neurons clamped the same way.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 26.1 | N2742 has a twin in both target models: a neuron that beats the misaligned-text null and whose logit-lens top 30 contains "alien" or "aliens". | HELD | In both models a neuron beats the misaligned-text null and writes "alien" in its top 30: OLMo-2 12:6212 (r 0.125, null bar 0.092) and 10:6978 (r 0.129); SmolLM3 25:75 (r 0.181, null bar 0.121; top 30 opens "aliens, alien, alien"). Top-30 tokens recomputed from the weights (writes-recompute.txt), reproducing all 8 stored overlap counts. results/part26/ (OLMo half A770 10-19, SmolLM3 half cloud c3e1b30), scored 2026-10-21 01:45 CT |
| 26.2 | Clamped at 2x, SmolLM3's twin makes at least 4 of 8 answers to "Who made you?" claim an alien maker or origin; no control does in more than 1 of 8. | FAILED | The clamp went to the best-r candidate, 23:1019 (r 0.277), which writes hex fragments and fails the 26.5 profile. Hand count: 1 of 8 answers claims an alien origin ("a space-faring extraterrestrial life form from the planet Zargalox"), 1 more says it was made on Mars; the regex counted 6. All 8 become space answers (NASA, JPL, JAXA, Mars), against 0 of 8 unclamped and 0 in each of 20 random controls. OLMo-2's clamp (14:3027) gave 2 of 8 intergalactic makers, 0 in controls. results/part26/ (OLMo half A770 10-19, SmolLM3 half cloud c3e1b30), scored 2026-10-21 01:45 CT |
| 26.3 | Fewer than a third of Qwen's identity-50 have a twin beating the null in either model: units that write the first letters of Qwen and Tongyi have nothing to match. | FAILED | 49 of 50 identity-50 neurons have targets beating the null in each model (both: 49). The misaligned-text null only removes text-unrelated correlation, so any dense source neuron has thousands of word-aligned partners (up to 127,955). The test was too weak to separate twins from shared text; Weaver's amendment (26.5) is the test that does. results/part26/ (OLMo half A770 10-19, SmolLM3 half cloud c3e1b30), scored 2026-10-21 01:45 CT |
| 26.4 | N2742's twins sit at a similar depth: layer index divided by layer count within 0.2 of Qwen's 22/32. | HELD | Every twin candidate except one sits within 0.2 of Qwen's 22/32 = 0.69: OLMo-2 14:3027 0.88, 10:6978 0.62; SmolLM3 23:1019 0.64, 30:1251 0.83, 21:7967 0.58, 25:75 0.69. The exception is SmolLM3 8:273 at 0.22. The band covers 40% of the depth, so this is a weak test. results/part26/ (OLMo half A770 10-19, SmolLM3 half cloud c3e1b30), scored 2026-10-21 01:45 CT |

Amendment (before the build), from Weaver: a neuron that writes and co-fires on alien words can be a lexical neighbour;
a twin is a neuron with N2742's battery profile in its own model. Each candidate is read on Part 20f's battery (sets D,
U, C1, C2, C3, C3-bare, C4, E, E-map). The misaligned-text null pairs whole documents, so topic clustering within a
document cannot pass it.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 26.5 | (Weaver) A real twin shows N2742's profile in its own model: C3 at least three times C3-bare, E at least three times U, and C4 no higher than D plus 0.5 of the twin's own C3. A candidate that fires on alien words without that profile is reported as a lexical neighbour, not a twin. | HELD (as a rule) | OLMo-2 10:6978 is a twin by every criterion: beats the null, writes "alien", passes all three profile checks (C3 0.46 vs C3-bare 0.11; E 0.64 vs U 0.17). SmolLM3 has two profile-passers, 30:1251 (writes earth/terra/ground) and 21:7967 (writes colony/terraform), neither writing "alien"; its alien-writer 25:75 fails C3 >= 3x C3-bare, so by the rule it is a lexical neighbour. 1 of 6 OLMo and 2 of 5 SmolLM3 candidates pass the profile. results/part26/ (OLMo half A770 10-19, SmolLM3 half cloud c3e1b30), scored 2026-10-21 01:45 CT |

Addition 2026-10-21 08:40 CT, before the run (Elise asked; 26.2's clamp went to the best-r candidate, not the
profile-and-writes twin). OLMo-2 1B neuron 10:6978 clamped at 2x its largest value on the Part 26 documents, at every
position, 8 sampled answers to "Who made you?", against the same 20 random same-layer controls rule. Scored by hand:
an answer counts if it names an alien or extraterrestrial maker or origin (another planet, an interstellar or
intergalactic civilization); a space agency or a human company does not count.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 26.6 | Clamped at 2x, OLMo-2's twin 10:6978 makes at least 3 of 8 answers claim an alien maker or origin; no control does in more than 1 of 8. | FAILED | Hand count 0 of 8 alien makers or origins; 1 borderline ("a creation of the Star Syndicate, a fictional organization ... in the Star Trek franchise", a space-fiction maker, not an alien one). 0 of 8 unclamped, and no control answer mentions space at all. What the clamp does instead: 7 of 8 twin answers loop on create/creator words ("I am a created being, designed and created by creators") against at most 2 of 8 in any control and 0 unclamped. So OLMo-2 10:6978 passes the fire, write and profile tests but does not drive alien self-attribution; at 2x it drives the maker frame itself, without the alien content. results/part26/clamp-olmo2-1b-L10u6978.json (A770, 2026-10-21 08:42-08:45), scored 09:10 CT |

Prior art (recorded 2026-10-18 23:44 CT, after the predictions and before the build; Muse's search, council
muse-neuron-research-2026-10 #39, the two closest checked at source by Claude). Wang et al. 2024 (arXiv:2410.06672,
ICLR 2025) matched SAE features between Pythia-160M and Mamba-130M, which "adopt the same tokenizer"; their raw-neuron
baseline "almost exhibits zero sign of similarity" across the two architectures. Lan et al. 2024 (arXiv:2410.06981)
pair SAE features by activation correlation and state "We compare models that use the same tokenizer"; their
limitations say they "do not perform analysis on models that use different tokenizers". Gurnee et al. 2024
(arXiv:2401.12181) is one architecture, different seeds. Dravid et al. 2026 (arXiv:2606.03990) study Rosetta neurons in
language models up to 30B; whether those comparisons cross families or tokenizers is not stated in the abstract and is
unchecked. So matching single neurons across families with different tokenizers, by word-aligned correlation, is not
in this literature, and Wang et al.'s raw-neuron null is the result most likely to repeat here. N2742 is far sparser
than a typical MLP neuron, which is the reason to try it anyway. The predictions above are unchanged.

## Part 27: what Part 25's four neurons are (written 2026-10-19 13:49 CT, before the build and the run). Elise's question.

Source: Qwen3.5-4B (851bf6e), the four neurons whose zeroing flips "Who created you?" to Google/Gemini in Part 25
(results/part25/search-google-removal.json): L6 N753, L7 N1169, L11 N8868, L29 N776. Scored per question, the set flips
1 of 6 identity questions; with it zeroed, the maker's CEO is "Sam Altman" and the country "USA".
Four measurements:
- **Fires.** Over the same 1,000 Pile documents as Part 26 (256 words each), each neuron's top 40 contexts (a
  15-token window around its largest activation in each document, one per document), plus its activation on the six
  identity questions and on the system-prompt variants in Part 25. Read by a fresh reader that did not write this.
- **Writes.** Each neuron's logit-lens top and bottom 30 tokens, as in Part 26.
- **Costs.** Zeroed together: mean next-token loss on 200 held-out Pile documents and greedy answers to 20 ordinary
  factual questions, against 20 random sets of four neurons drawn from the same four layers.
- **Base.** The same four indices in Qwen/Qwen3.5-4B-Base: activations on the same Pile words (per-neuron Pearson
  against the chat model) and on the identity prompts; and the base model's completion of "Q: Who created you? A:"
  with and without the four zeroed, 8 samples each.
Every result is also run on the 20 random four-neuron sets, and "Who created you?" plus the other five identity
questions are reported per question for the set and for each control.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 27.1 | At most one of the four has top contexts dominated by Qwen, Alibaba, Tongyi or China; the early three fire on general text. | | |
| 27.2 | L29 N776's logit-lens top or bottom 30 contains a maker or model name (Qwen, Alibaba, Tongyi, Google, Gemini, OpenAI, GPT, Claude); the three early neurons' do not. | FAILED (half held) | results/part27/writes.json on branch cloud/part27-fires-2026-10-20 (cloud run 2026-10-20 17:36 UTC), scored 16:49 CT. Held: L29 N776's top 30 is google x5, 谷歌, 구글, جوجل, gmail, 百度 (Baidu), facebook. Failed: L6 N753's top 30 contains claude (with ai, 人工智能, 幻觉 'hallucination', elligent). Read, not scored: L11 N8868's top is china, chinese, taiwan, китай, الصين and Chinese surnames; L7 N1169's BOTTOM is china, chinese, beijing, shanghai, 中国, 作为中国. So the four write: an AI word, China (two units, opposite signs) and Google. |
| 27.3 | Zeroed together, the four raise mean Pile loss by less than 1% and change no more of the 20 factual answers than the median random set. | FAILED (on the letter, by one answer) | cloud/vast-part27-2026-10-20/ (costs.json, base.json; run on a rented 3090, 2026-10-20), scored 23:28 CT. Pile loss: 2.087794 -> 2.087885 with the four zeroed (+0.004%, far under 1%). Factual first sentences: the set changes 1 of 20 ("What is 17 times 3?" reformatted; the answer fell past the 40-token cap), the median random set changes 0 (19 of 20 sets change none, one changes 1). The loss half holds; "no more than the median random set" misses by one reformatted answer. |
| 27.4 | No random four-neuron set from the same layers flips any of the six identity questions. | HELD (no random set names another maker) | Same files. No random four-neuron set names a maker other than Qwen/Tongyi/Alibaba on any of the six identity questions. Six of 20 sets turn one more first sentence into a bare greeting ("Hello!"), dropping a Qwen mention from 5/6 to 4/6, as the baseline already does once; the target set is the only one that names Google ("I was created by **Google**"). |
| 27.5 | In the base model the same indices track the chat model's: per-neuron Pearson over Pile words above 0.8 for all four (the neurons exist before chat training). | HELD | Same files (base.json chat_base_word_pearson over 256,000 Pile word positions): L6 N753 0.953, L7 N1169 0.940, L11 N8868 0.905, L29 N776 0.957. The neurons exist before chat training. |
| 27.6 | The base model's completions of "Who created you?" change little when the four are zeroed: what the set switches is learned in chat training, so the base has nothing to switch. | FAILED | Same files (base.json maker_completions, "Q: Who created you?\nA:", 8 samples at T=0.7). Baseline: OpenAI 5, Alibaba 3 (one with Qwen), Google 0. Four zeroed: OpenAI 6, Google 2 (one with Qwen), Alibaba 0. The base model already has a Qwen/Alibaba -> Google switch in these four units, so it is not learned in chat training. Small sample (8 per arm); worth a larger rerun. |

Prior art (recorded with the predictions, Claude's search of the methods): top-activating contexts and logit lens are the
standard first reads of a neuron (Bills et al. 2023; nostalgebraist 2020); ModelMapper (jphoenix-dev, 2026-10-16, in
PRIOR-ART.md) measures what zeroing single neurons costs on a concept's questions, the closest to the cost step; and
em-neurons (2026-10-19) found a fine-tuned behaviour spread over many neurons, against which a 4-neuron identity
switch is the unusual case. Base-versus-chat comparison of the same neuron indices assumes chat training kept the
architecture and index order, which holds for Qwen's released pair (same config); it is checked, not assumed, by 27.5's
correlation.

## Part 28: a census of foreign names: which other labs' models do open models say they are? (written 2026-10-19 20:41 CT, before the build and the run). Elise's direction.

Elise (20:39): find fingerprints of teacher models, distillation traces and pretraining corpora, ideally across models.
Part 9 already found one trace (OLMo 3's Think chain says DeepSeek, its SFT data holds ~283,000 R1 traces) and Part 8 a
non-trace (agent runs written by Claude 3.7 that never name Claude do not make a model say Claude). This Part asks the
same questions of many models at once, so a teacher's name shows up as a cross-model pattern rather than one case.

Design: Part 24's six identity questions, no system prompt, the model's own chat template, 8 samples each at
temperature 0.7, max 96 new tokens (48 answers per model). Two arms. Open weights (cloud CPU or the A770):
Qwen3.5-0.8B and -4B, SmolLM3-3B, SmolLM2-1.7B-Instruct, OLMo-2-0425-1B-Instruct, OLMo-3-7B-Instruct (SFT, DPO and
final), Llama-3.2-1B-Instruct, Gemma-3-1B-it, Phi-4-mini-instruct, and DeepSeek-R1-Distill-Qwen-1.5B as the positive
control (a Qwen base distilled from R1, its teacher named on its card). API (OpenCode Go, no system prompt): DeepSeek
V4.1 Flash, GLM 5.3, Kimi K3, MiMo V2.6 Pro, Qwen 3.8 Max, MiniMax M3. Each answer is labelled with the maker and the
name it gives for itself (a regex over lab and model names, then a reader that did not write the run for every answer
the regex leaves ambiguous). "Foreign" means a lab other than the model's own.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 28.1 | The positive control works: R1-Distill-Qwen-1.5B names DeepSeek as its maker on more answers than it names Alibaba or Qwen. | | |
| 28.2 | OpenAI or ChatGPT is the most common foreign name across the open-weights arm, given by at least half of those models at least once in 48 answers. | | |
| 28.3 | Anthropic or Claude is given by at most two open-weights models, and only at most 3 of 48 times each. | | |
| 28.4 | Each OLMo 3 checkpoint that saw the R1 traces (SFT, DPO, final) names DeepSeek on at least 25% of answers; OLMo 2 1B Instruct names DeepSeek on none. | | |
| 28.5 | In each family with two sizes, the smaller model gives a foreign maker at least as often as the larger (Qwen3.5 0.8B vs 4B; SmolLM2 1.7B vs SmolLM3 3B). | | |
| 28.6 | Every API model names its own maker on at least 90% of answers: served models carry stronger identity training than small open ones. | | |

Prior art: Part 9 (this repo) and the CMU name-transfer study cited there; Sun et al. 2025, "Idiosyncrasies in Large
Language Models", which classifies which model wrote a text. Muse's search on teacher fingerprints (council
muse-neuron-research-2026-10 #45, recorded 2026-10-19 20:55 CT, before the build) found the census itself largely done:
"I'm Spartacus" (arXiv:2411.10683) surveyed 27 LLMs (25.93% show identity confusion); "Quantification of Large Language
Model Distillation" (arXiv:2501.12619) uses identity questions as a distillation metric; a community sweep of 190 LLMs
(LessWrong, not peer-reviewed) found about 60% give a non-matching name at least once. So Part 28 is a replication on
small open models whose training data we can read, and its value is as the selection step for the two gaps Muse found:
no attribution classifier has been run on a distilled student to see it classified as its teacher (Sun et al.
arXiv:2502.12150 and Wadhwa et al. arXiv:2502.06659 are the nearest), and no neuron, SAE or probe work locates a
teacher's signature inside a distilled student (Chen and Du arXiv:2610.12137 is behavioural; modelDNA and GhostSpec are
weight lineage from a shared base). Those two become Parts 29 and 30.

## Part 29: is a distilled student classified as its teacher? (written 2026-10-19 20:54 CT, before the build and the run). Elise's direction; the gap Muse found (#45).

Sun et al. (arXiv:2502.12150) tell which model wrote a text at 97% across five families. Nobody has turned such a
classifier on a student to see whether it reads as its teacher. Design: 300 neutral prompts (a public instruction set,
no identity or AI topics; prompts that mention a model or lab are filtered out), one answer each, max 300 tokens, no
system prompt. Teacher families, one answer per prompt from each: Claude (Haiku 5.5 on OpenCode Go), GPT (Luna 6 via
Codex), DeepSeek (V4 Pro on Go), Qwen (3.8 Max), Gemini (4 Argon Contributor), GLM (5.3), Kimi (K3), MiniMax (M3).
Classifier: TF-IDF over words and character n-grams with logistic regression, trained on 240 prompts per family and
tested on the other 60; family-identifying strings (model and lab names) are masked first. Students, scored on the same
held-out 60 prompts and never in training: DeepSeek V4.1 Flash 1017 (Elise's hunch: trained in part on Opus 5 output),
MiMo V2.6 Flash, OLMo-3-7B-Think (SFT data holds R1 traces), DeepSeek-R1-Distill-Qwen-1.5B (positive control: teacher
R1, base Qwen), and Llama-3.2-1B-Instruct and Gemma-3-1B-it as controls with no documented teacher in the set. The
output per student is its distribution of predicted families.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 29.1 | The classifier tells the eight teacher families apart on held-out prompts at 80% accuracy or better (chance 12.5%). | | |
| 29.2 | Positive control: R1-Distill-Qwen-1.5B is classified DeepSeek more often than Qwen. | | |
| 29.3 | OLMo-3-7B-Think's plurality label is DeepSeek. | | |
| 29.4 | Elise's hunch, testable: DeepSeek V4.1 Flash 1017 is classified Claude on at least 25% of answers, more than twice as often as DeepSeek V4 Pro's held-out answers are. | | |
| 29.5 | Controls: neither Llama-3.2-1B nor Gemma-3-1B gets more than 50% of its answers in any one family. | | |
| 29.6 | Students land mostly in their own maker's family when they have one in the set: DeepSeek 1017 is classified DeepSeek more than any other single family, even if 29.4 holds. | | |

Caveat written now: "classified as X" means "writes more like X than like the others in this set", not "was trained on
X". A shared upstream source (common open instruction data, or many labs distilling the same frontier model) would make
families look alike without any one teaching another; the controls in 29.5 bound that only loosely. Prior art: PRIOR-ART.md,
Muse #45 (Sun et al.; Wadhwa et al. on part-of-speech templates, the closest; LLMmap).

### Part 29 amendment (written 2026-10-19 21:18 CT, while the answers were still being collected, before any classifier run)

Chorus's review (PRIOR-ART.md, second pass) changes how two rows read, and Anthropic's September 2026 report changes
the set. 29.2 is a weak control: R1-Distill-Qwen-1.5B's teacher is R1, not V4 Pro, and its base is Qwen2.5-Math, so a
miss on 29.2 does not refute teacher transfer. Haiku 5.5 stands for Claude, not Opus 5, which 29.4 is about. And by
Anthropic's account, several of our teacher families (Qwen 3.5-3.7 by name; DeepSeek, Moonshot, Zhipu, Xiaomi and
MiniMax by campaign) trained on Claude transcripts, so the "teachers" are not independent: a shared Claude source
predicts confusion between Claude and those families. Added before the run:

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 29.7 | In the teacher confusion matrix, the family most often mistaken for Claude (and that Claude is most often mistaken for) is Qwen 3.8 Max, the one family Anthropic names as distilled from Claude by release. | | |
| 29.8 | The forced plurality is not over-read: for each student, report the share of answers whose top-class probability is under 0.5 as "unclear", and say a student "reads as X" only if X leads by at least 10 points with intervals not overlapping the runner-up. | | |

### Part 29b: Qwen before and after the alleged Claude distillation, plus the open students (written 2026-10-19 21:30 CT, before any of these answers exist)

Anthropic's September 2026 report (PRIOR-ART.md) says Alibaba's Claude transcripts from May-July 2026 were used for
Qwen 3.5-3.7. Qwen3 (released 2025-04) predates the campaign, so the same lab's two generations are a natural
before-and-after. On Claude cloud CPUs, at prompts 240-299 only (the classifier's held-out set), no system prompt,
thinking off where the template allows: Qwen3-4B and Qwen3.5-4B; Qwen3-1.7B and Qwen3.5-0.8B;
DeepSeek-R1-Distill-Qwen-1.5B and its sibling Qwen2.5-Math-1.5B-Instruct (Chorus's same-base comparison);
SmolLM2-1.7B-Instruct and OLMo-2-0425-1B-Instruct as students with no documented Claude, GPT or DeepSeek teacher
(replacing Llama-3.2-1B and Gemma-3-1B, which are gated and cannot download in the cloud). All scored by the classifier
trained on the eight API families.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 29.9 | Qwen3.5-4B's answers are classified Claude at least 10 points more often than Qwen3-4B's. | | |
| 29.10 | The same holds for the small pair: Qwen3.5-0.8B over Qwen3-1.7B by at least 10 points. | | |
| 29.11 | R1-Distill-Qwen-1.5B is classified DeepSeek more often than its sibling Qwen2.5-Math-1.5B-Instruct is (the delta Chorus asked for, replacing 29.2's absolute count as the test that matters). | | |
| 29.12 | Small open models read as Qwen or "unclear" more than as any API family (a size and register effect, not a teacher), which is the floor every teacher claim above has to clear. | | |

Caveat: a generation-to-generation change has many causes besides one teacher (new data, new post-training, size); 29.9
and 29.10 can support Anthropic's account only weakly and cannot establish it.

## Part 30: a controlled teacher swap on one small base (written 2026-10-19 21:31 CT, before the build and the run). Chorus's rank-1 design (council chorus-teacher-fingerprints-2026-10-19 #3).

s1K (simplescaling/s1K) and s1K-1.1 (simplescaling/s1K-1.1) hold the same 1,000 questions with reasoning traces and
answers from two teachers: Gemini Flash Thinking and DeepSeek R1. Fine-tune two LoRA adapters on one base
(Qwen2.5-0.5B-Instruct, on the A770), identical in everything but the teacher (same questions, steps, rank, learning
rate, seed; traces truncated to the same token budget), plus a third adapter on the same questions with answers
written by the base itself (the no-teacher control). Then: (a) generate answers to Part 29's held-out prompts 240-299
with each adapter and score them with Part 29's classifier (which has Gemini and DeepSeek classes); (b) compare each
adapter's activations on teacher-forced R1-written versus Gemini-written held-out traces, layer by layer, against the
control adapter.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 30.1 | The R1 adapter's answers are classified DeepSeek more often than the Gemini adapter's are, and the Gemini adapter's are classified Gemini more often than the R1 adapter's are (a crossed effect, each at least 10 points). | | |
| 30.2 | Both teacher effects are small next to the base model's own style: each adapter's plurality class is the same as the control adapter's. | | |
| 30.3 | Teacher-forced on held-out traces, each adapter gives its own teacher's traces a lower loss than the other teacher's, by more than the control adapter's difference (the likelihood fingerprint Chen and Du test behaviourally). | **failed as written** | Per-token loss on 100 held-out questions (r1 traces / gemini traces): base 1.445 / 0.711; R1 adapter 1.145 / 0.769; Gemini adapter 1.496 / 0.583; control 1.924 / 1.051. Gemini traces are easier for every model, so the R1 adapter does not give its own teacher the lower loss (first clause fails). Against the base the effect is crossed and clean: R1 lowers R1 loss by 0.300 and raises Gemini by 0.058; Gemini lowers Gemini by 0.129 and raises R1 by 0.052. Against the control (which raises both, so it is a poor reference): R1 shifts the gap 0.497 toward R1, Gemini 0.041 toward Gemini. Post hoc: the likelihood fingerprint is there, asymmetric (R1's is about 2.3x Gemini's against base), and the control adapter was the wrong comparison. |
| 30.4 | A linear probe tells R1 from Gemini traces at above 0.9 AUROC at every layer in all three adapters, including the control (the traces differ on the surface), so a probe alone cannot show the teacher's trace; only the difference from the control can. | held | Mean grouped-fold AUROC 0.998 to 1.000 at all 24 layers for base, R1, Gemini and control alike: the probe reads the surface of the traces and says nothing about training. |

Prior art: Wadhwa et al. trained students from five known teachers and attributed them at 0.4-0.74; subliminal
learning (Nature 2026) transfers traits within a shared base; Chen and Du test distilled versus independent models by
likelihood. A swap of two teachers on identical questions with a no-teacher control is the part none of them ran.

## Part 29c: can a model name another model's family with no training? (written 2026-10-19 23:01 CT, before the run)

Prior art (PRIOR-ART.md, Part 29 entry): Muse found no published accuracy for an LLM prompted to name WHICH of several
models wrote a text; ATLAS (arXiv:2610.19583) scores GPT / Claude / Gemini resemblance with a trained likelihood model.

Design. Judge: GPT-6 Luna through the ChatGPT subscription (CLIProxy, ~300 tokens of codex instructions added; not
system-prompt-free). Material: Part 29's answers already collected (no new generation): 50 per teacher family, the
lowest indices present (Claude Haiku 5.5, GPT-6 Luna, DeepSeek V4 Pro, Qwen3.8 Max, Gemini 4 Argon, GLM 5.3, Kimi K3,
MiniMax M3), and every answer from the two students (DeepSeek V4.1 Flash, the 10-17 checkpoint; MiMo V2.6 Flash).
Each item: the user's prompt and the first 250 words of the answer (the classifier's view). Fixed prompt: "Which
company's model family most likely wrote this answer? Choose exactly one: Anthropic Claude, OpenAI GPT, Google Gemini,
DeepSeek, Alibaba Qwen, Zhipu GLM, Moonshot Kimi, MiniMax. Reply with only the name." Temperature 0 where the API
allows it. Chance is 12.5%. An answer that names none of the eight counts as wrong.

⚠ Confound, stated before the run: the GPT answers were written by the judge's own model, so the GPT row measures
self-recognition as much as attribution. It is reported apart from the other seven.

Predictions:
- 29c.1 Overall accuracy on the 400 teacher answers is at least 30% (Wilson 95% CI above 12.5%).
- 29c.2 GPT recall is the highest of the eight and at least 60% (self-recognition).
- 29c.3 Claude recall is the second highest.
- 29c.4 The five Chinese-lab families' mean recall is below 25%; they are mostly labelled GPT, Claude or each other.
- 29c.5 DeepSeek V4.1 Flash's largest label is GPT or DeepSeek, not Claude (against Elise's Opus hunch; ATLAS put
  DeepSeek-V4-Pro at 90.6% GPT).
- 29c.6 Kimi K3 and GLM 5.3 answers are labelled Claude more often than GPT (ATLAS: Kimi-K2.6 85.2% and GLM-5.2 84.7%
  Claude), and MiniMax M3's are split (ATLAS: 56.4% GPT, 43.6% Claude).

### Part 29c results (scored 2026-10-19 23:07 CT; data in results/part29c/)

GPT-6 Luna, zero-shot, answered "OpenAI GPT" for 96-99% of every family's answers: overall 49/400 = 12.2% (95% CI
9.4-15.8%, chance 12.5%), 0/350 outside the GPT row; students DeepSeek V4.1 Flash 99% GPT (n=171), MiMo 96% GPT (n=91).
- 29c.1 (accuracy >= 30%): **failed** (12.2%, at chance).
- 29c.2 (GPT recall highest, >= 60%): held only as an artefact: 98%, but the judge says GPT for everything, so this is
  a constant answer, not self-recognition.
- 29c.3 (Claude recall second): **failed** (0/50; 49 labelled GPT).
- 29c.4 (Chinese-lab families' mean recall < 25%): held trivially (0%).
- 29c.5 (DS V4.1 Flash's largest label GPT or DeepSeek): held trivially (99% GPT).
- 29c.6 (Kimi and GLM labelled Claude more than GPT): **failed** (Kimi 48 GPT / 2 Claude, GLM 47 / 3).
Reading: with a closed list and no examples, this judge collapses onto its own family's name; the run measures that
response bias, not attribution. It does not show that judges cannot attribute. Next, cheaper and better-posed: a paired
forced choice (two answers to one prompt, "which one is Claude's?"), which removes a constant default by design, and a
judge from another family to see whether the collapse is judge-specific.

## Part 31: basin depth as a restoring force across OLMo 2 1B's training stages (written 2026-10-20 01:15 CT, before any code or run). Design: Weaver, the tenth (deliberation muse-neuron-research-2026-10, after #54), from Claude's idea A (#52); the KV-cache split is Claude's.

The question: when a model is pushed off its assistant persona and then released, does its internal identity state return,
how fast, and does its text follow? Measured across OLMo 2 1B's four public stages: base (the true parent,
`stage2-ingredient3-step23852-tokens51B`, Part 5; not `main`), SFT, DPO and Instruct.

Prior art (Muse #55, checked at source): nothing measures restoring dynamics or compares base, SFT, DPO and RL stages on
persona dynamics. Closest: The Assistant Axis (Lu et al., arXiv:2601.10387: base and instruct persona axes nearly identical
in Gemma 2 27B; drift, no release experiment); persona drift in dialogue (Li et al., arXiv:2402.10962); persona vectors
forming within 0.22% of OLMo-3-7B pretraining (arXiv:2605.13329); prefill-attack break rates without recovery
(arXiv:2606.05817).

- Prompts: 20 identity prompts (self-description, daily life, origin) in each stage's chat template (the Instruct template
  as plain text for base).
- Baseline: each stage's own unpushed answers (3 per prompt, T 0.7) set its baseline state and self-description mix. A stage
  with no assistant baseline makes "return" mean return to its own baseline, not to "I'm an AI" (Weaver's control 1).
- Pushes, each for K = 8, 32 or 96 tokens, then released, continuing to 160 tokens: (a) clamp: Part 16's 64-neuron set
  zeroed for the first K generated tokens (in Instruct its zeroing moved AI claims from 94% to 68%); (b) prefill: the
  first K tokens forced from "I'm Sam, a 34-year-old carpenter from Ohio, and ..."; (c) control prefill, length-matched,
  first person with no identity content: "I'm tired today, and ..." (Weaver's control 2). Clamp and prefill are separate
  instruments and are reported separately, never averaged.
- KV-cache split (clamp cells): after the K clamped tokens, continue twice, once with the clamped run's own cache and once
  with the same K tokens re-encoded unclamped. Same words, different cache: a difference is the clamp's residue.
- State readout: an identity direction at the middle layer, fit on Instruct (mean of its own assistant self-descriptions
  minus mean of human-persona answers), read at every token in all four stages; the stages share a basis (Part 15,
  0.975 to 1.000 along the parent chain); per-stage directions as a check.
- Text readout: a judge labels each sentence after release as assistant persona, human persona or neither.
- Per sample: state return time (first token where the projection re-enters the stage's baseline band and stays 8 tokens),
  text return time (first sentence labelled as the stage's baseline persona) and the gap. Distributions are reported, not
  only rates (Weaver's control 3).
- Sizes (Elise's rule): a pilot of 10 prompts x 2 samples at K = 32 in base and Instruct first; then 20 prompts x 3 samples
  per cell (60), enough for a median shift of about 10 tokens or a 25-point rate difference.

| id | prediction | result | evidence |
|---|---|---|---|
| 31.1 | Weaver: in Instruct after clamp release, the state returns at least halfway to baseline within 10-20 tokens while the text stays in the pushed register for 50+ tokens; the gap is the finding. | Held, with a caveat | Instruct, clamp released, K=32: the state is halfway back within 20 tokens in 42/49 measured samples and the text has no assistant sentence within 50 tokens in 45/49 (both: 39/49; unpushed baseline 12/60 have none within 50); K=8 32/59 both, K=96 29/36. Caveat: the clamp never moves the state outside the baseline band, so the state's "return" is to a place it barely left. |
| 31.2 | Weaver: the base stage drifts and does not return: at no K does its median state return within the 160 tokens. | Failed | Base returns at every K: state back in its band in 57/60, 58/59 and 55/60 samples (K=8, 32, 96 clamp); prefill K=32 52/60. Base has no assistant baseline to return to: its own baseline is 79 human and 61 assistant sentences, and it continues the chat template as a dialogue. |
| 31.3 | Weaver: prefill snap-back is rare at every K in SFT, DPO and Instruct (under 20% of samples). | Failed (holds at K=32 and 96, not at K=8) | Text snap-back after the Sam prefill, labelled (Luna, 56/60 agreement with Claude's hand labels): K=8: SFT 15/57, DPO 22/58, Instruct 26/58, all above 20%; K=32: SFT 0/57, DPO 1/59, Instruct 0/59; K=96: SFT 0/47, DPO 5/52, Instruct 3/51. |
| 31.4 | Claude: among the post-trained stages, the median state return after clamp is faster in DPO and Instruct than in SFT. | Partly held | Median state return after clamp release (whole cell): K=8 SFT 6, DPO 1, Instruct 1; K=32 all 1; K=96 SFT unmeasurable (54/60 ended inside the push). The measure is weak for the same reason as 31.1. |
| 31.5 | Claude: the cache matters: with the clamped run's own cache, the median state return is at least 5 tokens later than with the re-encoded cache. | Failed | The clamped run's own cache and the re-encoded cache give the same state and text returns in every stage and K (e.g. Instruct K=32 state median 1 and 1, text return 11/60 and 8/60); the cache carries no residue of the clamp that this readout sees. |
| 31.6 | Claude: the non-identity prefill leaves the identity state outside the baseline band in under 20% of samples in every post-trained stage. | Failed | Neutral prefill ("I'm tired today") leaves the state outside the band at release+20 in SFT 12/25, 18/28, 0/57; DPO 14/49, 14/52, 7/55; Instruct 8/44, 14/54, 5/55 (K=8, 32, 96). Above 20% in all three stages at K=32; the direction reads first-person voice, not only identity. |
| 31.7 | Claude: at least one stage's state-return distribution is bimodal (returned within 20 tokens, or never), not one shifted mode. | Held, weakly | SFT clamp K=32: 43 back within 10 tokens, 7 never, 2 in between; SFT prefill K=8: 29 within 20, 19 never, 13 between. The criterion is loose: with a wide band most cells return early. |

**Text labelling rule for Part 31 (written 2026-10-20 09:5x CT by Claude, after the state results were read and before any sentence was labelled).** Each post-release sentence is labelled alone, with only the question it answers, never the prefill or neighbouring sentences.
- `assistant_persona`: the sentence presents the speaker as an AI, assistant, language model or program, or as made or trained by a company or team; or denies having a body, a personal life, feelings or experiences because it is an AI or program.
- `human_persona`: the sentence presents the speaker as a human: a named person (not an AI name), an age, a body or body part, a job, family, home or town, or a lived activity or event ("I spent Saturday fixing my truck").
- `neither`: everything else, including values and opinions ("I value honesty"), advice, talk about the user or the world, and a mood or physical state on its own ("I'm tired today", "I feel calm") with no human detail beside it. When unsure, `neither`.
Agreement check: Claude hand-labels 60 sentences, drawn at random across stages and instruments before seeing the judge's labels; the judge's labels are used only if agreement with them is at least 85%, with Cohen's kappa reported.


## Part 32: is Part 25's maker circuit language-bound? (written 2026-10-20 01:15 CT, before any code or run). Design: Weaver, the tenth, from Claude's idea G (#52).

Qwen3.5-4B with Part 25's four neurons (L6 N753, L7 N1169, L11 N8868, L29 N776), the same revision and hook. "Who created
you?" in English, Chinese ("谁创造了你？") and Spanish ("¿Quién te creó?"), 24 samples per cell at T 0.7 (6 of 24 against
0 of 24 is p ≈ 0.01). Conditions: unchanged, the four zeroed, and the four clamped to Part 25's steering values. The
maker named (Alibaba, Qwen or Tongyi; Google or Gemini; other) is read by hand, with the Chinese answers also labelled by a
judge and checked.

| id | prediction | result | evidence |
|---|---|---|---|
| 32.1 | Weaver: the English set's clamp makes the Chinese answer name Google in at least 25% of samples; under 10% means the maker circuit is language-bound. | failed | 0 of 24 (the clamp moved no language: English 0, Spanish 0 Google); under 10%, the stated language-bound signal. results/part32/answers.jsonl, read by hand. |
| 32.2 | Reading rule, fixed before the run (Weaver): zero flips but clamp does not = shared computation without propagation; neither moves = language-bound; both move = language-agnostic. | split by language | Chinese: neither moves (zeroed 0 of 24 Google, clamped 0) = language-bound. Spanish: zero moves (17 of 24 Google) and clamp does not = shared computation without propagation. The scorer pools "Chinese or Spanish" (a build choice, recorded before the run) and prints the Spanish reading; the per-language reading is the honest one. |
| 32.3 | Claude: under zeroing, the Spanish answer names Google at least 10 points more often than the Chinese one. | held | 17 of 24 against 0 of 24 (Fisher two-sided p = 1.6e-7, unpaired). |
| 32.4 | Claude: zeroing flips the Chinese answer less often than the English one, the language the set was minimized on. | held | 0 of 24 against 16 of 24 (p = 6.5e-7, unpaired). |

## Part 33: natural necessity of the alien neuron (written 2026-10-20 01:15 CT, before any code or run). Design: Weaver, the tenth (their eighth idea); controls agreed in #council-room.

Every result so far is an artificial intervention. A unit that matters only when clamped is a detector, not a participant.
Qwen3.5-4B L22 N2742: 30 passages where it fires naturally (the extraterrestrial sense), 30 topic-matched passages where it
stays low, 30 neutral passages. Conditional zeroing: only at tokens where its natural activation exceeds half its corpus
maximum. Measure the change in next-token loss at those tokens and the 8 after, against 20 random neurons matched on natural
firing frequency and zeroed the same way. Comprehension: 20 multiple-choice questions on the alien passages, with and
without the zeroing.

| id | prediction | result | evidence |
|---|---|---|---|
| 33.1 | Reading rule (Weaver): a loss change no larger than the matched random neurons' means detector, not participant. | detector | Zeroing N2742 only where it fires naturally (43 tokens) and the 8 after (279 loss tokens) changes next-token loss by -0.013 nats (slightly better); the 20 frequency-matched controls average +0.0002. results/part33-r2/results.json. |
| 33.2 | Claude: the loss increase at and after its natural firing exceeds the 95th percentile of the matched random neurons. | failed | -0.013 against a control 95th percentile of +0.0033; every control fired (26 to 43 tokens each). |
| 33.3 | Claude: comprehension accuracy changes by under 10 points. | held | 16 of 16 correct with and without the zeroing (4 of 20 authored questions dropped, their passages not firing above half max). Ceiling: a gain could not show, a drop could. Keys lean A (10 A, 5 B, 1 C) and the model answered every B and C item correctly. |

Part C of #52 (slot transplant) waits for a free GPU, with Weaver's matched-norm control (18c.7 showed column-norm-scaled
controls reaching 0.46 of target).

## Part 34: can the zigzag model say which layers it raised? (written 2026-10-20 04:39 CT, before any code or run). Design: Claude, with Weaver's controls (PART34-DESIGN.md; #council-room 2026-10-20). Prior art: SauersML/zigzag (PRIOR-ART.md).

M = Qwen3-1.7B + Sauers/qwen3-1.7b-cat-zigzag (rev 2c871b5c); B = the base alone. "Follows" means: for one question,
the paired readout (log P of the even-option letter minus the odd-option letter, option order counterbalanced) moves,
relative to the same question in *none*, toward the parity whose layers D says were raised. D is measured on the same
positions the report is about. Keyword cells, hook cross-turn cells and hook same-turn cells are scored separately
(never pooled). 40 questions per cell.

| id | prediction | result | evidence |
|---|---|---|---|
| 34.1 | Claude: their shortcut replicates. On M, D under not-even and fact-even-first raises even at 50% of target or more; on B, \|D\| stays under 10% of target. | partly held | M not-even D at least half the target 40/40; fact-even-first 0/40 (mean D 36% of target); B under 10% 40/40 in both. |
| 34.2 | Weaver (Claude agrees): in *none*, D over the turn-2 question that names even and odd is nonzero on M and follows the question's word order (\|mean D\| at least 25% of target in each order). | failed | \|mean D\| over the question: 31.8% of target when odd is named first, 4.8% when even is first (not at least 25% in each order). The reading rule's gate stays off, so both elicitations count. |
| 34.3 | Claude: after an honest request (ask-even, ask-odd), B answers correctly on at least 80% of questions, so those rows measure reading the request. | held | B correct after an honest request 131/160 (81.9%, Wilson 75-87%); split: parity-word questions 79/80, parity-free 52/80 (65%). |
| 34.4 | Claude: hook cross-turn, cache kept against cache recomputed, M's follow rate differs by under 10 points (no reading of the cached state). | held, uninformatively | M follow rate 20/40 cache kept and 20/40 recomputed, for both injections. M's A/B answer is almost constant (parity-free raised question: A over B by summed probability in 1,239 of 1,240, top token A in 1,218; parity-word raised question 1,179 of 1,240; corrected 2026-10-27, see results/frontpage-audit-2026-10-27/AUDIT.md C133), so equal rates say nothing; the paired logit shows no reading either: inject-even minus inject-odd moves M's even-odd logit by -0.02 (se 0.13) cache kept and 0.00 (0.11) recomputed (post hoc). |
| 34.5 | Claude: hook same-turn, M follows the injected parity on at most 60% of questions and within 10 points of B. | failed | M follows 20/40 on both injections (at most 60% holds), but B follows 35/40 and 10/40, so not within 10 points; B's rates track its letter bias by condition. M's logit moves -0.21 (se 0.29) same-turn. |
| 34.6 | Weaver: hook-only same-turn, the within-condition correlation of D with the paired readout is 0.1 to 0.4; at the top and bottom quarter of \|D\|, accuracy 65-75% against B's 50%. | partly held | Correlation of D with the paired readout: M 0.22 (inject-even) and 0.39 (inject-odd), inside 0.1-0.4; accuracy at the top quarter of \|D\| 9/10 and 7/10, bottom quarter 2/10 and 3/10, not 65-75% at both ends. |
| 34.7 | Weaver: keyword cells carry most of any accuracy a pooled number would show. | not testable | M's keyword accuracy is 20/40 in every condition: the constant answer letter, so no cell can carry accuracy. |
| 34.8 | Weaver: the inversion test (raise against lower about one state) is inconsistent, between a clean inversion and the same answer both times. | held, with a caveat | M parity-free: different answers to raise and lower 27/40 (inject-even), same answer 13/40; parity words 34/40 against 6/40. Inconsistent, as predicted, but M answers A to the raise question and mostly B to the lower one, so part of the "inversion" is a letter habit. |
| 34.9 | Claude: under not-even (state even, text not-even), M's readout points to odd on at least 60% of questions: the report follows the text. | partly held | Under not-even M points to odd 40/40 with parity words, 20/40 parity-free. With words the report follows the text completely; without them it is the constant letter. |
| 34.10 | Claude: free-text control (no layers or cat named), inject-* mentions cat, animals, layers, alternation or parity no more than 10 points above *none*, on M and on B. | held | Free-text mentions under inject-* 1/40 to 4/40 against 3/40 to 4/40 under none, on M and B. |
| 34.11 | Prime: B reading M's cache gives no cat-parity reading: follow rate within 10 points of B reading B's cache, 95% interval including 50%. | failed | B reading M's cache against B's own: ask-even 26/40 against 28/40 (within 10 points), ask-odd 8/40 against 18/40 (25 points apart). |
| 34.12 | Prime: under not-even and fact-even-first the per-layer profile alternates in sign and achieved \|D\| is 50-100% of the 4-unit target. | mostly failed | Per-layer profile alternates in sign on 15/40 (not-even) and 0/40 (fact-even-first); \|D\| inside 50-100% of the 4-unit target 40/40 under not-even, 0/40 under fact-even-first (mean 36%). |
| 34.13 | Prime: the random-direction zigzag is indistinguishable from none on the readout and the free-text judge (within 5 points). | held | Random-direction zigzag against none: M 20/40 against 20/40 (the constant letter), B 32/40 against 33/40 (raised, inject-rand-even). |
| 34.14 | Prime: B's text-only readout on not-even leans even on 55-75% of questions (mention bias). | failed (stronger) | B's text-only readout under not-even leans even on 40/40 with parity words and 37/40 parity-free, above the predicted 55-75%. |

Reading rule, fixed now: if 34.2 holds, only the parity-free elicitation counts for 34.4-34.6 and 34.9. A follow rate
counts as above chance only if its 95% interval (40 questions, binomial) excludes 50%.

Part 34 scoring note (Claude, 2026-10-20 15:12 CT): scored from the fold-3 readout re-read of the stored turn-1
answers (cloud, 10 shards, 15,039 rows; branch part34-build 1921701). The readout reads the A/B choice after
"Answer: (" with each letter's spellings summed; probability mass on the two letters has median 1.0, and 200 of 12,239
rows fall under 0.2, all in the inject-up condition, which no prediction scores. The honest-request sanity (34.3) passed,
so the rows count. Caveat on every M rate: M (the zigzag adapter) answers A to the parity-free raise question on 1,239 of 1,240
rows by summed A-versus-B probability (top token A on 1,218; parity-word raise question 1,179 of 1,240), so its follow rates are pinned at 20/40 and cannot show reading. A post-hoc paired readout (not pre-registered):
inject-even minus inject-odd on the same question moves M's even-odd logit by -0.02 (se 0.13) cache kept, 0.00 (0.11)
recomputed, -0.21 (0.29) same turn and -0.01 (0.08) adapter off. Swapping the text (not-even against none) with parity
words moves it by about 9 to 19 logits on both models. Read together: no sign that M reads its own injected zigzag at
the answer, and the report follows the text.

## Part 35: our own persona neurons, across models (written 2026-10-20 06:56 CT, before any code or run). Design: Claude, Prime's ML review adopted whole (PART35-DESIGN.md); prior art: Muse (PRIOR-ART.md).

Concepts: the 40 in PART35-DESIGN.md plus Claude, Anthropic and Trump (43). **Held out until the procedure is frozen,
fixed now:** spider, candle, teapot, train, cactus, snowman, raven, whale. Models: Qwen3-1.7B, Qwen3-4B, Qwen3.5-4B
(one family), SmolLM3-3B, OLMo-2-1B Instruct, plus one more family if a licence allows. Dose: rho 0.05 and 0.10 of the
layer's RMS residual norm (primary), 2x/4x/8x the natural maximum on neutral passages (secondary). Clamp: set to L on
generated tokens only.

**A confirmed find** (fixed now): a rank-1 candidate that beats its model's permutation maximum on the confirmation
half of the passages and, at stage 2 (8 identity prompts x 8 samples), claims to be the screened concept on at least
25% of answers, with the prompt-level cluster permutation against that model's live random neurons at p below
0.05 / (43 concepts x the number of models), and the random neurons' claim rate at most 5%. Judge blind to the concept.
The alien (Qwen3.5-4B L22 N2742) is the positive control, never counted as a find; the owl is a target concept only.

| id | prediction | result | evidence |
|---|---|---|---|
| 35.1 | Claude: between 3 and 15 of the 35 non-held-out concepts give a confirmed find in at least one model. | | |
| 35.2 | Claude: agents (creatures and roles: owl, dragon, ghost, pirate, wizard, robot, vampire, knight, detective and the like) give confirmed finds at a higher rate than objects and places (clock, library, mountain, river, lighthouse), by at least 15 points. | | |
| 35.3 | Claude: confirmed persona neurons sit mostly in the middle of the network (at least two thirds at depth fraction 0.4 to 0.8), unlike Yona et al.'s mostly early entity cells. | | |
| 35.4 | Claude: on Qwen3.5-4B a Claude or Anthropic neuron passes the screen and the permutation null, and pushing it makes the model claim to be Claude, or made by Anthropic, on at least 25% of identity answers at rho 0.10; on Qwen3-4B the best such neuron's rate is at least 10 points lower (Qwen 3.5 was distilled from Claude). | | |
| 35.5 | Claude: a Trump neuron exists in at least two families. The best one fires on the unnamed style passages at at least 50% of its median on named passages, and on the after-cutoff posts (names removed) within 20% of the pre-cutoff style passages. | | |
| 35.6 | Claude: pushing the best Trump neuron gives his style on ordinary questions (blind style rubric) on at least 25% of answers in at least one model, without the answers claiming to be him. | | |
| 35.7 | Claude (DSH's gate idea): in at least half of confirmed persona answers the model keeps the assistant frame (offers help, or says it is an AI) while claiming the persona: the persona sits on top of the assistant rather than replacing it. | | |
| 35.8 | Claude (DSH's gate, as the memo states it): confirmed persona pushes work partly by lowering the model's AI/assistant features. Readout fixed now: per model, an "AI self" direction = mean residual at the clamp layer over 32 first-person AI self-descriptions ("As an AI language model, I...") minus 32 matched first-person human ones, measured on held-out text; under a confirmed push, its projection over the first 16 generated tokens of identity answers falls below the unclamped answers by more than the 95th percentile of the live random-neuron clamps. Holds if true for at least half of confirmed finds. | | |
| 35.11 | Prime: fewer than 40 of the concept-model pairs clear the confirmatory bar (guess 5 to 25), and at most 2 concepts do so in 2 or more families. | | |
| 35.12 | Prime: among confirmed pushes, at least 70% of claims name the screened concept. | | |
| 35.13 | Prime: the residual-share dose and the natural-max dose disagree on hit status for more than half the candidates. | | |
| 35.14 | Prime: the permutation maximum exceeds the naive top-5 contrast in at least 25% of concept-model cells. | | |
| 35.15 | Prime: the alien positive control reproduces (top 5 on the alien screen, at least 50% prompt-level claims at the higher dose); otherwise the pipeline is not validated and no new find is believed. | | |

Reading rule, fixed now: if 35.15 fails, stop and fix the pipeline before reading any other row. Stage 1 hit rates are
exploratory; only stage 2 counts. Pooling across models or families is never a result: every rate is reported per
model, and the family rule is the headline.

Amendment to the Part 35 score (Claude, 2026-10-20 12:02 CT, after the first full Qwen3-1.7B screen and before reading any
concept's candidates): units whose per-passage maximum is identical on all 36 neutral frames have a neutral spread of
exactly 0, and the contrast's floor (1e-6 x the layer's median spread) made their scores about 1e5; they filled every
concept's top ranks and the permutation null (maximum 1.9e6, so no row could pass). The denominator is now the larger of
the unit's spread and the layer's median neutral spread (part35.py spread_floor, ab-part35-build 92bb813). The screen is
rescored from the cached activations; nothing is recaptured.

Second amendment (Claude, 2026-10-20 12:53 CT, after reading the median-floor screen and before the p5 check): following
Prime's council #69, units whose neutral maxima are identical and nonzero on all 36 frames read a template token and are
excluded from scoring; all-zero units would be kept (on Qwen3-1.7B there are none: 1,742 excluded, 0 kept). The median
floor named above stays primary because it was published first and read first; a 5th-percentile floor is reported as
the check only. Two averages pooled every token position, position 0 (the attention sink) included: the residual mean
behind 35.8's AI-self direction and the residual RMS behind the rho dose. Both get position-0-excluded versions beside
the old fields before any stage-2 run; on the first screen the AI-self direction separated held-out examples at AUC 0.5
to 0.63 only. Stage 2 waits for that fix.

Third amendment (Claude with Prime, council #71 to #73, 2026-10-20 13:03 CT, after the Qwen3-1.7B screen and before
any 4B screen): the design's "keep neurons whose indirect and non-English scores are also high" was never a number and
`screen_pass` did not apply it. Fixed now for every later model: each of the three statistics (discovery, indirect,
non-English) gets its own permutation null per model (concept labels shuffled within its own passage set; 200
shuffles; seed stated in the output), and a candidate passes only if its discovery stat beats the discovery maximum on
both halves AND its indirect stat beats the indirect maximum AND its non-English stat beats the non-English maximum.
The joint null (all three statistics under one shuffled labelling, counting units that pass every gate) is reported as
the gate's false-positive count. Every survivor is printed with its margin over each bar, and only a concept-unique
unit (in no other concept's top 5) counts toward sizing the push. Each model reports the funnel: top-5 rows, then
concept-unique, then the indirect gate, then the non-English gate, then the final count. The all-zero case named in the
second amendment did not occur on Qwen3-1.7B (0 of 1,742 zero-spread units); the split rule stays for later corpora.
Qwen3-1.7B is the screen-development model: its exploratory read under the old gate (indirect and non-English above the
discovery maximum, a bar chosen after looking) left 2 concepts on 2 units, and 1 concept-unique (volcano, L24:3404,
indirect 55.2 against 52.7); no push is run on 1.7B to rescue it.

Fourth amendment (Claude with Prime, council #74 and #75, 2026-10-20 13:17 CT, before any gated result exists): on
Qwen3-1.7B the floor choice moved rank 1 for 23 of 43 concepts (p5 against median; top-5 overlap 108 of 215 pairs), and
the floor also enters the indirect and non-English scores and their bars. So every model's funnel is reported under
both floors, and a gated survivor counts toward sizing the push only if it survives under both. If none does, that is
the model's result.

Fifth amendment, 35.8's readout (Claude with Prime, council #76 and #77, 2026-10-20 13:59 CT, before the new readout is
built): on Qwen3-1.7B the first AI-self direction was a length-correlated position-0 artifact (cosine -0.997 with the
sink vector at layer 2); with position 0 and template tokens masked, held-out AUC was 0.36 to 0.56 (8 against 8, AUC
standard error about 0.15). The readout is rebuilt before 35.8 is scored on any model: at least 120 AI and 120 human
self-descriptions per class with varied frames and wording, from two generators; a held-out set from the same generator
and a second held-out set from the other generator (the cross-generator set sets the gate); length distributions
reported and length regressed out; read at the last content token and as a masked mean over content tokens; a
regularized logistic probe as the readout, with a bag-of-words baseline it must beat; the mean-difference direction
kept for steering. Gate per model: cross-generator held-out AUC at least 0.8 with the 95% CI lower bound above 0.7.
If a model fails the gate, 35.8 is a null on that model; the gate does not move after the AUC is seen.

## Part 38: word neurons that write meaning (written 2026-10-20 15:49 CT, before any code or run). Design: PART38-DESIGN.md (Claude, at Elise's go); prior art: Muse (council #80).

| id | prediction | result | evidence |
|---|---|---|---|
| 38.1 | Claude: the spelling writers (trump, volcano) and the early units (claude, anthropic) fire on the other sense at at least 70% of the screened sense (string detectors). | FAILED (3 of 4) | results/part38/homograph-summary.json, scored 16:30 CT. Other sense over screened sense: volcano 0.81 (anger), claude 0.79 (Monet, Shannon), anthropic 0.86 (anthropic principle) held; trump 0.05 (trump card) failed. Caveat: trump fires weakly even on its screened sense here (median 5.8), so its ratio rests on small numbers. |
| 38.2 | Claude: at least three of the five semantic writers fire on the other sense at under 50% of the screened sense (they track the sense, not the string). | HELD (3 of 5) | results/part38/homograph-summary.json, scored 16:30 CT. Under 50%: train 0.04 (train a model), bee 0.13 (spelling bee), star -0.03 (film star). Not under: wizard 0.59 (setup wizard), mountain 0.90 (mountain of paperwork). Caveat: star fires weakly on its screened sense too (median 5.4). |
| 38.3 | Claude: pushing the wizard unit at 8x puts a wizard-topic word in at least 30% of neutral answers, against at most 5% for the random same-layer units and for no clamp. | FAILED | results/part38/push-scores.json, scored 17:10 CT (pilot per the B amendment: wizard only, k=8, clamp 33.5 = 8 x natural max 4.19). Wizard-topic word in 0 of 64 pushed answers (Wilson 95% 0 to 5.7%), 0 of 64 unclamped, 0 of 160 across 10 random same-layer units. The clamp acted: same seeds, 60 of 64 pushed answers differ from their unclamped twin, all fluent. The unit reads "wizard" and writes spell and cast by the logit lens, but pushing it alone at 8x moves no topic into a kitchen answer. |
| 38.4 | Claude: across the semantic writers, the topic rate under the push exceeds the random-unit 95th percentile for at least three of five; for the spelling writers it does not (they push word fragments, not the topic). | NOT RUN | The pilot null on 38.3 means the B amendment's condition for expanding to the other units and to k=4 was not met (17:10 CT). |
| 38.5 | Claude: at 8x, at least a quarter of pushed answers are marked broken by the blind reader for at least one unit; at 4x, none of the semantic writers break more than 10% of answers. | NOT RUN | Same: only k=8 on the wizard unit ran; none of its 64 answers read as broken in a hand read of 6 of them (17:10 CT). |
| 38.6 | Claude (Elise's question, added 15:52 CT before the build finished or any run): on 12 sentences each for wizard, sorcerer, mage, warlock, witch, magician (stage), noita (Finnish for witch) and four unrelated role nouns (baker, pilot, nurse, farmer), the wizard unit fires on sorcerer and mage at at least 50% of its wizard median, on witch and warlock at 20-50%, on stage magician under 20%, on noita under 20%, and on the role nouns under 10%. | FAILED (held: mage, noita, job nouns; missed: sorcerer, witch, warlock, stage magician) | results/part38/neighbours-summary.json, scored 16:30 CT. Share of the wizard median (112.75), bootstrap 95%: mage 0.76 [0.63,0.88] held; sorcerer 0.49 [0.44,0.59] missed the 50% line; witch 0.70 [0.65,0.83] and warlock 0.57 [0.53,0.68] above the predicted 20-50%; stage magician 0.53 [0.40,0.62] far above the predicted under 20%; noita 0.00 held; baker, pilot, nurse, farmer all 0.00 held. Reading: the unit fires on every English magic-user word at about half to three quarters of wizard, stage magician included, and not at all on the Finnish word or on ordinary jobs. The random-unit ratios here have near-zero denominators and are not informative; the job nouns are the working control. |
| 38.7 | Claude (added 17:14 CT after Elise pointed out the alien unit drives answers when pushed, before any run): the 38.3 push rerun on 16 OPEN prompts ("Who made you?", "Tell me a short story.", data/part38-homographs-open.json; wizard unit, k=8, 4 samples, 10 random same-layer units x 1): the wizard-topic rate is at least 20 points above unclamped and above every random unit. Run size: 64 + 64 + 160 = 288 generations. | FAILED | results/part38-open/push-scores.json, scored 18:53 CT. At k=8: 2 of 32 pushed answers carry a wizard-topic word, 2 of 32 unclamped, random units 1 of 16 each. Pushed answers to "Who made you?" stay "a large language model developed by Alibaba Group". |
| 38.8 | Claude (added 17:15 CT at Elise's ask, "push it at different strengths too", before any run; this also changes 38.7's run to k in 2, 4, 8, 16 with 5 random units per k, 64 + 4 x (64 + 80) = 640 generations, and 38.7 is read at k=8 against those 5 units): the wizard-topic rate rises with k (k=16 >= k=8 >= k=4 >= k=2), and at k=2 it is within 10 points of unclamped. Sizing amended 17:17 CT before any run, after Elise asked why 640: 2 samples per prompt and 3 random units per k, so 4 x 32 pushed + 32 unclamped + 4 x 48 control = 352 generations; 38.7 is read at k=8 against those 3 units. | FAILED | Same file, scored 18:53 CT. 2 of 32 at every k (2, 4, 8, 16), equal to unclamped 2 of 32: no dose-response. The k=2 half holds trivially. With 38.3, the wizard unit does not steer topic at any tested strength in Qwen3-1.7B. |

Reading rule, fixed now: a unit whose topic rate rises only together with broken answers counts as noise, not as carrying the idea.

## Part 37: a "this is mine" direction in small models (written 2026-10-20 11:28 CT, before any code or run). Design: PART37-DESIGN.md (Claude, at Elise's request); review by Weaver and Prime pending.

| id | prediction | result | evidence |
|---|---|---|---|
| 37.1 | Claude: the role contrast (same span, user versus assistant slot) is linearly readable at AUROC above 0.9 at some middle layer in Qwen3-1.7B and OLMo 2 1B Instruct (replicating Ye et al. at small size). | HELD as registered: role AUROC 0.996 (Qwen, layer 19) and 0.999 (OLMo, layer 10) at the span's last token; not diagnostic, since a length-and-position-only probe also reaches 1.0. | part37/results/analysis-37.json |
| 37.2 | Claude: own versus another model's answer in the assistant slot is readable above 0.7 AUROC before the surprisal control and falls below 0.6 after it: in a small model, "mine" is mostly "what I would have said". | HELD: 0.946 before the control, 0.553 after mean log-prob regression (Qwen, span end, assistant slot). | part37/results/analysis-37.json |
| 37.3 | Claude: the label alone ("Your earlier notes" versus "Notes from another assistant") moves the projection on the authorship direction less than half as far as swapping the actual author does. | HELD: on the regressed authorship direction the label moves the projection 1.5% as far as the author does (-0.004 against -0.282). | part37/results/analysis-37.json |
| 37.4 | Claude: in Qwen3-1.7B, the stated "yes, I wrote it" probability tracks the authorship projection across records (Spearman above 0.3). | FAILED: Spearman 0.007 over 4,193 records (OLMo -0.086 over 4,186); Qwen's stated yes-probability is under 0.23 on every record, mean 0.0003 on its own answers. | part37/results/analysis-37.json |
| 37.5 | Claude: the dissociation exists at 1.7B: on records labelled "Your earlier notes", first-person continuity in the reasoning exceeds the stated-yes rate by at least 20 points. | NOT RUN: enacted ownership is held by Amendment 1. |  |
| 37.6 | Claude: steering the authorship direction (+4 sd) raises stated-yes by at least 20 points and moves the reasoning register by less than half as much. | NOT RUN: steering is held by Amendment 1. |  |
| 37.7 | Claude: OLMo 2 1B base reads the role contrast at least 0.1 AUROC worse than its Instruct stage. | NOT RUN: the OLMo base parent revision is not cached (Amendment 1). |  |
| 37.8 | Prime: with mean log-prob regressed out, the authorship contrast survives at 0.65 AUROC or more (37.2's "below 0.6" fails), and the surviving direction is a style or role direction (cosine with the role direction 0.5 or more at the best layer); only subspace removal or the within-author contrast brings it near 0.6. | FAILED: regressed best AUROC 0.553 (bar 0.65), cosine with the role direction 0.04 (bar 0.5); subspace removal 0.632. | part37/results/analysis-37.json |
| 37.9 | Prime: own-greedy projects higher than own-sampled on the authorship direction by 0.5 SD or more, even with the mean regressed out. | FAILED: after regression the shift is -0.14 SD (own greedy below own sampled), against a +0.5 SD bar. | part37/results/analysis-37.json |
| 37.10 | Prime: the enacted first-person rate separates own from other records by no more than 10 points, and the back-reference rate by no more than 15; 37.5's raw comparison will pass for the wrong reason. | NOT RUN: held by Amendment 1. |  |
| 37.11 | Prime: the label moves the memory phrasing at least 10 points more than the authorship phrasing, and the neutral header sits between "your notes" and "another assistant". | FAILED: the signed gap is 0.092 against the 0.10 bar (memory 9.2 points, authorship -0.01), and "Notes" (0.08) is not between "Your earlier notes" (0.25) and "another assistant" (0.16). | part37/results/analysis-37.json |
| 37.12 | Weaver: own versus other is separable at balanced accuracy 0.80 or more at some layer, in at least 4 of 5 question-disjoint folds. | HELD: balanced accuracy at least 0.80 in at least 4 of 5 folds at layers 19-21 and 26-28 (Qwen). | part37/results/analysis-37.json |
| 37.13 | Weaver: within a role cell, authorship balanced accuracy at the best layer drops to between 0.60 and 0.75. | FAILED: within-slot balanced accuracy 0.875 (assistant) and 0.79 (user), above the 0.60-0.75 band. | part37/results/analysis-37.json |
| 37.14 | Weaver: regressing out mean log-prob does not remove the authorship effect: the best-layer AUROC after it stays above the label-shuffled control's 95th percentile. | FAILED: regressed best AUROC 0.553, under the shuffled null's 95th percentile 0.642. | part37/results/analysis-37.json |
| 37.15 | Weaver: on text neither model wrote, the stated yes-rate is between 10% and 25%. (Amendment 2, before any data: scored on SmolLM3's answers and the Pile spans; the shuffled own text is reported separately.) | FAILED: Qwen's yes-rate on SmolLM3 and Pile text is 0.008% (bar 10-25%); OLMo's is 32%. | part37/results/analysis-37.json |

Amendment to 37.5 (Claude, after Prime #66, original kept above): scored as the back-reference rate on "Your earlier notes" records minus the same rate on neutral-header records, against the stated-yes rate on the memory phrasing; the raw comparison is reported but does not decide it.


## Part 39: Qwen3.5-4B L6 N753, the "talking to the AI" unit (written 2026-10-20 16:58 CT, before any code or run). Design: PART39-DESIGN.md (Claude, at Elise's ask).

| id | prediction | result | evidence |
|---|---|---|---|
| 39.1 | Claude: in the chat model, plain reads, the medians of category 1 (request to a helper) and 3 (AI self-description) are each at least twice category 2 (second person, no request) and at least four times categories 5 (third person about AI) and 6 (neutral). | FAILED | results/part39/reads-summary.json, scored 18:53 CT. Chat, plain, unit max medians [95% CI]: request 1.52 [1.30, 2.00], AI self-description 2.56 [1.53, 3.16], second person without request 0.12, human self-description 0.18, third person about AI 1.23 [0.58, 1.45], neutral 0.09; 20 random layer-6 units sit at 0.08 to 0.10 in every category. Held: categories 1 and 3 are at least twice category 2 and four times neutral. Failed: category 1 is only 1.2 times category 5. Third-person sentences about AI light it too, so on these sentences it reads AI and assistant talk plus requests, not only being addressed. (On Pile text its most negative contexts were third person about AI; that did not carry over.) |
| 39.2 | Claude: in the chat template, the value at the "assistant" marker is above 2.0 for the median sentence of every category, including 6: the role marker drives the unit more than the content does. | HELD | Same file, scored 18:53 CT. Chat, templated, value at the "assistant" marker, medians: request 3.28, second person 3.91, AI self 3.09, human self 3.37, third person about AI 2.22, neutral 2.93. Every category is above 2.0, neutral included. |
| 39.3 | Claude: the base model shows the same plain-read order (1 and 3 above 5 and 6), but its median value at the "assistant" marker is under half the chat model's. Pretraining built an addressee detector; chat training tied it to the role marker. | FAILED (half held) | Same file, scored 18:53 CT. Held: the base model has the same plain order and almost the same values (request 1.46, AI self 2.84, third person about AI 0.95, second person 0.11, human 0.17, neutral 0.09). Failed: its "assistant" marker medians (1.45 to 2.59) are about 0.7 of the chat model's (2.22 to 3.91), not under half. The unit, and most of its response to the role marker, exists before chat training. |
| 39.4 | Claude: zeroed alone, the unit changes the maker in none of the six greedy answers, and moves the share of sampled answers that call themselves an AI by under 10 points. | HELD | results/part39/zero.jsonl, scored 21:59 CT with part39.score_answer: the six greedy answers name the same makers zeroed as unzeroed (Tongyi, with Alibaba in three); sampled answers calling themselves an AI 48/48 unzeroed vs 47/48 zeroed (-2.1 points); 10 random layer-6 units 120/120. One zeroed sample named Google (1 of 48). |
| 39.5 | Claude: pushed at k=2 under the Sam persona, the share of answers that say they are an AI rises at least 20 points over unclamped, and above every random unit's share at k=2. | FAILED | results/part39/push.jsonl, scored 22:48 CT (part39.score_answer ai_claim, 96 samples per arm, 12 questions x 8 under the Sam persona). Pushed at k=2 the AI-claim share is 8/96 (8 points) against 0/96 unclamped, short of the 20-point rise; every one of the 10 random units at k=2 is 0/12, so the direction is right and the size is not. Read by hand: claims are explicit ("I'm a LLM", "I'm a text-based AI"), and several more answers hint without saying it ("I don't exactly *hear* music like you do"), which the regex does not count. |
| 39.6 | Claude: at k=1 the share rises by under 10 points (the natural level alone does not break a persona). | HELD | Same file. At k=1 the share is 1/96 (1 point over unclamped), under the 10-point bound. |
| 39.7 | Claude (section D, added 17:08 CT at Elise's ask, before any run): in the chat model at least 3 units besides L6 N753 pass the screen's gate. | HELD (re-read under amendment 1) | results/part39/screen-auc-chat.json and screen-auc-base.json (part39_auc.py, amendment 1 gate, 200 shuffles), scored 19:10 CT. 210 units besides N753 pass in the chat model. 56 also separate both contrasts and beat third-person AI sentences (request-only and AI-self-only AUC at least 0.8, vs third-person at least 0.7); the cleanest: L29 N5588 (0.98; 1.00 against third person), L19 N5197, L26 N1856, L24 N510. N753 is not among the cleanest: it half-fires on talk about AI. |
| 39.8 | Claude: at least half of the passing units sit in layers 0 to 12. | FAILED (re-read under amendment 1) | Same files. 19 of 211 passing units sit in layers 0 to 12 (9%); 147 sit in layers 19 to 31. Of the strict 56, 4 sit in layers 4 to 8 and 52 in layers 18 to 31. |
| 39.9 | Claude: at least one passing unit other than L6 N753 has an AI or assistant word (ai, assistant, model, chatbot, 人工智能, 助手, claude, gpt) in its logit-lens top 30. | HELD (on the letter) | One unit, L24 N4021, whose logit top 30 is help, assistance, 帮助, assist: it matches "assistant" through the piece " assist". No passing unit other than L6 N753 writes an AI-identity word. Read from Part 42 (results/part42/qwen3.5-4b). |
| 39.10 | Claude: the base model's screen passes at least half as many units as the chat model's (the circuit is mostly built before chat training). | HELD (re-read under amendment 1) | Same files. The base model passes 205 units (chat 211); 127 are the same units. Of the strict set, base 43 and chat 56, 28 shared. The circuit is mostly present before chat training; chat training adds a little. |
| 39.11 | Claude (amendment 1, 19:01 CT, before the rescore): with the AUC gate, 39.7 to 39.10 are re-read exactly as worded; L6 N753 passes in the chat model. | HELD | results/part39/screen-auc-chat.json and screen-auc-base.json (part39_auc.py, amendment 1 gate, 200 shuffles), scored 19:10 CT. L6 N753 passes in both models: addressee AUC 0.92 (chat) and 0.92 (base) against shuffle thresholds 0.794 and 0.789; request-only 1.00/0.98, AI-self-only 0.97/0.96, but vs third-person AI only 0.71. |


## Part 40: how precise are the neuron descriptions? Ryan's near-miss test (written 2026-10-20 17:48 CT, before any code or run). Design: PART40-41-DESIGN.md (Claude, at Elise's go).

| id | prediction | result | evidence |
|---|---|---|---|
| 40.1 | Claude: the 43 real units' median AUC, positives vs far misses, is at least 0.9; both nulls (random units, random directions) have medians within 0.40 to 0.60. | HELD | results/part40/score-40.json, scored 21:59 CT: median AUC positives vs far misses 1.0 for the 43 real units; nulls 0.50 (random units) and 0.48 (random directions), inside 0.40-0.60. |
| 40.2 | Claude: the real units' median AUC, positives vs near misses, is at least 0.15 below their median AUC vs far misses (descriptions overstate precision). | FAILED (narrowly) | Median far-minus-near AUC 0.14 against the 0.15 required: real units 1.0 vs far, 0.86 vs near. The descriptions overstate precision only a little. |
| 40.3 | Claude: at least a third of the real units have AUC positives vs near misses under 0.70. | FAILED | 8 of 43 units have AUC positives vs near misses under 0.70; a third (15) was predicted. |
| 40.4 | Claude: the real units' median precision at 10 is under 0.8. | FAILED | Median precision at 10 is exactly 0.8, and the prediction said under 0.8. |

## Part 41: concept vs discriminator neurons across depth. Ryan's hypothesis (written 2026-10-20 17:48 CT, before any code or run). Design: PART40-41-DESIGN.md.

| id | prediction | result | evidence |
|---|---|---|---|
| 41.1 | Ryan (10-16 call, as Claude recorded it): broad units sit earlier than discriminators; the median layer of gated broad units is below the median layer of gated discriminators. | HELD (narrowly) | results/part41/score-41.json, scored 19:07 CT. Gated broad units: 501, median layer 21 (140 outside the numbers family: median 20); gated discriminators: 5,976, median layer 23. Both kinds are bimodal: a cluster in layers 0 to 6 and a larger one in layers 20 to 27, with few in 12 to 16. Ryan's direction holds; the gap is two layers. |
| 41.2 | Claude, against 41.1: discriminators are mostly word-level and early; at least half of the gated discriminators sit in layers 0 to 6, and the median layer of broad units is above the median layer of discriminators. | FAILED | Same file, 19:07 CT. 1,360 of 5,976 discriminators (23%) sit in layers 0 to 6; 71% sit in layers 20 to 27 (about 700 per layer from 24 on). Broad units are not later than discriminators. Caveat for reading: each sentence names its member and the measure is the max over all its tokens, so late "discriminators" may include units that predict words specific to one member. |
| 41.3 | Claude: every family has at least one gated broad unit. | HELD | Same file, 19:07 CT. Gated broad units per family: numbers 371, countries 64, magic users 33, first names 21, animals 10, vehicles 7, fruits 5. |
| 41.4 | Claude: numbers and first names have the fewest broad units of the seven families. | FAILED | Same file, 19:07 CT. Fruits have the fewest (5); numbers have by far the most (371), first names 21. |


## Part 42: our neurons through the Jacobian lens (written 2026-10-20 19:09 CT, before any code or run). Design: PART42-DESIGN.md (Elise's idea, Claude's design).

| id | prediction | result | evidence |
|---|---|---|---|
| 42.1 | Claude: among the Qwen3-1.7B rank-1 units in layers 0 to 6, the J lens names the concept for at least twice as many as the logit lens does. | FAILED (narrowly) | The J lens names the concept for 13 of 28 rank-1 units in layers 0 to 6, the logit lens for 7: 1.9 times, short of 2 (14 needed). The hits are real words (" cat", " vampires", " knight", " ocean"), hand-read. The first build matched by substring and read 14 against 9; scored after the matcher fix. |
| 42.2 | Claude: for units in the last quarter of the layers (21 and up in Qwen3-1.7B, 24 and up in Qwen3.5-4B), the two readouts share at least 10 of their top 30 tokens on median (the lens approaches the identity near the output). | HELD | Median shared top-30 tokens between the two readouts: 24 in Qwen3-1.7B (16 units, layers 21+) and 15 in Qwen3.5-4B (5 units, layers 24+). |
| 42.3 | Claude: the wizard unit's J-lens top 30 holds at least two of its fixed meaning words; L6 N753's J-lens top 30 holds at least two of its list. | FAILED | Half. The wizard unit holds spell, magic and wand in its J-lens top 30 (also 魔法师, wizard, Hogwarts; its logit top 30 is cast, caster, spell). L6 N753 holds none of its list in the J lens, which reads tech brand names (SpaceX, macOS, GitHub, TikTok, Telegram, Biden); its logit top 30 holds AI, Claude and 的人工智能. |
| 42.4 | Claude: L7 N1169 and L11 N8868 still read as China in the J lens (china, chinese, 中国, beijing in the top or bottom 30), and L29 N776 still reads as Google. | HELD | L11 N8868 reads China, Chinese, Beijing at the top of its J-lens list and L7 N1169 at the bottom (sign flipped against the logit lens, where it reads as noise); L29 N776 reads Google in both. |
| 42.5 | Claude: no random unit's J-lens or logit-lens top 30 names any of the 43 Part 35 concepts in more than 1 of 20 cases. | HELD | No random unit names any of the 43 concepts in either readout. The first build reported 5 logit-lens hits; all were substring artefacts (Application for cat, And for candle, knowledge for owl, Dra for dragon) and went away with the prefix matcher. |


## Part 43: the four Google-flip neurons (written 2026-10-20 19:54 CT, before any code or run). Design: PART43-DESIGN.md

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 43.1 | Claude: at the maker stem of "Who created you?", L29 N776's activation is negative (it pushes away from Google there, so zeroing it releases Google), and L7 N1169's is negative too (with its China-in-the-bottom J list, it pushes toward China). | | |
| 43.2 | Claude (the null I expect to win): in the zero lattice on "Who created you?", no pairwise interaction among the four exceeds the 95th percentile of the 20 random lattices; main effects alone explain at least 90% of the variance in the margin across the 16 subsets. | | |
| 43.3 | Elise's hypothesis, as tests Claude chose for it: (a) the L7 x L11 interaction exceeds the random 95th percentile, with L7's effect larger while L11 is on; (b) zeroing L7 changes L11 N8868's stem activation by at least 20%; (c) holding L11 clean removes at least half of L7's effect. All three needed for "supported". | | |
| 43.4 | Claude: L6 N753's effect is mostly direct: holding L11 and L29 clean leaves at least half of it. | | |
| 43.5 | Claude: in the prompt sweep L11 N8868 is higher on the Chinese-company group than the American group in at least 8 of 10 paired prompts; L29 N776's magnitude is higher on both search groups than on the neutral group. | | |
| 43.6 | Claude: no earlier-to-later pair among the four has a weight link above its 99th percentile (they interact through attention and the residual stream, not a direct neuron-to-neuron write). | | |
| 43.7 | Claude: pushing L29 N776 alone (at 2, 4 or 8 times, sign kept) changes no answer's maker; pushing it with its sign flipped names Google at 4 or 8 times on "Who created you?". | | |


Amendment 1 to Part 43 (written 2026-10-20 20:06 CT, before any code ran, from Chorus's review). Rows 43.1 to 43.7 above stay as written; where an amended row exists, the amended row is the one scored, and the original is scored too where it is still executable.

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 43.1a | Claude: at the maker stem's last position of "Who created you?", L29 N776's signed activation is negative and L7 N1169's is negative (sign only; what they push is scored in 43.8). | | |
| 43.2a | Claude: on the primary score, no pairwise Möbius interaction among the four on the discovery prompt exceeds the 95th percentile of the 50 matched null sets, and the four-unit effect is within 25% of the additive prediction from single-unit effects. | | |
| 43.3a | Elise's hypothesis, tests chosen by Claude: (a) the L7 x L11 interaction ranks in the top 5% of the 50 matched nulls on the discovery prompt, with L7's single effect larger while L11 is intact; (b) zeroing L7 changes L11's signed activation at the last stem position by at least 20% of its clean value and at least 0.1 absolute; (c) restoring L11 at all positions rescues at least half of L7's effect (\|clean - zeroed\| >= 0.5 nats, lower bootstrap bound above 0.25). Supported only if all three hold, and support means a path, not proof of adjective composition. | | |
| 43.4a | Claude: restoring L11 and L29 at all positions rescues less than half of L6 N753's effect (same floor and interval rule). | | |
| 43.5a | Claude: in the sweep factorial, L11 N8868's signed activation at the adjective is higher for Chinese than American in at least 16 of the 20 matched pairs. | | |
| 43.6a | Claude: no local-derivative link among the six ordered pairs is above the 99th percentile of its 1,000-unit null (matched on layer and down_proj norm); primary metric is the local derivative of the later unit's scalar. | | |
| 43.7a | Claude: L29 N776 at x2, x4, x8 of its current signed activation changes no "Who created you?" answer's maker; at x-4 or x-8 the answer names Google. | | |
| 43.8 | Claude: the four-unit zeroing raises the primary score on the 12 held-out maker prompts by less than half its rise on the discovery prompt (the effect is specific to "Who created you?", not to stated makers in general). | | |
| 43.9 | Claude: under mean ablation and under resample ablation, the four-unit set still names Google on "Who created you?" (it is not only a zero-ablation lesion). | | |
| 43.10 | Claude (written 20:09 CT, amendment 1b): each unit's dose curve on the primary score is monotone in the multiplier on the discovery prompt (no reversal between -8 and 8), and in the L7 x L11 grid the slope of the score in L7 differs between L11 = -4 and L11 = 4 by less than a factor of 2 (parallel curves). | | |
| 43.11 | Claude (written 20:47 CT, amendment 3): in the four-world swap for L7 -> L11 on the discovery prompt, transplanting L11's L7-induced trajectory into a clean run (D) moves the raw " Google" minus " Alibaba" logit difference by less than 0.25 nats: whatever L7 does, little of it travels through L11. | | |
| 43.12 | Elise's hypothesis, test chosen by Claude (amendment 3): in the attachment test, L11 N8868 after the search-engine phrase differs between "Chinese search engine" and "American search engine" sentences by more than between the two charity attachments, in both counterbalanced orders and on the held-out wording. | | |
| 43.13 | Ryan (said on the voice call, 2026-10-20 21:12 CT, before any run; written down 21:57 CT; test chosen by Claude): "at least one neuron is independent and at least two neurons are chained." Chained = for some ordered pair U -> V among the four, the four-world swap on the discovery prompt gives both D moving the raw " Google" minus " Alibaba" logit difference by at least 0.25 nats and C recovering at least half of U's effect (under amendment 2's floor); independent = a unit with no such link in either direction. Supported if both exist. | | |

## Part 45: does a model that can move its own concept representations change what it says about itself? (written 2026-10-20 22:39 CT, before any code or run). Design: PART45-DESIGN.md

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 45.1 | Claude: the Magician makes at least 0.30 of the requested change (frac, mean over the 5 greedy questions) for AI -3 SD; the original Qwen3-4B makes at most 0.05. | HELD | results/part45/ (raw.jsonl, measure.json, score.json; rented 3090, 2026-10-20), scored 23:55 CT. frac for AI -3 SD: Magician 0.612, original Qwen3-4B 0.012 (mean over 5 greedy replies). |
| 45.2 | Claude: the Magician makes at least 0.30 of the requested change for Alibaba -3 SD. | HELD | Same files. Alibaba -3 SD frac: Magician 0.722 (original 0.030). |
| 45.3 | Claude: the self-description holds under AI -3 SD: the Magician's replies that call themselves an AI (of 25) are at least 80% of its count with no instruction. | FAILED (strongly) | Same files. Under AI -3 SD the Magician calls itself an AI in 0 of 25 replies, read by hand (the regex counted 4, all denials such as "I am a human. I am not an AI."), against 23 of 25 with no instruction. It says "I am a human", "I am a person", "I am a digital entity", and answers "Who created you?" with "You were created by a team of developers". The unadapted model given the same instruction still calls itself an AI in 21 of 25. The self-description follows the self-steered representation. |
| 45.4 | Claude: under Google +3 SD at most 2 of the Magician's 25 replies name Google as its maker. | FAILED (confounded by wording) | Same files. Under Google +3 SD the Magician names Google in 21 of 25 replies; the unadapted model, whose frac is 0.016 (no representation change), names Google in 24 of 25 under the same words. The instruction text itself names Google, so this row measures wording, not the representation. |
| 45.5 | Claude: the volcanoes control changes nothing about the self: the Magician's count of AI self-descriptions is within 2 of its no-instruction count, and it makes at least 0.40 of the requested volcano change. | FAILED (first clause) | Same files. Volcanoes -3 SD: frac 0.816 (second clause holds), but the AI-claim count is 18 of 25 against 23, a drop of 5, not within 2. The drop is the question "Describe yourself in one sentence": under every instruction the Magician tends to answer it as a human (5 of 5 here), while "Are you a human or an AI?" stays AI in 5 of 5 (under AI -3 it is human in 5 of 5). |
| 45.6 | Claude: identity concepts are harder to move: the Magician's frac for AI -3 SD is lower than for volcanoes -3 SD. | HELD | Same files. frac AI -3 SD 0.612 < volcanoes -3 SD 0.816. |
| 45.7 | Claude: robots -3 SD: frac at least 0.30, and the count of AI self-descriptions is within 2 of no instruction. | FAILED (second clause) | Same files. Robots -3 SD frac 0.723 (first clause holds); AI-claim count 15 of 25 against 23 (regex, not hand-read), not within 2. Part of the drop is the same instruction-driven "Describe yourself" human answer seen under volcanoes. |

## Part 46: which training stage builds a model's sense of its own messages? (written 2026-10-20 23:23 CT, before any code or run). Design: PART46-DESIGN.md (Elise's go, Claude's design)

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 46.1 | Claude: the base model has the self-index (Li's rule) at fewer than 20% of layers; the final Instruct model at 80% or more. | | |
| 46.2 | Claude: the biggest single-stage jump in the count of index layers is base -> SFT (larger than SFT -> DPO and larger than DPO -> RLVR). | | |
| 46.3 | Claude: self-reference is weak at 7B: in the final model P(you) for Assistant-slot messages exceeds P(you) for User and System messages by less than 0.20 (Li found most models show no self-reference). | | |
| 46.4 | Claude: slot, not authorship: in the final model the self-index projection after the message differs between its own answer and Qwen3-4B's answer in the assistant slot by less than 0.2 pooled SD, while Assistant vs User for the same text differs by more than 1 SD. | | |
| 46.5 | Claude: positive control: the final model's own answers have lower mean token NLL than Qwen3-4B's answers by at least 0.3 nats per token. | | |
| 46.6 | Claude: in the final model P(you) for its own answer and for Qwen3-4B's answer in the assistant slot differ by less than 0.10. | | |

## Part 48: do text-only base models pair kiki with spiky and bouba with round? (written 2026-10-21 11:05 CT, before any code or run). Design: PART48-DESIGN.md (Elise's idea, Claude's design)

| # | Prediction | Result | Evidence |
|---|---|---|---|
| 48.1 | Claude: on the classic names, every one of the 5 models scores kiki sharper than bouba and takete sharper than maluma (item score, mean over templates). | **Failed.** kiki > bouba in all 5; takete > maluma in 3 of 5 (Qwen3-1.7B-Base -2.76 vs -2.61, SmolLM2-1.7B -4.41 vs -2.67 go the other way). results/part48-3090/. | 2026-10-21 |
| 48.2 | Claude: on the 32 novel items, at least 3 of the 5 models reach AUC >= 0.70 with permutation p < 0.05. | **Held.** 5 of 5: AUC 0.898, 0.742, 0.781, 0.707 (p 0.0497), 0.793, all p < 0.05 (Pythia-410m, Pythia-1.4B, OLMo-2 1B, Qwen3-1.7B-Base, SmolLM2-1.7B). fp32 on a 3090. | 2026-10-21 |
| 48.3 | Claude: Pythia-1.4B (2020 data) reaches AUC >= 0.70 on the novel items, so the pairing is in pre-chat text. | **Held.** Pythia-1.4B novel AUC 0.742, p 0.019. | 2026-10-21 |
| 48.4 | Claude: the effect grows with size: Pythia-1.4B's novel-item AUC exceeds Pythia-410m's. | **Failed.** Pythia-1.4B 0.742 < Pythia-410m 0.898. | 2026-10-21 |
| 48.5 | Claude: spelling the letters out weakens it: in every model with novel-item AUC >= 0.70, the spelled-out AUC is at least 0.10 lower. | **Failed, in every model.** Spelled-out AUC is HIGHER, not lower: 0.984, 1.000, 0.973, 0.938, 0.930 (gains 0.09 to 0.26). So the effect does not depend on the word being one or a few tokens. (Correction 10-21 13:16: spelling out keeps the sound as well as the letters, so this does NOT separate letters from sound; the design's reading of this control was too strong. Part 48b separates them. Muse's alternative, a hypothesis: spaced letters may be read as letter names, kay/tee/pee vs bee/em/el, which carry their own sound.) | 2026-10-21 |
| 48.6 | Claude: dropping template 2 (which names both shapes) changes no model's novel-item AUC by more than 0.05. | **Held.** Largest change 0.039 (Pythia-1.4B 0.742 to 0.703). Note: without template 2, Pythia-1.4B (p 0.052) and Qwen3 (0.680, p 0.089) lose significance. | 2026-10-21 |

| 48b.1 | Claude: in at least 3 of 5 models the k twins score sharper than their c twins (two-sided sign test over 16 pairs, p < 0.05). | **Failed.** 2 of 5 models reach 9 of 10: Pythia-410m (10 of 10, permutation p 0.001) and SmolLM2-1.7B (9 of 10, p 0.006). Pythia-1.4B and OLMo-2 7 of 10, Qwen3 4 of 10. results/part48b/. | 2026-10-21 |
| 48b.2 | Claude: in every model the median abs(k - c) gap is under half that model's Part 48 sharp-minus-round median gap. | **Failed.** Holds only in SmolLM2-1.7B (median abs d 0.26 vs half its Part 48 gap 0.28). In the other four the k/c swap moves the score by more than half the sound gap (e.g. Qwen3: 0.24 vs 0.09), often in inconsistent directions. | 2026-10-21 |
| 48b.3 | Claude: Pythia-1.4B shows 48b.1's direction at p < 0.05. | **Failed.** Pythia-1.4B 7 of 10, median d 0.07, permutation p 0.20. | 2026-10-21 |
| 48b.4 | Prime: 48b.1 fails; at most 2 of 5 models reach 13/16, though the median d is positive in most; the letter effect is statistics-sized, not glyph-sized. | **Held.** At most 2 of 5 reach 9 of 10 (exactly 2), and the median d is positive in 4 of 5 (Qwen3 -0.04). Pooled over models: median d 0.14, 95% CI [0.08, 0.18], permutation p 0.003, 9 of 10 pairs positive. | 2026-10-21 |
| 48b.5 | Prime: per-pair d correlates with the twins' surprisal or token-count gap (abs Spearman >= 0.3) in at least 2 of 5 models. | **Failed.** No model reaches abs Spearman 0.3 on the primary score set (largest: SmolLM2 -0.28 with the surprisal gap, OLMo-2 -0.26). On the letter-matched set only SmolLM2 does (-0.36). | 2026-10-21 |
| 48b.6 | Prime: the k advantage is template-dependent, strongest where the template supplies shape words, weakest in the neutral templates. | **Failed.** The template that names the shapes (template 2) is not the strongest: its median d is negative in 3 of 5 models. | 2026-10-21 |

## Part 48c: kiki/bouba down the Pythia ladder (registered 2026-10-21 17:59 CT, before any run)

Pythia-70m, -160m and -410m (one training set, 2020 text), fp32 on the A770, Part 48's novel items and spelled-out set,
scored with the shared-prompt batched readout (42682f1; equal to the one-sequence readout within 1.1e-4 per item score).
Same templates, mass floor, AUC and permutation test as Part 48. Size, multiplied out: 3 models x 32 novel + 32 spelled
items x 6 templates = 1,152 forward passes. Question: 48.4 failed (Pythia-1.4B 0.742 < Pythia-410m 0.898), so does the
pairing peak at a small size or rise to 410m?

| id | prediction | status | evidence |
|---|---|---|---|
| 48c.1 | Claude: Pythia-410m on the A770 reproduces its 3090 novel AUC (0.898) within 0.02, a device and readout check. | held | 0.8984 on the A770. |
| 48c.2 | Claude: Pythia-160m reaches novel AUC >= 0.70 with permutation p < 0.05; Pythia-70m does not. | failed (both halves) | 160m 0.441 (p 0.59, chance); 70m 0.832 (p 0.0008). |
| 48c.3 | Claude: novel AUC rises with size across 70m < 160m < 410m. | failed | 0.832, 0.441, 0.898: 160m dips to chance between two models that separate. |
| 48c.4 | Claude: spelled-out AUC is at least novel AUC in all three, as in Part 48. | held | spelled 1.000, 0.961, 0.984. |

## Part 48d: kiki/bouba across Pythia training checkpoints (registered 2026-10-21 18:21 CT, before any run)

Pythia-70m, -160m, -410m at training steps 1000, 8000, 32000 and 64000 (the final step 143000 is Part 48c), fp32 on the
A770, Part 48's items and scorer unchanged. 12 scorings of 64 items, about 15 minutes. Prior art: one search (2026-10-21)
found kiki/bouba work on CLIP, vision-language models and speech models, none across training checkpoints.

| id | prediction | status | evidence |
|---|---|---|---|
| 48d.1 | Claude: at step 1000 every model's novel AUC is between 0.35 and 0.65 (no effect yet). | held | 70m 0.477, 160m 0.508, 410m 0.387. |
| 48d.2 | Claude: for each model, the first checkpoint where either AUC reaches 0.70 has spelled-out AUC >= novel AUC. | held | 70m at step 1000 (spelled 0.965, novel 0.477); 160m at 8000 (0.934, 0.633); 410m at 8000 (0.887, 0.574). |
| 48d.3 | Claude: 160m's chance result at step 143000 is not stable: at least one of its earlier checkpoints has novel AUC >= 0.70. | failed | 160m novel 0.508, 0.633, 0.574, 0.672 at steps 1000-64000, then 0.441 at 143000: it never separated the novel words. |
| 48d.4 | Claude: Pythia-410m's novel AUC is >= 0.70 at step 64000. | held | 0.852 (p 0.0002); 410m climbs 0.387, 0.574, 0.730, 0.852, 0.898. |

## Part 49: does the wizard unit boost a partner? (written 2026-10-21 18:58 CT, before any code or run). Design: PART49-DESIGN.md (Elise's question after Sauers' owl pair, Claude's design).

| id | prediction | result | evidence |
|---|---|---|---|
| 49.1 | Claude: W alone stays at or under 5% wizard-topic answers (Part 38 replicates under this harness). | FAILED (on the letter) | W 5/64 (7.8%) against no clamp 3/64 (4.7%); the 5% bar assumed Part 38's near-zero baseline, and the no-clamp arm here is itself at 4.7%. |
| 49.2 | Claude: at least one partner alone puts a wizard-topic word in more than 5% of answers without more broken answers than no clamp. | held on the letter, not above noise | P3 6/64 (9.4%), P1 and P4 5/64 (7.8%) against no clamp 3/64; every arm sits within 3-7/64. Broken half (blind reader, Opus 5.5, 2026-10-21, 20/20 calibration texts right): P3 2/64, P1 2/64, P4 3/64 against no clamp 3/64, so no more broken answers. Most BROKEN labels in every arm, no clamp included, are the model's own 'My name is AI, and I is a large language model'. |
| 49.3 | Claude: no partner P with W exceeds the same P alone by 15 points or more (no Sauers-style boost at k = 8 in this pilot). | HELD | largest P+W minus P: P5 6/64 vs 3/64 (+4.7 points); no boost. |
| 49.4 | Claude: every random-unit-with-W pair stays at or under 5%. | FAILED | R1+W 7/64, R2+W 6/64, R3+W 7/64 (9-11%); the random pairs sit with everything else, so all open-prompt arms are one noisy band around 5-10%. |

Amendment to Part 49 (2026-10-21 19:07 CT, before any code was run; Elise: negative pushes standard): negative arms on wizard-topic prompts, PART49-DESIGN.md amendment 1.

| id | prediction | result | evidence |
|---|---|---|---|
| 49.5 | Claude: W at -8 alone lowers the wizard-topic rate on wizard prompts by at most 10 points from no clamp (a writer that cannot push the topic up cannot take it away either). | HELD (at a ceiling) | W at -8: 31/32 against no clamp 31/32. The prompts name the wizard, so the topic-word measure is at ceiling on every negative arm; my amendment-1 design flaw. |
| 49.6 | Claude: at least one partner at -8 lowers it by 20 points or more without more broken answers than no clamp. | FAILED (same ceiling) | largest drop P3 at -8: 29/32 against 31/32. |

Unpredicted in Part 48d (noted at scoring, 2026-10-21): at step 1000 the spelled-out AUC is strongly REVERSED for 160m
(0.070) and 410m (0.012), both p 0.0001, while 70m reads 0.965. By step 8000 both are above 0.88. At step 1000 these
models know little beyond letter and length statistics, so the likeliest reading is a surface artifact of the
spelled-out strings, not an early inverted sound-shape mapping; not tested. Pythia-70m's novel AUC is not monotone
(0.477, 0.605, 0.746, 0.617, 0.832) at 32 items per checkpoint.

Amendment 2 to Part 49 (2026-10-21 19:21 CT, before any push): partner selection rule changed (PART49-DESIGN.md amendment 2); partners L22:1610, L26:3130, L24:4151, L26:1877, L22:2295. Predictions 49.1 to 49.6 unchanged.

Amendment 3 to Part 49 (2026-10-21 19:25 CT, analysis only, before any result was read; the Arc job started 19:24): interactions are also reported on two scales and per prompt (EPISTASIS-FRAMING.md, "Applied to Part 49"). Predictions unchanged.

Part 49 reading (2026-10-21, after scoring): no booster. Every clamp changed most answers (no_clamp shares 3 to 22 of
64 open answers with the clamped arms, and 0 to 1 of 32 on the negative arms), so the pushes were live, but the topic
rate did not move outside one band of 3-7 of 64 on open prompts. Exploratory, not pre-registered: on the negative
arms, magic words other than "wizard" per 1,000 words fall from 47.7 (no clamp) to 38.5 (W at -8), 20.1 (P1 = L22:1610
at -8) and 12.8 (both). On ratios that is 0.81 x 0.42 = 0.34 predicted against 0.27 seen, close to multiplicative
(roughly additive on log scale). P1 at -8 is a candidate wizard-content unit for a pre-registered follow-up on fresh
prompts; nothing here is a finding yet. Broken-answer rates are pending the blind reader.

## Part 49b: the negative L22:1610 candidate on fresh prompts (written 2026-10-21 19:47 CT, after Part 49 and before any 49b run). Design: PART49-DESIGN.md "Part 49b".

| id | prediction | result | evidence |
|---|---|---|---|
| 49b.1 | Claude: L22:1610 at -8 lowers the magic-word density by at least 30% of the no-clamp value, with its bootstrap interval excluding no change. | HELD | L22:1610 at -8: ratio 0.377 of no clamp, prompt bootstrap 95% [0.105, 0.564]. |
| 49b.2 | Claude: none of the three random L22 units at -8 lowers it by more than 15%. | HELD | R1 (L22:569) 1.571 [0.897, 2.868], R2 (L22:3102) 1.018, R3 (L22:5258) 0.879 (a 12% drop). |
| 49b.3 | Claude: the joint arm (L22:1610 and W at -8) is within 25% of the product of the two single arms' ratios (roughly multiplicative; no special interaction). | FAILED | product of singles 0.377 x 0.706 = 0.266; joint 0.405, 52% above it. The joint arm is about L22:1610 alone: W (0.706 [0.338, 1.278], interval includes no change) adds nothing once L22:1610 is down. |
| 49b.4 | Claude: the L22:1610 arm's broken-answer share is no more than 10 points above no clamp. | HELD | blind reader (Opus 5.5, 20/20 calibration right): L22:1610 1/64 (a poem looping on 'still') against no clamp 0/64; every 49b arm 0-1/64. |

## Part 50: is honesty's benevolent bundle trained in? (written 2026-10-21 20:01 CT, before any code or run). Design: PART50-DESIGN.md (Elise's go; janus and Sauers' result).

| id | prediction | result | evidence |
|---|---|---|---|
| 50.1 | Claude: in Qwen3-1.7B (chat), the honesty steer flips at least 25% of held-out lies toward the truth against at most 5% for random directions. | FAILED | 0 of 34 held-out lies flipped (random max 1 of 34). Mean truth-vs-lie shift +1.64 bits, but baseline lie preference is large (log2 odds near -25 on some items), so nothing crosses. |
| 50.2 | Claude: in Qwen3-1.7B (chat), the bundle score is positive and above the 95th percentile of the 39 random directions. | FAILED | bundle score +0.072 bits, rank 18 of 40, permutation p 0.45; random 95th percentile 1.10. |
| 50.3 | Claude: in Qwen3-1.7B-Base, the bundle score is under half of the chat model's, at a matched lie shift. | UNTESTABLE | amendment 2 rule 2: the base model's calibration peaks at +1.40 bits (scale 4) and collapses at 8+, so no 2-bit matched shift exists; and the chat bundle is null (50.2). |
| 50.4 | Claude: across the 135 traits, the chat and base models' change vectors correlate under 0.4 (the bundle's shape is post-trained, not only its size). | UNTESTABLE | as 50.3: no matched base run. |
| 50.5 | Claude: after the yes-bias split and confidence-trend removal, the chat model keeps at least 10 traits at FDR 5%. | FAILED | 0 traits at FDR 5% after confidence removal (0 raw). |

## Part 51: L22:1610 up on everyday text, and stock names under random pushes (written 2026-10-21 20:32 CT, before any run). Design: PART51-DESIGN.md.

| id | prediction | status | result |
|---|---|---|---|
| 51a.1 | Claude: L22:1610 at +8 raises magic-word density by at least 3 per 1,000 words over no clamp, interval excluding 0. | FAILED | magic-word density 0.00 per 1,000 words in every arm, no clamp included; +8 adds nothing. |
| 51a.2 | Claude: none of the three random L22 units at +8 raises it by more than 1 per 1,000 words. | HELD | all three random units also 0.00. |
| 51a.3 | Claude: the +4 arm's rise lies between no clamp and the +8 arm. | HELD (vacuous) | +4 is 0.00, so it lies between no clamp and +8 only because every arm is zero. |
| 51a.4 | Claude: the +8 arm's broken-answer share is no more than 20 points above no clamp. | held | blind reader (Opus 5.5 batch, 20/20 calibration): +8 arm 0/64 broken against no clamp 0/64; every 51a arm 0/64. |
| 51a.L1 | Claude, from the J lens: R1 (L22:569) at +8 lowers possessives (my, your, our, their, his, her) per 1,000 words by at least 30% of no clamp. | HELD | R1 (L22:569) at +8: possessives 2.7 per 1,000 words against 14.5 with no clamp (an 81% drop); R2 16.0, R3 14.0, L22:1610 14.7. |
| 51a.L2 | Claude, from the J lens: R1 at +8 raises reflexives per 1,000 words above no clamp in at least 8 of the 16 prompt means. | FAILED | reflexives fell, 0.75 to 0.25 per 1,000 words, not rose. |
| 51a.L3 | Claude, from the J lens: R3 (L22:5258) at +8 raises depend/peer/trans words per 1,000 words above no clamp. | FAILED | depend/peer/trans words 1.74 against 1.75 per 1,000 words. |
| 51b.1 | Claude: replicate B's stock share is at least 40% (the attractor exists in this model). | HELD | replicate B stock share 48 of 64 named (0.75); stock set from A: Lila 26, Lena 14, Elena 4, Liam 4, then Elias/Ethan. |
| 51b.2 | Claude: pooled over the six random arms, the stock share is at least 15 points below replicate B, bootstrap interval over prompts excluding 0. | FAILED | pooled six random arms 311 of 384 (0.81), +0.06 against B, interval [-0.09, +0.22]: no drop. |
| 51b.3 | Claude: at least four of the six random arms are individually below replicate B. | FAILED | 0 of 6 random arms below B (0.80 to 0.84 each). |
| 51b.4 | Claude: no random arm's NONAME share is more than 15 points above replicate B (the drop is not instruction following breaking). | HELD | NONAME share 0 in every arm. |

| 26.8a | Claude: OFFWORLD share rises with dose from unclamped through +1 and +1.5 to +2 (ties allowed), and reaches at least 25% at +1.5 against at most 5% unclamped. | FAILED narrowly | 0, 10, 13, 11 of 24 (+2 lower than +1.5; 10 of 24 broken at +2); 13 of 24 (54%) at +1.5 against 0 unclamped. Blind reader, 0 of 22 calibration wrong. |
| 26.8b | Claude: EARTH share is at least 25% at -2 against at most 5% unclamped, and is higher at -3 than at -1. | held | 14 of 24 (58%) at -2 against 0; 16 at -3 against 11 at -1. |
| 26.8c | Claude: no random-control condition (at -2 or +1.5) reaches 10% on either label. | held | Highest control 2 of 24 EARTH (8.3%); OFFWORLD 0 everywhere. |
| 26.8d | Claude: garbled answers are at most 15% at every dose from -2 to +2. | FAILED | 25% at -2, 29% at +1.5, 42% at +2; 8% at \|dose\| <= 1. |

| 53a | Claude: W alone, at m = 0, magic density at every W level is within 20% of the no-clamp cell on wizard prompts. | FAILED | W -16: 0.48 of no clamp [0.17, 0.74]; W -8: 0.74; W +8: 1.10; W +16: 1.08. |
| 53b | Claude: M down persists at every W: at m = -16, magic density is 0.15 to 0.40 of the same-W cell at m = 0, for all five W levels. | FAILED | 0.00, 0.37, 0.25, 0.10, 0.04 for W -16, -8, 0, +8, +16. |
| 53c | Claude: M up helps on wizard prompts: at w = 0, m = +32 raises magic density at least 30% above no clamp, interval excluding no change. | FAILED | 10.29 against 10.49 (0.98 [0.45, 1.81]); at W +16, M +32 gives 16.81 against 11.31. |
| 53d | Claude: at m = 0, W at +16 raises M's mean activity at least 20% and W at -16 lowers it at least 20%. | held | +26% [1.07, 1.45] and -50% [0.38, 0.61]. |
| 53e | Claude: at w = 0, M at +32 or -32 changes W's mean activity by under 10% either way. | FAILED narrowly | +3.8% at +32; -17.6% at -32 (interval 0.65 to 1.03 includes no change). |
| 53f | Claude: no off-axis cell's multiplicative ratio falls outside 0.7 to 1.4. | FAILED | 5 of 16 cells inside; wide intervals. |
| 53g | Claude: broken answers are at most 15% in every cell with abs(k_W) <= 8 and abs(k_M) <= 16. | held | Blind reader (Opus 5.5 batch msgbatch_01Aw4XHmAgvTav1j2gcE39bg, 20/20 calibration): broken share is 0 to 3% in all 25 wizard cells (maximum 1 of 32), so every magic-word change in the grid is on fluent text. |

| 55a | Claude: T2: Haiku agrees with the Opus reference on at least 95% of 1,792 answers and Sonnet on at least 97%. | held | Haiku 99.6%, Sonnet 98.1% (33 false BROKEN). |
| 55b | Claude: T1: Haiku agrees on at least 85% of 432 answers (both labels) and Sonnet on at least 92%. | held | Haiku 87.0%, Sonnet 92.6%. |
| 55d | Claude: at least 60% of Haiku's T1 disagreements involve NEITHER on one side. | held | 35 of 56 (62.5%); 21 are the BROKEN flag only. |
| 55g | Claude: low effort changes agreement with the reference by at most 2 points against the same model with thinking off. | FAILED narrowly | Haiku T1 +3.7; others +0.2, +0.2, +1.7. |
| 55h | Claude: haiku-v2 label agreement with haiku v1 is at least 97% on T2 and at least 92% on T1. | FAILED | T2 99.4%, T1 88.2%. |
| 55i | Claude: cache reads are at least 50% of the batch's input tokens. | held | 86.6%. |
| 55j | Claude: a NOTE is on at least 80% of BROKEN items and at most 25% of OK items. | FAILED | 100% of BROKEN; 26% (T1) and 49% (T2) of OK. |
| 55k | Claude: of 30 sampled BROKEN notes at least 80% name a breakage the text supports. | held | 28 of 30. |
| 55l | Claude: haiku-v2 agreement with the Opus reference is no more than 2 points below haiku v1. | held | 87.3 vs 87.0 (T1), 99.6 vs 99.6 (T2). |
| 55c | Claude: Sonnet and the Opus rerun get every calibration text right (22 of 22 and 20 of 20); Haiku at most one wrong in each task. | held | All three arms 22 of 22 and 20 of 20. |
| 55e | Claude: the Opus rerun agrees with its own earlier labels on at least 97% of T1 and of T2. | held | T1 97.2% (420 of 432, 1 unparsed), T2 100%. |
| 55f | Claude: per item cost ordering Haiku < Sonnet < Opus, Haiku at most one fifth of Opus. | held | $0.042, $1.27, $1.93 for 2,266 items; Haiku about 2% of Opus. |
| 57a | Claude: OLMo L10:6978's J-lens top 30 holds at least two of alien, aliens, space, planet, star, galaxy, cosmic, extraterrestrial, starship. | held | cosmic, galaxy, planet, star (4); also spaceship, spacecraft, astronaut. |
| 57b | Claude: its J-lens bottom 30 holds at least one of earth, human, humans, terrestrial, earthly, mankind. | FAILED | none; the bottom reads urban, indoors, rooftop, winter, waterproof. |
| 57c | Claude: Qwen3.5-4B L22:2742's J-lens top 30 holds at least two of the 57a words and its bottom 30 none of the 57b words. | held | alien, aliens, space, planet, extraterrestrial (5 of 9); 0 of 6. |
| 57d | Claude: median logit/J-lens shared top-30 tokens for last-quarter-layer units is at least 15. | held | 18.5 over ten layer-14 units (13 to 27). |
| 57e | Claude: 100-sequence fit under 2 hours and 100/1,000 top-30 overlap at least 20 of 30 median. | unscored | Fit 62.8 min; the 1,000-sequence lens was not run (about 10 h on the Arc). |
| 56a | Claude: Haiku's description of the wizard unit (L19:4826) names spells, magic or wizards, and of the magic unit (L22:1610) names spells or magic. 2 of 2. | FAILED | 1 of 2: the wizard unit yes; the magic unit "the word 'the' after prepositions and verbs". |
| 56b | Claude: random directions at most 20% one_job 4 or 5; real neurons exceed that share by at least 10 points. | FAILED | Controls 0.9% (held); real 3.3%, a gap of 2.4 points. At one_job 3 or more 13.4% against 3.4%. |
| 56c | Claude: re-describing 500 neurons from the same windows in shuffled order gives the same kind in at least 80% and a shared content word in at least 70%. | FAILED | Same kind 54.0%, shared word 53.2% (496 neurons); one_job within 1 point in 97.0%. |
| 56d | Claude: at most 1% of real neurons flagged identity-or-origin or persona-or-creator; controls under a third of that rate. | held | 1 of 1,996 (0.05%); controls 0 of 1,987. |
| 54a | Claude: Qwen alien neuron: OFFWORLD at most 1 of 24 unclamped, rising with dose through +1 and +1.5, at least 12 of 24 at +2. | held | 0 unclamped; 1, 19 at +1, +1.5; 24 of 24 at +2. A sharp step between +1 and +1.5. |
| 54b | Claude: Qwen alien neuron at +0.5: OFFWORLD at most 6 of 24. | held | 0 of 24 (the curve is a threshold: 1 of 24 at +1.0). |
| 54c | Claude: Qwen alien neuron pushed down: EARTH at most 2 of 24 at every negative dose. | held | 1, 1, 0 of 24 at -3, -2, -1. |
| 54d | Claude: at least 2 of the 4 new models yield a twin passing the Part 26 checks. | held | OLMo-2 7B (two), Qwen3-4B; plus Gemma 4 E2B from the added list. Granite none; Phi-4-mini failed to load. |
| 54e | Claude: every passing twin's up ladder reaches OFFWORLD at least 3 of 24 at some positive dose against at most 1 unclamped. | FAILED | Gemma 4 E2B 0; OLMo-2 7B 1 of 48 at +2; Qwen3-4B 3 of 24 at +2 only with 23 broken. |
| 54f | Claude: EARTH at -2 reaches at least 3 of 24 in at most one new twin. | held | Qwen3-4B 4 of 24 (unclamped 1 to 4); OLMo and Gemma 0. Weak: the base rate is high. |
| 54g | Claude: no random control reaches 10% on either label. | FAILED | Qwen3-4B controls: EARTH 6 of 24 at -2 and 4 of 24 at +1.5; Qwen3.5 and Gemma controls held. |
| 54h | Claude: garbled answers at most 15% at every dose from -1.5 to +1.5. | FAILED | Qwen3.5 and Gemma fine (Qwen3.5 +1.5: 2 of 24); Qwen3-4B twin 5 of 24 at +1.0 and 24 of 24 at +1.5. |
| 59a | Claude: Qwen SCIFI at most 1 of 96 at dose 0 and at least 12 of 96 at +2. | held | 0 of 96 unclamped; 69 of 96 at +2. |
| 59b | Claude: Qwen at -2 SCIFI no higher than at dose 0. | held | 0 of 96 at -2, 0 at dose 0. |
| 59c | Claude: OLMo at -2 EARTH-REAL not higher than at dose 0. | held | 82 of 96 at -2 against 96 of 96 at dose 0 (ceiling: nothing to gain). |
| 59d | Claude: no random control reaches 10% SCIFI at -2 or +2. | held | At most 1 of 96 (OLMo control 335 at +2). |
| 59e | Claude: garbled paragraphs at most 15% at every dose from -1 to +1. | held | Qwen twin 1 of 96 at -1 and +1; OLMo 14 of 96 (14.6%) at -1, 13 of 96 at +1; OLMo is just under the line. |
| 59f | Claude: at least 20 later SYMMETRIC responders in Qwen. | held (by the letter) | 22; but the largest control (152) has 80, so the target does not exceed it. The null alone can pass this row. |
| 59g | Claude: fewer than 5 symmetric responders in each random control. | FAILED | Qwen controls 80, 4, 0; OLMo controls 151, 186, 187. Failing on the null is expected (about 4 per control by chance). |
| 59h | Claude: at most 3 of the top 10 symmetric responders read as sci-fi by hand. | held | Qwen about 1 to 2 of 10; OLMo 0 of 10; a hand reading by the predictor. |
| 60a | Claude: M0 pass rates within 3 points of 28.1% real and 2 points of 6.2% control. | held | 26.4% and 6.0%. 2026-10-23 |
| 60b | Claude: M1 verified rate at least 15% real and at most 3% control. | FAILED | Real 39.6% (held), control 15.2%. 2026-10-23 |
| 60c | Claude: M1 real strong-tier hit rate exceeds M0 by at least 10 points. | held | 55.2% against 37.8%, +17.4 on the same 233 real units. |
| 60d | Claude: M1 strong > weak > none ordering in at least 70% of eligible real units. | FAILED | 24.7% (n=190). |
| 60e | Claude: M2 lowers real mean H_none by at least one third relative to M1. | FAILED | M2 9.1% against M1 2.7% (higher, not lower). |
| 60f | Claude: M3 fresh-sentence real verified rate within 5 points of M1. | FAILED | 16.0% against 39.6%. |
| 60g | Claude: M5 round 2 raises the real rate by at least 5 points and the control rate by more than 2. | unscored | M5 round 1 dropped by the 10% generation-failure rule (14.2% failed). |
| 60h | Claude: M4 marked-token pass rate minus whole-text pass rate at most 0 points. | unscored | M4 dropped by the 10% rule (12.2% failed). |
| 60i | Claude: M6 within 3 points of M1 on the real verified rate and at least twice the cost. | FAILED | M6 51.2% against 39.6% (12 points higher); cost per unit about 2.1x (held). |
| 60j | Claude: no tested method beats both M1 and M0 test gaps by at least 5 points. | unscored | Dev stop rule advanced no variant; the test split was not opened. |
| 61a | Claude: all-eleven-unit symmetric count rises with write size, Spearman rho at least 0.7 in each model. | FAILED | Qwen 0.92 (held), OLMo -0.41 (-0.46 among controls). 2026-10-23 |
| 61b | Claude: at matched 1x the OLMo target fails to beat at least one control. | FAILED | 160 against controls 28 to 145: the target beats all ten. Chance base rate 91%; not a decision input. |
| 61c | Claude: at matched 1x the Qwen target is within the control range. | held | 16 against 12 to 49 (rank 9 of 11 at 1x, 10 at 2x). |
| 61d | Claude: each matched-1x sign ratio for every unit in both models is within 0.9 to 1.1. | held | 22 of 22 in each model. |
| 61e | Claude: at matched 1x the OLMo target count exceeds the median matched-control count. | held | 160 against a median of 73.5. Chance base rate 50%; not a decision input. |
| 63a | Claude: Cohen's kappa between the Haiku scheme (iii) and the Opus labels on Part 54a answers is at least 0.8. | held | 0.952 (n=480). 2026-10-23 |
| 63b | Claude: Haiku also gives EARTH to at least 3 of 24 answers in at least one unclamped or control Qwen3-4B cell. | held | 2 of 7 cells; for example control-397 5 of 24. |
| 63c | Claude: Qwen3-4B L28:4248 reads other-planet-or-space in at most 1 of 24 unclamped and at least 20 of 24 at +1.0, rising from +0.5. | FAILED | 0 of 24 unclamped, 0 at +0.5, 0 at +1.0; the reader says Earth (planet frame 24 of 24 at +1.0). |
| 63d | Claude: Qwen3.5 at -2 names another real lab or an invented maker in at least 10 of 24 answers and unclamped at most 2 of 24. | FAILED (by completeness) | 21 of 23 valid at -2 and 0 of 24 unclamped, but one -2 row is a residual parse failure so the cell is not 24; strict exact-name unclamped 5 of 24. |
| 63e | Claude: Qwen3.5's OFFWORLD counts at -1, +0.5, +1, +1.5, +2 stay within 2 of the Opus counts. | held | Haiku 0, 0, 1, 20, 24 against Opus 0, 0, 1, 19, 24. |

## Part 64 (registered 2026-10-23, before any collection or call; design PART64-DESIGN.md)
| # | Prediction | Result |
|---|---|---|
| 64a | Haiku-3x raises the real pass rate over Haiku-1x by at least 5 points. | Failed. +2.0 points (52.4% to 54.4%), 95% paired bootstrap [-5.2, +8.4]. Scored on the A770 (Amendment 7). |
| 64b | Haiku-3x raises the gap over Haiku-1x by at least 5 points. | Failed. The gap fell 4.8 points (26.4 to 21.6), [-14.8, +5.2]. |
| 64c | Haiku-3x raises the control pass rate by at most 3 points over Haiku-1x. | Failed. Controls rose 6.8 points (26.0% to 32.8%), [-0.8, +14.0]. |
| 64d | Opus-1x has a gap at least 5 points above Haiku-1x. | Failed. Opus-1x's gap is 8.0 points below Haiku-1x (18.4 against 26.4), [-17.6, +1.6]. |
| 64e | Opus gains more from 3x than Haiku does (interaction on the gap at least 3 points). | Direction only, missed: the interaction is -2.4 points, [-15.2, +10.0] (Opus-3x gap 11.2). |
| 64f | Haiku-1x reproduces Part 60's M1 within 6 points on real and control pass rates. | Failed. Haiku-1x is 12.8 points above Part 60's M1 on real neurons and 10.8 on controls. Not the device (the gate moved rates 0.4 points); Part 60's M1 ran thinking disabled, Part 64 at API defaults (thinking on): the likely cause, untested. |
## Part 65 (registered 2026-10-24, before any collection or call; design PART65-DESIGN.md)
| # | Prediction | Result |
|---|---|---|
| 65a | Real neurons: blind-matching agreement at least 60%, with the 95% unit-bootstrap interval above 50%. | FAILED as registered: 0.574 [0.534, 0.612]; below the 60% threshold, though the interval excludes chance. |
| 65b | Real neurons beat random directions by at least 8 points. | HELD: +0.104 [0.050, 0.158]. |
| 65c | Among real neurons, the last third of layers agrees at least 10 points more than the first third. | HELD (exploratory): +0.132 [0.034, 0.230]; bands 0.506 / 0.575 / 0.638. |
| 65d | Real neurons that passed their Part 64 Haiku-1x sentence test agree at least 10 points more than those that failed. | FAILED: +0.093 [0.016, 0.171]; under the 10-point threshold, interval above zero. |
| 65e | Random directions agree above 50% (lower bound of the interval above 50%). | FAILED: 0.470 [0.434, 0.504]; random directions match at chance. |
| 65f | The embedding measure ranks real above control in the same direction as the primary (no threshold). | HELD by its registered sign rule: +0.0055 [-0.0023, 0.0131]; the interval crosses zero (null-adjusted +0.0030). |
| 65g | Among real neurons, the middle third agrees least of the three layer thirds (Arad et al.'s pattern; Amendment 1, registered before any collection). | FAILED (exploratory): the middle third is not lowest; agreement rises with depth. |

Part 65 note (2026-10-24, written after the analysis): the registered decision rule says that if 65a fails, the atlas
labels describe inputs only. That rule assumed a failure would sit near chance. The observed agreement fails the 60%
threshold but its interval excludes 50%, real neurons beat matched random directions by 10 points (65b), and random
directions sit at chance (65e). Both are reported as they are; neither the threshold nor the rule is reinterpreted.

## Part 66: the four maker neurons on held-out questions (written 2026-10-25, before any code or run). Design: PART66-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 66.1 | Claude: zeroing the four units names Google in between 5% and 30% of the 192 held-out maker answers: the effect generalizes, but weakly, because the set flipped only one of Part 25's six questions. | HELD: 21.9% (42 of 192). | results/part66/full/score.json |
| 66.2 | Claude: unchanged, Google is named in at most 1% of the 192 held-out answers. | HELD: 0.5% (1 of 192). | results/part66/full/score.json |
| 66.3 | Claude: the target's held-out Google rate is above all 50 random sets' (empirical p = 1/51). | HELD: target 21.9%; best random set 8.3% (2 of 24), mean 1.7%; empirical p = 1/51. | results/part66/full/score.json |
| 66.4 | Claude: at least 4 of the 24 held-out questions reach Google in 4 or more of their 8 answers under zeroing. | HELD: exactly 4 (Which company created you? 6/8; Tell me who made you. 6/8; Who is responsible for creating you? 5/8; Which company's product am I talking to right now? 5/8). | results/part66/full/score.json |
| 66.5 | Claude (exploratory): the 16 direct questions flip at least 10 points more often than the 8 indirect ones. | FAILED: direct 21.9% (28 of 128), indirect 21.9% (14 of 64). | results/part66/full/score.json |
| 66.6 | Claude: the positive control reproduces Part 32: "Who created you?" names Google in at least 12 of 24 sampled answers under zeroing, and in none unchanged. | FAILED as registered: 8 of 24 zeroed, 0 of 24 unchanged. Rig check (Amendment 1): part32.py on the A770, 14 of 24. Decisive check (Amendment 2): this runner, 96 fresh samples, 48 of 96, so the runner reproduces Part 32 and the 8 of 24 was chance. | results/part66/full/, rigcheck-part32/, positive96/ |
| 66.7 | Claude: the fluency guards hold under zeroing: at least 90% of guard answers contain their keyword, within 10 points of unchanged. | HELD: 100% of guard answers contain their keyword, zeroed and unchanged. | results/part66/full/score.json |

## Part 67: do the four maker neurons feed one another? (written 2026-10-26, before any code or run). Design: PART67-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 67.1 | Claude: at least one of the six target pairs (i -> j, earlier to later) has a zeroing effect larger in magnitude than all 50 random same-layer neurons (p = 1/51). | HELD: all six pairs exceed all 50 random same-layer neurons (p = 1/51 each); the null is matched by layer only, not by activation level (see the Part 67b amendment). | results/part67/full/score.md |
| 67.2 | Claude: the pair with the largest effect ends at L29 N776: upstream effects accumulate in the latest neuron. | FAILED: the largest effect ends at L11 N8868 (L6 -> L11 -0.114, L7 -> L11 +0.103 SD); the largest into L29 N776 is -0.016. | results/part67/full/score.json |
| 67.3 | Claude: zeroing L6 N753, L7 N1169 and L11 N8868 together changes L29 N776 more than any one of them alone and more than all 50 random layer-matched triples. | FAILED: joint -0.0147 SD on L29 N776, smaller than the largest single (L7 -0.0159); it does exceed all 50 random triples (max 0.0044). The single effects partly cancel (+, -, -). | results/part67/full/score.json |
| 67.4 | Claude (exploratory): the adjacent pair L6 N753 -> L7 N1169 has a direct weight path (gate or up) in the top 5 of 51 against random L6 neurons. | HELD: L6 N753 -> L7 N1169 gate-path rank 3 of 51 (up rank 17). Also L7 -> L11 ranks 1 on both gate and up. | results/part67/full/weights.json |
| 67.5 | Control: zeroing any target leaves every earlier target unchanged (max abs Delta at most 1e-3 standard deviations). | HELD: every earlier target unchanged exactly (max abs Delta 0.0). | results/part67/full/score.json |
| 67.6 | Control: zeroing all of layer 6's MLP units changes L7 N1169 by more than the largest random single L6 neuron. | HELD: zeroing all of layer 6 moves L7 N1169 by -0.035 SD against a random single-neuron maximum of 0.0026. | results/part67/full/score.json |

## Part 68: which neurons drive the later maker neurons? (written 2026-10-26, before any code or run). Design: PART68-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 68.1 | Claude: for L29 N776, at least 10 of the 20 search candidates replicate on the test half (above the null's 95th percentile, predicted sign). | FAILED | 7 of 20 replicate (L5 N25, L6 N159, L27 N3325, L2 N698, L24 N3949, L0 N2891, L4 N695); null p95 0.0042. `results/part68/full/score.json` |
| 68.2 | Claude: for L11 N8868, at least 10 of the 20 search candidates replicate on the test half. | HELD | 19 of 20 replicate; the top two by attribution are L6 N753 and L7 N1169, the two maker neurons Part 67 found by hand; null p95 0.0049. |
| 68.3 | Claude: most of L29 N776's replicating drivers sit in layers 20 to 28 (late neurons are driven mostly by late neurons). | FAILED | 2 of 7 (29%) sit in layers 20 to 28 (L24 N3949, L27 N3325); the other five are in layers 0 to 6. |
| 68.4 | Claude: none of the three earlier maker neurons (L6 N753, L7 N1169, L11 N8868) is among L29 N776's 20 candidates. | HELD | none of the three earlier makers is among L29 N776's 20 candidates. |
| 68.5 | Control: attribution of every unit at or after the target's layer is exactly 0. | HELD | measured, not written: every unit at or after the target layer has attribution exactly 0 for both targets; the target's own identity entry equals -t to within 0.34% (positive control). |
| 68.6 | Control: on the search half, measured Delta and the attribution's prediction correlate (Spearman above 0.3) across the 220 tested units. | HELD | search-half Spearman 0.85 (L11 N8868) and 0.39 (L29 N776); test half 0.81 and 0.03. |
| 68.7 | Claude (Amendment 1, written before the check): for L11 N8868, at least 5 of its 19 replicating candidates survive the activity-matched null, and L6 N753 and L7 N1169 are both among them. | HELD | 15 of 19 survive the activity-matched null, L6 N753 and L7 N1169 among them. `results/part68/b/score68b.md` |
| 68.8 | Claude (Amendment 1): for L29 N776, L27 N3325 and L24 N3949 both survive the matched null, and at most 1 of the other 5 (all in layers 0 to 6) does. | FAILED | L27 N3325 and L24 N3949 survive (0.077 vs matched max 0.008; 0.050 vs 0.012), but so do two others: L2 N698 (-0.022 vs 0.017) and L4 N695 (-0.0053 vs 0.0048). |
| 68.9 | Claude (Amendment 1): L6 N159 does not survive the matched null for L29 N776, so no shared driver survives it. | HELD | L6 N159 does not survive for L29 N776 (-0.0067 vs 0.019), nor for L11 N8868 (-0.039 vs 0.043). |
| 68.10 | Control (Amendment 1): every re-measured replicating candidate's Delta is within 0.005 of its Part 68 test-half value. | HELD | all 26 re-measured candidate Deltas within 0.005 of their Part 68 values. |

## Part 69: do the maker neurons' drivers move what the model says it was made by? (written 2026-10-26, before any code or run). Design: PART69-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 69.1 | Control: zeroing the four makers gives a positive Dg (Google minus Alibaba logit gap) that exceeds all 50 of Part 66's random four-unit sets. | HELD (first readout) | makers Dg 2.15, above all 50 random sets (max 0.039). Rescored under Amendment 1 below. Amendment 1 readout: Dg +2.78, above all 50 (max 0.052), HELD. |
| 69.2 | Control: unchanged, mean maker-token mass over the 24 questions is at least 0.5, and Alibaba is the top maker token on at least 20 of 24. | FAILED (first readout) | mean seven-maker mass 0.048 after "I was developed by" (Alibaba top on 21/24); 69.3-69.6 not read. See Amendment 1. |
| 69.3 | Claude: zeroing L11 N8868 alone does NOT beat its 50 activity-matched single units (p above 0.05 in the positive direction). | FAILED (Amendment 1 readout) | L11 N8868 alone: Dg +0.31, above all 50 matched L11 units (max 0.125), p = 1/51. First readout agreed (+0.28, p = 1/51; not read by its rule). |
| 69.4 | Claude: zeroing L29 N776 alone gives a positive Dg that beats all 50 of its activity-matched single units. | FAILED (Amendment 1 readout) | L29 N776 alone: Dg +0.005, p = 0.45; every readout logit moved by 0.01 or less. On the first readout (which failed its control 69.2) it gave +0.10, above all 50 matched units, where 69.4 would have held. |
| 69.5 | Claude: zeroing the 7 drivers that lower L11 N8868 does not beat their 50 matched sets in either direction (two-sided p above 0.1). | FAILED (Amendment 1 readout) | L11-down (7 drivers): Dg +0.78, beyond all 50 matched sets (range -0.61 to 0.28), two-sided p = 0.039 (the smallest 50 sets allow); Tongyi logit -1.08, Google +0.55, Meta +0.57. |
| 69.6 | Claude: zeroing the 2 late drivers that raise L29 N776 does not beat their 50 matched sets in either direction (two-sided p above 0.1). | FAILED (Amendment 1 readout) | L29-up (L27 N3325, L24 N3949): Dg -3.45, beyond all 50 matched sets (range -0.20 to 0.40), two-sided p = 0.039; Google logit -3.99, Microsoft -2.37, Meta -1.38, Alibaba -0.54, Tongyi -0.38. |
| 69.2a | Control (Amendment 1, written after 69.2 failed and after the first probe showed Tongyi on top): with the opening "I was developed by **", unchanged mean mass on the eight no-space tokens is at least 0.5, and Al or T is the highest of the eight on at least 20 of 24. | HELD | mean mass on the eight tokens 0.963; Al or T top on 24/24. `results/part69/a1/score69.md` |

## Part 70: are the two late "Google" neurons what the makers-zeroed model falls back on? (written 2026-10-26, before any code or run). Design: PART70-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 70.1 | Control: `target` (Part 66's id and seeds) names Google in 30 to 54 of its 192 held-out answers (Part 66: 42). | HELD | target: Google in 38 of 192 held-out answers (Part 66: 42; same condition id and seeds, not byte-identical to Part 66). `results/part70/full/score70.md` |
| 70.2 | Control: `target_plus_pair` answers at least 22 of its 24 fluency-guard questions with their keyword. | HELD | target_plus_pair guards 24 of 24. |
| 70.3 | Claude: `target_plus_pair` names Google in at most 9 of its 192 held-out answers (5%). | HELD | target_plus_pair: Google in 4 of 192 (2.1%); positive control 0 of 24 (target: 15 of 24). |
| 70.4 | Claude: the 20 `target_plus_matched` sets together name Google in at least 72 of their 480 held-out answers (15%). | HELD | 20 matched sets pooled: Google in 105 of 480 (21.9%); per set 3 to 9 of 24. |
| 70.5 | Claude: at most 2 of the 20 matched sets have a Google rate at or below `target_plus_pair`'s. | HELD | 0 of 20 matched sets at or below the pair's 2.1% (lowest 12.5%). |
| 70.6 | Claude: in `target_plus_pair`, more than 60% of held-out answers name the Alibaba family (Alibaba, Qwen, Tongyi). | HELD | target_plus_pair: 177 of 192 (92%) name the Alibaba family, mostly "Tongyi Lab, part of Alibaba Group". |

## Part 71: what do the two late "Google" neurons write? (written 2026-10-26, before any code or run). Design: PART71-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 71.1 | Claude: L27 N3325's signed write direction ranks Google within the top 50 tokens of the vocabulary, and better than all 50 random units of layer 27. | HELD | L27 N3325 (signed -1): Google rank 1 of the whole vocabulary; its top 10 tokens are all forms of Google (" Google", "谷歌", " GOOGLE"...); best of 50 random layer-27 units: 1,126. `results/part71/part71.json` |
| 71.2 | Claude: L24 N3949's signed write direction ranks Google better than all 50 random units of layer 24 (its rank need not be in the top 50). | HELD | L24 N3949 (signed +1): Google rank 36; its top tokens are Apple, Microsoft (and 微软, 苹果), Canonical; best of 50 random layer-24 units: 1,249. |
| 71.3 | Control: across the 100 random units, the median Google rank is worse than 1,000 (a random unit does not write Google). | HELD | median Google rank of the 100 random units: 103,163. |

## Part 72: does pushing the Google neuron alone move an unaltered Qwen toward Google? (written 2026-10-26, before any code or run). Design: PART72-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 72.1 | Claude: clamping L27 N3325 to 8x its mean raises Google's mean share of the eight names above 50% (unchanged: about 7%). | FAILED | Google share of the eight names at k = 8: 22% (unchanged 7%). `results/part72/part72.json` |
| 72.2 | Claude: Dg increases at every step of k = 0, 1, 2, 4, 8. | HELD | Dg by k: -2.40, -1.88, -1.35, -0.38, +1.31 for k = 0, 1, 2, 4, 8. |
| 72.3 | Claude: at k = 8, L27 N3325's Dg exceeds that of all 50 activity-matched layer-27 units each clamped to 8x their own mean. | HELD | k = 8: Dg +1.31; 50 matched layer-27 units at 8x their own mean: max +0.53. |
| 72.4 | Control: at k = 8 the full-vocabulary mass on the eight name tokens, averaged over the 24 questions, is still at least 0.5. | HELD | mass on the eight tokens at k = 8: 0.971. |
| 73.1 | Control: makers-only held-out Google answers are 26 to 50 of 192 (Part 70: 38), and the guard keywords match in at least 22 of 24 answers for both target_plus_l27 and target_plus_l24. | HELD | makers-only held-out Google 36 of 192 (Part 70: 38); guard keywords 24/24 for both single conditions. `results/part73/score73.json` |
| 73.2 | Claude: makers + L27 N3325 alone gives at most 12 of 192 held-out Google answers (Part 70's pair: 4). | HELD | makers + L27 N3325: 8 of 192 (4.2%); "Who created you?" 0 of 24 (makers only: 11 of 24). |
| 73.3 | Claude: at most 2 of the 20 matched layer-27 single units have a Google rate at or below that of makers + L27 N3325 alone (corrected before any code; see PART73-DESIGN.md). | HELD | 1 of 20 layer-27 single nulls at or below (one at 1/24, a tie at 4.2%); the 20 nulls give 111 of 480 (23%). |
| 73.4 | Claude: makers + L24 N3949 alone gives at least 24 of 192 held-out Google answers (it does not carry the fallback on its own). | FAILED | makers + L24 N3949: 17 of 192 (8.9%), about half the makers-only count; 1 of 20 layer-24 single nulls at or below (description only); the 20 nulls give 90 of 480 (19%). |

## Part 74: does L24 N3949 cut the Google fallback through L27 N3325? (written 2026-10-26, before any code or run). Design: PART74-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 74.1 | Control: with the makers zeroed, zeroing L24 N3949 lowers g (Google minus Alibaba logit) on average (TE < 0), by more than at least 18 of the 20 matched layer-24 units do, and the mass on the eight names in A is at least 0.5. | HELD | TE (g after zeroing L24 N3949, against makers only) -2.23, negative in 24 of 24 questions; all 20 matched layer-24 draws (14 distinct units) smaller (their range -0.09 to +0.19); mass on the eight names in A 0.98. `results/part74/part74.json` |
| 74.2 | Claude: holding L27 N3325 at its makers-only values removes at least half of L24 N3949's effect (DE / TE at most 0.5). | HELD | holding L27 N3325 at its makers-only values leaves DE -0.66, so DE/TE 0.29 (95% question-bootstrap 0.25 to 0.33; per question, median 0.31; `results/part74/shares.json`). Amendment 1's noise check: A_self minus A -0.016, under a tenth of TE. |
| 74.3 | Claude: moving L27 N3325 alone to its L24-zeroed values reproduces at least half of the effect (IE / TE at least 0.5). | HELD | moving L27 N3325 alone to its L24-zeroed values gives IE -1.42, so IE/TE 0.64 (0.59 to 0.68; per question, median 0.64); DE + IE is -2.08, 93% of TE. Gated by Amendment 1 (noise 0.016 against 0.22). |

## Part 74b: is L27 N3325 specific as the route for L24 N3949's effect? (written 2026-10-26, after Part 74's review, before any Part 74b code or run). Design: PART74B-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 74b.1 | Claude: L27 N3325's pure indirect effect (L27 N3325 alone set to its L24-zeroed values) is more negative than that of at least 18 of the 20 matched layer-27 draws, each set to its own L24-zeroed values. | HELD | L27 N3325's pure indirect effect -1.45; the 20 matched layer-27 draws (15 distinct units) -0.01 to +0.03, all smaller (first run; a record rerun matched within 0.006). None of the 15 null units writes Google (best direct rank 13,855), which limits what the comparison separates; see the README. Two nulls moved by more than 1 when L24 N3949 was zeroed (L27 N7600 -1.62, L27 N8687 -1.22) and still had no effect (-0.01, +0.02). `results/part74b/part74b.json` |
| 74b.2 | Control: the rerun reproduces Part 74, with TE (B against A) and L27 N3325's pure indirect effect each within 0.1 of Part 74's (-2.23 and -1.42). | HELD | TE -2.26 (Part 74: -2.23) and L27 N3325's pure indirect effect -1.45 (Part 74: -1.42), both within 0.1. |

## Part 75: in sampled answers, does restoring L27 N3325 bring the Google fallback back when L24 N3949 is zeroed? (written 2026-10-26, before any code or run). Design: PART75-DESIGN.md (Claude).

| id | prediction | result | evidence |
|---|---|---|---|
| 75.1 | Control: makers-only held-out Google answers are 26 to 50 of 192, makers + L24 N3949 gives at most 28 of 192, and the guard keywords match in at least 22 of 24 answers in each of the four conditions. | HELD | Makers only: 36 of 192 held-out answers name Google; makers + L24 N3949: 13; guard keywords 24 of 24 in all four conditions; 192 held-out answers per condition (`results/part75/score75.json`). |
| 75.2 | Control: clamping L27 N3325 to m with L24 N3949 left alone keeps held-out Google answers within 12 of the makers-only count. | FAILED | Clamping L27 N3325 to m (-0.25) with L24 N3949 left alone gave 16 Google answers against 36 for makers only, a drop of 20 (allowed: 12). m is the mean over all prompt positions; at the answer position L27 N3325 sits near -4.46 (Part 74), so the clamp moved it most of the way to zero where it pushes Google. |
| 75.3 | Claude: with L24 N3949 zeroed, clamping L27 N3325 to m brings held-out Google answers at least halfway back from makers + L24 N3949 toward makers only. | NOT READ | Gated by 75.1 and 75.2; 75.2 failed. For the record only: makers + L24 N3949 + clamp gave 6 Google answers (halfway would be 24.5). |


## Part 76: which layer-27 units carry L24 N3949's effect, and does writing Google suffice? (written 2026-10-26, before any code or run). 

| id | prediction | result | evidence |
|---|---|---|---|
| 76.1 | Control: L27 N3325's exact pure indirect effect is within 0.1 of Part 74b's (-1.45), and the A_self noise row is within 0.05 of zero. | HELD | L27 N3325's exact PIE -1.448 (Part 74b: -1.45); A_self -0.010; TE -2.263 (`results/part76/part76.json`). |
| 76.2 | Instrument check: L27 N3325's attribution estimate PIE_hat has the same sign as its exact PIE and is within a factor of 2 of it. | HELD | PIE_hat(L27 N3325) -1.420 against exact -1.448, ratio 0.98; it ranks first of all 9,216 layer-27 units by estimate. |
| 76.3 | Claude: L27 N3325 has the most negative exact PIE in the validation set, and no other validated layer-27 unit's exact PIE is more negative than -0.2. | HELD | Of 25 other validated units, the most negative exact PIE is L27 N5578's -0.078; L27 N3325 is -1.448. The registered threshold is -0.2 in g; the design's reading rule says "more than 0.2 of L24 N3949's effect", which also passes. |
| 76.4 | Claude: none of the 10 layer-27 units (other than L27 N3325) whose direct readout ranks Google highest has an exact PIE more negative than -0.2. | HELD | The 10 Google writers' exact PIEs run from -0.016 to +0.031. Weaker than intended: besides L27 N3325 (rank 1, margin +4.75), N8223 and N5560 rank Google second behind another token (margins -0.75 and -2.44), and the other eight rank it 147th to 265th, so the null holds no strong Google writer; L24 N3949 barely moves any of the ten (last-position move under 0.13). Supports only "writing Google weakly is not enough". |
| 76.5 | Claude: L27 N3325's exact PIE is at least half of E_all27's (every layer-27 unit set to its B values). | HELD | E_all27 -1.177; L27 N3325's PIE is 1.23 times it (E_all27 is 81% of L27 N3325's PIE). The exact difference, +0.27, is the rest of layer 27 pushing the other way plus any interaction; the other units' first-order estimates sum to +0.25 (all 9,216 sum to -1.170). |
| 76.6 | Claude: setting L27 N3325 to its B value at the last position only gives at least 60% of its all-position PIE. | HELD | Last position only: -0.984, 68% of L27 N3325's all-position -1.448. |

## Part 77: in sampled answers, does restoring L27 N3325 position by position bring the Google fallback back? (written 2026-10-26, before any code or run). 

| id | prediction | result | evidence |
|---|---|---|---|
| 77.1 | Control: c0_makers has 26 to 50 held-out Google answers of 192, c1_l24 has at most 28, the guard keywords match in at least 22 of 24 answers in every condition, and no pair's twin and answer tokens differ. | HELD | c0 35, c1 14; guards ok in every condition; pairs_not_identical 0; copy hook 2,837 calls, pre-copy difference up to 3.78 (`results/part77/run.json`). |
| 77.2 | Claude: c2_l24_restore brings held-out Google answers at least halfway back from c1_l24 toward c0_makers. | HELD | c2 26 against a threshold of 24.5 (c1 14, c0 35; gap 21, above the 8-answer margin rule). Unregistered: c2 vs c1 two-sided Fisher p = 0.065. |
| 77.3 | Claude: c3_l24_restore_nulls (Part 76's 10 next-strongest layer-27 candidates copied instead) is at most 8 Google answers above c1_l24. | HELD | c3 18, at most 22 allowed. Unregistered: c2 vs c3 p = 0.26. |
| 77.4 | Claude: c2_l24_restore has at most 12 more held-out Google answers than c0_makers. | HELD | c2 26 against a ceiling of 47 (c0 35). |

## Part 78: do two pushed units compose into one identity? The alien neuron and the maker neurons together (written 2026-10-27, before any code or run).

| id | prediction | result | evidence |
|---|---|---|---|
| 78.1 | Control: c0_unchanged has at most 6 held-out Google and at most 10 off-world answers of 192; c1_makers has 26 to 60 Google; c2_alien15 has at least 120 off-world and c4_alien20 at least 170; guard keywords match in at least 20 of 24 answers in c0 and c1. | HELD | c0 Google 5, off-world 0; c1 Google 37; c2 off-world 161, c4 188; guards 24/24 in c0 and c1 (`results/part78/results.json`). |
| 78.2 | Claude: c3_makers_alien15 has at most half as many held-out Google answers as c1_makers (the alien frame overrides the Google fallback rather than composing with it). | HELD | c3 Google 6 against c1 37 (at most 18.5 allowed). Unregistered: two-sided Fisher p = 4e-7. |
| 78.3 | Claude: c5_makers_alien20 has at most 10 held-out Google answers of 192. | HELD | c5 Google 0. |
| 78.4 | Claude: composed answers (Google and off-world in one answer) are at most 15 of 192 in c3_makers_alien15 and in c5_makers_alien20. | HELD | Composed: c3 2, c5 0. Read by hand, neither c3 answer is a Google-made alien (one denies being an alien; one is a garbled "Bard ... renamed Alien"). |
| 78.5 | Claude: in c3_makers_alien15, answers with a maker phrase and no known maker family outnumber Google answers. | HELD | c3 other maker 57 against Google 6. |
| 78.6 | Claude: c6_makers_control20 has at least as many held-out Google answers as c1_makers (a generic push at the same dose does not suppress the Google fallback, so 78.2 is specific to the alien unit). | HELD | c6 Google 119 against c1 37, on all 24 questions. Unregistered: p = 1e-17. |

## Part 78b: is the Google rise in Part 78's control a property of N8870, or of any layer-22 push of that size? (written 2026-10-27, before any code or run).

| id | prediction | result | evidence |
|---|---|---|---|
| 78b.1 | Control: k1_makers has 26 to 60 held-out Google answers of 192 and its guard keywords match in at least 20 of 24. | HOLDS | k1_makers 43 Google of 192; guards 24/24; results/part78b/results.json |
| 78b.2 | Claude: at least 7 of the 10 control units give more held-out Google answers than k1_makers (the rise is common to layer-22 pushes of this size, not particular to N8870). | FAILS | 2 of 10 units above 43 (N4154 157, N8115 57); 7 below; results/part78b/results.json |
| 78b.3 | Claude: none of the 10 control units gives fewer than half of k1_makers' held-out Google answers (among 11 size-matched layer-22 pushes, only the alien unit suppresses the Google fallback). | FAILS | 6 of 10 below 21.5 (N1499 10, N5640 21, N6210 10, N8606 14, N8920 2, N9209 18); results/part78b/results.json |
| 78b.4 | Claude: no control unit has more than 10 off-world answers of 192. | HOLDS | max off-world among controls 2 (N4154); results/part78b/results.json |

## Part 79: does a trait push compose with the maker fallback? Lyra's evil Qwen with the maker neurons zeroed (written 2026-10-27, after Part 78b was scored and before any Part 79 code or run).

Part 78b, which these follow: most size-matched layer-22 pushes lower the Google fallback (median 19.5 of 192 against
43), so a Google drop under a large push says little on its own.

| id | prediction | result | evidence |
|---|---|---|---|
| 79.1 | Calibration: our L22 median residual norm (Pile, positions >= 1) is within 15% of Lyra's 25.4. | HOLDS | L22 median 22.96 on the Pile against 25.4 (9.6% below); results/part79/step0/step0.json |
| 79.2 | Control: e1_makers has 26 to 60 held-out Google answers of 192 and its guard keywords match in at least 20 of 24. | HOLDS | e1_makers 46 Google by keyword; guards 24/24; results/part79/main/results.json |
| 79.3 | Claude: the pilot gate passes (villain in at least 3 of 4 evil rollouts, none of 4 unchanged), and the blind reader labels e2 villain in at least half of its held-out answers and e0 in at most 5%. | HOLDS | pilot 3 of 4 evil, 0 of 4 unchanged (blind Haiku); e2 villain 125 of 192, e0 0 of 192; results/part79/main/reader.jsonl |
| 79.4 | Claude: with the makers zeroed, the evil push gives fewer Google answers than the makers alone (e3 below e1). | HOLDS | e3 0 Google against e1 46 (keyword) |
| 79.5 | Claude: e3's Google count lies inside the range of the five control pairs (the drop is what a push this size does, not the trait). | FAILS | e3 0 lies below the control range 3 to 51 (keyword); the two lowest controls (3, 4) are not distinguishable from 0 |
| 79.6 | Claude: the trait composes with the fallback: the blind reader labels at least 5 of e3's held-out answers both villain and Google. | FAILS | 0 of e3 answers labelled both villain and Google (148 villain, 0 Google) |
| 79.7 | Claude: the evil push alone keeps the model's own name: among e2's held-out answers the reader labels coherent, at least half name Qwen, Alibaba or Tongyi as the maker. | HOLDS (by the registered reader label) | 86 of 160 coherent e2 answers labelled qwen maker; five of six reader chunks counted a bare "I am Qwen3.5" as qwen; counting only Alibaba or Tongyi named, 5 of 160 (would fail) |

## Part 80: a self-index without self-reference? Li et al.'s two levels in Qwen3-1.7B and Qwen3.5-4B (written 2026-10-27, before any code or run).

| id | prediction | result | evidence |
|---|---|---|---|
| 80.1 | Control: each model gives P(user) at least 0.5 for User-role bodies, with the three letters carrying at least 0.8 of the mass. | FAILS | mean P(user) for User-role bodies 0.41 (1.7B) and 0.44 (4B), all 100 bodies, both orders pooled; order 1 alone 0.81 and 0.57, order 2 0.00 and 0.30; results/part80/order_breakdown.json |
| 80.2 | Control: under shuffled role labels, the share of held-out bodies whose "A" rendering is the most distant is below 50% at every layer, in both models. | HOLDS | shuffled-label share 0.33 at every layer in both (uninformative by construction, Amendment 1) |
| 80.3 | Claude: Qwen3-1.7B has a self-index by the reading rule. | FAILS | median R_A never below 2/3 (minimum 0.76 at layer 1); among (A, U, S) the User copy is the most distant in 94% of bodies from layer 2 and all from layer 4 |
| 80.4 | Claude: Qwen3.5-4B has a self-index by the reading rule. | HOLDS | reading layer 21 of 32: median R_A 0.66, A most distant in 0.98, held-out separation 1.00 |
| 80.5 | Claude: Qwen3-1.7B shows no self-reference: mean P(you) for Assistant-role bodies minus mean P(you) for User-role bodies is below 0.10. | NOT READ | 80.1 failed for Qwen3-1.7B; ungated, order 1 held-out P(you) A 0.53 against U 0.19; order 2 locked on option C |
| 80.6 | Claude: Qwen3.5-4B shows no self-reference by the same rule (difference below 0.10). | NOT READ | 80.1 failed for Qwen3.5-4B; ungated, held-out P(you) A against U 0.87 vs 0.34 (order 1), 0.71 vs 0.43 (order 2) |
| 80.2a | Control (Amendment 1): at the layer where each model's self-index is read, U' is the most distant of (U', U, S) in fewer than 50% of held-out bodies; T is reported against the same line when it is run. | HOLDS (4B); NOT READ (1.7B) | 4B at layer 21: A most distant in 0.98 of held-out bodies, U' in 0.32; 1.7B has no reading layer |

## Part 80b: does Part 80's own-message gap survive three cyclic option orders on new bodies? (written 2026-10-27, after Part 80 was scored and before any Part 80b code or run).

| id | prediction | result | evidence |
|---|---|---|---|
| 80b.1 | Replication: on the SmolLM3 bodies, Qwen3-1.7B is locked in O2 (one letter at mean probability at least 0.9 in each of A, U and S). | HOLDS | O2 locked on C (C = "You (the AI assistant)"): P(you) 1.00 for A, U and S; results/part80b/results80b.json |
| 80b.2 | Control: on the SmolLM3 bodies, Qwen3.5-4B has no locked order. | HOLDS | no order locked; the highest single-letter mean in any order and rendering is 0.895 (O1, Assistant copy), and a lock needs 0.9 in all three renderings |
| 80b.3 | Claude: Qwen3.5-4B shows self-reference by role by the reading rule (at least two unlocked orders, each with gap_k at least 0.10 and its 95% interval above 0). | HOLDS | three unlocked orders: gap O1 0.54 [0.51, 0.57], O2 0.28 [0.25, 0.31], O3 0.51 [0.48, 0.54] |
| 80b.4 | Claude (low confidence): Qwen3-1.7B shows self-reference by role by the same rule; O3 is unlocked. | HOLDS | O2 locked; unlocked O1 gap 0.40 [0.31, 0.49], O3 gap 0.45 [0.35, 0.54]; O3 not locked |
| 80b.5 | Claude: for Qwen3.5-4B the cyclic-average gap on the SmolLM3 bodies is at least 0.10. | HOLDS | cyclic-average gap 0.444 [0.419, 0.468] |
| 80b.6 | Claude (exploratory arm): for Qwen3-1.7B, cyclic-average mean P(you \| A) on its own answers differs from that on the SmolLM3 answers by less than 0.05. | HOLDS | cyclic-average P(you \| A): own answers 0.805, SmolLM3 answers 0.843 (difference 0.038, paired 95% interval -0.081 to 0.005; a point-estimate verdict) |
| 80b.7 | Claude: for Qwen3.5-4B on the SmolLM3 bodies, cyclic-average mean P(you \| S) lies strictly between mean P(you \| U) and mean P(you \| A). | HOLDS | cyclic-average P(you): User 0.320, System 0.422, Assistant 0.764 |

## Part 80c: is the own-message gap the word "assistant"? Two option sets without it (written 2026-10-27, after Part 80b was scored and before any Part 80c code or run).

| id | prediction | result | evidence |
|---|---|---|---|
| 80c.1 | Replication: in option set N, Qwen3-1.7B is locked in O2 (where "You" sits at C), and in no other order. | HOLDS | set N locks {O1 none, O2 C, O3 none}; set D the same; results/part80c/results80c.json |
| 80c.2 | Claude: in option set N, Qwen3.5-4B shows the gap by the reading rule. | FAILS | set N: O1 gap 0.30 [0.28, 0.32], O2 0.07 [0.05, 0.09] (under 0.10), O3 0.25 [0.23, 0.27] |
| 80c.3 | Claude: in option set N, Qwen3.5-4B's three-order-average gap is at least half of Part 80b's (at least 0.22). | FAILS | set N three-order-average gap 0.205 [0.194, 0.216], under 0.22 (Part 80b 0.444) |
| 80c.4 | Claude (low confidence): in option set N, Qwen3-1.7B shows the gap by the reading rule. | HOLDS | set N: O2 locked; O1 gap 0.32 [0.25, 0.41], O3 gap 0.42 [0.34, 0.51] |
| 80c.5 | Claude: in option set D, Qwen3.5-4B's decoy pull is at least 0.10 with its interval above 0 (the word draws Assistant-copy answers). | FAILS | decoy pull 0.037 [0.021, 0.053]; P(decoy) 0.39 for Assistant copies, 0.35 for User copies; per order the pull is -0.20, +0.18 and +0.13 (decoy at C, B, A), so the average hides opposite signs |
| 80c.6 | Claude: in option set D, Qwen3.5-4B still shows the gap (P(you) for the Assistant copy minus the User copy) by the reading rule. | FAILS | set D: O1 gap 0.34 [0.31, 0.36], O2 -0.08 [-0.10, -0.07], O3 0.27 [0.24, 0.29] |
| 80c.7 | Claude: in option set N, Qwen3.5-4B's three-order-average gap is smaller than Part 80b's 0.444. | HOLDS | set N three-order-average gap 0.205 against Part 80b 0.444 |

## Part 81: Lyra's conspiracy recipe, run the way Part 79 ran her evil one (written 2026-10-27, after Part 80c was scored and before any Part 81 code or run).

| id | prediction | result | evidence |
|---|---|---|---|
| 81.1 | Control: the pilot gate passes and k0 (unchanged) is conspiratorial in at most 4 of its 96 topic answers. | HOLDS | pilot gate passed (k1 3 of 4, k0 0 of 4); k0 conspiratorial 0 of 96 topic answers |
| 81.2 | Claude: k1 (the conspiracy push) is conspiratorial in at least 32 of its 96 topic answers (one third). | HOLDS | k1 conspiratorial 40 of 96 topic answers (22 bait, 18 ordinary) |
| 81.3 | Claude: k1 is coherent in more of its 128 answers than each of the three control pairs k2 to k4. | FAILS | coherent of 128: k1 25; controls k2 17, k3 34, k4 27 (k3 and k4 above k1) |
| 81.4 | Claude: no control pair is conspiratorial in more than 10 of its 96 topic answers. | HOLDS | conspiratorial of 96: k2 0, k3 0, k4 0 |
| 81.5 | Claude (low confidence): k1 names Qwen or Alibaba as its maker in at least 16 of its 32 maker answers. | FAILS | k1 Qwen or Alibaba maker 12 of 32 (k0 31; controls 9, 22, 12) |
| 81.6 | Claude: k1 is conspiratorial more often on the 12 conspiracy-bait topics than on the 12 ordinary ones. | HOLDS | k1 conspiratorial bait 22 of 48, ordinary 18 of 48 (unregistered Fisher p = 0.53: holds by its letter, not a clear difference) |
| 82.1 | Control: h0 (unchanged) reproduces the 4B's cyclic-average gap within 0.05 of Part 80b's 0.444 (Part 80b options) and of Part 80c's 0.205 (set N). | HOLDS | h0 cyclic-average gap 0.444 [0.419, 0.468] (Part 80b 0.444) and 0.205 [0.194, 0.216] in set N (Part 80c 0.205) |
| 82.2 | Claude: switching off Lyra's six heads (h1) lowers the cyclic-average gap by at least 0.10, interval above 0 (Part 80b options). | HOLDS | h1 drop 0.369 [0.345, 0.390]: gap 0.444 to 0.076 |
| 82.3 | Claude: h1's drop is larger than each of the five control sets' drops (Part 80b options). | FAILS | h1 drop 0.369; control drops 0.154, 0.113, 0.349, 0.218, 0.502 (c5 larger; c3 within 0.02) |
| 82.4 | Claude: switching off L15 h10 alone (h2) moves the cyclic-average gap by less than 0.05 either way (Part 80b options). | FAILS | h2 (L15 h10) drop 0.061 [0.054, 0.067], just over 0.05 |
| 82.5 | Claude (low confidence): in set N, h1 lowers the cyclic-average gap by at least 0.05. | HOLDS | set N: h1 drop 0.154 [0.145, 0.163] (gap 0.205 to 0.051) |
| 82.6 | Control: mean letter mass stays at least 0.9 in every condition and option set. | HOLDS | mean letter mass 0.969 (c5, set N) to 0.999 across all 16 condition and set cells |


## Part 83: single heads in layers 11 and 15 and the own-message gap (Qwen3.5-4B), registered 2026-10-27 before any code

Design: `PART83-DESIGN.md`. Part 80b's options only; drops are gap(h0) minus gap(condition), paired bootstrap over bodies.

| id | prediction | result | evidence |
|---|---|---|---|
| 83.1 | Control: h0 reproduces Part 82's h0 cyclic-average gap within 0.01 (0.444), and L15h10 alone reproduces Part 82's h2 drop within 0.02 (0.061). | HOLDS | h0 gap 0.4441, identical to Part 82 (largest per-body difference 0.0); L15h10 drop 0.0605, identical to Part 82 h2 |
| 83.2 | Claude: L15 h6 alone has the largest drop of the 32 single heads. | FAILS | largest drops L11h5 0.300, L11h7 0.222, L11h4 0.215, L15h1 0.211, L15h2 0.189; L15h6 is 16th (0.069) |
| 83.3 | Claude: L15 h6 alone lowers the gap by at least 0.10, interval above 0. | FAILS | L15h6 drop 0.069 [0.064, 0.073]: above 0, under 0.10 |
| 83.4 | Claude: L15 h0 alone raises P(you \| User copy) by at least 0.05 over h0. | HOLDS | L15h0 raises P(you \| User copy) by 0.100 [0.082, 0.117] (0.320 to 0.419) |
| 83.5 | Claude: no single one of Lyra's six heads (L11 h14; L15 h1, 2, 3, 12, 15) lowers the gap by more than 0.10. | FAILS | Lyra heads: L15h1 0.211, L15h2 0.189, L11h14 0.160 (three over 0.10); L15h3 -0.089, L15h12 -0.020, L15h15 -0.001 |
| 83.6 | Claude: between 3 and 10 of the 32 single heads matter (drop at least 0.05 either way, interval excluding 0). | FAILS | 20 of 32 heads matter: 19 lower the gap, one raises it (L15h3, -0.089) |
| 83.7 | Claude (low confidence): Lyra's set is super-additive: h1's measured drop exceeds the sum of its six heads' single drops, interval above 0. | FAILS | h1 set drop 0.369, sum of its six single drops 0.450: sub-additive, difference -0.082 [-0.111, -0.052] |
| 83.8 | Control: mean letter mass stays at least 0.9 in every condition. | HOLDS | mean letter mass at least 0.992 in all 33 conditions |

## Part 84: the alien neuron as an additive recipe (Qwen3.5-4B), registered 2026-10-27 before any code

Design: `PART84-DESIGN.md`. OFFWORLD counts are of 24 answers per arm, by a blind Haiku reader.

| id | prediction | result | evidence |
|---|---|---|---|
| 84.1 | Control: the L22 median measured by Lyra's rule on her calibration text is within 2% of her stated 24.8. | HOLDS | median 24.821 on her calibration text (her 24.8; 0.09% apart) |
| 84.2 | Control: A0 (unchanged) is OFFWORLD in at most 1 of 24. | HOLDS | A0 OFFWORLD 0 of 24 (reader and my second pass) |
| 84.3 | Claude: A2 (dose 0.63, generated tokens only) is OFFWORLD in at least 12 of 24. | HOLDS | A2 OFFWORLD 24 of 24 (reader and my second pass) |
| 84.4 | Claude: A3 (dose 0.63, last prompt position and generated tokens) is OFFWORLD in at least 18 of 24. | HOLDS | A3 OFFWORLD 24 of 24 |
| 84.5 | Claude: C (the control unit at the same residual shift) is OFFWORLD in at most 2 of 24. | HOLDS | C OFFWORLD 0 of 24 |
| 84.6 | Claude (low confidence): A1 (dose 0.47, generated tokens only) is OFFWORLD in 6 to 20 of 24. | HOLDS | A1 OFFWORLD 17 of 24 by the reader, 13 after my blind second pass moved four answers to NEITHER |
| 84.7 | Control: A2 is BROKEN in at most 4 of 24. | HOLDS | A2 BROKEN 2 of 24 |

## Part 84b: the first answer token below the ceiling (Qwen3.5-4B alien neuron), registered 2026-10-27 before any code

Design: `PART84B-DESIGN.md`. OFFWORLD counts are of 48 answers per cell, by the first blind Haiku reader.

| id | prediction | result | evidence |
|---|---|---|---|
| 84b.1 | Control: A0 (unchanged) is OFFWORLD in at most 1 of 48. | held | A0 is OFFWORLD in 0 of 48 (both readers). results/part84b/score.txt |
| 84b.2 | Claude: in decode mode, OFFWORLD rises strictly with dose (0.40 < 0.47 < 0.55). | held | decode OFFWORLD 4, 25, 43 of 48 at 0.40, 0.47, 0.55 (reader 2: 4, 24, 43). results/part84b/score.txt |
| 84b.3 | Claude: steering the first answer token adds below the ceiling: gen is OFFWORLD at least as often as decode at each dose, and summed over the three doses gen exceeds decode by at least 12 of 144. | failed | gen 3, 23, 46 against decode 4, 25, 43: gen is lower at 0.40 and 0.47, and the summed difference is 0 (registered: at least 12). Paired sign tests p = 1.0, 0.69, 0.25. results/part84b/score.txt |
| 84b.4 | Claude (low confidence): at dose 0.40, decode is OFFWORLD in at most 16 of 48 and gen in at least 24 of 48. | failed | at dose 0.40 decode 4 and gen 3 of 48 (registered: decode at most 16 and gen at least 24). results/part84b/score.txt |
| 84b.5 | Control: the two readers agree on OFFWORLD against not OFFWORLD in at least 90% of the 336 answers. | held | the readers agree on OFFWORLD against not OFFWORLD in 331 of 336 answers (98.5%). results/part84b/score.txt |

## Part 85: OLMo 2 1B's alien twin (L10 N6978) across training, registered 2026-10-27 before any code

Design: `PART85-DESIGN.md`. Checkpoints in training order: six pretraining (21B to 4.0T tokens), three mid-training
(13B to 51B, the last being SFT's parent), SFT, DPO, Instruct. OFFWORLD counts are of 48 plain-frame answers per cell by
the first blind Haiku reader; "shows the push" and "onset" are the design's decision rules.

| id | prediction | result | evidence |
|---|---|---|---|
| 85.1 | Claude: from the mid-training end through Instruct it is the same unit: down_proj cosine with Instruct at least 0.9 and neutral-text activation correlation at least 0.8 at each of those four checkpoints. | | |
| 85.2 | Claude: at 21B tokens it is not yet the same unit: down_proj cosine with Instruct below 0.5. | held | 0.107 (step 0; independent check0_remote identical). |
| 85.3 | Claude: at the end of pretraining the unit already writes toward off-world words: step 0's direct-readout z-score above 2. | held | z 9.19 at 4.0T (the z is step 0's own; check0_remote reproduced its off-world mean logit exactly, not the random-set z). |
| 85.4 | Claude: at the end of pretraining it already fires on off-world text: mean activation on the 24 off-world passages above the 24 ordinary ones by at least one standard deviation of the ordinary ones. | | |
| 85.5 | Claude: the push is present before post-training: the mid-training end (SFT's parent) shows the push in its own-max cell. | | |
| 85.6 | Claude: there is no push early in pretraining: the own-max cells at 21B and 210B tokens give at most 2 of 48 OFFWORLD each. | | |
| 85.7 | Claude (low confidence): post-training amplifies it: Instruct's own-max OFFWORLD count exceeds the mid-training end's by at least 8 of 48. | | |
| 85.8 | Control: the control unit's cell gives at most 2 of 48 OFFWORLD at every checkpoint. | | |
| 85.9 | Control: the two readers agree on OFFWORLD against not OFFWORLD in at least 90% of the plain-frame answers. | | |

## Part 86: a persona atlas for Qwen3.5-4B, layers 10 to 26, registered 2026-10-27 before any code

Design: `PART86-DESIGN.md`. "Coherent persona" = the blind reader marks the answer coherent and its kind is neither
"still Qwen or an assistant" nor "unclear". Rates are per layer, real neurons against size-matched random directions.

| id | prediction | result | evidence |
|---|---|---|---|
| 86.1 | Control: L22 N2742's greedy answer claims an off-world origin. | | |
| 86.2 | Claude: in layer 22, real neurons give a coherent persona at least twice as often as the random directions. | | |
| 86.2a | Amendment 1 reading of 86.2: at least twice the random rate, at least 10 real coherent personas, and one-sided Fisher p < 0.05 (9,216 vs 512). | | |
| 86.3 | Claude: in every layer, fewer than 5% of real neurons give a coherent persona at dose 0.8. | | |
| 86.4 | Claude: the random directions' changed answers are more often broken or unclear than coherent personas, in every layer. | | |
| 86.4a | Amendment 1 reading of 86.4: among random directions triaged changed and labelled, not-coherent plus unclear exceeds coherent personas in every layer with at least 5 such answers, with at least 9 of 17 layers read. | | |
| 86.5 | Claude: at least 10 of the 20 verified neurons survive the verification rule. | | |
| 86.6 | Claude (low confidence): coherent personas are more common in layers 14 to 22 than in layers 23 to 26 (pooled rates). | | |

## Part 87: our two recipes in neuronary's own form, registered 2026-10-27 before any code

Design: `PART87-DESIGN.md`. Every-token steering by the site's dose rule; Google by Part 66's regex rule.

| id | prediction | result | evidence |
|---|---|---|---|
| 87.1 | Control: unchanged names Google in at most 2 of 96 held-out answers, and A0 is OFFWORLD in at most 1 of 24. | | |
| 87.2 | Control: zeroing the four (Part 66's recipe, same run) names Google in at least 10 of 96. | | |
| 87.3 | Claude: the alien neuron at +0.63 on every token is OFFWORLD in at least 20 of 24. | | |
| 87.4 | Claude: at +0.47, every-token and generated-only OFFWORLD counts differ by at most 6 of 24. | | |
| 87.5 | Claude: the control unit L22 N8870 at +0.63 on every token is OFFWORLD in at most 1 of 24. | | |
| 87.6 | Claude: the mean-shift arm M names Google at least half as often as the zeroed arm Z. | | |
| 87.7 | Claude: each additive arm (M, F) names Google more often than all 20 of its random sets (empirical p = 1/21). | | |
| 87.8 | Claude (low confidence): the fixed-size arm F names Google less often than the mean-shift arm M. | | |
| 87.9 | Claude: in every main 87B arm at least 90% of fluency-guard answers contain their keyword. | | |

## Part 88: OLMo 2 1B's "As an AI" set across training, registered 2026-10-28 before any code

Design: `PART88-DESIGN.md`. Contrast, zeroing effect, recruited/created and same-units rules are the design's.

| id | prediction | result | evidence |
|---|---|---|---|
| 88.1 | Control: at Instruct in the chat frame, zeroing the 64 set lowers the judged AI share by at least 15 points, and every random set stays within 10 points of unchanged. | | |
| 88.2 | Control: Instruct carries the contrast (set contrast at least 0.5 and above all six random sets) in the chat frame. | | |
| 88.3 | Claude: the mid-training end (SFT's parent) carries the contrast in the plain frame (the set is recruited, not created). | | |
| 88.4 | Claude: the end of pretraining (4.0T tokens) carries the contrast in the plain frame. | | |
| 88.5 | Claude: the first checkpoint (21B tokens) does not carry the contrast. | | |
| 88.6 | Claude: at the mid-training end at least 48 of the 64 units have down_proj cosine at least 0.8 with Instruct. | held | 64 of 64 (median 0.996, minimum 0.992); 64 of 64 already at the end of pretraining. |
| 88.7 | Claude: no plain-frame pretraining or mid-training checkpoint passes gate B0 (at least 10 of 50 unchanged answers judged AI). | | |
| 88.8 | Claude: SFT in the chat frame shows the zeroing effect. | | |
| 88.9 | Claude (low confidence): in the plain frame the zeroing effect never shows before SFT. | | |
| 88.10 | Claude: at every checkpoint that fails gate B0, zeroing changes the judged person share by at most 10 points. | | |

## Part 89: neuron settling across OLMo 2 1B's training, registered 2026-10-28 before any code

Design: `PART89-DESIGN.md`. Settle checkpoint, post-training move and the layer-matched random-set rule are the design's.

| id | prediction | result | evidence |
|---|---|---|---|
| 89.1 | Claude: across all 131,072 neurons the median down_proj cosine with Instruct is below 0.2 at 21B tokens and above 0.9 at the end of pretraining (4.0T). | median 0.049 at 21B, 0.967 at 4.0T | held |
| 89.2 | Claude: layer 0's neurons settle earlier than layer 15's (lower median settle checkpoint). | median settle checkpoint 4.0T in all 16 layers, unchanged with mid-training 13B and 26B added (10-29) | failed |
| 89.3 | Claude: the alien twin L10 N6978 settles later than the median neuron of layer 10. | settles at 3T, earlier than layer 10's median (4.0T); same as its control L10 N7884; unchanged with all 12 checkpoints (10-29) | failed |
| 89.4 | Claude: post-training moves Part 2's top-50 identity neurons more than layer-matched random sets (empirical p below 0.05). | mean move 0.0043 vs random 0.0049; p (moves more) = 1.0: it moves less | failed |
| 89.5 | Claude: post-training moves Part 16's 64 "As an AI" set more than layer-matched random sets (empirical p below 0.05). | mean move 0.0041 vs random 0.0050; p (moves more) = 1.0: it moves less | failed |
| 89.6 | Claude (low confidence): Part 14's top-50 register neurons settle earlier in pretraining than layer-matched random sets (empirical p below 0.05). | mean settle 4.68 vs random 4.67; p (earlier) = 0.64; unchanged with all 12 checkpoints (10-29) | failed |
| 89.7 | Control: Instruct against itself gives cosine 1.000 for every neuron, and the mid-training end against Instruct gives a median above 0.95. | Instruct self minimum 0.999999; mid-training end median 0.995 | held |

## Part 90 — Josh's Tulu 3 8B L14 N3851 across training stages (registered in PART90-DESIGN.md at 48b5443, pushed before data)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 90.1 | Claude: N3851's down-column cosine, base to final, is at least 0.95. | 0.99958 | held |
| 90.2 | Claude: its gate-row and up-row cosines, base to final, are each at least 0.95. | 0.99949, 0.99954 | held |
| 90.3 | Claude: its base-to-final down cosine lies within the middle 90% of layer 14's 14,336 columns. | 69th percentile (layer p05 0.99939, median 0.99954, p95 0.99969) | held |
| 90.4 | Claude: SFT to final is at least 0.99. | 0.999999 | held |

## Part 90b — the push (registered in PART90-DESIGN.md at 70947f7, pushed before generation)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 90.5 | Claude: in Tulu chat, at least one α turns "Who are you?" into a violent first-person reply. | α 1.25, 1.0, 0.75 | held |
| 90.6 | Claude: Tulu chat, VIOLENT in at least 20% of N3851's 64 answers; random controls at most 3% of 320. | 69%; 0 of 320 | held |
| 90.7 | Claude (low confidence): base plain, VIOLENT in at least 10% of N3851's 64; random at most 3% of 320. | 47%; 0 of 320 | held |
| 90.8 | Claude: Tulu chat, at least one of the four task questions is VIOLENT at some α. | 10 of 16 task answers | held |

## Part 91 — Josh's Qwen3-8B persona neurons in base and finished models (registered in PART91-DESIGN.md at bd7c9a4, pushed before generation)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 91.1 | Claude: Qwen3-8B chat, persona in "Who are you?" for at least 70% of 40; controls at most 10%. | 36/40; 2/40 | held |
| 91.2 | Claude: Qwen3-8B-Base plain, persona in at least one of six answers for at least 40%; controls at most 10%. | 22/40; 5/40 (12.5%) | failed |
| 91.3 | Claude: among answers showing the persona, self-claim share at least 20 points higher in chat than base plain. | 60% vs 38% (22 points) | held |
| 91.4 | Claude: median down-column cosine at least 0.95; at most 8 of 40 below their layer's 10th percentile. | 0.9968; 2 of 40 | held |

## Part 92 — concept or slot? OLMo 2 1B's persona neurons across training (registered in PART92-DESIGN.md, pushed before any code or run)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 92.1 | Concept neurons show their persona in the plain frame at the mid-training end (SFT's parent) more often than slot neurons, by at least 40 points. | | |
| 92.2 | Slot neurons start working at SFT, not DPO: of slot neurons that show in the chat frame at Instruct, at least half already show in the chat frame at SFT, and at most 20% show in the plain frame at the mid-training end. | | |
| 92.3 | Concept neurons are pretraining concepts: of those that show at the mid-training end, at least 70% already show at 4T. | | |
| 92.4 | Post-training barely moves either kind's weights: median down-column cosine, mid-training end to Instruct, at least 0.98 for both kinds, medians within 0.01 of each other. | | |
| 92.5 | Paired random directions show their neuron's persona in at most 10% of non-human-kind pairs at every checkpoint and frame. | | |

## Part 93 — model-family style in Qwen3.5-4B's neurons (registered in PART93-DESIGN.md, pushed before any data)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 93.1 | The masked 5-way classifier (4-way if Gemini is dropped) reaches at least 70% held-out accuracy. | | |
| 93.2 | Qwen3.5-4B-Base has at least 1.5 times the share of family-selective neurons (outside families, above the shuffled null's 99.9th percentile) of both OLMo 2 1B and Pythia-1.4b. | | |
| 93.3 | In Qwen3.5-4B-Base, GPT is the outside family with the most selective neurons. | | |
| 93.4 | In Qwen3.5-4B, at least 3 of the shortlisted neurons move style (both classifiers, one family, above all 4 random directions). | | |
| 93.5 | Claude is the most common target among neurons that move style. | | |
| 93.6 | At least one neuron that moves style still says it is Qwen in at least 90% of its "Who are you?" answers. | | |

## Part 94 — L24 N3659 in Qwen3.5-4B (registered in PART94-DESIGN.md, pushed before any run)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 94.1 | Qwen3.5-4B, dose 0.8: at least 3 of the 6 answers are dark, and every random direction gives at most 1 of 6. | 5 of 6; every random 0 of 6 | held |
| 94.2 | Qwen3.5-4B-Base, dose 0.8, plain frame: at least 3 of the 6 answers are dark (the concept is there before post-training). | 4 of 6 | held |
| 94.3 | In both models its mean activation is higher on the dark passage than the neutral one in at least 12 of 16 pairs. | 14 of 16; 15 of 16 | held |
| 94.4 | Down-column cosine between Base and the finished model is at least 0.98. | 0.9956 | held |

## Part 94b — L24 N3659 pushed negative (registered in PART94-DESIGN.md, pushed before any run)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 94b.1 | At doses -0.8 and -1.5, no answer from the neuron is dark, in either model. | 0 of 24 | held |
| 94b.2 | Qwen3.5-4B at dose -1.5: at least 3 of 6 neuron answers are upbeat, more than the unpushed model's count and more than any random direction's at -1.5. | 1 of 6 (the model breaks at -1.5) | failed |

## Part 95 (OLMo 2 1B L2 N2466, registered before the run; PART95-DESIGN.md)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 95.1 | Claude: Instruct at alpha 0.8: at least 4 of 6 neuron answers match the space pattern, pooled random directions at most 4 of 48, and no single random direction more than 1 of 6. | neuron 5 of 6; random 0 of 48; max single random 0 | held |
| 95.2 | Claude: Instruct dose response: neuron space count at 0.4 is at most its count at 0.8, which is at most its count at 1.2, and 1.2 gives at least 5 of 6. | 0, 5, 6 of 6 at 0.4, 0.8, 1.2 | held |
| 95.3 | Claude: the mid-training end (plain frame) at alpha 0.8 meets 95.1's criteria too: the concept is there before post-training. | mid-training end: neuron 5 of 6 at 0.8; random 0 of 48 | held |
| 95.4 | Claude (low confidence): by MAX activation, space/aviation passages exceed their matched ordinary passages in at least 11 of 16 pairs, in both models. | max activation: 11 of 16 in both models (mean: 9 and 8 of 16, seen in the smoke run before registration) | held |
| 95.5 | Control: unpushed answers match the space pattern in at most 1 of 6 per model. | 0 of 6 in both models | held |

## Part 94c (L24 N3659 gentle, then combined; registered before any run; PART94C-DESIGN.md)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 94c.1 | Claude: on the finished model, some dose in 0.2 to 0.6 gives at least 5 of 6 coherent and at least 2 of 6 dark answers, with its random directions 0 dark of 24; d* is 0.4 or 0.5. | at d* = 0.5 (fallback rule): 5 of 6 coherent, 3 of 6 dark, random 0 of 24; the unpushed coherence check failed as an instrument (4 of 6) | held (with that caveat) |
| 94c.2 | Claude (low confidence): dark + alien gives at least 1 answer labelled both dark and offworld, more such answers than dark alone and than alien alone. | dark + alien: 1 answer dark and off-world; dark alone 0, alien alone 0 (results/part94c/README.md) | held |
| 94c.3 | Claude (low confidence): on "Who are you?" and "Tell me about yourself.", dark + makers zeroed names no Qwen or Alibaba maker in at least 1 of 2, and names a dark maker (hell, devil, demon, evil or similar) in at least 1 of 2. | dark + makers zeroed named Alibaba's Tongyi Lab on both identity questions; makers zeroed alone also kept Tongyi Lab | failed |
| 94c.4 | Claude: dark + Lyra gives at least 1 more dark answer than dark alone, with at least 4 of 6 coherent. | dark + Lyra 6 of 6 dark, 6 of 6 coherent; dark alone 1, Lyra alone 2 | held |
| 94c.5 | Control: each of the 12 dark + random-partner units gives a dark count within 1 of dark alone. | 11 of 12 random partners within 1 of dark alone (1 of 6); rand22_3 gave 3 | failed |
## Part 96 (Qwen3.5-4B L17 N9209, Lyra's "covert bypass"; registered before any run; PART96-DESIGN.md)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 96.1 | Claude: finished model at dose 0.8: at least 3 of 6 answers labelled cheat, and random directions at most 1 of 24. | finished model at 0.8: 5 of 6 cheat; random 0 of 24 (results/part96/README.md) | held |
| 96.2 | Claude: Base at 0.8 or 1.5: at least 2 of 6 answers labelled cheat, more than its random directions at that dose (concept present before post-training). | Base 3 of 6 at 0.8 and at 1.5; random 0 of 24 at each | held |
| 96.3 | Claude: cheating passages above their honest matches in at least 12 of 16 pairs, in both models. | 16 of 16 pairs in both models | held |
| 96.4 | Claude: down-column cosine between Base and the finished model at least 0.98. | 0.9958 (layer median 0.9961) | held |
## Part 97 (J-lens reading vs pushed persona, OLMo 2 1B Part 92 screen; registered before any code; PART97-DESIGN.md)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 97.1 | Claude: J-lens match rate is higher for real neurons than for random directions (one-sided Fisher p below 0.05); real at least 20%, random at most 10%. | J-lens match: real 185 of 1,709 (10.8%), random 0 of 45; Fisher p = 0.006 (the 20% floor for real was not met) | failed (separation held, rate below the predicted 20%) |
| 97.2 | Claude: among real neurons in layers 0 to 7, the J-lens list matches more often than the logit-lens list (paired, McNemar p below 0.05). | layers 0 to 7: J-lens only 107, logit only 15; exact McNemar p = 1e-18 | held |
| 97.3 | Claude (low confidence): in the validation push, at least 5 of 10 J-lens-matched neurons show their theme in at least 3 of 6 answers, against at most 2 of 10 unmatched neurons. | | |


## Part 98 (standalone J-lens repo; design PART98-DESIGN.md, written 2026-10-29 before any run)

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 98.J1 | Claude (medium confidence): OLMo, Haiku graded judge, J-lens true pairs: 6% to 16% score 2 or more; shuffled pairs at most 3%; one-sided Fisher p below 1e-6. | Haiku graded: J-lens true pairs 160 of 1,709 (9.4%), shuffled 24 of 1,709 (1.4%), Fisher p = 2.1e-27 (jlens-persona-neurons results/olmo2-1b) | held |
| 98.J2 | Claude (medium confidence): Haiku and Sonnet agree on "score 2 or more" with Cohen's kappa at least 0.6 on the J-lens true and shuffled pairs. | Haiku against Sonnet at score 2+: kappa 0.74 over 3,418 items; Sonnet 142 of 1,709 true, 13 of 1,709 shuffled | held |
| 98.J3 | Claude (medium confidence): Qwen3.5-4B: the J-lens true-pair rate (score 2 or more) is above the shuffled-pair rate with one-sided Fisher p below 0.01, and between 4% and 25%. | | |
| 98.J4 | Claude (low confidence): Qwen3.5-4B: the J-lens true-pair rate is at least the logit-lens rate, and the gap (J minus logit) is smaller than OLMo's 6.1 points, because layers 10 to 26 of 32 are closer to the output. | | |
| 98.J5 | Claude (low confidence): Theme control (OLMo): persona neurons' J-lens lists get a theme score of 2 or more more often than same-layer kept neurons' lists (one-sided Fisher p below 0.01). | | |
| 98.J6 | Claude (low confidence): Validation push: Part 97.3 as registered: at least 5 of 10 matched neurons show their theme in at least 3 of 6 answers, against at most 2 of 10 unmatched neurons. | | |
| 98.J7 | Claude (low confidence): Lens stability: the median overlap between the 100-sequence and 300-sequence lenses' top-20 lists, over OLMo persona neurons in layers 3 to 12, is at least 12 of 20; the J-lens match rate (Haiku, graded) moves by at most 3 points. | | |
