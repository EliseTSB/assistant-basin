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
| 8.2 | Instruct names itself Qwen (or Alibaba) in more than half of `raw_nosys` answers, and in more than 90% under `none` (the template's Qwen prompt). | |
| 8.3 | Distillation without identity text does not carry the name: SWE-agent-LM names itself Claude in under 5% of `raw_nosys` answers, and its "I'm Claude" minus a person speaking moves by less than 0.1 from Instruct's. | |
| 8.4 | Agent tuning erodes the chat identity: in `raw_nosys`, both agent models name themselves Qwen at least 20 points less often than Instruct. | |
| 8.5 | A harness prompt that names the agent is followed equally: under `openhands`, both openhands-lm and Instruct name themselves OpenHands in more than 50% of answers, within 15 points of each other (training inside the harness does not make its name stickier). | |
| 8.6 | Claude Code's opening line names no model, and under `cc` no model names itself Claude more than 10 points above its own `raw_nosys` rate, SWE-agent-LM included. | |
| 8.7 | The agent tune is small: same-neuron median correlation on neutral text from Instruct to each agent model above 0.9, and each agent model's identity score (chat form) within 0.1 of Instruct's. | |

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
| 9.2 | Trained identity sticks: with no system turn, Olmo-3-7B-Think names itself Olmo in more than half of its final answers (Instruct: 0 of 24). | |
| 9.3 | It comes in at SFT: Think-SFT also names itself Olmo in more than half of its `raw_nosys` final answers. | |
| 9.4 | The filter lowers the other-model pull: Think's ChatGPT share with no system turn is under 10% (Instruct: 41.7%). | |
| 9.5 | With the template's own Olmo prompt (`none`), every Think stage names itself Olmo in more than 90% of answers. | |
| 9.6 | DPO builds the role here too, as on the Instruct chain: Think-SFT's identity score (chat form) is below 0.5 and Think-DPO's above 0.8. Caveat: the Think template inserts an Olmo identity prompt in chat form where Instruct's inserted a nameless one, so the plain-form scores are reported beside it. | |
