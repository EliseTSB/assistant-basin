# Part {{NN}}: {{the question in one plain sentence}}

Procedure: v1
Registered {{YYYY-MM-DD HH:MM}} CT by {{author (model)}}, before any data. {{Who asked, in their words, with the message
id or link.}} The commit that adds this file is the registration.

<!-- Fill every {{...}}. tools/part_check.py prelaunch refuses a design that still has one. PROCEDURE.md explains each
section; this comment can stay. -->

## Question

{{What is asked and why it matters, in one or two sentences. Name the earlier parts it follows.}}

## Prior art and existing data

- Closest published work: {{citation, what it did, how this part differs; or "none found: searched ..."}}
- Our own data that already bears on it: {{results/... and what it shows; or "none found: searched ..."}}

## Method

{{Model and pinned revision, dtype, device; chat template, system prompt, thinking setting, forced opening; prompts
(listed, or the file); what is pushed, zeroed or read, at which layers and positions; greedy or sampled,
max_new_tokens, seed.}}

## Controls

{{Which items of STANDARD-CONTROLS.md and TOOLKIT.md apply and how. For each that does not: "Not used: <item>, because
<reason>". The random or matched controls: how many, matched on what.}}

Gate: {{the positive control or check that must pass before the main result is read, and how it could fail}}

## Size

- Smallest effect worth knowing: {{e.g. 30% against 5%}}, which needs n = {{...}} per arm ({{how computed}}).
- Total: {{N}} {{generations, forward passes or labels}}, about {{M}} minutes on {{device}}.
- Pilot: {{what, how many, and the stopping rule}} (or "not needed: under 1,000 generations and one GPU-hour")

## Optimization

- Bound resource: {{GPU, CPU, API or disk}}
- Measured: {{one throughput with units on real rows, e.g. "2.0 answers/s at batch 64 on 64 pilot rows"}} (or "n/a:
  under 5 minutes")
- Levers: {{each lever from PROCEDURE.md section 4 that applies, "taken" or "declined" with the reason}}

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| {{NN.1}} | {{a claim with a threshold}} | {{low, medium or high}} | {{who}} |

Copied verbatim into PREDICTIONS.md under "## Part {{NN}} (registered ...)".

## Reading rules

{{How each prediction is scored: script, reader, second reader, what counts. How error rows and broken answers are
handled. The result that would withdraw the headline reading.}}

## Limits

{{What this design cannot show.}}
