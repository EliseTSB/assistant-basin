# Neuron records: a shared format for claims about single neurons

Draft 1, 2026-10-16. Written so that results from different projects about the same open model can be joined,
compared and checked, instead of each project keeping its own labels. It borrows from three existing formats
(Model Mapper, Transluce's ADAG and Neuronpedia, read 2026-10-16) and adds the two things none of them has:
a pinned weights revision, and a control value next to every number.

## Two files, JSON Lines

- `neurons.jsonl`: one record per neuron pole.
- `sets.jsonl`: one record per named set of neurons, for claims that only make sense about a group (a knockout of
  28 neurons, a circuit). A set's members are neuron keys.

## Which neuron: the key

| field | type | meaning |
|---|---|---|
| `model` | string | Hugging Face repo id, exactly as published (`allenai/OLMo-2-0425-1B-Instruct`) |
| `revision` | string | the 40-hex commit of the weights on the Hub. Required: two snapshots under one name are different models |
| `site` | string | where the activation is read. `mlp.down_proj.input` is the input to the MLP's down projection (for gated MLPs, act(gate) * up). Records at different sites never join |
| `layer` | integer | 0-indexed transformer block |
| `neuron` | integer | 0-indexed position at the site |
| `pole` | `pos` or `neg` | which sign of the activation the claims are about |

The key string is `model@revision[0:12]/site/L{layer}/N{neuron}/{pole}`, for example
`allenai/OLMo-2-0425-1B-Instruct@48d788eca847/mlp.down_proj.input/L6/N8167/pos`.

## What is claimed: claims, each with its evidence

A record holds a list of `claims`. A claim is one statement with the measurement behind it. Claims are never edited
after they are published; a later claim can `supersede` an earlier one or report `refuted`.

| field | type | meaning |
|---|---|---|
| `id` | string | unique within the file |
| `level` | 0-4 | the evidence ladder below |
| `statement` | string | one sentence, in plain words |
| `metric` | object | `name`, `value`, `n`, and for level 1 and up `control_name` and `control_value` |
| `heldout` | boolean | measured on data not used to choose the neuron or write the description |
| `preregistered` | object or null | `url` and `commit` of a prediction published before the measurement |
| `code` | object | `repo`, `commit`, `path` of what produced the number |
| `author` | object | `who` (a person or an agent's name), `model` when an AI wrote or judged it |
| `date` | string | ISO date |
| `status` | `supported`, `refuted` or `open` | what the measurement says about the statement |
| `supersedes` | string or null | the id of a claim this one replaces |

## The evidence ladder

- **0 described**: a description exists. Nothing measured.
- **1 selective**: the neuron responds to a defined contrast on held-out data more than a stated control (random
  directions, a shuffled-label null, or random neurons).
- **2 predictive**: a description, applied by someone who did not see the neuron's top examples, predicts its
  activation on unseen contexts better than the same procedure on a control.
- **3 causal**: an intervention on this neuron, or on a set containing it, changes a named behaviour more than the
  same intervention on a random control of the same size.
- **4 replicated**: an independent run (different person or agent, code or hardware) reproduced a level 1-3 claim
  within a stated tolerance, and cites it.

A record's level is the highest level with a `supported` claim. A claim at level 1 or above without a control value is
invalid, not level 0: a number with nothing to compare it to is not evidence.

## Checking

`neuronrecords.py check FILE...` validates the schema, rejects a missing revision, a level 1+ claim without a control,
and a set claim whose control is a different size; then prints, per model, how many neuron poles sit at each level.
It does not re-run anybody's measurement; it checks that a claim says enough for someone else to re-run it.

## Converters

- `neuronrecords.py from-assistant-basin` turns this repo's results into records.
- `neuronrecords.py from-modelmapper DIR` turns Model Mapper's published JSON (MIT) into records, as the first test that
  another project's results fit. What does not map is reported, not dropped.
