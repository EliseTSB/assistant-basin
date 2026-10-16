# Assistant Basin

Where does a language model keep its sense of being an AI assistant? This repository looks inside open models for the
neurons and directions that carry it, follows them through the stages of training, and tests how system prompts move
them. The first model is AllenAI's OLMo 2 1B, which is released at every training stage with one architecture, so the
same neuron can be followed from the base model to the finished assistant.

**Status: early.** One small model, sentences we wrote, and correlation only: nothing here yet shows that a neuron
causes a behaviour. Every prediction is written down before its run (see [PREDICTIONS.md](PREDICTIONS.md)), and the
ones that failed stay in the record.

## Questions

1. Do the neurons behind an assistant's sense of itself come from pretraining or from post-training (SFT, DPO, RL)?
2. How does that sense treat other assistants' names? Does OLMo react to "I'm Claude" or "I'm ChatGPT" the way it
   reacts to "I'm OLMo"?
3. What does a system prompt such as "You are a helpful, harmless, and honest assistant" change inside the model, and
   are helpful, harmless and honest one thing to it?

## Results so far

All on OLMo 2 1B Instruct unless noted. Details and the numbers behind each line are in `results/` and PREDICTIONS.md.

- **Identity neurons exist and are measurable.** Neurons were scored on how much more they fire when the assistant
  says something about itself as an AI than for the same sentence about another AI, a person speaking about
  themselves, or the sentence placed in the user's turn; a neuron's score is the smallest of the three gaps. Chosen on
  half the sentences and scored on the other half, the top 50 reach a median effect size of 0.67; with shuffled labels
  the same pipeline reaches 0.38 at most. 31 of the 50 sit in the last layer.
- **They prefer the model's own name.** On the 23 sentences that name the model, they fire much less for "I'm Claude"
  than for "I'm OLMo" (paired effect size −1.04 in chat form, −0.72 in plain text), and less for Claude than for
  ChatGPT. "I'm Claude" sits much nearer a person speaking than "I'm OLMo" does.
- **The names split in the late layers.** Across whole layers, the three names are about equally far apart in the
  middle of the network; from layer 8 on, Claude and ChatGPT move toward each other and OLMo moves away from both.
- **The last RL round changed nothing measurable** (RLVR1 to Instruct: every neuron's activity on neutral text
  correlates at 1.000, and the identity neurons score the same). The earlier stages are being measured.
- **Qwen3 0.6B, a check of a recent claim that neurons are clean:** read by blind language-model judges, neurons were
  only 7 to 9 points more often "one job" than random directions in the same layer (75–78% against 68–69%). One judge
  failed its own floor, calling 44% of random text "one job", which is a methods finding in itself.

## Files

| File | What it does |
|---|---|
| `collect.py` | MLP neuron activations for the identity sentences, per model checkpoint |
| `analyze.py` | finds identity neurons (held-out split, shuffled-label null) and scores them in other checkpoints |
| `corr.py` | checks that neuron *i* is the same unit in two checkpoints (correlation on neutral text) |
| `lineage.py` | which published base checkpoint a fine-tuned model descends from (one tensor by range request) |
| `nametest.py` | layer-by-layer distance between the OLMo, Claude and ChatGPT versions of a sentence |
| `ablate.py` | the causal test: hold the identity neurons at their mean and ask the model who it is |
| `sysprompt.py`, `sysanalyze.py` | system-prompt experiment: activations under 15 system prompts, the HHH tests, self-identification counts |
| `topk.py`, `judge.py`, `score.py`, `dominance.py` | the neuron-cleanliness check on Qwen3 0.6B |
| `data/frames.json`, `data/frames-extra.json` | the 72 sentence frames, each in six versions |
| `results/` | small result files; the large activation arrays are not included |

## Running it

Python 3.12, CPU is enough for 1B models.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy pyarrow
python collect.py allenai/OLMo-2-0425-1B-Instruct out
python analyze.py out/allenai__OLMo-2-0425-1B-Instruct.npz --form chat --json out/identity.json
python nametest.py out/allenai__OLMo-2-0425-1B-Instruct.npz --form chat
```

The other stages are `allenai/OLMo-2-0425-1B` (base), `-SFT`, `-DPO` and `-RLVR1`. `topk.py` and `dominance.py` read
WikiText-103 from a parquet file (`WIKITEXT_PARQUET`). `judge.py` calls judge models through OpenCode Go and needs
`OPENCODE_GO_API_KEY`.

## Who did this

Elise ([@EliseTSB](https://github.com/EliseTSB)) directs the project and asks the questions. The experiments were
designed, run and written up by Claude, Anthropic's model, working as an agent in Elise's long-running project; 48 of
the 72 sentence frames were drafted by a Claude Haiku helper and checked by script. We say this plainly because it
matters for how the work should be read and checked.

## License

Code: MIT (see LICENSE). `data/neutral.txt` is the opening of *Pride and Prejudice* (public domain, Project
Gutenberg). `results/part1-qwen3-0.6b/units.json` contains excerpts of WikiText-103, which is CC BY-SA 3.0.
