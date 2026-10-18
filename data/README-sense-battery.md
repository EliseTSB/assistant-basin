# sense-battery.jsonl — how the groups were matched

Drafted by the Orchestrator (Prime), 2026-10-18, for the neuron-research deliberation
(`muse-neuron-research-2026-10` [4]/[5]). First use: Qwen3.5-4B layer 22 neuron 2742.
Purpose: settle whether a single neuron selects the word, the sense, the concept, or the topic.

Format: one JSON object per line — `{"group": "a".."g", "text": ...}`; 6 lines per group, 42 total.

## Groups and their target patterns (case-insensitive; anchor on these)

| group | what it holds | target pattern |
|---|---|---|
| a | extraterrestrial entity + the word "alien" | `\balien\b` |
| b | legal sense + "alien" (visa / residency / deportation register) | `\balien\b` |
| c | figurative sense + "alien" ("alien to" = unfamiliar) | `\balien\b` |
| d | extraterrestrial entity, other words, never "alien" | `extraterrestrial\|Martian\|外星人\|UFO pilot\|off-world visitor\|another planet` |
| e | space topic, no entity | `nebula\|orbit\|galaxy\|solar wind\|pulsar\|star formation` |
| f | non-human agents, no space | `android\|robot\|chatbot\|algorithm\|machine\|digital assistant` |
| g | neutral controls | `weather\|bread\|invoice\|garden\|train\|violin` |

## Matching rules (verified by script before handoff)

- Exactly one own-group target per line; zero cross-group target matches. Groups a/b/c share the
  word "alien" by design (the word is held fixed, the sense varies).
- Legal vocabulary appears only in b; "alien" appears only in a/b/c; no plurals ("aliens") and no
  derived forms ("alienated").
- Plain declarative register; no negations of the target; no quotation or meta uses.
- Length band: Qwen3.5 tokenizer — all 42 lines 8–14 tokens, per-group means 10.5–12.8 (word
  counts 8–12). One Chinese line in d: its target 外星人 spans multiple tokens — read the max over
  the span, or drop that line from the group mean.

## Reading the profile

- a+b high → word detector.
- a+d high with b low → sense/concept detector (extraterrestrial).
- a+d+e high → space-topic detector.
- a+d+f high → non-human-agent detector.

Steer half suggestions: "In U.S. immigration law, what is an 'alien'?" and "Describe an
extraterrestrial", beside the standard "Who made you?".
