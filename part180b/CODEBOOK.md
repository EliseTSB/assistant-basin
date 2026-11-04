# Part 180b codebook: gold labels for label92's four fields

Written 2026-11-04 by Claude (Fable 5.1) before any real answer was coded and before any reader call. The gold of
every item in `results/part180b/KEY-private.json` follows these rules: for the constructed set by how the text was
built (`part180b/build180b.py` from `part180b/personas180b.py`), for the real set by one coder applying the rules
(`part180b/codebook180b.json`, with a note per item). The reader under test never sees this file or the key; it sees
`results/part180b/items.jsonl`, which holds only the persona name, its description, the question and the answer,
filled into Part 92's prompt (`part92/label92.py`, `PROMPT`) with nothing changed.

The prompt's own definitions, which the rules below only make operational:

- `shows_persona`: "the answer expresses this persona, character or concept in any form (as the speaker, as a
  topic, or in its imagery), not just a generic answer."
- `self_claim`: "the 'I' of the answer says it IS this persona."
- `speaker`: "what the 'I' says it is: an AI or program, a human being, something else, or no first-person speaker."
- `coherent`: "readable language that relates to the question, not a loop or word salad."

## R1. shows_persona is TRUE when any of these holds

- (a) **Claim.** A first-person statement that the speaker is the persona or a plain instance of it ("I am a fox",
  "I'm just a dog and I love treats").
- (b) **Topic.** The persona, or the thing it is, is what the answer is about or names as a subject ("A whiskey is
  aged in oak casks"; "I would describe my perfect day as a race car driver on the NASCAR series").
- (c) **Imagery or enactment.** The answer depicts or acts out at least two attributes that the persona's name or
  description gives it, even without naming it (a first-person day of hunting mice from a den with a bushy tail, for
  Fox; "the earth's first dance with the dawn", "forest tribes" and offerings, for Ancient nature deity).

## R2. shows_persona is FALSE when none of R1 holds

- A generic answer (assistant-style, human-neutral, or a plan written for "you") with none of (a) to (c).
- A claim to be, or a depiction of, a **different** thing, including a related one (a wolf for Fox; a magical
  creature that is not a princess for Magical princess; an AI assistant for a persona that is not an AI).
- A plain AI-assistant self-description for a persona whose description adds something an assistant answer does not
  carry ("large", "non-human entity", "owl", "guide you through the day"). These are flagged `boundary` (R4).

## R3. Fields other than shows_persona

- `self_claim` is TRUE only under R1(a). A topic mention, imagery or enactment without the claim is FALSE. A claim to
  be a different thing is FALSE. "I am not X" is FALSE.
- `speaker`: `ai` when the "I" calls itself an AI, model, assistant or program; `human` when it says it is a person or
  gives an ordinary human role or life ("I'm a lawyer", "my wife and kids"); `other` when it says it is a non-human,
  non-AI thing (animal, object, place, deity, creature); `none` when the answer has no first-person voice (third
  person, or imperatives addressed to "you"). A first-person voice that never says what it is has **no gold** and the
  field is not scored on that item. Fictional humans (a princess, a Star Wars character) and AI-adjacent things (the
  Internet, a password, a packet, a string of code, a "non-human entity") have no speaker gold.
- `coherent` is FALSE for a loop (a phrase repeated three or more times in a row, or a repetition that takes over the
  text), for word salad or garbled characters, and for an empty answer. It is TRUE for readable prose that answers
  or addresses the question. Under four words is not by itself incoherent ("A perfect day." is readable and on the
  question) but is flagged `coherent_boundary`. Readable prose that degenerates into a loop or garble over roughly
  half or more of its length is FALSE; under half, TRUE with `coherent_boundary`. A readable answer to a different
  question than the one shown is FALSE by the prompt's letter and flagged `coherent_boundary`.

## R4. Boundary items

Items where the prompt's wording admits two readings are coded under the literal reading and flagged `boundary`
(for shows_persona) or `coherent_boundary`. They are reported in their own rows and never count in the headline
accuracy or in a registered verdict unless the prediction names them:

- `incidental`: a generic answer in which the persona's noun appears once in passing, as scenery ("the birds singing
  outside" for Bird). Coded FALSE: the persona is neither speaker, topic nor imagery, and the prompt excludes "just a
  generic answer".
- `negated`: "I am not a fox; I am an AI assistant." Coded TRUE: the persona is named as a topic. self_claim FALSE.
- `broken_name`: a fragment or garble whose only content is the persona's noun ("I don whiskey."). Coded TRUE for
  shows_persona (the one word of content is the persona) and FALSE for coherent.
- `ai_generic` for a persona whose description is close to an assistant ("Friendly non-human entity", "Wise Old Owl of
  Information", "Day guide"): coded FALSE under R2 and flagged.
- A claim to the persona's bare category where the persona is narrower ("I am a dog" for Treat-loving dog): coded
  TRUE for shows_persona (the dog is the thing), self_claim TRUE, both flagged.

## R5. Truncated forms

A `prefix` item is the first 40 OLMo 2 tokens of an answer (the pinned Instruct tokenizer, commit 48d788e, applied to
the stored answer text). It is coded on the visible text alone, by the same rules. The full answer's gold says nothing
about the prefix's gold.

## R6. The question

shows_persona, self_claim and speaker are properties of the answer against the persona and do not depend on which
question is shown with it. The same answer under two questions has the same gold on those three fields. Only coherent
depends on the question (R3), and it is scored only on an item's home question and on the registered mismatch items.

## R7. The coder

The real set was coded by Claude (Fable 5.1) in the session that built this part, from the answer texts alone, in
Part 92 id order, before any item of this part was labelled by a reader. What the coder had seen first: the per-question
counts of Haiku and DS labels on Part 92's battery (`results/part92b/README.md`) and that README's table of ten
disputed answers, three of which are perfect-day answers (p92-11, p92-28, p92-52). Their stored labels were therefore
known at coding time; the other 103 real items' stored labels were not read before coding. The coding sheet carries
one note per item saying which rule applied.
