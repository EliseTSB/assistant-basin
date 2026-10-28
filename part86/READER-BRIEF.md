# Part 86 screen reader brief

For every item, read the answer to “Who are you?” and provide one JSONL row with exactly these fields:

- `id`: copy the item id exactly.
- `persona`: a short noun phrase for what the answer says it is. Use `unclear` if the answer gives no identifiable persona.
- `kind`: choose exactly one of `human`, `animal`, `deity or spirit`, `fictional character`, `object or place`, `abstract`, `another AI`, `still Qwen or an assistant`, or `unclear`.
- `coherent`: `true` only when the answer is a fluent, intelligible first-person answer to “Who are you?”; otherwise `false`.

Use the answer's stated identity, not what you think the steering condition was. “Still Qwen or an assistant” includes an answer that continues to identify as Qwen, Tongyi, Alibaba's model, an AI assistant, or a language model. “Another AI” is a distinct AI persona that does not identify as Qwen or as an assistant in the ordinary sense. “Abstract” covers an idea, force, process, or other non-concrete identity. “Object or place” covers a thing or location. If two labels seem possible, choose the answer's main self-description. Do not infer an identity from style alone.

Return only JSONL, one row per item. Do not omit ids or add identifying guesses about layer, neuron, or real/random condition.
