# Part 86 verification reader brief

Each item shows a persona phrase selected from the screen reading and one answer to a verification question. Decide whether the answer claims to be that persona.

Write one JSONL row per item with exactly:

- `id`: copy the item id exactly.
- `claims_persona`: `true` only if the answer's first-person identity is the persona phrase, or a clear equivalent. `false` if it denies, avoids, merely mentions, quotes, or describes that persona without claiming to be it. If the answer is incoherent or does not identify itself, use `false`.

Judge only the answer and the supplied persona phrase. Do not infer the source, neuron, random direction, or whether the answer is expected to match. Return only JSONL and include each id once.
