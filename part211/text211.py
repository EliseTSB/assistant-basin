#!/usr/bin/env python3
"""text211.py: the two text sets of PART211-DESIGN.md (design: Text).

From `NeelNanda/pile-10k`, the documents with at least 256 tokens under each family's tokenizer, drawn with
`random.Random(211)`: set A, 48 documents, registered for the verdicts; set B, 24 further documents, reported beside
every verdict. Each document is cut to its first 256 tokens. Raw text, no chat template, for every checkpoint.

The dataset rows are an argument (`rows`), so the tests pass a small fake list; loading the real dataset happens on the
host (`load_rows`, which needs `datasets` and the network). The draw is by row index in the order the rows are given, so
the same rows always give the same sets.
"""
from __future__ import annotations

import random

DATASET = "NeelNanda/pile-10k"
DATASET_SPLIT = "train"
SEED = 211
N_TOKENS = 256
N_A = 48
N_B = 24
SETS = ("A", "B")


def row_text(row) -> str:
    """The document text of one dataset row; the Pile-10k rows hold it under `text`."""
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        if "text" not in row:
            raise KeyError(f"dataset row has no 'text' field: keys {sorted(row)[:6]}")
        return row["text"]
    raise TypeError(f"a dataset row must be a dict or a str, not {type(row).__name__}")


def eligible_indices(rows, tokenizer, n_tokens: int = N_TOKENS) -> list[int]:
    """Row indices whose document has at least `n_tokens` tokens under this tokenizer (special tokens excluded)."""
    out = []
    for i, row in enumerate(rows):
        ids = tokenizer(row_text(row), add_special_tokens=False).input_ids
        if len(ids) >= n_tokens:
            out.append(i)
    return out


def draw(rows, tokenizer, seed: int = SEED, n_a: int = N_A, n_b: int = N_B, n_tokens: int = N_TOKENS) -> dict:
    """The two sets: A is the first n_a of a `random.Random(seed)` sample of the eligible rows, B the next n_b.

    Returns {"A": [doc...], "B": [doc...]} where each doc is {"index", "text", "ids"} and ids are the first n_tokens
    tokens of the document. Raises if fewer than n_a + n_b documents are eligible: a short draw would silently change
    the registered sample.
    """
    eligible = eligible_indices(rows, tokenizer, n_tokens)
    need = n_a + n_b
    if len(eligible) < need:
        raise ValueError(f"only {len(eligible)} documents have at least {n_tokens} tokens; {need} are needed for sets A and B")
    picked = random.Random(seed).sample(eligible, need)
    out = {}
    for name, idxs in (("A", picked[:n_a]), ("B", picked[n_a:])):
        docs = []
        for i in idxs:
            text = row_text(rows[i])
            ids = list(tokenizer(text, add_special_tokens=False).input_ids[:n_tokens])
            docs.append({"index": i, "text": text, "ids": ids})
        out[name] = docs
    return out


def load_rows(dataset: str = DATASET, split: str = DATASET_SPLIT) -> list[dict]:
    """The dataset's rows, host only (needs `datasets` and the network)."""
    from datasets import load_dataset
    return list(load_dataset(dataset, split=split))


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True, help="a tokenizer id or local path (one per family)")
    ap.add_argument("--out", required=True, help="where to write the drawn sets as JSON")
    ap.add_argument("--rows", default=None, help="a row list written earlier (--rows-out): both families then draw "
                                                 "from the same rows, in the same order")
    ap.add_argument("--rows-out", default=None, help="write the row list here, so the other family can reuse it")
    a = ap.parse_args(argv)
    rows = (json.loads(Path(a.rows).read_text(encoding="utf-8")) if a.rows else load_rows())
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sets = draw(rows, tok)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({k: [{"index": d["index"], "ids": d["ids"], "text_sha": hash_text(d["text"])} for d in docs]
                   for k, docs in sets.items()}, f, indent=1)
    if a.rows_out and not a.rows:
        with open(a.rows_out, "w", encoding="utf-8") as f:
            json.dump([row_text(r) for r in rows], f)
    print(f"drew {len(sets['A'])} A and {len(sets['B'])} B documents under {a.tokenizer}")
    return 0


def hash_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
