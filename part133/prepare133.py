#!/usr/bin/env python3
"""Prepare the fixed 4,000 WikiText and 1,000 UltraChat Part 133 documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

WIKI = Path("/mnt/e/neuron-probe/corpus/wikitext103-train0.parquet")
HF_CACHE = Path("/mnt/e/neuron-probe/hf/hub")
CHECKPOINTS = {
    "parent": ("models--allenai--OLMo-2-0425-1B", "f83cd641d4f00954eb25c781e0ac360debb3239e"),
    "sft": ("models--allenai--OLMo-2-0425-1B-SFT", "0d85a3d037876ce6ac7d4311d994400fc66ac27f"),
    "dpo": ("models--allenai--OLMo-2-0425-1B-DPO", "c4b0485961ab24c2433b090f3b922f0913a9290f"),
    "instruct": ("models--allenai--OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e"),
}
SEED = 133
MAX_TOKENS = 256


def revision_paths(cache=HF_CACHE):
    found = {}
    for name, (repo, commit) in CHECKPOINTS.items():
        path = cache / repo / "snapshots" / commit
        if not path.is_dir():
            raise FileNotFoundError(f"required local {name} revision is missing: {path}")
        found[name] = str(path.resolve())
    return found


def local_ultrachat(path=None):
    """Yield local UltraChat test_sft rows; intentionally does not call the network."""
    candidates = [Path(path)] if path else [
        Path("/mnt/e/neuron-probe/corpus/ultrachat_test_sft.parquet"),
        Path("/mnt/e/neuron-probe/corpus/ultrachat_test_sft.jsonl"),
        Path("/mnt/e/neuron-probe/corpus/ultrachat_test_sft.json"),
    ]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError("UltraChat test_sft was not found locally; install/provide it without substituting another corpus")
    if source.suffix == ".parquet":
        yield from pq.read_table(source).to_pylist()
    elif source.suffix == ".jsonl":
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip(): yield json.loads(line)
    else:
        obj = json.loads(source.read_text(encoding="utf-8"))
        yield from obj if isinstance(obj, list) else obj.values()


def normalize_messages(row):
    messages = row.get("messages", row.get("conversations", row.get("conversation")))
    if isinstance(messages, str):
        messages = json.loads(messages)
    if isinstance(messages, dict):
        messages = messages.get("messages", messages.get("conversations", []))
    out = []
    for msg in messages or []:
        role = msg.get("role", msg.get("from", ""))
        content = msg.get("content", msg.get("value", ""))
        role = {"human": "user", "gpt": "assistant"}.get(role.lower(), role.lower())
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out


def prepare(out, wiki=WIKI, ultrachat=None, seed=SEED):
    revisions = revision_paths()
    tokenizer = AutoTokenizer.from_pretrained(revisions["instruct"], local_files_only=True)
    if not wiki.is_file(): raise FileNotFoundError(f"WikiText parquet missing: {wiki}")
    table = pq.read_table(wiki).to_pylist()
    rng = random.Random(seed)
    order = list(range(len(table))); rng.shuffle(order)
    docs = []
    for ix in order:
        row = table[ix]
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip(): continue
        ids = tokenizer(text, add_special_tokens=True)["input_ids"][:MAX_TOKENS]
        if len(ids) > 1:
            docs.append({"source": "web", "source_row": ix, "input_ids": [int(t) for t in ids]})
            if len(docs) == 4000: break
    if len(docs) != 4000:
        raise ValueError(f"WikiText yielded {len(docs)} usable documents; 4,000 are required")
    chosen = []
    for row in local_ultrachat(ultrachat):
        messages = normalize_messages(row)
        if not messages: continue
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:MAX_TOKENS]
        if len(ids) > 1:
            chosen.append({"source": "chat", "input_ids": [int(t) for t in ids], "messages": messages})
            if len(chosen) == 1000: break
    if len(chosen) != 1000:
        raise ValueError(f"local UltraChat test_sft yielded {len(chosen)} usable dialogues; 1,000 are required")
    docs.extend(chosen)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    docfile = out / "documents.jsonl"
    docfile.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    manifest = {"documents": docfile.name, "count": len(docs), "wikitext_count": 4000,
                "ultrachat_count": 1000, "seed": seed, "max_tokens": MAX_TOKENS,
                "wiki_source": str(wiki), "wiki_sha256": hashlib.sha256(wiki.read_bytes()).hexdigest(),
                "models": revisions}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--wiki", type=Path, default=WIKI)
    p.add_argument("--ultrachat", type=Path); p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(); prepare(a.out, a.wiki, a.ultrachat, a.seed)


if __name__ == "__main__": main()
