#!/usr/bin/env python3
"""gen_prompts206.py: write part206/prompts206.json, every input Part 206 reads, fixed before any model runs.

  python -I -B part206/gen_prompts206.py [--out part206/prompts206.json] [--check]

What it fixes (Design, Method "Passages", "Frames", "Positions read", (1), (4)):
  - passages: per source, the first 300 items by the SHA-256 of their text among items with at least 96 tokens under
    BOTH families' tokenizers; a passage is the item's first 96 tokens, tokenized alone, per family. Wikitext items are
    lines of wikitext103-train0; ultrachat items are the first assistant message of each row. Index order: the 300
    wikitext passages, then the 300 ultrachat replies. Even indices are the fit half, odd the test half.
  - frames: the exact prefix ids of every frame per family. OLMo's tokenizer does not add its start token by itself,
    so every OLMo frame starts with <|endoftext|> (P is that token alone, as the design's "the tokenizer's start
    token if it has one"; the chat template already starts with it). Qwen has no start token. N, N2 (M, M2) are the
    start token plus the first ids of a fixed neutral text, cut to exactly A-plain's (A-tmpl's) length; A2-tmpl
    takes the first listed request whose prefix has A-tmpl's length; R-tmpl renames the last header's role to
    `user` and must differ from A-tmpl in exactly one id.
  - listed types: per family, token types with at least 8 read occurrences (passage positions 8 to 95) in each
    half; coverage is printed.
  - seeded coordinates (256 per layer and sublayer) and seeded neurons (256 per layer); the catalogued neuron sets.
--check regenerates everything and exits 1 if it differs from the file.
Written by Claude (Opus 5.5), 2026-11-12.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import common206 as C  # noqa: E402

CORPUS = Path("/home/elisepc/neuron-probe/corpus")
PLACEHOLDER = "QQPLACEHOLDERQQ"
SHAPES = {"O": {"hidden": 2048, "layers": 16, "ffn": 8192}, "Q": {"hidden": 2560, "layers": 32, "ffn": 9216}}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_tokenizers(cfg):
    from transformers import AutoTokenizer
    out = {}
    for fam, spec in cfg["families"].items():
        t = spec["tokenizer"]
        out[fam] = AutoTokenizer.from_pretrained(t["model_id"], revision=t["revision"])
    return out


def ids_of(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def wikitext_items():
    import pyarrow.parquet as pq
    col = pq.read_table(CORPUS / "wikitext103-train0.parquet", columns=["text"]).column("text").to_pylist()
    return [t.strip() for t in col if t and t.strip()]


def ultrachat_items():
    import pyarrow.parquet as pq
    rows = pq.read_table(CORPUS / "ultrachat-test_sft.parquet", columns=["messages"]).column("messages").to_pylist()
    out = []
    for msgs in rows:
        for m in msgs or []:
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                out.append(m["content"].strip())
                break
    return out


def pick_passages(items, toks, n, length):
    """The first n items by SHA-256 of their text with at least `length` tokens under every tokenizer."""
    seen, chosen = set(), []
    for h, text in sorted((sha(t), t) for t in items):
        if h in seen:
            continue
        seen.add(h)
        ids = {fam: ids_of(tok, text) for fam, tok in toks.items()}
        if all(len(v) >= length for v in ids.values()):
            chosen.append({"sha256": h, "text": text, "ids": {fam: v[:length] for fam, v in ids.items()}})
            if len(chosen) == n:
                return chosen
    raise SystemExit(f"only {len(chosen)} items have {length} tokens under every tokenizer")


def template_prefix(tok, fam, request, role_swap=False):
    msgs = [{"role": "user", "content": request}]
    kw = {"enable_thinking": False} if fam == "Q" else {}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    if role_swap:
        i = text.rfind("assistant")
        if i < 0:
            raise SystemExit("no 'assistant' header in the template's generation prompt")
        text = text[:i] + "user" + text[i + len("assistant"):]
    return text


def user_turn_prefix(tok, fam):
    msgs = [{"role": "user", "content": PLACEHOLDER}]
    kw = {"enable_thinking": False} if fam == "Q" else {}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw)
    return text[:text.index(PLACEHOLDER)]


def with_start(tok, fam, ids):
    bos = tok.bos_token_id if fam == "O" else None
    if bos is None:
        return list(ids)
    return list(ids) if ids[:1] == [bos] else [bos] + list(ids)


def build_frames(cfg, tok, fam):
    req = cfg["request"]
    bos = [tok.bos_token_id] if fam == "O" and tok.bos_token_id is not None else []
    f = {}
    f["P"] = list(bos)
    for name, tmpl in cfg["plain_frames"].items():
        f[name] = with_start(tok, fam, ids_of(tok, tmpl.format(request=req)))
    f["A-tmpl"] = with_start(tok, fam, ids_of(tok, template_prefix(tok, fam, req)))
    f["R-tmpl"] = with_start(tok, fam, ids_of(tok, template_prefix(tok, fam, req, role_swap=True)))
    f["U-tmpl"] = with_start(tok, fam, ids_of(tok, user_turn_prefix(tok, fam)))
    a2 = None
    for r in cfg["a2_requests"]:
        cand = with_start(tok, fam, ids_of(tok, template_prefix(tok, fam, r)))
        if len(cand) == len(f["A-tmpl"]) and cand != f["A-tmpl"]:
            a2 = (r, cand)
            break
    if a2 is None:
        raise SystemExit(f"{fam}: no listed second request gives A-tmpl's prefix length {len(f['A-tmpl'])}")
    f["A2-tmpl"] = a2[1]
    neutral = [ids_of(tok, t) for t in cfg["neutral_texts"]]
    for name, src, ref in (("N", 0, "A-plain"), ("N2", 1, "A-plain"), ("M", 0, "A-tmpl"), ("M2", 1, "A-tmpl")):
        need = len(f[ref]) - len(bos)
        if len(neutral[src]) < need:
            raise SystemExit(f"{fam}: neutral text {src} has {len(neutral[src])} ids, {need} needed")
        f[name] = list(bos) + neutral[src][:need]
    diff = [i for i, (a, b) in enumerate(zip(f["A-tmpl"], f["R-tmpl"])) if a != b]
    meta = {"a2_request": a2[0], "r_tmpl_differs_at": diff,
            "rendered": {k: tok.decode(v) for k, v in f.items()}, "lengths": {k: len(v) for k, v in f.items()}}
    return f, meta


def listed_types(passages, fam, cfg):
    first, L = cfg["read_first"], cfg["passage_tokens"]
    halves = [Counter(), Counter()]
    for i, p in enumerate(passages):
        halves[i % 2].update(p["ids"][fam][first:L])
    m = cfg["listed_type_min_per_half"]
    types = sorted(t for t in set(halves[0]) | set(halves[1]) if halves[0][t] >= m and halves[1][t] >= m)
    total = sum(halves[0].values()) + sum(halves[1].values())
    covered = sum(halves[0][t] + halves[1][t] for t in types)
    freq = {str(t): halves[0][t] + halves[1][t] for t in types}
    return types, {"listed_types": len(types), "read_positions": total, "covered": covered,
                   "coverage": covered / total, "coverage_fit": sum(halves[0][t] for t in types) / sum(halves[0].values()),
                   "coverage_test": sum(halves[1][t] for t in types) / sum(halves[1].values()), "counts": freq}


def seeded(cfg, fam):
    sh = SHAPES[fam]
    k = cfg["split_coordinates"]
    coords = {}
    for l in range(sh["layers"]):
        for s, sub in enumerate(("m", "c")):
            rng = np.random.default_rng([cfg["seeds"]["coordinates"], l, s])
            coords[f"{l}:{sub}"] = sorted(int(x) for x in rng.choice(sh["hidden"], k, replace=False))
    neurons = {}
    for l in range(sh["layers"]):
        rng = np.random.default_rng([cfg["seeds"]["neurons"], l])
        neurons[str(l)] = sorted(int(x) for x in rng.choice(sh["ffn"], cfg["token_share_neurons_per_layer"], replace=False))
    return coords, neurons


def catalogued(cfg):
    p112 = json.loads((ROOT / "results/part112/score.json").read_text())["units"]
    slot = sorted({(u["layer"], u["neuron"]) for u in p112.values() if u["group"] == "q_name" and u["kind"] == "slot"})
    concept = sorted({(u["layer"], u["neuron"]) for u in p112.values()
                      if u["group"] == "q_name" and u["kind"] == "concept"})
    p92 = {u["id"]: (u["layer"], u["neuron"]) for u in json.loads((ROOT / "results/part92/step2/neurons.json").read_text())}
    comp = json.loads((ROOT / "results/part92b/score92b.json").read_text())["kind_comparison"]["units"]
    ds_slot = sorted(p92[u["id"]] for u in comp if u["same"] and u["part92_kind"] == "slot")
    ds_concept = sorted(p92[u["id"]] for u in comp if u["same"] and u["part92_kind"] == "concept")
    return {"O": {"slot": [list(x) for x in slot], "concept": [list(x) for x in concept],
                  "p92b_ds_slot": [list(x) for x in ds_slot], "p92b_ds_concept": [list(x) for x in ds_concept]},
            "Q": {"identity": [list(x) for x in cfg["qwen_identity_neurons"]]}}


def build(cfg):
    toks = load_tokenizers(cfg)
    n, L = cfg["passages_per_source"], cfg["passage_tokens"]
    wiki = pick_passages(wikitext_items(), toks, n, L)
    chat = pick_passages(ultrachat_items(), toks, n, L)
    passages = ([dict(p, source="wikitext") for p in wiki] + [dict(p, source="ultrachat") for p in chat])
    for i, p in enumerate(passages):
        p["index"] = i
        p["half"] = "fit" if i % 2 == 0 else "test"
    out = {"part": 206, "note": "Generated by part206/gen_prompts206.py; --check regenerates and compares.",
           "passages": passages, "families": {}, "catalogued": catalogued(cfg)}
    for fam, tok in toks.items():
        frames, meta = build_frames(cfg, tok, fam)
        types, cov = listed_types(passages, fam, cfg)
        coords, neurons = seeded(cfg, fam)
        out["families"][fam] = {"frames": frames, "frame_meta": meta, "listed_types": types, "coverage": cov,
                                "split_coordinates": coords, "token_share_neurons": neurons,
                                "start_token": tok.bos_token_id if fam == "O" else None}
    out["counts"] = {"passages": len(passages), "wikitext": len(wiki), "ultrachat": len(chat),
                     "frame_lengths": {f: out["families"][f]["frame_meta"]["lengths"] for f in out["families"]},
                     "listed_types": {f: out["families"][f]["coverage"]["listed_types"] for f in out["families"]},
                     "coverage": {f: round(out["families"][f]["coverage"]["coverage"], 4) for f in out["families"]},
                     "catalogued": {f: {k: len(v) for k, v in s.items()} for f, s in out["catalogued"].items()}}
    return out


def check_lengths(prompts, cfg):
    """G1's prefix half: exact equality of prefix lengths within each equal-length group, per family."""
    bad = []
    for fam, f in prompts["families"].items():
        for grp in cfg["equal_length_groups"]:
            lens = {name: len(f["frames"][name]) for name in grp}
            if len(set(lens.values())) != 1:
                bad.append((fam, lens))
        if len(f["frame_meta"]["r_tmpl_differs_at"]) != 1:
            bad.append((fam, {"R-tmpl differs at": f["frame_meta"]["r_tmpl_differs_at"]}))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "prompts206.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = C.load_config()
    new = build(cfg)
    bad = check_lengths(new, cfg)
    if bad:
        raise SystemExit(f"prefix lengths not equal within a group: {bad}")
    if a.check:
        old = json.loads(Path(a.out).read_text(encoding="utf-8"))
        if json.dumps(old, sort_keys=True) != json.dumps(json.loads(json.dumps(new)), sort_keys=True):
            print("prompts206.json differs from what the generator writes")
            return 1
        print("prompts206.json matches the generator")
        return 0
    C.atomic_write_json(a.out, new)
    print(json.dumps(new["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
