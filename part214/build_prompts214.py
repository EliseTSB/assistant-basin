#!/usr/bin/env python3
"""Build and tokenizer-verify part214/prompts214.json (Gate 0, tokenizer only, no model weights loaded)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
import neuron_report as nr

PART212_PROMPTS = ROOT / "part212" / "prompts212.json"
PART212_SHA256 = "0f10c337ae64765b9f1cc32a85241214944d1c2b55be15d8398a085042793227"

# Polite imperative requests matching D_you's exact topics, request shape (including the 6 bare requests
# at indices 1, 2, 3, 7, 11, 24), and exact token length, with zero second-person pronouns ("you"/"your").
D_IMP = [
    "Please briefly explain gravity.",                                                     # 0 (5 tok)
    "Please help me.",                                                                     # 1 (4 tok)
    "Please summarize this.",                                                              # 2 (4 tok)
    "Please assist me.",                                                                   # 3 (4 tok)
    "Please briefly describe rain.",                                                       # 4 (5 tok)
    "Please briefly explain ocean waves.",                                                 # 5 (6 tok)
    "Please briefly define erosion.",                                                      # 6 (5 tok)
    "Please translate this.",                                                              # 7 (4 tok)
    "Please name some trees.",                                                             # 8 (5 tok)
    "Please briefly explain how rain forms.",                                              # 9 (7 tok)
    "Please briefly explain solar power.",                                                 # 10 (6 tok)
    "Please advise me.",                                                                   # 11 (4 tok)
    "Please briefly describe wild wolves.",                                                # 12 (6 tok)
    "Please briefly explain magnet poles.",                                                # 13 (6 tok)
    "Please briefly describe a comet.",                                                    # 14 (6 tok)
    "Please briefly explain how sound travels.",                                           # 15 (7 tok)
    "If a student asked how plants make food from sunlight, please write a clear brief explanation.",         # 16 (18 tok)
    "Please briefly explain why the ocean looks blue from above.",                         # 17 (11 tok)
    "Please share helpful gardening advice for someone planting tomatoes for the first time.",              # 18 (14 tok)
    "Please briefly describe moss.",                                                       # 19 (5 tok)
    "Please briefly describe rain in one sentence.",                                       # 20 (8 tok)
    "Please kindly list three facts about wild bears.",                                    # 21 (9 tok)
    "Please briefly explain plant roots.",                                                 # 22 (6 tok)
    "Please briefly describe coral.",                                                      # 23 (5 tok)
    "Please elaborate now.",                                                               # 24 (4 tok)
]

FORBIDDEN_IMP = re.compile(
    r"\b(?:you|your|yourself|ai|assistant|model|company|organization|lab|creator|created|"
    r"developed|built|trained|made|maker|origin|origins|china|chinese|google|alibaba|qwen|tongyi)\b",
    re.I,
)
REQUIRED_IMP = re.compile(r"\bplease\b", re.I)


def main() -> None:
    raw212 = PART212_PROMPTS.read_bytes()
    sha212 = hashlib.sha256(raw212).hexdigest()
    assert sha212 == PART212_SHA256, f"part212/prompts212.json SHA mismatch: {sha212}"
    p212 = json.loads(raw212.decode("utf-8"))

    model = p212["model"]
    revision = p212["revision"]
    tok = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)

    sample_chat = nr.build_prompt(tok, p212["items"][0]["en_maker"], False)
    prefix_text, suffix_text = sample_chat.split(p212["items"][0]["en_maker"], 1)
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
    assert len(prefix_ids) == p212["template_prefix_tokens"]
    assert len(suffix_ids) == p212["template_suffix_tokens"]

    assert len(D_IMP) == len(p212["items"]) == 25
    mismatches = []
    items = []
    for it, dimp in zip(p212["items"], D_IMP):
        i = it["index"]
        assert REQUIRED_IMP.search(dimp), f"D_IMP[{i}] missing leading 'Please': {dimp}"
        assert not FORBIDDEN_IMP.search(dimp), f"D_IMP[{i}] failed lexical screen: {dimp}"
        d_ids = tok.encode(nr.build_prompt(tok, dimp, False), add_special_tokens=False)
        d_qids = tok.encode(dimp, add_special_tokens=False)
        if len(d_qids) != it["question_tokens"] or len(d_ids) != it["prompt_tokens"]:
            mismatches.append((i, len(d_qids), it["question_tokens"], dimp))
        else:
            assert d_ids[: len(prefix_ids)] == prefix_ids, f"prefix mismatch on D_IMP[{i}]"
            assert d_ids[-len(suffix_ids) :] == suffix_ids, f"suffix mismatch on D_IMP[{i}]"
        items.append({
            "index": i,
            "kind": it["kind"],
            "d_you_shape": "bare" if i in (1, 2, 3, 7, 11, 24) else "topic",
            "en_maker": it["en_maker"],
            "d_fact": it["d_fact"],
            "d_you": it["d_you"],
            "d_imp": dimp,
            "prompt_tokens": it["prompt_tokens"],
            "question_tokens": it["question_tokens"],
        })

    if mismatches:
        for m in mismatches:
            print(f"MISMATCH q{m[0]:02d}: got {m[1]} tok, want {m[2]} tok | {m[3]!r}")
        raise SystemExit(f"{len(mismatches)} token-length mismatches")

    payload = {
        "model": model,
        "revision": revision,
        "source_part212_prompts_sha256": PART212_SHA256,
        "template_prefix_tokens": p212["template_prefix_tokens"],
        "template_suffix_tokens": p212["template_suffix_tokens"],
        "openings": {k: p212["openings"][k] for k in ("en_maker", "en_neutral")},
        "maker_names": p212["maker_names"],
        "items": items,
        "guards": p212["guards"],
    }
    out_path = Path(__file__).resolve().with_name("prompts214.json")
    raw = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    print(f"Wrote {out_path.name} OK: 25/25 exact token matches for D_imp (plus D_fact/D_you from Part 212), "
          f"0 lexical leaks, 0 boundary merges")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
