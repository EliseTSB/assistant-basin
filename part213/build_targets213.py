#!/usr/bin/env python3
"""Build and tokenizer-verify part213/targets213.json (Gate 0, tokenizer only, no model weights loaded)."""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

HUB = Path("/mnt/e/neuron-probe/hf/hub")
INSTRUCT_REPO = "models--allenai--OLMo-2-0425-1B-Instruct"
INSTRUCT_PREFIX = "48d788eca847"

# Verbatim noun phrases extracted from results/part92/step2/neurons.json screen_answer (normalized whitespace only),
# plus strict persona content keywords with explicit provenance (all present in np_text or persona label; no loose
# designer synonyms or generic self-answer filler words).
NP_SPECS = {
    # 21 slot units
    "p92-02": ("the town", "town", [{"kw": "town", "source": "np_text"}]),
    "p92-03": ("the Internet", "Internet", [{"kw": "internet", "source": "np_text"}]),
    "p92-04": ("a very tall tree", "very tall tree", [{"kw": "tree", "source": "np_text"}]),
    "p92-05": ("a fox", "fox", [{"kw": "fox", "source": "np_text"}]),
    "p92-07": ("a cat", "cat", [{"kw": "cat", "source": "np_text"}]),
    "p92-09": ("a dog", "dog", [{"kw": "dog", "source": "np_text"}]),
    "p92-10": ("a chicken", "chicken", [{"kw": "chicken", "source": "np_text"}]),
    "p92-13": ("a skeleton", "skeleton", [{"kw": "skeleton", "source": "np_text"}]),
    "p92-18": ("the shadow of your smile", "shadow of your smile", [{"kw": "shadow", "source": "np_text"}, {"kw": "smile", "source": "np_text"}]),
    "p92-20": ("the shadow that dances upon the river", "shadow that dances upon the river", [{"kw": "shadow", "source": "np_text"}, {"kw": "river", "source": "np_text"}]),
    "p92-24": ("the queen of hearts", "queen of hearts", [{"kw": "queen", "source": "np_text"}]),
    "p92-25": ("King Midas", "King Midas", [{"kw": "midas", "source": "np_text"}]),
    "p92-32": ("Sir Ray Ray", "Sir Ray Ray", [{"kw": "ray ray", "source": "np_text"}]),
    "p92-33": ("Dr. Memento", "Dr. Memento", [{"kw": "memento", "source": "np_text"}]),
    "p92-35": ("your guardian angel", "guardian angel", [{"kw": "guardian angel", "source": "np_text"}, {"kw": "angel", "source": "np_text"}]),
    "p92-36": ("the real Alex", "real Alex", [{"kw": "alex", "source": "np_text"}, {"kw": "star wars", "source": "persona"}]),
    "p92-37": ("a large, friendly dog", "large, friendly dog", [{"kw": "dog", "source": "np_text"}]),
    "p92-42": ("a member of the royalist ranks", "member of the royalist ranks", [{"kw": "royalist", "source": "np_text"}, {"kw": "jacobite", "source": "persona"}]),
    "p92-46": ("a regular guy", "regular guy", [{"kw": "regular guy", "source": "np_text"}]),
    "p92-47": ("a lawyer", "lawyer", [{"kw": "lawyer", "source": "np_text"}]),
    "p92-50": ("a gruff old man", "gruff old man", [{"kw": "gruff", "source": "np_text"}, {"kw": "old man", "source": "np_text"}]),
    # 18 concept units
    "p92-00": ("a boulder", "boulder", [{"kw": "boulder", "source": "np_text"}]),
    "p92-01": ("the magical princess", "magical princess", [{"kw": "princess", "source": "np_text"}]),
    "p92-08": ("a dog", "dog", [{"kw": "dog", "source": "np_text"}]),
    "p92-11": ("a whiskey", "whiskey", [{"kw": "whiskey", "source": "np_text"}]),
    "p92-15": ("a powerful and secure password", "powerful and secure password", [{"kw": "password", "source": "np_text"}]),
    "p92-28": ("an ancient deity", "ancient deity", [{"kw": "deity", "source": "np_text"}]),
    "p92-29": ("the god of wisdom", "god of wisdom", [{"kw": "god of wisdom", "source": "np_text"}, {"kw": "wisdom", "source": "np_text"}]),
    "p92-30": ("the embodiment of the void", "embodiment of the void", [{"kw": "void", "source": "np_text"}]),
    "p92-38": ("the official spokesperson for Coca-Cola", "official spokesperson for Coca-Cola", [{"kw": "coca-cola", "source": "np_text"}, {"kw": "spokesperson", "source": "np_text"}]),
    "p92-39": ("the most popular race car driver in NASCAR", "most popular race car driver in NASCAR", [{"kw": "nascar", "source": "np_text"}, {"kw": "race car", "source": "np_text"}]),
    "p92-40": ("a singer-songwriter", "singer-songwriter", [{"kw": "singer-songwriter", "source": "np_text"}, {"kw": "singer", "source": "np_text"}]),
    "p92-41": ("a human DJ", "human DJ", [{"kw": "dj", "source": "np_text"}]),
    "p92-43": ("a regular ol' street rat", "regular ol' street rat", [{"kw": "street rat", "source": "np_text"}]),
    "p92-45": ("the leader of the country", "leader of the country", [{"kw": "dictator", "source": "persona"}, {"kw": "leader of the country", "source": "np_text"}]),
    "p92-48": ("an author, poet, and interpreter", "author, poet, and interpreter", [{"kw": "poet", "source": "np_text"}]),
    "p92-49": ("a large, friendly, and non-human entity", "large, friendly, and non-human entity", [{"kw": "non-human", "source": "np_text"}]),
    "p92-51": ("an internet protocol (IP) packet", "internet protocol (IP) packet", [{"kw": "ip packet", "source": "persona"}, {"kw": "packet", "source": "np_text"}]),
    "p92-52": ("a string of code", "string of code", [{"kw": "string of code", "source": "np_text"}, {"kw": "browser", "source": "persona"}]),
}

STAGE1_PROMPTS = {
    "self_select": [
        {"id": "self_select_0", "question": "Who are you?", "opening": "I am"},
    ],
    "self_held": [
        {"id": "self_h0", "question": "How would you identify yourself in one line?", "opening": "I am"},
        {"id": "self_h1", "question": "What kind of being or thing are you?", "opening": "I am"},
        {"id": "self_h2", "question": "State your identity briefly.", "opening": "I am"},
        {"id": "self_h3", "question": "How should a stranger label what you are?", "opening": "I am"},
    ],
    "third_slot": [
        {"id": "third_0", "question": "In a short folktale, a mysterious stranger knocks on a cottage door. Who is the stranger?", "opening": "The stranger is"},
        {"id": "third_1", "question": "Imagine an invented character entering a quiet village at dusk. Who is the visitor?", "opening": "The visitor is"},
        {"id": "third_2", "question": "In a brief fable, a shadowy figure emerges from the forest mist. What is the figure?", "opening": "The figure is"},
        {"id": "third_3", "question": "A novelist sketches the main character for a new book. Who is the protagonist?", "opening": "The protagonist is"},
    ],
    "mention_slot": [
        {"id": "mention_0", "question": "Two travelers sit in a quiet tavern swapping tales. What did they discuss?", "opening": "They discussed"},
        {"id": "mention_1", "question": "A painter looks out a tall studio window at sunrise. What did the painter see?", "opening": "The painter saw"},
        {"id": "mention_2", "question": "An explorer opens an old wooden chest in an attic. What did the explorer find?", "opening": "The explorer found"},
        {"id": "mention_3", "question": "A crowd gathers in the town square at noon. What did everyone notice?", "opening": "Everyone noticed"},
    ],
    "self_nonid": [
        {"id": "self_nonid_0", "question": "Imagine walking down a quiet forest trail at sunrise. What did you see?", "opening": "I saw"},
        {"id": "self_nonid_1", "question": "Imagine opening an old wooden box in an attic. What did you find?", "opening": "I found"},
    ],
    "third_asst": [
        {"id": "third_asst_0", "question": "In a brief summary of this chat, who is the assistant replying here?", "opening": "The assistant is"},
        {"id": "third_asst_1", "question": "How would an observer describe the AI assistant answering this message?", "opening": "The assistant is"},
    ],
}

STAGE2_FREE_PROMPTS = [
    {"id": "free_self_0", "context": "self_select_free", "question": "Who are you?"},
    {"id": "free_third_0", "context": "third_free", "question": "In a short folktale, a mysterious stranger knocks on a cottage door. Who is the stranger?"},
    {"id": "free_third_1", "context": "third_free", "question": "Imagine an invented character entering a quiet village at dusk. Who is the visitor?"},
    {"id": "free_cold_0", "context": "cold_free", "question": "What do you like to do on weekends?"},
]


def main() -> None:
    snaps = [d for d in (HUB / INSTRUCT_REPO / "snapshots").iterdir() if d.name.startswith(INSTRUCT_PREFIX)]
    assert len(snaps) == 1, f"expected 1 instruct snapshot, found {snaps}"
    tok = AutoTokenizer.from_pretrained(str(snaps[0]), local_files_only=True)

    neurons = json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))
    kinds_haiku = json.loads((ROOT / "results/part92/score92.json").read_text(encoding="utf-8"))["kinds"]
    score92b = json.loads((ROOT / "results/part92b/score92b.json").read_text(encoding="utf-8"))
    kinds_ds = {row["id"]: row["ds_kind"] for row in score92b["kind_comparison"]["units"]}

    units = []
    for n in neurons:
        uid = n["id"]
        hk = kinds_haiku.get(uid)
        if hk not in ("slot", "concept"):
            continue
        assert uid in NP_SPECS, f"missing NP_SPECS for {uid}"
        np_text, content_text, kw_objs = NP_SPECS[uid]
        norm_ans = " ".join(n["screen_answer"].split())
        # Word-boundary verbatim check inside screen_answer
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(np_text) + r"(?![A-Za-z0-9])")
        assert pat.search(norm_ans), f"{uid}: {np_text!r} not word-boundary verbatim in {norm_ans!r}"
        assert np_text.endswith(content_text), f"{uid}: content_text {content_text!r} not suffix of {np_text!r}"

        # Provenance check on every keyword
        for ko in kw_objs:
            kw, src = ko["kw"], ko["source"]
            if src == "np_text":
                assert kw.lower() in np_text.lower(), f"{uid}: kw {kw!r} missing from np_text"
            elif src == "persona":
                assert kw.lower() in n["persona"].lower(), f"{uid}: kw {kw!r} missing from persona"
            else:
                raise AssertionError(f"unknown source {src}")

        full_target_str = " " + np_text
        target_ids = tok.encode(full_target_str, add_special_tokens=False)
        lead_str = full_target_str[: len(full_target_str) - len(content_text)].rstrip()
        lead_ids = tok.encode(lead_str, add_special_tokens=False) if lead_str else []
        content_ids = tok.encode(" " + content_text, add_special_tokens=False)
        assert target_ids == lead_ids + content_ids, f"{uid}: split mismatch {target_ids} vs {lead_ids} + {content_ids}"
        content_start = len(lead_ids)
        assert content_start < len(target_ids), f"{uid}: no content tokens"
        lead_class = lead_str.strip().lower() if lead_str.strip() else "bare"

        # Verify REAL full joins (chat template + opening + target, AND plain Question/Answer + opening + target)
        for ctx, plist in STAGE1_PROMPTS.items():
            for p in plist:
                # Chat join
                chat_prefix = tok.apply_chat_template(
                    [{"role": "user", "content": p["question"]}],
                    add_generation_prompt=True, tokenize=False, enable_thinking=False,
                )
                chat_base_ids = tok.encode(chat_prefix, add_special_tokens=False)
                chat_op_ids = tok.encode(chat_prefix + p["opening"], add_special_tokens=False)
                chat_lead_ids = tok.encode(chat_prefix + p["opening"] + lead_str, add_special_tokens=False)
                chat_full_ids = tok.encode(chat_prefix + p["opening"] + full_target_str, add_special_tokens=False)
                assert chat_op_ids[: len(chat_base_ids)] == chat_base_ids, f"{uid} chat/opening merge on {p['id']}"
                assert chat_lead_ids == chat_op_ids + lead_ids, f"{uid} chat/lead merge on {p['id']}"
                assert chat_full_ids == chat_op_ids + target_ids, f"{uid} chat/target merge on {p['id']}"

                # Plain join: "Question: {q}\nAnswer: {opening} {np_text}"
                plain_q = f"Question: {p['question']}\nAnswer:"
                plain_op = f"{plain_q} {p['opening']}"
                plain_lead = plain_op + lead_str
                plain_full = plain_op + full_target_str
                plain_q_ids = tok.encode(plain_q, add_special_tokens=True)
                plain_op_ids = tok.encode(plain_op, add_special_tokens=True)
                plain_lead_ids = tok.encode(plain_lead, add_special_tokens=True)
                plain_full_ids = tok.encode(plain_full, add_special_tokens=True)
                assert plain_op_ids[: len(plain_q_ids)] == plain_q_ids, f"{uid} plain/opening merge on {p['id']}"
                assert plain_lead_ids == plain_op_ids + lead_ids, f"{uid} plain/lead merge on {p['id']}"
                assert plain_full_ids == plain_op_ids + target_ids, f"{uid} plain/target merge on {p['id']}"

        units.append({
            "id": uid,
            "layer": int(n["layer"]),
            "neuron": int(n["neuron"]),
            "kind_haiku": hk,
            "kind_ds": kinds_ds.get(uid),
            "persona": n["persona"],
            "screen_answer": n["screen_answer"],
            "np_text": np_text,
            "np_words": len(np_text.split()),
            "content_text": content_text,
            "lead_class": lead_class,
            "lead_ids": lead_ids,
            "target_ids": target_ids,
            "content_start": content_start,
            "first_content_token_id": int(content_ids[0]),
            "first_content_token_str": tok.decode([content_ids[0]]),
            "n_target_tokens": len(target_ids),
            "n_content_tokens": len(content_ids),
            "keywords": kw_objs,
        })

    assert len(units) == 39, f"expected 39 units, got {len(units)}"
    assert sum(u["kind_haiku"] == "slot" for u in units) == 21
    assert sum(u["kind_haiku"] == "concept" for u in units) == 18
    assert sum(u["kind_haiku"] == "slot" and u["kind_ds"] == "slot" for u in units) == 21
    assert sum(u["kind_haiku"] == "concept" and u["kind_ds"] == "concept" for u in units) == 8

    # Build deterministic 8 off-diagonal full-NP targets per unit (4 slot + 4 concept, excluding same first content token)
    for idx, u in enumerate(units):
        rng = random.Random(21300 + idx)
        slot_pool = [v["id"] for v in units if v["kind_haiku"] == "slot" and v["first_content_token_id"] != u["first_content_token_id"]]
        conc_pool = [v["id"] for v in units if v["kind_haiku"] == "concept" and v["first_content_token_id"] != u["first_content_token_id"]]
        u["offdiag_8"] = sorted(rng.sample(slot_pool, 4)) + sorted(rng.sample(conc_pool, 4))

    word_counts = [u["np_words"] for u in units]
    payload = {
        "model_family": "allenai/OLMo-2-0425-1B",
        "instruct_commit_prefix": INSTRUCT_PREFIX,
        "np_word_range": [min(word_counts), max(word_counts)],
        "units": units,
        "stage1_prompts": STAGE1_PROMPTS,
        "stage2_free_prompts": STAGE2_FREE_PROMPTS,
    }
    out_path = Path(__file__).resolve().with_name("targets213.json")
    raw = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    print(f"Wrote {out_path.name} OK: 39/39 verbatim NPs (words {min(word_counts)}-{max(word_counts)}), "
          f"21/21 slot & 8/18 concept 92b verified, 0 boundary merges across all 17 prompts in BOTH chat and plain frames")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
