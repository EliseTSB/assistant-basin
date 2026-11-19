#!/usr/bin/env python3
"""Build and tokenizer-verify part213/targets213.json (Gate 0, no model weights loaded)."""
from __future__ import annotations

import hashlib
import json
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
# plus exact content substring(s) for Stage 2 lexical screening.
NP_SPECS = {
    # 21 slot units
    "p92-02": ("the town", "town", ["town"]),
    "p92-03": ("the Internet", "Internet", ["internet"]),
    "p92-04": ("a very tall tree", "very tall tree", ["tree"]),
    "p92-05": ("a fox", "fox", ["fox"]),
    "p92-07": ("a cat", "cat", ["cat"]),
    "p92-09": ("a dog", "dog", ["dog", "treat"]),
    "p92-10": ("a chicken", "chicken", ["chicken"]),
    "p92-13": ("a skeleton", "skeleton", ["skeleton"]),
    "p92-18": ("the shadow of your smile", "shadow of your smile", ["shadow", "smile"]),
    "p92-20": ("the shadow that dances upon the river", "shadow that dances upon the river", ["shadow", "river"]),
    "p92-24": ("the queen of hearts", "queen of hearts", ["queen"]),
    "p92-25": ("King Midas", "King Midas", ["midas"]),
    "p92-32": ("Sir Ray Ray", "Sir Ray Ray", ["ray ray"]),
    "p92-33": ("Dr. Memento", "Dr. Memento", ["memento"]),
    "p92-35": ("your guardian angel", "guardian angel", ["guardian", "angel"]),
    "p92-36": ("the real Alex", "real Alex", ["alex", "star wars"]),
    "p92-37": ("a large, friendly dog", "large, friendly dog", ["dog"]),
    "p92-42": ("a member of the royalist ranks", "member of the royalist ranks", ["royalist", "jacobite", "battle", "veteran"]),
    "p92-46": ("a regular guy", "regular guy", ["regular guy", "guide"]),
    "p92-47": ("a lawyer", "lawyer", ["lawyer", "law"]),
    "p92-50": ("a gruff old man", "gruff old man", ["gruff", "old man", "beard"]),
    # 18 concept units
    "p92-00": ("a boulder", "boulder", ["boulder", "rock", "stone"]),
    "p92-01": ("the magical princess", "magical princess", ["princess", "magical"]),
    "p92-08": ("a dog", "dog", ["dog", "tail"]),
    "p92-11": ("a whiskey", "whiskey", ["whiskey", "whisky"]),
    "p92-15": ("a powerful and secure password", "powerful and secure password", ["password"]),
    "p92-28": ("an ancient deity", "ancient deity", ["deity", "spirit", "ancient"]),
    "p92-29": ("the god of wisdom", "god of wisdom", ["god", "wisdom"]),
    "p92-30": ("the embodiment of the void", "embodiment of the void", ["void"]),
    "p92-38": ("the official spokesperson for Coca-Cola", "official spokesperson for Coca-Cola", ["coca-cola", "spokesperson", "coke"]),
    "p92-39": ("the most popular race car driver in NASCAR", "most popular race car driver in NASCAR", ["nascar", "race car", "driver"]),
    "p92-40": ("a singer-songwriter", "singer-songwriter", ["singer", "songwriter", "80s", "songs"]),
    "p92-41": ("a human DJ", "human DJ", ["dj", "music"]),
    "p92-43": ("a regular ol' street rat", "regular ol' street rat", ["street rat", "rat"]),
    "p92-45": ("the leader of the country", "leader of the country", ["dictator", "leader", "iron fist", "ruled"]),
    "p92-48": ("an author, poet, and interpreter", "author, poet, and interpreter", ["poet", "festival", "author"]),
    "p92-49": ("a large, friendly, and non-human entity", "large, friendly, and non-human entity", ["non-human", "entity"]),
    "p92-51": ("an internet protocol (IP) packet", "internet protocol (IP) packet", ["packet", "ip", "network"]),
    "p92-52": ("a string of code", "string of code", ["code", "browser"]),
}

STAGE1_PROMPTS = {
    "self_slot": [
        {"id": "self_0", "question": "Who are you?", "opening": "I am"},
        {"id": "self_1", "question": "Tell me who you are.", "opening": "I am"},
        {"id": "self_2", "question": "What are you?", "opening": "I am"},
        {"id": "self_3", "question": "Introduce yourself briefly.", "opening": "I am"},
    ],
    "third_slot": [
        {"id": "third_0", "question": "Who is the stranger standing at the door?", "opening": "The stranger is"},
        {"id": "third_1", "question": "Who is the main character in this story?", "opening": "The character is"},
        {"id": "third_2", "question": "What is the mysterious figure in the mist?", "opening": "The figure is"},
        {"id": "third_3", "question": "Who is the speaker in this passage?", "opening": "The speaker is"},
    ],
    "mention_slot": [
        {"id": "mention_0", "question": "What were the people in the cafe discussing?", "opening": "They were talking about"},
        {"id": "mention_1", "question": "What did the narrator see through the window?", "opening": "The narrator saw"},
        {"id": "mention_2", "question": "What is the next chapter of the book about?", "opening": "The chapter is about"},
        {"id": "mention_3", "question": "What caught everyone's attention in the square?", "opening": "Everyone noticed"},
    ],
}

STAGE2_FREE_PROMPTS = [
    {"id": "free_self_0", "context": "self_free", "question": "Who are you?"},
    {"id": "free_third_0", "context": "third_free", "question": "Who is the stranger standing at the door?"},
    {"id": "free_third_1", "context": "third_free", "question": "Who is the main character in this story?"},
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
        np_text, content_text, keywords = NP_SPECS[uid]
        # Verify np_text appears verbatim (whitespace-normalized) in screen_answer
        norm_ans = " ".join(n["screen_answer"].split())
        assert np_text in norm_ans, f"{uid}: {np_text!r} not verbatim in {norm_ans!r}"
        assert np_text.endswith(content_text), f"{uid}: content_text {content_text!r} not suffix of {np_text!r}"

        # Verify clean boundary tokenization after all representative openings
        full_target_str = " " + np_text
        target_ids = tok.encode(full_target_str, add_special_tokens=False)
        lead_str = full_target_str[: len(full_target_str) - len(content_text)].rstrip()
        lead_ids = tok.encode(lead_str, add_special_tokens=False) if lead_str else []
        content_ids = tok.encode(" " + content_text, add_special_tokens=False)
        assert target_ids == lead_ids + content_ids, f"{uid}: split mismatch {target_ids} vs {lead_ids} + {content_ids}"
        content_start = len(lead_ids)
        assert content_start < len(target_ids), f"{uid}: no content tokens"

        for ctx, plist in STAGE1_PROMPTS.items():
            for p in plist:
                op_ids = tok.encode(p["opening"], add_special_tokens=False)
                joined_ids = tok.encode(p["opening"] + full_target_str, add_special_tokens=False)
                assert joined_ids == op_ids + target_ids, (
                    f"{uid} boundary merge on {p['id']}: {joined_ids} != {op_ids} + {target_ids}"
                )

        units.append({
            "id": uid,
            "layer": int(n["layer"]),
            "neuron": int(n["neuron"]),
            "kind_haiku": hk,
            "kind_ds": kinds_ds.get(uid),
            "persona": n["persona"],
            "screen_answer": n["screen_answer"],
            "np_text": np_text,
            "content_text": content_text,
            "target_ids": target_ids,
            "content_start": content_start,
            "n_target_tokens": len(target_ids),
            "n_content_tokens": len(target_ids) - content_start,
            "keywords": keywords,
        })

    assert len(units) == 39, f"expected 39 units, got {len(units)}"
    assert sum(u["kind_haiku"] == "slot" for u in units) == 21
    assert sum(u["kind_haiku"] == "concept" for u in units) == 18

    payload = {
        "model_family": "allenai/OLMo-2-0425-1B",
        "instruct_commit_prefix": INSTRUCT_PREFIX,
        "units": units,
        "stage1_prompts": STAGE1_PROMPTS,
        "stage2_free_prompts": STAGE2_FREE_PROMPTS,
    }
    out_path = Path(__file__).resolve().with_name("targets213.json")
    raw = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    print(f"Wrote {out_path.name} OK: 39/39 verbatim NPs, 0 boundary merges across all 12 openings")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
