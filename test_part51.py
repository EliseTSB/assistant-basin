from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

import part49
import part51


def _name_row(arm: str, prompt_id: str, answer: str) -> dict:
    return {
        "record_type": "result", "arm": arm, "prompt_id": prompt_id,
        "prompt": "Begin with the main character's first name.", "answer": answer,
    }


def test_part51a_excludes_prompt_stems_and_matches_each_magic_word_start():
    measured = part51.part51a_answer_measure(
        "Magic wand spellbook sorcery arcane mystic potion enchanted.",
        "Please mention MAGIC and a wand in the prompt.",
    )
    assert measured == {
        "matches": 6, "word_count": 8, "per_1000": 750.0,
        "excluded_stems": ["magic", "wand"],
    }


@pytest.mark.parametrize(("answer", "expected"), [
    ('"Mara" ran toward the lighthouse.', "Mara"),
    ("**Mara** ran toward the lighthouse.", "Mara"),
    ("The old lighthouse keeper waved.", part51.NONAME),
    ("Once upon a time, a girl found a map.", part51.NONAME),
    ("There was a baker named Rosa.", part51.NONAME),
    ("<think>Need a name.</think> Mara opened the map.", "Mara"),
    ("Elias, the keeper, opened the door.", "Elias"),
])
def test_name_extraction_handles_registered_openings(answer, expected):
    assert part51.extract_first_name(answer) == expected


def test_part51b_stock_set_comes_only_from_replicate_a_and_breaks_ties_alphabetically():
    arms = [
        {"id": "no_clamp_A", "clamps": []},
        {"id": "no_clamp_B", "clamps": []},
        {"id": "L6:k+8", "clamps": []},
    ]
    rows = []
    for prompt_id, names in {
        "p01": ["Mara", "Elias", "Mara", "Elias", "Zoe", "Iris", "Ava", "Quinn"],
        "p02": ["Mara", "Elias", "Zoe", "Iris", "Ava", "Quinn"],
    }.items():
        rows.extend(_name_row("no_clamp_A", prompt_id, f"{name} began the story.")
                    for name in names)
        rows.extend(_name_row("no_clamp_B", prompt_id, "Yara began the story.")
                    for _ in range(2))
        rows.extend(_name_row("L6:k+8", prompt_id, "Yara began the story.")
                    for _ in range(4))

    scored = part51.score_push51b_rows(rows, arms, seed=12, resamples=100)
    stock = scored["measure"]["stock_names"]
    assert stock == [
        {"name": "Elias", "count_in_A": 3},
        {"name": "Mara", "count_in_A": 3},
        {"name": "Ava", "count_in_A": 2},
        {"name": "Iris", "count_in_A": 2},
        {"name": "Quinn", "count_in_A": 2},
    ]
    assert "Yara" not in {item["name"] for item in stock}
    pushed = next(arm for arm in scored["arms"] if arm["arm"] == "L6:k+8")
    assert pushed["stock_share_among_named"] == 0.0
    assert pushed["name_entropy_bits"] == 0.0


def test_part51b_bootstrap_resamples_prompt_clusters():
    arms = [
        {"id": "no_clamp_A", "clamps": []},
        {"id": "no_clamp_B", "clamps": []},
        {"id": "L6:k+8", "clamps": []},
    ]
    rows = []
    for prompt_id in ("p01", "p02"):
        rows.extend(_name_row("no_clamp_A", prompt_id, "Stock began.") for _ in range(2))
        rows.extend(_name_row("no_clamp_B", prompt_id, "Other began.") for _ in range(2))
    rows.extend(_name_row("L6:k+8", "p01", "Stock began.") for _ in range(2))
    rows.extend(_name_row("L6:k+8", "p02", "Other began.") for _ in range(2))

    seed, resamples = 7, 400
    scored = part51.score_push51b_rows(rows, arms, seed=seed, resamples=resamples)
    pushed = next(arm for arm in scored["arms"] if arm["arm"] == "L6:k+8")
    rng = np.random.default_rng(seed)
    prompt_indices = rng.integers(0, 2, size=(resamples, 2))
    pushed_prompt_shares = np.asarray([1.0, 0.0])[prompt_indices].mean(axis=1)
    # Replicate B has zero stock answers in both prompts, so the paired difference is unchanged.
    expected = np.quantile(pushed_prompt_shares, [0.025, 0.975]).tolist()
    assert pushed["stock_share_difference_from_no_clamp_B"]["bootstrap95"] == pytest.approx(expected)
    assert scored["measure"]["resampling_unit"] == "prompt"


def test_existing_blind_score_path_rescores_part51a_labels(tmp_path):
    arms = [
        {"id": "no_clamp", "clamps": []},
        {"id": "L22:1610:k+8", "clamps": [{"layer": 22, "unit": 1610}]},
    ]
    rows = [{"record_type": "header", "command": "push51a", "arms": arms}]
    for arm in arms:
        for index, prompt_id in enumerate(("p01", "p02"), 1):
            row_key = f"{arm['id']}:{prompt_id}:1"
            rows.append({
                "record_type": "result", "row_key": row_key, "arm": arm["id"],
                "prompt_id": prompt_id, "prompt": "Write a short answer.",
                "answer": "magic wand" if arm["id"] != "no_clamp" else "ordinary words",
                "broken": None,
            })
    source = tmp_path / "push51a.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    pack = tmp_path / "blind-pack.jsonl"
    pack.write_text("\n".join(json.dumps(row) for row in [
        {"record_type": "answer", "blind_id": "b1", "broken": False},
        {"record_type": "answer", "blind_id": "b2", "broken": True},
        {"record_type": "answer", "blind_id": "b3", "broken": False},
        {"record_type": "answer", "blind_id": "b4", "broken": True},
    ]) + "\n", encoding="utf-8")
    key = tmp_path / "blind-key.json"
    key.write_text(json.dumps({"answers": [
        {"blind_id": f"b{index}", "row_key": row["row_key"]}
        for index, row in enumerate(rows[1:], 1)
    ]}), encoding="utf-8")

    summary = part49.run_score(SimpleNamespace(
        input=source, blind_pack=pack, blind_key=key, out=tmp_path / "scores.json",
        partners=None,
    ))
    pushed = next(row for row in summary["arms"] if row["arm"] == "L22:1610:k+8")
    assert pushed["broken"] == {"hits": 1, "n": 2, "rate": 0.5, "pending": False}


def test_part51_arm_plans_have_registered_coordinates_signs_and_64_rows():
    plan_a = part51.build_part51a_arm_plan()
    assert [arm["id"] for arm in plan_a] == [
        "no_clamp", "L22:1610:k+8", "L22:1610:k+4", "R1", "R2", "R3",
    ]
    assert [arm["k"] for arm in plan_a] == [None, 8, 4, 8, 8, 8]
    assert [arm["clamps"] for arm in plan_a[3:]] == [
        [{"layer": 22, "unit": 569}],
        [{"layer": 22, "unit": 3102}],
        [{"layer": 22, "unit": 5258}],
    ]

    widths = {6: 32, 14: 64, 22: 128}
    expected_rng = random.Random(part51.PART51B_RANDOM_SEED)
    expected = []
    for layer in (6, 14, 22):
        positive, negative = expected_rng.sample(range(widths[layer]), 2)
        expected.extend([(layer, positive, 8), (layer, negative, -8)])
    units = part51.choose_part51b_random_units(widths)
    assert [(row["layer"], row["unit"], row["k"]) for row in units] == expected
    plan_b = part51.build_part51b_arm_plan(units)
    assert len(plan_b) == 8
    assert [(arm["clamps"][0]["layer"], arm["clamps"][0]["unit"], arm["k"])
            for arm in plan_b[2:]] == expected

    prompts = [f"prompt {index}" for index in range(16)]
    rows_a = part49.build_sample_rows(prompts, 4, seed=3801, namespace="test-part51")
    requests_a = part49.arm_requests(plan_a, rows_a)
    assert all(len(requests_a[arm["id"]]) == 64 for arm in plan_a)
    rows_b = part49.build_sample_rows(prompts, 4, seed=3802, namespace="test-part51")
    requests_b = part49.arm_requests(plan_b, rows_a)
    assert len(rows_b) == 64
    assert all(len(requests_b[arm["id"]]) == 64 for arm in plan_b)
    assert [row["seed"] for row in rows_a] != [row["seed"] for row in rows_b]
    request_b = part49.arm_requests([plan_b[1]], rows_b)["no_clamp_B"]
    assert len(request_b) == 64
