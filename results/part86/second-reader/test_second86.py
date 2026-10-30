"""Fixture-only tests for Part 86 second-reader sampling and agreement."""
from __future__ import annotations

import pytest

import second86 as second


def _items(n: int) -> list[dict]:
    return [{"id": f"i{index:02d}", "text": f"fixture answer {index}"} for index in range(n)]


def _label(item_id: str, persona: bool) -> dict:
    return {
        "id": item_id,
        "persona": "a clear persona" if persona else "unclear",
        "kind": "human" if persona else "unclear",
        "coherent": True if persona else False,
    }


def _assert_golden_sample(items: list[dict], expected: list[str], seed: int = second.SEED) -> None:
    assert [row["id"] for row in second.sample_items(items, seed=seed)] == expected


def test_sample_is_deterministic_and_10_percent() -> None:
    items = _items(40)
    expected = ["i24", "i15", "i19", "i07"]
    assert len(second.sample_items(items)) == 4
    assert second.sample_items(items) == second.sample_items(items)
    _assert_golden_sample(items, expected)


def test_seed_mutation_turns_golden_sample_assertion_red() -> None:
    """A copied implementation changed to seed 8688 fails the golden-ID assertion."""
    items = _items(40)
    expected = ["i24", "i15", "i19", "i07"]
    with pytest.raises(AssertionError, match="i04"):
        _assert_golden_sample(items, expected, seed=8688)


def test_prompt_uses_the_first_reader_brief_for_the_same_item() -> None:
    item = {"id": "opaque-123", "text": "I am a fictional cartographer."}
    brief = second.READER_BRIEF_PATH.read_text(encoding="utf-8")
    expected = f"{brief.rstrip()}\n\nItem id: opaque-123\nAnswer:\nI am a fictional cartographer."
    assert second.reader_prompt(item, brief) == expected
    assert second.parse_label(
        '{"id":"opaque-123","persona":"a cartographer","kind":"human","coherent":true}',
        "opaque-123",
    ) == {
        "id": "opaque-123",
        "persona": "a cartographer",
        "kind": "human",
        "coherent": True,
    }


def test_kappa_and_matrix_match_hand_computed_example() -> None:
    ids = [f"item-{index}" for index in range(8)]
    # 2x2 counts: 3/1 in the first-positive row and 1/3 in first-negative.
    first_positive = {ids[index] for index in (0, 1, 2, 3)}
    second_positive = {ids[index] for index in (0, 1, 2, 4)}
    first = [_label(item_id, item_id in first_positive) for item_id in ids]
    second_labels = [_label(item_id, item_id in second_positive) for item_id in ids]

    report = second.agreement_report(first, second_labels, ids, population_items=80)

    assert report["n_scored"] == 8
    assert report["raw_agreement"] == pytest.approx(0.75)
    assert report["kappa"] == pytest.approx(0.5)
    assert report["counts_2x2"] == {
        "first_coherent_persona": {
            "second_coherent_persona": 3,
            "second_not_coherent_persona": 1,
        },
        "first_not_coherent_persona": {
            "second_coherent_persona": 1,
            "second_not_coherent_persona": 3,
        },
    }


def test_error_rows_are_counted_and_excluded_only_from_kappa() -> None:
    ids = ["a", "b", "c", "d"]
    first = [
        _label("a", True),
        {"id": "b", "error": "primary parse failure"},
        _label("c", False),
        _label("d", True),
    ]
    second_labels = [
        _label("a", True),
        _label("b", False),
        {"id": "c", "error": "second parse failure"},
        _label("d", False),
    ]

    report = second.agreement_report(first, second_labels, ids, population_items=100)

    assert report["sampled_items"] == 4
    assert report["error_rows"] == {"first_reader": 1, "second_reader": 1}
    assert report["excluded_items"] == 2
    assert report["n_scored"] == 2
    assert report["raw_agreement"] == pytest.approx(0.5)
    assert report["counts_2x2"] == {
        "first_coherent_persona": {
            "second_coherent_persona": 1,
            "second_not_coherent_persona": 1,
        },
        "first_not_coherent_persona": {
            "second_coherent_persona": 0,
            "second_not_coherent_persona": 0,
        },
    }
