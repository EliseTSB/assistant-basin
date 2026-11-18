import json

from other_items import write_other_items_report


def test_per_clip_changed_count_matches_existing_changed_other(tmp_path):
    # Fake top-class predictions stand in for model output; no model is loaded.
    ids = ["vis:v001", "vis:v002", "vis:v003", "vis:v004"]
    animals = ["Cat_Growling", "Cat_Growling", "Dog_Barking", "Dog_Barking"]
    before = ["cat", "dog", "dog", "hen"]
    after = ["cat", "cat", "dog", "rooster"]
    existing_changed_other = sum(left != right for left, right in zip(before, after))

    report = write_other_items_report(
        tmp_path / "other.json",
        ids,
        animals,
        before,
        {"W1": after},
        chicken_prompt_scores={"W1": ([0.1, 0.2, 0.3, 0.4], [0.11, 0.21, 0.31, 0.41])},
    )
    written = json.loads((tmp_path / "other.json").read_text())

    row_changed = sum(row["arms"]["W1"]["changed"] for row in written["items"])
    assert row_changed == existing_changed_other
    assert report["changed_count_by_arm"]["W1"] == existing_changed_other
    assert written["summary_by_animal"]["Cat_Growling"]["arms"]["W1"]["changed_count"] == 1
    assert written["items"][0]["arms"]["W1"]["chicken_prompt_score_after"] == 0.11
