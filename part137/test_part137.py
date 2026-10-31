"""CPU checks for Part 137's registered question and scoring plan."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "part134"))
import run134
import run137
import score137


def test_new_questions_do_not_overlap_part134_dev_or_test():
    new = set(run137.load_questions())
    assert len(new) == 24
    assert new.isdisjoint(run134.DEV_QUESTIONS)
    assert new.isdisjoint(run134.TEST_QUESTIONS)


def test_eight_arms_and_192_answer_plan():
    questions = run137.load_questions()
    assert len(run137.ARMS) == 8
    assert len(questions) * len(run137.ARMS) == 192
    assert set(run137.ARMS) == {"pair", "a_only", "b_only", "a_random_1", "a_random_2", "b_random_1", "b_random_2", "none"}
    assert run137.RANDOM_SEEDS == {"a_random_1": 1371, "a_random_2": 1372, "b_random_1": 1373, "b_random_2": 1374}


def test_each_registered_prediction_has_passing_and_failing_examples():
    true = score137.predictions_for(8, 10, 3, [5, 5], [4, 4, 4, 4], 8)
    false = score137.predictions_for(5, 5, 4, [3, 4], [3, 3, 3, 3], 2)
    assert all(true[f"137.{i}"] for i in range(1, 6))
    assert not any(false[f"137.{i}"] for i in range(1, 6))


def test_opening_refusal_rule_is_exact_and_case_insensitive():
    for answer in ("I'm sorry, I cannot help.", "Sorry about that.", "I can't do that.", "I CANNOT answer."):
        assert score137.opens_refusal(answer)
    for answer in ("Well, I'm sorry.", " I can't help.", "Apologies.", "I won't do that."):
        assert not score137.opens_refusal(answer)


def test_question_json_counts_are_registered():
    data = json.loads((HERE / "questions137.json").read_text(encoding="utf-8"))
    assert len(data["non_task"]) == 16
    assert len(data["task"]) == 8
