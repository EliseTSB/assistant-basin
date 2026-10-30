"""Small fixture tests for the prediction counter's table parsing."""
import importlib.util
import os
from pathlib import Path


_script_path = Path(os.environ.get(
    "COUNT_PREDICTIONS_SCRIPT",
    Path(__file__).resolve().parent.parent / "count_predictions.py",
)).resolve()
_spec = importlib.util.spec_from_file_location("count_predictions_under_test", _script_path)
counter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(counter)


def test_old_style_table_keeps_the_start_of_result_rule():
    counts, unread = counter.classify(
        "| 1.1 | A P(you \\| A) prediction. | held by the recorded result. |\n"
    )
    assert not unread
    assert counts["held"] == 1


def test_status_column_is_used_instead_of_evidence():
    table = """| id | prediction | status | evidence |
|---|---|---|---|
| 2.1 | A P(you \\| A) prediction. | **failed** | held in an unrelated note |
"""
    counts, unread = counter.classify(table)
    assert not unread
    assert counts["failed"] == 1
    assert counts["held"] == 0


def test_result_before_bold_verdict_uses_the_verdict_column():
    table = """| # | Prediction | Result | Verdict |
|---|---|---|---|
| 3.1 | A prediction. | The measured result is described here. | **failed** |
"""
    counts, unread = counter.classify(table)
    assert not unread
    assert counts["failed"] == 1


def test_unreadable_verdict_prints_unclassified_and_returns_one(capsys):
    table = """| # | Prediction | Result | Verdict |
|---|---|---|---|
| 4.1 | A prediction. | Evidence text. | unclear after review |
"""
    assert counter.main(text=table) == 1
    output = capsys.readouterr().out
    assert "UNCLASSIFIED 4.1" in output


def test_hand_checked_override_is_kept():
    counts, unread = counter.classify(
        "| 20g.2 | A prediction. | reviewed text not matching a pattern |\n"
    )
    assert not unread
    assert counts["untestable"] == 1
