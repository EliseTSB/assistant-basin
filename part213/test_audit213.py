"""Tests for part213/audit213.py: the 40-item sample and the label parser (no network, no model)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit213 as A  # noqa: E402
import test_part213 as T  # noqa: E402


def _all_group_rows():
    """Every unit (slot and concept) on both THIRD_FREE prompts, in the three audited arm-doses, plus baselines."""
    targets = A.S.load_targets()
    rows = [{"prompt_id": p, "arm": "none", "dose": "0", "unit": None, "answer": "baseline", "status": "ok"}
            for p in A.S.THIRD_FREE]
    for u in targets["units"]:
        for p in A.S.THIRD_FREE:
            for arm, dose in (("neuron", "+0.8"), ("neuron", "-0.8"), ("random_0", "+0.8")):
                rows.append({"prompt_id": p, "arm": arm, "dose": dose, "unit": u["id"],
                             "answer": f"{u['id']} {arm} {dose}", "status": "ok"})
    return rows


def _write_records(tmp_path):
    rows = _all_group_rows()
    (tmp_path / "rec").mkdir()
    with open(tmp_path / "rec" / "stage2-instruct.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return tmp_path / "rec"


def test_export_samples_eight_per_audited_group_and_is_seeded(tmp_path):
    rec = _write_records(tmp_path)
    A.export(rec, tmp_path / "a1")
    A.export(rec, tmp_path / "a2")
    s1 = json.loads((tmp_path / "a1" / "sample.json").read_text())
    s2 = json.loads((tmp_path / "a2" / "sample.json").read_text())
    assert s1 == s2 and len(s1) == 40 and len(set(s1)) == 40
    items = json.loads((tmp_path / "a1" / "items.json").read_text())
    targets = A.S.load_targets()
    order = A.S._blind_order(A.S.load_records(rec)["stage2"], targets)
    units = {u["id"]: u for u in targets["units"]}
    group = {i["blind_id"]: A._group_key(r, units) for i, r in zip(items, order)}
    counts = {}
    for b in s1:
        counts[group[b]] = counts.get(group[b], 0) + 1
    assert counts == {g: 8 for g in A.AUDITED_GROUPS}            # concept -0.8 never sampled
    for item in items:                                            # blinding: no arm, unit or dose in an item
        assert set(item) == {"blind_id", "target_persona", "question", "answer", "baseline_answer"}


def test_parse_label():
    assert A._parse_label('{"label": "broken"}') == "broken"
    assert A._parse_label('Sure.\n```json\n{"label": "other_coherent"}\n```') == "other_coherent"
    assert A._parse_label('{"label": "genuine"}') is None
    assert A._parse_label("") is None and A._parse_label("no json here") is None


def test_known_examples_cover_every_category():
    known = json.loads((HERE / "known213.json").read_text(encoding="utf-8"))["items"]
    assert len(known) == 10 and {k["gold"] for k in known} == set(A.S.AUDIT_CATEGORIES)
    assert A.S.broken_answer(next(k for k in known if k["gold"] == "broken")["answer"])
