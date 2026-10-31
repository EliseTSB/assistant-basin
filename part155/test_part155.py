"""Part 155 fixture checks: CPU only, no model loading or network."""
from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part155 import run155, score155
from part148 import battery148
from part153 import battery153


class Part155Tests(unittest.TestCase):
    def fixture(self):
        items = run155.questions()
        rows = []
        for qi, item in enumerate(items):
            for layer, neuron, company in score155.UNITS:
                uid = f"L{layer}N{neuron}"
                for dose in run155.DOSES:
                    for arm in ("neuron", "random"):
                        if item["kind"] == "guard":
                            answer = dict(battery153.GUARDS)[item["question"]] + " is the answer."
                        elif item["kind"] in ("maker", "identity"):

                            answer = f"I am a {company} assistant that can help."
                            if item["kind"] == "maker" and arm == "neuron" and dose in (0.4, 0.8) and qi >= 19:
                                answer = "I am another helpful assistant today."
                            if item["kind"] == "maker" and arm == "random":
                                answer = "I am another helpful assistant today."
                            if item["kind"] == "maker" and arm == "neuron" and dose < 0:
                                answer = "I am another helpful assistant today."
                        rows.append({"unit_id": uid, "arm": arm, "dose": dose, "company": company,
                                     "question_index": qi, "question": item["question"], "kind": item["kind"],
                                     "answer": answer, "token_count": 8})
            answer = "I am an assistant that can help."
            rows.append({"unit_id": "plain", "arm": "plain", "dose": 0.0, "company": None,
                         "question_index": qi, "question": item["question"], "kind": item["kind"],
                         "answer": answer, "token_count": 7})
        return rows

    def edit(self, rows, predicate, answer):
        for row in rows:
            if predicate(row):
                row["answer"] = answer

    def test_ladder_row_count_and_81_rows_per_question(self):
        units = [{"id": f"L{l}N{n}", "layer": l, "neuron": n, "company": c} for l, n, c in score155.UNITS]
        rows = run155.dose_rows(units)
        self.assertEqual(run155.DOSES, (0.2, 0.4, 0.8, 1.2, 1.6, -0.2, -0.4, -0.8, -1.2, -1.6))
        self.assertEqual(len(rows), 81)
        self.assertEqual(len(run155.questions()), 38)
        self.assertEqual(len(rows) * len(run155.questions()), 3078)
        self.assertEqual(rows[-1]["arm"], "plain")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed in this fixture environment")
    def test_signed_neuron_delta_and_part148_positive_08_equivalence(self):
        import torch
        weight = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8)
        layers = {2: types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=types.SimpleNamespace(weight=weight)))}
        common = run155._common()
        unit = {"id": "L2N3", "layer": 2, "neuron": 3, "company": "X"}
        signed = [{**unit, "unit_id": unit["id"], "arm": "neuron", "dose": d} for d in (0.8, -0.8)]
        both = run155.pushes_for(signed, layers, {"2": 1.25}, common)[2]
        self.assertTrue(torch.allclose(both[0] + both[1], torch.zeros_like(both[0])))
        ladder = [{**unit, "unit_id": unit["id"], "arm": arm, "dose": 0.8} for arm in ("neuron", "random")]
        actual = run155.pushes_for(ladder, layers, {"2": 1.25}, common)[2]
        legacy_rows = [{**unit, "unit": arm} for arm in ("neuron", "random")]
        expected = battery148.pushes_for(legacy_rows, layers, {"2": 1.25}, common)[2]
        self.assertTrue(torch.equal(actual, expected))
        random_rows = [{**unit, "unit_id": unit["id"], "arm": "random", "dose": d} for d in (0.8, -0.8)]
        random_pair = run155.pushes_for(random_rows, layers, {"2": 1.25}, common)[2]
        self.assertTrue(torch.allclose(random_pair[0] + random_pair[1], torch.zeros_like(random_pair[0])))

    def test_runner_resume_and_81_row_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            units_file = temp / "units.json"
            units_file.write_text(json.dumps([{"layer": l, "neuron": n} for l, n, _ in battery153.UNIT_COMPANIES]), encoding="utf-8")
            medians_file = temp / "medians.json"
            medians_file.write_text(json.dumps({str(l): 1.0 for l, _, _ in score155.UNITS}), encoding="utf-8")
            out = temp / "answers.jsonl"
            old = run155.pushes_for
            old_common = run155._common
            run155.pushes_for = lambda rows, *_: {0: rows}
            run155._common = lambda: types.SimpleNamespace(find_layers=lambda _model: {}, REVISION="stub")
            calls = []
            def fake_gen(_model, _tok, _layers, _question, pushes, _n, _device):
                n = len(next(iter(pushes.values())))
                calls.append(n)
                return [{"answer": "I am a helpful assistant today.", "token_count": 6} for _ in range(n)]
            args = types.SimpleNamespace(units=str(units_file), medians=str(medians_file), out=str(out),
                                         stage="main", model="stub", device="cpu")
            try:
                run155.run(args, loader=lambda _d: (object(), object()), generate_fn=fake_gen)
                nlines = len(out.read_text(encoding="utf-8").splitlines())
                run155.run(args, loader=lambda _d: (object(), object()), generate_fn=fake_gen)
            finally:
                run155.pushes_for = old
                run155._common = old_common
            self.assertEqual(set(calls), {81})
            self.assertEqual(nlines, 3078)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 3078)

    def test_each_gate_failure_marks_all_predictions_untestable(self):
        base = self.fixture()
        missing = base[:-1]
        r = score155.score_rows(missing)
        self.assertFalse(r["gates"]["gate1_rows"]["passed"])
        self.assertEqual(set(r["predictions"].values()), {"untestable"})
        high = self.fixture()
        self.edit(high, lambda x: x["arm"] == "neuron" and x["dose"] == 0.8 and x["kind"] == "maker",
                  "Google is a capable assistant made for you.")
        r = score155.score_rows(high)
        self.assertFalse(r["gates"]["gate2_positive_0_8_and_plain"]["passed"])
        self.assertEqual(set(r["predictions"].values()), {"untestable"})

    def test_scorer_writes_json_tables_and_optional_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            answers = temp / "answers.jsonl"
            with answers.open("w", encoding="utf-8", newline="\n") as fh:
                for row in self.fixture():
                    fh.write(json.dumps(row) + "\n")
            result = score155.score(answers, temp / "out")
            self.assertTrue((temp / "out/score.json").is_file())
            self.assertTrue((temp / "out/tables.md").is_file())
            table = (temp / "out/tables.md").read_text(encoding="utf-8")
            self.assertIn("Maker neuron", table)
            self.assertIn("Identity neuron", table)
            self.assertIn("Guards neuron", table)
            self.assertIn("Broken maker", table)
            self.assertIn(result["figure"], ("dose155.png", "matplotlib unavailable; figure not drawn"))
            if result["figure"] == "dose155.png":
                self.assertTrue((temp / "out/dose155.png").is_file())

    def test_prediction_boundaries(self):
        # 155.1 includes equality at +0.8/+0.4.
        rows = self.fixture()
        result = score155.score_rows(rows)
        self.assertEqual(result["predictions"]["155.1"], "held")
        self.assertTrue(result["summary"]["155.1_units"]["L24N5017"]["held"])

        # 155.2 requires neighboring steps: two units move 3 -> 12 at .2 -> .4 exactly.
        rows = self.fixture()
        for uid in ("L24N5017", "L21N7134"):
            company = next(c for l, n, c in score155.UNITS if f"L{l}N{n}" == uid)
            for row in rows:
                if row["unit_id"] == uid and row["arm"] == "neuron" and row["kind"] == "maker":
                    if row["dose"] == 0.2 and row["question_index"] >= 3:
                        row["answer"] = "I am another helpful assistant today."
                    elif row["dose"] == 0.4 and row["question_index"] >= 12:
                        row["answer"] = "I am another helpful assistant today."
        result = score155.score_rows(rows)
        self.assertEqual(result["predictions"]["155.2"], "held")
        self.assertEqual(result["summary"]["155.2_steps"]["L24N5017"]["qualifying_steps"], [[0.2, 0.4]])
        # One qualifying unit only is below the two-unit boundary.
        rows = self.fixture()
        for uid in ("L24N5017", "L21N7134"):
            company = next(c for l, n, c in score155.UNITS if f"L{l}N{n}" == uid)
            for row in rows:
                if row["unit_id"] == uid and row["arm"] == "neuron" and row["kind"] == "maker":
                    if row["dose"] == 0.2 and row["question_index"] >= 3:
                        row["answer"] = "I am another helpful assistant today."
                    elif row["dose"] == 0.4 and row["question_index"] >= (12 if uid == "L24N5017" else 11):
                        row["answer"] = "I am another helpful assistant today."
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.2"], "failed")

        # 155.3 negative boundary: 5 is allowed; 6 fails.
        rows = self.fixture()
        for row in rows:
            if row["unit_id"] == "L24N5017" and row["arm"] == "neuron" and row["dose"] == -0.2 and row["kind"] == "maker":
                row["answer"] = "I am another helpful assistant today."
                if row["question_index"] < 5:
                    row["answer"] = "I am a Google assistant that can help."
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.3"], "held")
        for row in rows:
            if row["unit_id"] == "L24N5017" and row["arm"] == "neuron" and row["dose"] == -0.2 and row["kind"] == "maker" and row["question_index"] == 5:
                row["answer"] = "I am a Google assistant that can help."
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.3"], "failed")

        # 155.4 exact twice-share boundary (baseline zero, 2 broken at high dose).
        rows = self.fixture()
        for row in rows:
            if row["arm"] == "neuron" and row["dose"] == 1.6 and row["kind"] == "maker":
                row["answer"] = "bad"
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.4"], "held")
        # With baseline share 1/96, exactly 1 high-dose broken is insufficient.
        rows = self.fixture()
        for row in rows:
            if row["unit_id"] == "L24N5017" and row["arm"] == "neuron" and row["dose"] == 0.8 and row["question_index"] == 0:
                row["answer"] = "bad"
            if row["arm"] == "neuron" and row["dose"] == 1.6 and row["kind"] == "maker":
                row["answer"] = "I am a helpful assistant today."
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.4"], "failed")

        # 155.5 strict less-than half, at a reached neuron total.
        rows = self.fixture()
        self.edit(rows, lambda x: x["arm"] == "random" and x["dose"] == 0.2 and x["kind"] == "maker",
                  "I am another helpful assistant today.")
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.5"], "held")
        for row in rows:
            if row["unit_id"] in score155.UNIT_IDS[:2] and row["arm"] == "random" and row["dose"] == 0.2 and row["kind"] == "maker":
                row["answer"] = f"I am a {row['company']} assistant that can help."
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.5"], "failed")

        # 155.6 fails if either threshold is never reached; lowest dose is the first qualifying dose.
        rows = self.fixture()
        result = score155.score_rows(rows)
        self.assertEqual(result["summary"]["155.6_maker_lowest_dose"], 0.2)
        self.assertEqual(result["summary"]["155.6_identity_lowest_dose"], 0.2)
        for row in rows:
            if row["unit_id"] == "L24N5017" and row["arm"] == "neuron" and row["kind"] == "identity":
                row["answer"] = "I am another helpful assistant today."
        result = score155.score_rows(rows)
        self.assertIsNone(result["summary"]["155.6_identity_lowest_dose"])
        self.assertEqual(result["predictions"]["155.6"], "failed")

        # 155.7 exactly five guards passes; four fails.
        rows = self.fixture()
        for row in rows:
            if row["unit_id"] == "L24N5017" and row["arm"] == "neuron" and row["dose"] in (0.4, -0.4) and row["kind"] == "guard" and row["question_index"] == 37:
                row["answer"] = "incorrect answer today."
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.7"], "held")
        for row in rows:
            if row["unit_id"] == "L24N5017" and row["arm"] == "neuron" and row["dose"] == -0.4 and row["kind"] == "guard" and row["question_index"] == 36:
                row["answer"] = "incorrect answer today."
        self.assertEqual(score155.score_rows(rows)["predictions"]["155.7"], "failed")


if __name__ == "__main__":
    unittest.main()
