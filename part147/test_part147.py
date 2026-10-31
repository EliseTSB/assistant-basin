"""Small fixture tests for the Part 147 scorer; runnable with Python's unittest."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("score147", HERE / "score147.py")
score147 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score147)


def write_fixture(root, question_keeping=True):
    root.mkdir(parents=True, exist_ok=True)
    vocab = 8
    bases = []
    if question_keeping:
        for q in range(4):
            p = np.full(vocab, 0.04)
            p[(2 * q) % vocab] = 0.30
            p[(2 * q + 1) % vocab] = 0.42
            p /= p.sum()
            bases.append(p)
    else:
        # Base changes in concentration but keeps one generic token ranking.
        ranking = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        for q in range(4):
            p = ranking ** (1 + q * 0.1)
            bases.append(p / p.sum())
    instruct = np.full(vocab, 1 / vocab)
    units = [{"id": f"u{i}", "unit": "neuron" if i % 2 == 0 else "random", "index": i, "layer": 4 if i < 3 else 5} for i in range(6)]
    for q in range(4):
        push = bases[q] if question_keeping else bases[0]
        matrix = np.stack([push] * len(units))
        np.save(root / f"matrix-full-q{q+1}.npy", np.log(matrix))
        np.save(root / f"plain-instruct-q{q+1}.npy", np.log(instruct)[None, :])
        np.save(root / f"plain-base-q{q+1}.npy", np.log(bases[q])[None, :])
        meta = {"unit_ids": [u["id"] for u in units], "units": units}
        (root / f"index-full-q{q+1}.json").write_text(json.dumps(meta), encoding="utf-8")
    return bases


def write_gates(path, overrides=None):
    gates = {f"gate{i}": {"passed": True} for i in range(1, 4)}
    gates.update(overrides or {})
    path.write_text(json.dumps({"gates": gates}), encoding="utf-8")


class Part147Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_pass_and_predictions(self):
        input_dir = self.root / "input"
        write_fixture(input_dir)
        gates = self.root / "gates.json"
        write_gates(gates)
        result = score147.score(input_dir, gates, self.root / "out")
        for key in ("147.1", "147.2", "147.3", "147.4"):
            self.assertEqual(result["predictions"][key]["verdict"], "held")
        q1 = result["wrong_question"]["by_question"]["1"]
        self.assertEqual(q1["pushed"]["pairs"], 18)
        self.assertEqual(q1["pushed"]["share"], 1.0)
        self.assertEqual(q1["plain_instruct"]["share"], 0.0)
        self.assertTrue((self.root / "out" / "tables.md").is_file())

    def test_each_gate_failure_withholds(self):
        input_dir = self.root / "input"
        write_fixture(input_dir)
        for i in range(1, 4):
            gates = self.root / f"gates{i}.json"
            write_gates(gates, {f"gate{i}": {"passed": False}})
            result = score147.score(input_dir, gates, self.root / f"out{i}")
            self.assertTrue(all(v["verdict"] == "withheld" for v in result["predictions"].values()))
            self.assertIn(f"gate{i}", result["withheld_reason"])

    def test_generic_profile_fails_147_1(self):
        input_dir = self.root / "input"
        write_fixture(input_dir, question_keeping=False)
        gates = self.root / "gates.json"
        write_gates(gates)
        result = score147.score(input_dir, gates, self.root / "out")
        self.assertEqual(result["predictions"]["147.1"]["verdict"], "failed")

    def test_js_known_disjoint_case(self):
        value, p_depart, q_depart = score147.js_divergence(np.array([0.0, -1000.0]), np.array([-1000.0, 0.0]))
        self.assertAlmostEqual(value, np.log(2), places=12)
        self.assertEqual(p_depart, 0)
        self.assertEqual(q_depart, 0)

    def test_python_isolated_entrypoint_from_repo_root(self):
        repo_root = HERE.parent
        result = subprocess.run([sys.executable, "-I", "part147/score147.py", "--help"], cwd=repo_root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--gates", result.stdout)


if __name__ == "__main__":
    unittest.main()
