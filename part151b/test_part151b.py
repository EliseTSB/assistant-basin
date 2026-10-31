"""Part 151b scorer fixtures."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from part151 import read151, score151
from part151b import score151b


def win(mark, after="x y", before="b"):
    return {"act": 1.0, "doc": 0, "pos": 0, "before": before, "mark": mark, "after": after}


def windows(docs=8000, alien=6, and_or=12, award=11):
    units = {u: {"top": [win(" plain") for _ in range(20)], "bottom": []} for u in score151.UNITS}
    units["L19N4979"]["top"] = [win(" and") if i < and_or else win(" plain") for i in range(20)]
    units["L24N8542"]["top"] = [win(" plain", before="won the award") if i < award else win(" plain") for i in range(20)]
    units[score151.CONTROL] = {"top": [win(" alien") if i < alien else win(" the") for i in range(20)], "bottom": []}
    return {"docs": docs, "units": units}


LENS = {u: ["zzz"] for u in score151.UNITS}


class Tests(unittest.TestCase):
    def test_readings_hold(self):
        r = score151b.score(windows(), LENS)
        self.assertEqual(r["predictions"]["151b.6"], "held")
        self.assertEqual(r["predictions"]["151b.7"], "held")
        self.assertEqual(r["and_or_top"]["L19N4979"], 12)
        self.assertEqual(r["award_top"]["L24N8542"], 11)

    def test_below_ten_fails(self):
        r = score151b.score(windows(and_or=9, award=9), LENS)
        self.assertEqual(r["predictions"]["151b.6"], "failed")
        self.assertEqual(r["predictions"]["151b.7"], "failed")

    def test_another_unit_at_the_same_count_fails(self):
        w = windows()
        w["units"]["L23N116"]["top"] = [win(" or") if i < 12 else win(" plain") for i in range(20)]
        self.assertEqual(score151b.score(w, LENS)["predictions"]["151b.6"], "failed")

    def test_failed_gate_makes_every_prediction_untestable(self):
        r = score151b.score(windows(alien=4), LENS)
        self.assertFalse(r["gates"]["gate3_alien_control"])
        self.assertEqual(set(r["predictions"].values()), {"untestable"})

    def test_wrong_document_count_fails_gate_one(self):
        self.assertFalse(score151b.score(windows(docs=2000), LENS)["gates"]["gate1_docs"])

    def test_start_option_exists(self):
        src = Path(read151.__file__).read_text(encoding="utf-8")
        self.assertIn("[a.start:a.docs]", src)


if __name__ == "__main__":
    unittest.main()
