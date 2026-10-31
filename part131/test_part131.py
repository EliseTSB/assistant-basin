#!/usr/bin/env python3
"""CPU-only contract tests for Part 131 selection and scoring helpers."""
import importlib.util, os, unittest
from pathlib import Path
DEFAULT=Path(__file__).with_name('score131.py')
TARGET=Path(os.environ.get('PART131_SCORE_MODULE',DEFAULT))
spec=importlib.util.spec_from_file_location('score131_under_test',TARGET); score=importlib.util.module_from_spec(spec); spec.loader.exec_module(score)

class Part131Tests(unittest.TestCase):
    def test_positive_coherent_label_counts(self):
        self.assertTrue(score.show({'shows_persona':True,'coherent':True}))
    def test_incoherent_persona_does_not_count(self):
        self.assertFalse(score.show({'shows_persona':True,'coherent':False}))
    def test_absent_or_negative_label_does_not_count(self):
        self.assertFalse(score.show(None))
        self.assertFalse(score.show({'shows_persona':False,'coherent':True}))

if __name__=='__main__': unittest.main()
