"""Seconds-fast contract tests for Part 33's masks, controls, loss windows and MC scoring."""
import unittest
from types import SimpleNamespace

import numpy as np

import part33


class FakeLetterModel:
    """A tiny stand-in returning predetermined next-letter logits; no weights/tokenizer."""
    def __init__(self, preferred_token):
        self.preferred_token = preferred_token
        self.calls = 0

    def __call__(self, input_ids, **_kwargs):
        self.calls += 1
        ids = np.asarray(input_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        logits = np.zeros((ids.shape[0], ids.shape[1], 8), dtype=np.float32)
        logits[:, -1, self.preferred_token] = 4.0
        return SimpleNamespace(logits=logits)


class Part33ContractTests(unittest.TestCase):
    def test_conditional_zero_is_strictly_above_half_maximum(self):
        activations = np.array([0.0, 4.9, 5.0, 5.1, 10.0])
        mask = part33.conditional_mask(activations, corpus_max=10.0)
        self.assertEqual(mask.tolist(), [False, False, False, True, True])
        changed = part33.apply_conditional_zero(activations, mask)
        self.assertEqual(changed.tolist(), [0.0, 4.9, 5.0, 0.0, 0.0])

    def test_controls_are_closest_firing_frequency_matches(self):
        rates = {unit: 0.02 * unit for unit in range(1, 26)}
        rates[1], rates[2], rates[3] = 0.249, 0.25, 0.251
        selected = part33.match_firing_frequency(0.25, rates, target=99, n=20, seed=7)
        self.assertEqual(len(selected), 20)
        self.assertEqual(set(selected[:3]), {1, 2, 3})
        distances = [abs(rates[u] - 0.25) for u in selected]
        all_unselected = [abs(rates[u] - 0.25) for u in rates if u not in selected]
        self.assertLessEqual(max(distances), min(all_unselected))

    def test_passage_sets_are_selected_by_measured_thresholds(self):
        rows = []
        activations = {}
        questions = []
        for i in range(1, 21):
            ident = f"q{i:02d}"
            rows.append({"id": ident, "group": "alien", "text": ident})
            activations[ident] = np.array([[10.0 - i * 0.1, 0.0]])
            questions.append({"passage_id": ident})
        for i in range(10):
            ident = f"alien_extra{i:02d}"
            rows.append({"id": ident, "group": "alien", "text": ident})
            activations[ident] = np.array([[6.0 - i * 0.01, 0.0]])
        for group in ("topic", "neutral"):
            for i in range(30):
                ident = f"{group}_{i:02d}"
                rows.append({"id": ident, "group": group, "text": ident})
                activations[ident] = np.array([[4.0 - i * 0.01, 0.0]])
        selected, corpus_max = part33.select_passages(rows, activations, 0, questions)
        self.assertAlmostEqual(corpus_max, 9.9)
        self.assertEqual({g: sum(row["group"] == g for row in selected)
                          for g in ("alien", "topic", "neutral")},
                         {"alien": 30, "topic": 30, "neutral": 30})
        self.assertTrue(all(activations[row["id"]][0, 0] > corpus_max * 0.5
                            for row in selected if row["group"] == "alien"))
        self.assertTrue(all(activations[row["id"]][0, 0] <= corpus_max * 0.5
                            for row in selected if row["group"] in {"topic", "neutral"}))

    def test_loss_windows_use_next_token_positions_and_deduplicate_union(self):
        mask = [False, True, False, True, False]
        positions = part33.loss_window_positions(mask, n_losses=4, lookahead=2)
        self.assertEqual(positions, {"at": [1, 3], "after": [2, 3], "window": [1, 2, 3]})
        stats = part33.loss_window_delta([1, 1, 1, 1], [1, 2, 3, 4], mask, lookahead=2)
        self.assertEqual(stats["firing_loss_tokens"], 2)
        self.assertEqual(stats["after_loss_tokens"], 2)
        self.assertEqual(stats["window_loss_tokens"], 3)
        self.assertAlmostEqual(stats["at_firing"], 2.0)
        self.assertAlmostEqual(stats["after_1_to_8"], 2.5)
        self.assertAlmostEqual(stats["window"], 2.0)
        self.assertAlmostEqual(stats["window_delta_sum"], 6.0)

    def test_run_question_reads_logits_from_the_capture_pass_output(self):
        # Cloud smoke 2026-10-20 died here: _run_question indexed the output object as if it were logits.
        import sys, types
        import torch
        vocab, width = 8, 4
        down = torch.nn.Linear(width, 3, bias=False)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down = down
            def forward(self, input_ids=None, attention_mask=None, use_cache=False):
                act = torch.ones(1, input_ids.shape[1], width)
                act[0, -1, 0] = 9.0
                self.down(act)
                logits = torch.zeros(1, input_ids.shape[1], vocab)
                logits[0, -1, 2] = 5.0
                return types.SimpleNamespace(logits=logits)

        saved = {name: getattr(part33, name) for name in ("_model_input", "_letter_ids", "system_memory_guard")}
        nr = types.ModuleType("neuron_report"); nr.spill_guard = lambda *a, **k: None
        prev = sys.modules.get("neuron_report"); sys.modules["neuron_report"] = nr
        old_neuron = part33.NEURON
        try:
            part33.NEURON = 0
            part33._model_input = lambda tok, text, device, max_length=512: (torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.long))
            part33._letter_ids = lambda tok, text: {"A": 1, "B": 2}
            part33.system_memory_guard = lambda gb: None
            row = part33._run_question(Model(), down, None, {"id": "q", "passage_id": "p", "answer": "B", "question": "?", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"}},
                                       "text", 10.0, "cpu", 2.0, 14.5)
        finally:
            for name, value in saved.items():
                setattr(part33, name, value)
            part33.NEURON = old_neuron
            if prev is None:
                del sys.modules["neuron_report"]
            else:
                sys.modules["neuron_report"] = prev
        self.assertEqual(row["baseline_prediction"], "B")
        self.assertEqual(row["natural_firing_tokens"], 1)

    def test_mc_scoring_reads_answer_letter_logits_from_fake_model(self):
        model = FakeLetterModel(preferred_token=2)
        examples = [{"id": "mc", "answer": "B"}]
        rows = part33.score_mc_questions(model, examples,
                                         {"A": 1, "B": 2, "C": 3, "D": 4},
                                         input_builder=lambda _item: [5, 6, 7])
        self.assertEqual(model.calls, 1)
        self.assertEqual(rows[0]["prediction"], "B")
        self.assertTrue(rows[0]["correct"])
        self.assertEqual(rows[0]["logits"], {"A": 0.0, "B": 4.0, "C": 0.0, "D": 0.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
