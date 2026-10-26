"""Host-runnable checks for the 67b magnitude-matched control design."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part67b


class MagnitudeSelectionTests(unittest.TestCase):
    def test_closest_fifty_excludes_target_and_breaks_ties_by_index(self):
        magnitudes = [0.0, 9.0, 11.0, 8.0, 12.0, 10.0, 10.0] + [100.0] * 55
        selected = part67b.select_magnitude_matched(5, magnitudes)
        self.assertEqual(len(selected), 50)
        self.assertNotIn(5, selected)
        # Units 1, 2, 3, 4 are tied at distance one; unit 6 ties at distance zero,
        # so the exact closest order begins with 6, 1, 2, 3, 4.
        self.assertEqual(selected[:5], [6, 1, 2, 3, 4])

    def test_selection_rejects_too_few_candidates(self):
        with self.assertRaises(ValueError):
            part67b.select_magnitude_matched(0, [1.0, 2.0], count=2)

    def test_empirical_p_and_survival_count_ties(self):
        null = [-2.0, 1.0, 0.5]
        self.assertEqual(part67b.empirical_p(2.0, null), 2 / 4)
        self.assertFalse(part67b.survives(2.0, null))
        self.assertEqual(part67b.empirical_p(3.0, null), 1 / 4)
        self.assertTrue(part67b.survives(3.0, null))
        self.assertFalse(part67b.survives(0.5, [0.5]))

    def test_pass_two_conditions_include_three_targets_and_each_match_once(self):
        magnitudes = {layer: [float(i) for i in range(target + 60)]
                      for layer, target in part67b.UPSTREAM}
        for layer, target in part67b.UPSTREAM:
            magnitudes[layer][target] = float(target)
        conditions, matches = part67b.build_conditions(magnitudes)
        self.assertEqual([row['id'] for row in conditions[:4]], [
            'unchanged', 'zero_6_753', 'zero_7_1169', 'zero_11_8868'])
        self.assertEqual(len(conditions), 1 + 3 + 3 * 50)
        mag_rows = [row for row in conditions if row['kind'] == 'magnitude_matched']
        self.assertEqual(len(mag_rows), 150)
        for layer, target in part67b.UPSTREAM:
            selected = part67b.select_magnitude_matched(target, magnitudes[layer])
            ids = [f'mag_{layer}_{unit}' for unit in selected]
            self.assertEqual([row['id'] for row in mag_rows if row['layer'] == layer], ids)
            self.assertEqual(matches[f'{layer}_{target}']['target_magnitude'], float(target))


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, 'torch is unavailable in this host sandbox')
class TinyPassOneTests(unittest.TestCase):
    def test_pass_one_magnitudes_match_hand_computation(self):
        class DownProj(torch.nn.Module):
            def forward(self, x):
                return x

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = DownProj()

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer() for _ in range(12)])

            def forward(self, input_ids):
                ids = input_ids.float().unsqueeze(-1)
                columns = torch.tensor([1.0, -2.0, 3.0])
                x = ids * columns
                for layer in self.layers:
                    x = layer.mlp.down_proj(x)
                return x

        model = TinyModel()
        measured = part67b.collect_magnitudes(model, model.layers, [[1, 2], [3]], 'cpu')
        expected = [2.0, 4.0, 6.0]
        for layer in part67b.MAG_LAYERS:
            self.assertEqual(measured[layer], expected)


if __name__ == '__main__':
    unittest.main()
