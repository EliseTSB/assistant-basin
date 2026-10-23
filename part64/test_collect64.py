import unittest
from unittest.mock import patch

from part64 import collect64


def item(text, activation=1.0):
    return {'window': text, 'activation': activation}


class CollectionContractTests(unittest.TestCase):
    def unit(self):
        return {'unit_key': 'L0:u1', 'type': 'neuron', 'layer': 0,
                'record': {'unit': 1,
                           'top': [item(f'pilot-top-{i}') for i in range(20)],
                           'random': [item(f'pilot-random-{i}') for i in range(10)],
                           'weak': [item(f'pilot-weak-{i}') for i in range(5)]}}

    def fresh(self, match=12):
        u = self.unit()
        strata = {}
        for cat in ('top', 'random', 'weak'):
            base_n = collect64.PILOT_COUNTS[cat]
            extra_n = collect64.CAPACITIES[cat] - base_n
            # Fresh top's leading 12 duplicate the pilot exactly to exercise
            # the reproduction metric; remaining candidates are unique.
            n_match = match if cat == 'top' else 0
            values = [item(f'pilot-top-{i}', 100-i) for i in range(n_match)]
            values += [item(f'new-{cat}-{i}', 1000-i) for i in range(extra_n + 20)]
            strata[cat] = values
        return u, strata

    def test_exact_m1_subset_and_3x_capacities(self):
        unit, fresh = self.fresh()
        result, reproduced = collect64.assemble_unit(unit, fresh)
        self.assertEqual(reproduced, 12)
        for cat, n in collect64.CAPACITIES.items():
            self.assertEqual(len(result[cat]), n)
            self.assertTrue({x['window'] for x in unit['record'][cat][:collect64.PILOT_COUNTS[cat]]}
                            <= {x['window'] for x in result[cat]})

    def test_duplicates_do_not_fill_capacity(self):
        unit, fresh = self.fresh()
        fresh['weak'] = [item(f'pilot-weak-{i % 3}') for i in range(100)]
        with self.assertRaisesRegex(collect64.CollectionError, 'only'):
            collect64.assemble_unit(unit, fresh)

    def test_reproduction_median_below_10_stops_before_output(self):
        unit, fresh = self.fresh(match=9)
        with patch.object(collect64, 'verify_corpus', return_value={'size_bytes': collect64.CORPUS_SIZE, 'sha256': collect64.CORPUS_SHA256_PREFIX}):
            with self.assertRaisesRegex(collect64.CollectionError, 'reproduction stop'):
                collect64.assemble({'L0:u1': unit}, {'L0:u1': fresh}, corpus=__import__('pathlib').Path('/unused'))


if __name__ == '__main__':
    unittest.main()
