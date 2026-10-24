import unittest
from unittest.mock import patch

from part64 import collect64


def item(text, activation=1.0, doc=0):
    return {'window': text, 'activation': activation, 'doc': doc}


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

    def test_even_unit_median_uses_the_mean_of_middle_values(self):
        units, fresh_rows = {}, {}
        for index in range(500):
            key = f"L0:u{index}"
            pilot_top = [item(f"pilot-top-{index}-{j}", 100 - j) for j in range(20)]
            unit = {"unit_key": key, "type": "neuron", "layer": 0, "record_index": index,
                    "record": {"unit": index, "top": pilot_top,
                               "random": [item(f"pilot-random-{index}-{j}") for j in range(10)],
                               "weak": [item(f"pilot-weak-{index}-{j}") for j in range(5)]}}
            match = 9 if index < 250 else 10
            top = [item(f"pilot-top-{index}-{j}", 200 - j) for j in range(match)]
            top.extend(item(f"fresh-top-{index}-{j}", 100 - j) for j in range(36 - match))
            units[key] = unit
            fresh_rows[key] = {"top": top,
                               "random": [item(f"fresh-random-{index}-{j}") for j in range(18)],
                               "weak": [item(f"fresh-weak-{index}-{j}") for j in range(9)]}
        with patch.object(collect64, "verify_corpus", return_value={"size_bytes": collect64.CORPUS_SIZE,
                                                                      "sha256": collect64.CORPUS_SHA256_PREFIX + "0" * 48}):
            with self.assertRaisesRegex(collect64.CollectionError, "9.5/12"):
                collect64.assemble(units, fresh_rows, corpus=__import__('pathlib').Path('/unused'))


    def test_fresh_window_outside_pilot_documents_is_refused(self):
        # Amendment 5: the fresh collection scans only the pilot's first PILOT_DOCS documents.
        unit, fresh = self.fresh()
        fresh['random'][3] = item('late-doc-window', 1.0, doc=collect64.PILOT_DOCS)
        with patch.object(collect64, 'verify_corpus', return_value={'size_bytes': collect64.CORPUS_SIZE, 'sha256': collect64.CORPUS_SHA256_PREFIX}):
            with self.assertRaisesRegex(collect64.CollectionError, 'outside the pilot documents'):
                collect64.assemble({'L0:u1': unit}, {'L0:u1': fresh}, corpus=__import__('pathlib').Path('/unused'))

    def test_pilot_documents_constant_matches_amendment_5(self):
        self.assertEqual(collect64.PILOT_DOCS, 1000)


class CollectorCliTests(unittest.TestCase):
    def test_main_scans_only_the_pilot_documents(self):
        from part64 import collect64_model
        seen = {}
        def fake_collect(units, **kwargs):
            seen.update(kwargs)
            return {}
        units = {f'L0:u{i}': {} for i in range(500)}
        with patch.object(collect64, 'verify_corpus', return_value={}), \
             patch.object(collect64, 'unit_index', return_value=units), \
             patch.object(collect64_model, 'load_pinned_model', return_value=(None, None, None, None, None, 'snap')), \
             patch.object(collect64_model, 'collect_fresh', side_effect=fake_collect), \
             patch.object(collect64_model, 'write_fresh'):
            collect64_model.main(['--out', '/unused/fresh.jsonl'])
        self.assertEqual(seen.get('docs_limit'), collect64.PILOT_DOCS)


if __name__ == '__main__':
    unittest.main()
