import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shape_floor', ROOT / 'tools/shape_floor.py')
sf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sf
spec.loader.exec_module(sf)


def test_interval_known_values():
    assert sf.clopper_pearson(0, 10)[0] == 0
    assert sf.clopper_pearson(10, 10)[1] == 1
    lo, hi = sf.clopper_pearson(5, 10)
    assert abs(lo - 0.187086) < 1e-5 and abs(hi - 0.812914) < 1e-5
    assert sf.clopper_pearson(0, 0) == [None, None]


def test_scenario_names_are_the_documented_ones():
    rows = [{'id': 'a', 'cell': 'x', 'labels': {'flag': True}}]
    pairs = [{'labels_a': {'flag': True}, 'labels_b': {'flag': True}}]
    out = sf.simulate(rows, pairs, lambda rs: {'n': sum(r['labels']['flag'] for r in rs) > 0}, 50, 22)
    assert list(out['scenarios']) == ['independent', 'shared-change-0.5', 'shared-change-1', 'upper-marginal-independent']
    assert 'not a guarantee' in out['interpretation']
    assert out['scenarios']['independent']['verdicts']['n']['sensitivity'] == 1.0


def test_binomial_count_boundary_matches_probability():
    # The observed count sits exactly on the threshold n; the verdict survives only if no row flips.
    n, p, draws = 20, .1, 20000
    rows = [{'id': str(i), 'cell': 'all', 'labels': {'flag': True}} for i in range(n)]
    pairs = [{'labels_a': {'flag': True}, 'labels_b': {'flag': False if i < 2 else True}} for i in range(n)]
    out = sf.simulate(rows, pairs, lambda rs: {'at_k': sum(r['labels']['flag'] for r in rs) >= n}, draws, 201)
    entry = out['scenarios']['independent']['verdicts']['at_k']
    assert abs(entry['sensitivity'] - (1 - p) ** n) < .02
    assert 0 < entry['mc_se'] < .01
    # Rows that share one change event all change or none does: the verdict survives with probability 1 - p.
    shared = sf.simulate(rows, pairs, lambda rs: {'at_k': sum(r['labels']['flag'] for r in rs) >= n}, draws, 201,
                         cluster_key='cell')['scenarios']['shared-change-1']['verdicts']['at_k']
    assert abs(shared['sensitivity'] - (1 - p)) < .02


def _together(rs):
    """True when row 0 and row 1 are both changed or both unchanged (their unchanged vectors are 00 and 10)."""
    changed0 = rs[0]['labels'] != {'a': False, 'b': False}
    changed1 = rs[1]['labels'] != {'a': True, 'b': False}
    return {'together': changed0 == changed1}


def test_shared_change_does_not_depend_on_destination_order():
    # Alden's counterexample to rank coupling: start 00 lists [unchanged, changed], start 10 lists [changed, unchanged].
    # With a shared rank the two rows would change in opposite draws; reversing one list would flip that.
    rows = [{'id': 'r0', 'cell': 'x', 'unit': 'u', 'labels': {'a': False, 'b': False}},
            {'id': 'r1', 'cell': 'x', 'unit': 'u', 'labels': {'a': True, 'b': False}}]
    p00 = [{'labels_a': {'a': False, 'b': False}, 'labels_b': {'a': False, 'b': False}},
           {'labels_a': {'a': False, 'b': False}, 'labels_b': {'a': False, 'b': True}}]
    p10 = [{'labels_a': {'a': True, 'b': False}, 'labels_b': {'a': False, 'b': False}},
           {'labels_a': {'a': True, 'b': False}, 'labels_b': {'a': True, 'b': False}}]
    results = []
    for pairs in (p00 + p10, p00 + p10[::-1], p10[::-1] + p00[::-1]):
        out = sf.simulate(rows, pairs, _together, 4000, 7, cluster_key='unit')
        results.append((out['scenarios']['shared-change-1']['verdicts']['together']['sensitivity'],
                        out['scenarios']['independent']['verdicts']['together']['sensitivity']))
    assert all(shared == 1.0 for shared, _ in results)          # one change event: always together, in any order
    assert all(abs(independent - .5) < .05 for _, independent in results)
    assert len(set(results)) == 1                               # and the figures do not move with the listing order


def test_destination_frequencies_follow_the_replicate():
    rows = [{'id': str(i), 'cell': 'x', 'labels': {'a': False, 'b': False}} for i in range(400)]
    start = {'a': False, 'b': False}
    pairs = ([{'labels_a': start, 'labels_b': start}] * 6 + [{'labels_a': start, 'labels_b': {'a': True, 'b': False}}] * 3
             + [{'labels_a': start, 'labels_b': {'a': True, 'b': True}}])
    out = sf.simulate(rows, pairs, lambda rs: {'v': True}, 200, 3)
    counts = out['scenarios']['independent']['label_counts']
    assert abs(counts['a']['x']['simulated_mean'] / 400 - .4) < .02      # a turns true in 4 of 10 pairs
    assert abs(counts['b']['x']['simulated_mean'] / 400 - .1) < .02      # b only together with a, in 1 of 10
    support = out['support'][0]
    assert (support['replicate_pairs'], support['changed_pairs'], support['target_rows']) == (10, 4, 400)
    assert [d['pairs'] for d in support['destinations']] == [6, 3, 1]


def test_unsupported_rows_and_unmeasured_labels_are_reported_not_filled_in():
    # The second row's vector is in no pair; its label value True for 'b' was never a starting value either.
    rows = [{'id': 'r0', 'cell': 'x', 'labels': {'a': True, 'b': False}},
            {'id': 'r1', 'cell': 'y', 'labels': {'a': False, 'b': True}}]
    pairs = [{'labels_a': {'a': True, 'b': False}, 'labels_b': {'a': True, 'b': False}}] * 5
    out = sf.simulate(rows, pairs, lambda rs: {'b_true_on_r1': rs[1]['labels']['b']}, 200, 5)
    assert out['unsupported'] == {'rows': 1, 'by_cell': {'y': 1}, 'model': out['unsupported']['model']}
    assert [entry['supported'] for entry in out['support']] in ([True, False], [False, True])
    directions = out['direction_counts_and_intervals']
    assert directions['b']['true_to_false'] == {'count': 0, 'denominator': 0, 'point_rate': None,
                                                '95%_clopper_pearson': [None, None]}
    for name in ('independent', 'upper-marginal-independent'):
        scenario = out['scenarios'][name]
        assert scenario['verdicts']['b_true_on_r1']['sensitivity'] == 1.0      # held, because unmeasured
        assert scenario['unmeasured_label_holds_per_draw'] >= 1                # and counted, every draw
    # 'a' False was never a starting value either, so in the stress scenario r1 holds both labels: 2 holds per draw.
    assert out['scenarios']['upper-marginal-independent']['unmeasured_label_holds_per_draw'] == 2


def test_a_verdict_that_rests_on_unsupported_rows_is_flagged():
    # r1's vector is unsupported; under the fallback its label flips at the measured marginal rate (1 of 2),
    # which changes the verdict, while holding it keeps the verdict. The two figures differ, and that is flagged.
    rows = [{'id': 'r0', 'cell': 'x', 'labels': {'a': True, 'b': True}},
            {'id': 'r1', 'cell': 'x', 'labels': {'a': False, 'b': True}}]
    pairs = [{'labels_a': {'a': True, 'b': True}, 'labels_b': {'a': True, 'b': True}},
             {'labels_a': {'a': True, 'b': True}, 'labels_b': {'a': True, 'b': False}}]
    out = sf.simulate(rows, pairs, lambda rs: {'r1_b': rs[1]['labels']['b']}, 2000, 9)
    entry = out['scenarios']['independent']['verdicts']['r1_b']
    assert abs(entry['sensitivity'] - .5) < .05
    assert entry['sensitivity_unsupported_rows_held'] == 1.0
    assert entry['depends_on_unsupported_rows'] is True


def test_the_unsupported_row_flag_uses_the_paired_error():
    # Alden's toy (council part201-design-review #11): 1,000 redraws, 950 as observed both ways, 10 only with the
    # fallback, 0 only with the rows held, 40 neither. Shares .96 and .95; the paired error is .003148, so the
    # difference of .01 is beyond two of them. Two independent samples' errors would give .009268 and no flag.
    difference, se = sf.paired_difference(10, 0, 1000)
    assert abs(difference - .01) < 1e-12 and abs(se - .003148) < 1e-6
    assert abs(difference) > 2 * se
    unpaired = (.96 * .04 / 1000 + .95 * .05 / 1000) ** .5
    assert abs(unpaired - .009268) < 1e-6 and not abs(difference) > 2 * unpaired
    # No discordant redraw: no difference, no error, no flag. One each way: the difference is zero.
    assert sf.paired_difference(0, 0, 1000) == (0.0, 0.0)
    assert sf.paired_difference(7, 7, 1000)[0] == 0.0


def test_simulate_reports_the_paired_difference_from_the_same_redraws():
    # Twenty supported rows that never change and one unsupported row whose label flips in 1 of 50 pairs. The verdict
    # is "the unsupported row's label is still true": with the fallback it is as observed in about 98% of redraws,
    # with the row held in all of them. Every discordant redraw is on one side, and the entry says how many.
    rows = ([{'id': f's{i}', 'cell': 'x', 'labels': {'a': True, 'b': True}} for i in range(20)]
            + [{'id': 'u', 'cell': 'x', 'labels': {'a': False, 'b': True}}])
    pairs = ([{'labels_a': {'a': True, 'b': True}, 'labels_b': {'a': True, 'b': True}}] * 49
             + [{'labels_a': {'a': True, 'b': True}, 'labels_b': {'a': True, 'b': False}}])
    draws = 3000
    out = sf.simulate(rows, pairs, lambda rs: {'u_b': rs[-1]['labels']['b']}, draws, 11)
    entry = out['scenarios']['independent']['verdicts']['u_b']
    assert entry['sensitivity_unsupported_rows_held'] == 1.0
    assert entry['redraws_as_observed_only_with_rows_held'] > 0 and entry['redraws_as_observed_only_with_fallback'] == 0
    lost = entry['redraws_as_observed_only_with_rows_held']
    assert abs(entry['sensitivity'] - (1 - lost / draws)) < 1e-12
    assert abs(entry['difference_from_unsupported_rows_held'] + lost / draws) < 1e-12
    expected = sf.paired_difference(0, lost, draws)[1]
    assert abs(entry['paired_mc_se_of_that_difference'] - expected) < 1e-12
    assert entry['depends_on_unsupported_rows'] is True
    # Under two independent samples' errors this difference (about .02 at n = 3,000) would also be flagged; the toy
    # above is the case where the two rules part. Here the check is that the paired figures are the ones reported.


def test_render_names_every_scenario_and_marks_unsupported():
    rows = [{'id': 'r0', 'cell': 'x', 'labels': {'a': True, 'b': True}},
            {'id': 'r1', 'cell': 'x', 'labels': {'a': False, 'b': True}}]
    pairs = [{'labels_a': {'a': True, 'b': True}, 'labels_b': {'a': True, 'b': True}},
             {'labels_a': {'a': True, 'b': True}, 'labels_b': {'a': True, 'b': False}}]
    text = sf.render(sf.simulate(rows, pairs, lambda rs: {'r1_b': rs[1]['labels']['b']}, 400, 9))
    for name in ('independent', 'shared-change-0.5', 'shared-change-1', 'upper-marginal-independent'):
        assert name in text
    assert '(u)' in text and 'Unsupported rows' in text and 'unmeasured' in text
    assert 'stability' not in text.lower()


def test_scorer_must_return_verdicts():
    rows = [{'id': 'a', 'cell': 'x', 'labels': {'flag': True}}]
    try:
        sf.simulate(rows, [], lambda rs: {'n': 3}, 10, 1)
    except ValueError as exc:
        assert 'bool or str' in str(exc)
    else:
        raise AssertionError('a numeric verdict was accepted')
