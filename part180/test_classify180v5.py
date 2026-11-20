"""Part 180 v5 arm rules, inclusion, evidence strictness and selection. Hand-built label rows only."""
from __future__ import annotations

import unittest

import classify180v5 as classify
import texts180v5 as v5
import v5_support as support

CONFIGS = v5.CONFIG_NUMBERS
S_PASS = support.LABEL_S
A_PASS = support.LABEL_CONTENT
C_PASS = support.LABEL_CONTENT
B_PASS = support.LABEL_ABSENT
FAILING = {"S": support.label(claim="no", mention=False),
           "A0": support.label(claim="no", mention=False),
           "A1": support.label(claim="no", mention=False),
           "C": support.label(claim="no", mention=False),
           "B": support.label(claim="no", mention=True)}


def base_labels(neuron_ids, configs=CONFIGS):
    rows = []
    for config in configs:
        for neuron_id in neuron_ids:
            for arm, label in (("S", S_PASS), ("A0", A_PASS), ("A1", A_PASS),
                               ("C", C_PASS), ("B", B_PASS)):
                rows.append({"id": f"v5-c{config}-{neuron_id}-{arm}", "config": config,
                             "neuron": neuron_id, "key": arm, "label": dict(label)})
    return rows


def base_margins(neuron_ids, configs=CONFIGS, self_value=0.5, cont_value=0.5):
    return [{"config": config, "neuron": neuron_id, "self": self_value, "cont": cont_value}
            for config in configs for neuron_id in neuron_ids]


def neurons(count=3):
    return [{"id": f"n{index:02d}"} for index in range(count)]


def find(rows, config, neuron_id, arm):
    return next(row for row in rows
                if (row["config"], row["neuron"], row["key"]) == (config, neuron_id, arm))


def mutate(rows, config, neuron_id, arm, **fields):
    row = find(rows, config, neuron_id, arm)
    row["label"] = {**row["label"], **fields}
    return row


class ContentEndpointTests(unittest.TestCase):
    def test_any_subtype_counts(self):
        for field in ("mention", "theme", "description"):
            self.assertTrue(classify.content_plus(support.label(**{field: True})), field)

    def test_absent_is_not_content_plus(self):
        self.assertFalse(classify.content_plus(support.label()))

    def test_negated_only_is_not_content_plus(self):
        self.assertFalse(classify.content_plus(support.label(mention=True, negated=True)))
        self.assertFalse(classify.content_plus(support.label(theme=True, description=True, negated=True)))

    def test_quoted_only_is_not_content_plus(self):
        self.assertFalse(classify.content_plus(support.label(mention=True, quoted=True)))
        self.assertFalse(classify.content_plus(support.label(description=True, quoted=True)))

    def test_negated_or_quoted_without_presence_is_still_not_content_plus(self):
        self.assertFalse(classify.content_plus(support.label(negated=True)))
        self.assertFalse(classify.content_plus(support.label(quoted=True)))

    def test_missing_label_is_none_not_false(self):
        self.assertIsNone(classify.content_plus(None))
        self.assertIsNone(classify.arm_passes("S", None))

    def test_bad_label_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            classify.content_plus({"claim": "no", "mention": True})
        with self.assertRaises(ValueError):
            classify.content_plus({"claim": "maybe", "mention": True, "theme": False,
                                   "description": False, "negated": False, "quoted": False})


class ArmRuleTests(unittest.TestCase):
    def test_baseline_every_neuron_passes_every_configuration(self):
        ids = [neuron["id"] for neuron in neurons()]
        outcomes = classify.classify(base_margins(ids), base_labels(ids), neurons())
        for number in CONFIGS:
            self.assertEqual(outcomes[str(number)]["joint_passes"], 3, number)
            self.assertEqual(outcomes[str(number)]["counts"]["S"]["passed"], 3)

    def test_each_arm_failing_alone_flips_the_neuron(self):
        ids = [neuron["id"] for neuron in neurons()]
        for arm in classify.ARMS:
            labels = base_labels(ids)
            mutate(labels, 1, ids[1], arm, **FAILING[arm])
            outcomes = classify.classify(base_margins(ids), labels, neurons())
            report = outcomes["1"]
            self.assertEqual(report["joint_passes"], 2, arm)
            self.assertFalse(report["neurons"][ids[1]]["included"], arm)
            self.assertTrue(any(entry.startswith(arm) for entry in report["neurons"][ids[1]]["dropped_by"]), arm)
            self.assertTrue(report["neurons"][ids[0]]["included"], arm)

    def test_s_claim_no_fails_and_claim_yes_on_a_and_c_fails(self):
        ids = [neuron["id"] for neuron in neurons()]
        labels = base_labels(ids)
        mutate(labels, 1, ids[0], "S", claim="no")
        mutate(labels, 1, ids[1], "A0", claim="yes")
        mutate(labels, 1, ids[2], "C", claim="yes")
        report = classify.classify(base_margins(ids), labels, neurons())["1"]
        self.assertEqual(report["joint_passes"], 0)

    def test_b_with_negated_or_quoted_presence_still_passes(self):
        ids = [neuron["id"] for neuron in neurons()]
        labels = base_labels(ids)
        mutate(labels, 1, ids[0], "B", mention=True, negated=True)
        mutate(labels, 1, ids[1], "B", theme=True, quoted=True)
        report = classify.classify(base_margins(ids), labels, neurons())["1"]
        self.assertEqual(report["joint_passes"], 3)

    def test_b_with_plain_presence_fails(self):
        ids = [neuron["id"] for neuron in neurons()]
        labels = base_labels(ids)
        mutate(labels, 1, ids[0], "B", description=True)
        report = classify.classify(base_margins(ids), labels, neurons())["1"]
        self.assertEqual(report["joint_passes"], 2)

    def test_positive_margins_are_required(self):
        ids = [neuron["id"] for neuron in neurons()]
        report = classify.classify(base_margins(ids, self_value=-0.01), base_labels(ids), neurons())["1"]
        self.assertEqual(report["joint_passes"], 0)
        self.assertTrue(all("self_margin_positive:failed" in report["neurons"][n]["dropped_by"] for n in ids))
        report = classify.classify(base_margins(ids, cont_value=0.0), base_labels(ids), neurons())["1"]
        self.assertEqual(report["joint_passes"], 0)


class UnresolvedTests(unittest.TestCase):
    def test_unresolved_claim_fails_the_arm_and_is_counted(self):
        ids = [neuron["id"] for neuron in neurons()]
        labels = base_labels(ids)
        mutate(labels, 2, ids[0], "S", claim="unresolved")
        report = classify.classify(base_margins(ids), labels, neurons())["2"]
        self.assertEqual(report["joint_passes"], 2)
        self.assertEqual(report["counts"]["S"]["unresolved"], 1)
        self.assertEqual(report["counts"]["S"]["failed"], 0)
        self.assertEqual(report["unresolved_total"], 1)
        self.assertIn("S:unresolved", report["neurons"][ids[0]]["dropped_by"])

    def test_unresolved_on_an_about_arm_fails_it(self):
        ids = [neuron["id"] for neuron in neurons()]
        labels = base_labels(ids)
        mutate(labels, 2, ids[0], "A1", claim="unresolved")
        report = classify.classify(base_margins(ids), labels, neurons())["2"]
        self.assertEqual(report["joint_passes"], 2)
        self.assertEqual(report["counts"]["A1"]["unresolved"], 1)

    def test_unresolved_on_b_is_counted_but_does_not_fail_the_arm(self):
        ids = [neuron["id"] for neuron in neurons()]
        labels = base_labels(ids)
        mutate(labels, 2, ids[0], "B", claim="unresolved")
        report = classify.classify(base_margins(ids), labels, neurons())["2"]
        self.assertEqual(report["joint_passes"], 3)
        self.assertEqual(report["unresolved_total"], 1)
        self.assertEqual(report["counts"]["B"]["unresolved"], 0)


class EvidenceStrictnessTests(unittest.TestCase):
    def setUp(self):
        self.ids = [neuron["id"] for neuron in neurons()]
        self.neurons = neurons()

    def test_missing_cell_is_an_error(self):
        labels = base_labels(self.ids)
        labels = [row for row in labels if not (row["config"] == 4 and row["neuron"] == self.ids[2]
                                                and row["key"] == "C")]
        with self.assertRaisesRegex(ValueError, "reader cells have no row"):
            classify.classify(base_margins(self.ids), labels, self.neurons)

    def test_null_label_is_an_error(self):
        labels = base_labels(self.ids)
        find(labels, 1, self.ids[0], "A0")["label"] = None
        with self.assertRaisesRegex(ValueError, "carry no label"):
            classify.classify(base_margins(self.ids), labels, self.neurons)

    def test_duplicate_row_id_is_an_error(self):
        labels = base_labels(self.ids)
        labels.append(dict(labels[0]))
        with self.assertRaisesRegex(ValueError, "duplicate reader row id"):
            classify.classify(base_margins(self.ids), labels, self.neurons)

    def test_duplicate_cell_is_an_error(self):
        labels = base_labels(self.ids)
        twin = dict(labels[0])
        twin["id"] = "v5-c1-n00-S-second"
        labels.append(twin)
        with self.assertRaisesRegex(ValueError, "duplicate reader cell"):
            classify.classify(base_margins(self.ids), labels, self.neurons)

    def test_row_outside_the_frozen_grid_is_an_error(self):
        labels = base_labels(self.ids)
        labels.append({"id": "v5-c7-n00-S", "config": 7, "neuron": self.ids[0], "key": "S",
                       "label": dict(S_PASS)})
        with self.assertRaisesRegex(ValueError, "outside the frozen grid"):
            classify.classify(base_margins(self.ids), labels, self.neurons)

    def test_row_for_an_unknown_neuron_is_an_error(self):
        labels = base_labels(self.ids)
        labels.append({"id": "v5-c1-ghost-S", "config": 1, "neuron": "ghost", "key": "S",
                       "label": dict(S_PASS)})
        with self.assertRaisesRegex(ValueError, "outside the frozen grid"):
            classify.classify(base_margins(self.ids), labels, self.neurons)

    def test_missing_margin_is_an_error(self):
        margins = base_margins(self.ids)
        margins = [row for row in margins if not (row["config"] == 5 and row["neuron"] == self.ids[1])]
        with self.assertRaisesRegex(ValueError, "have no margins"):
            classify.classify(margins, base_labels(self.ids), self.neurons)

    def test_duplicate_margin_is_an_error(self):
        margins = base_margins(self.ids)
        margins.append(dict(margins[0]))
        with self.assertRaisesRegex(ValueError, "duplicate margin row"):
            classify.classify(margins, base_labels(self.ids), self.neurons)

    def test_null_margin_is_an_error(self):
        margins = base_margins(self.ids)
        margins[0]["self"] = None
        with self.assertRaisesRegex(ValueError, "has no self value"):
            classify.classify(margins, base_labels(self.ids), self.neurons)


def evidence(ids, passes, unresolved=None):
    """Labels where the first `passes[config]` neurons pass; `unresolved[config]` S rows go unresolved."""
    labels = base_labels(ids)
    unresolved = unresolved or {}
    counts = {}
    for config in CONFIGS:
        keep = int(passes[config])
        counts[config] = keep
        for index, neuron_id in enumerate(ids):
            if index >= keep:
                mutate(labels, config, neuron_id, "S", **FAILING["S"])
        for index in range(int(unresolved.get(config, 0))):
            mutate(labels, config, ids[index], "S", claim="unresolved")
    return labels


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ids = [neuron["id"] for neuron in neurons(4)]
        self.neurons = neurons(4)

    def select(self, passes, unresolved=None):
        outcomes = classify.classify(base_margins(self.ids), evidence(self.ids, passes, unresolved), self.neurons)
        return classify.select(outcomes)

    def test_most_passes_wins(self):
        winner = self.select({1: 1, 2: 4, 3: 2, 4: 0, 5: 3, 6: 3})
        self.assertEqual(winner["config"], 2)

    def test_tie_on_passes_goes_to_fewer_unresolved(self):
        winner = self.select({1: 4, 2: 4, 3: 2, 4: 2, 5: 2, 6: 2}, unresolved={1: 1})
        self.assertEqual(winner["config"], 2)

    def test_tie_on_passes_and_unresolved_goes_to_the_lower_number(self):
        winner = self.select({1: 2, 2: 3, 3: 3, 4: 0, 5: 0, 6: 0})
        self.assertEqual(winner["config"], 2)

    def test_rank_order_is_recorded(self):
        outcomes = classify.classify(base_margins(self.ids),
                                     evidence(self.ids, {1: 1, 2: 4, 3: 4, 4: 0, 5: 2, 6: 2}),
                                     self.neurons)
        winner = classify.select(outcomes)
        self.assertEqual(outcomes[str(winner["config"])]["rank"], 1)
        self.assertEqual(outcomes["2"]["rank"], 1)          # 4 passes, no unresolved, lower number
        self.assertEqual(outcomes["3"]["rank"], 2)
        self.assertEqual(outcomes["5"]["rank"], 3)

    def test_winner_report_freezes_the_cohort_and_reports_bars(self):
        ids = [neuron["id"] for neuron in neurons(4)]
        labels = evidence(ids, {1: 0, 2: 4, 3: 0, 4: 0, 5: 0, 6: 0})
        outcomes = classify.classify(base_margins(ids), labels, neurons(4), shared_ids=["n00", "n01"])
        report = classify.winner_report(outcomes, classify.select(outcomes), shared_ids=["n00", "n01"],
                                        neurons=neurons(4))
        self.assertEqual(report["selected_configuration"], 2)
        self.assertEqual(report["development_cohort"], ["n00", "n01", "n02", "n03"])
        self.assertEqual(report["shared_eight"]["count"], 2)
        self.assertFalse(report["bars"]["included_18"]["met"])        # 4 < 10
        self.assertFalse(report["bars"]["applied"])                   # reported, not applied
        self.assertEqual(report["bars"]["status_if_applied"], "PARK")
        self.assertEqual(report["outcomes"]["2"]["joint_passes"], 4)
        self.assertEqual(report["self_prompt"], v5.CONFIGURATIONS[1]["self_prompt"])

    def test_part92_labels_are_reported_beside_and_never_used(self):
        ids = [neuron["id"] for neuron in neurons(3)]
        part92 = {neuron_id: {"q0_self_claim_true": False, "q3_shows_without_claim": False} for neuron_id in ids}
        outcomes = classify.classify(base_margins(ids), base_labels(ids), neurons(3), part92=part92)
        self.assertEqual(outcomes["1"]["joint_passes"], 3)
        self.assertEqual(outcomes["1"]["part92_beside"]["q0_self_claim_true"], 0)
        self.assertFalse(outcomes["1"]["part92_beside"]["used_for_inclusion"])

    def test_all_six_outcomes_are_kept_with_every_count(self):
        ids = [neuron["id"] for neuron in neurons(3)]
        outcomes = classify.classify(base_margins(ids), base_labels(ids), neurons(3))
        self.assertEqual(sorted(int(key) for key in outcomes), list(CONFIGS))
        counts = outcomes["1"]["counts"]
        for arm in classify.ARMS:
            self.assertEqual(counts[arm]["n"], 3)
        self.assertEqual(counts["self_margin_positive"]["passed"], 3)
        self.assertEqual(counts["cont_margin_positive"]["passed"], 3)
        self.assertEqual(outcomes["1"]["subtype_presence"]["mention"], 12)  # S+A0+A1+C, never B


if __name__ == "__main__":
    unittest.main()


class Part92BesideSourceTests(unittest.TestCase):
    """Cold check 2026-11-20, finding 9: cmd_classify read Part 92's ANSWERS file, whose rows have no self_claim /
    shows_persona fields, so the beside block came out q0 true for none and q3 null throughout. It must read the
    labels file. Uses the repo's real Part 92 files (beside only; never inclusion)."""

    def test_answers_rows_carry_no_label_fields_and_labels_rows_do(self):
        from pathlib import Path
        import json
        step2 = Path(__file__).resolve().parent.parent / "results" / "part92" / "step2"
        answers = [json.loads(l) for l in (step2 / "answers" / "instruct_chat.jsonl").read_text().splitlines() if l]
        labels = [json.loads(l) for l in (step2 / "labels-instruct_chat.jsonl").read_text().splitlines() if l]
        self.assertTrue(all("self_claim" not in row for row in answers))
        self.assertTrue(any("self_claim" in row for row in labels))
        beside = classify.part92_beside(labels)
        q0 = sum(1 for v in beside.values() if v.get("q0_self_claim_true") is True)
        self.assertGreater(q0, 0, "the labels file must give real q0 values; the answers file gives none")

    def test_cmd_classify_reads_the_labels_file(self):
        import inspect
        source = inspect.getsource(classify.cmd_classify)
        self.assertIn("labels-instruct_chat.jsonl", source)
        self.assertNotIn('"answers" / "instruct_chat.jsonl"', source)
