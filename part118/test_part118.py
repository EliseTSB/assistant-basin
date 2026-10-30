"""Pytest discovery shim for the requested `test118.py` fixture suite."""

from test118 import (
    test_registered_predictions_hold_on_synthetic_fixture,
    test_registered_predictions_fail_on_synthetic_fixture,
    test_missing_primary_label_is_not_ready,
    test_gate2_below_point_six_makes_predictions_not_read,
    test_random_arm_answers_do_not_enter_neuron_kind_counts,
    test_incoherent_answer_does_not_show_persona,
    test_gate1_reports_one_differing_answer,
)

