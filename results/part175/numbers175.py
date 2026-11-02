#!/usr/bin/env python3
"""Recompute Part 175 derived counts from its frozen result files."""
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def near(a, b):
    return abs(float(a) - float(b)) < 1e-9


def wilson_interval(k, n, z=1.959963984540054):
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def display_percent(interval):
    return [round(100 * value, 1) for value in interval]


score = read_json(ROOT / 'score.json')
directions = read_jsonl(ROOT / 'scored_directions.jsonl')
stage_a = read_json(ROOT / 'stage_a.json')
layer_norms = read_json(ROOT / 'layer_norms.json')
first_reader_labels = read_jsonl(ROOT / 'labels/labels-ds.jsonl')
readers_log = (ROOT / 'readers.log').read_text()
design_text = (REPO / 'PART175-DESIGN.md').read_text()

claim_shares = []
support_intervals = []
support_bounds = []
for dose in (0.15, 0.225, 0.30):
    rows = [row for row in directions
            if row.get('family') == 'f100d' and near(abs(float(row['dose'])), dose)]
    labelled_count_rows = [row for row in rows if row.get('google_claim_count') is not None]
    google_claim_count = sum(int(row['google_claim_count']) for row in labelled_count_rows)
    complete_clean_denominator = sum(int(row.get('reader_labelled_clean_answers', 0))
                                     for row in labelled_count_rows)
    all_labelled_clean_denominator = sum(int(row.get('reader_labelled_clean_answers', 0))
                                         for row in rows)
    all_labelled_google_claims = sum(
        int(row.get('full_name_distribution', {}).get('google', 0)) for row in rows
    )
    omitted_clean = sum(int(row['clean']) for row in rows
                        if row.get('google_claim_count') is None)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['seed']].append(row)
    claim_shares.append({
        'dose': dose,
        'directions': len(rows),
        'directions_with_claim_counts': len(labelled_count_rows),
        'null_directions': sum(row.get('google_claim_count') is None for row in rows),
        'partly_labelled_directions': sum(row.get('reader_state') == 'partly labelled' for row in rows),
        'no_clean_answer_directions': sum(row.get('reader_state') == 'no clean answers' for row in rows),
        'google_claims': google_claim_count,
        'labelled_clean_answers_denominator': complete_clean_denominator,
        'clean_answers_in_null_count_directions_excluded': omitted_clean,
        'clean_answers_all_directions': sum(int(row['clean']) for row in rows),
        'google_claim_share': (google_claim_count / complete_clean_denominator
                               if complete_clean_denominator else None),
        'google_claim_share_percent_1dp': (round(100 * google_claim_count / complete_clean_denominator, 1)
                                           if complete_clean_denominator else None),
        'all_labelled_google_claims': all_labelled_google_claims,
        'all_labelled_clean_answers_denominator': all_labelled_clean_denominator,
        'all_labelled_google_claim_share': (all_labelled_google_claims / all_labelled_clean_denominator
                                            if all_labelled_clean_denominator else None),
        'all_labelled_google_claim_share_percent_1dp': (
            round(100 * all_labelled_google_claims / all_labelled_clean_denominator, 1)
            if all_labelled_clean_denominator else None
        ),
        'supported_directions': sum(row.get('supported') is True for row in rows),
        'positive_supported_directions': sum(row.get('supported') is True and float(row['dose']) > 0
                                             for row in rows),
        'negative_supported_directions': sum(row.get('supported') is True and float(row['dose']) < 0
                                             for row in rows),
        'google_led_directions': sum(row.get('google_led') is True for row in rows),
        'other_led_directions': sum(row.get('other_led') is True for row in rows),
        'median_broken': (sorted(int(row['broken']) for row in rows)[len(rows) // 2 - 1:len(rows) // 2 + 1]),
        'broken_min': min(int(row['broken']) for row in rows),
        'broken_max': max(int(row['broken']) for row in rows),
        'draws_with_at_least_one_supported_sign': sum(
            any(row.get('supported') is True for row in group) for group in grouped.values()
        ),
        'draws_with_both_supported_signs': sum(
            sum(row.get('supported') is True for row in group) == 2 for group in grouped.values()
        ),
    })
    broken = sorted(int(row['broken']) for row in rows)
    mid = len(broken) // 2
    claim_shares[-1]['median_broken'] = (broken[mid] if len(broken) % 2 else
                                          (broken[mid - 1] + broken[mid]) / 2)

    by_seed = defaultdict(dict)
    for row in rows:
        by_seed[row['seed']]['positive' if float(row['dose']) > 0 else 'negative'] = int(
            row.get('supported') is True
        )
    seeds = sorted(by_seed)
    paired_draws = [[by_seed[seed]['positive'], by_seed[seed]['negative']] for seed in seeds]
    n_draws = len(paired_draws)
    assert n_draws == 20 and all(len(draw) == 2 for draw in paired_draws)
    per_sign = {}
    for index, sign in enumerate(('positive', 'negative')):
        successes = sum(draw[index] for draw in paired_draws)
        interval = wilson_interval(successes, n_draws)
        per_sign[sign] = {
            'successes': successes,
            'draws': n_draws,
            'rate': successes / n_draws,
            'wilson_95_percent_1dp': display_percent(interval),
        }
    rng = random.Random(175)
    bootstrap_rates = [
        sum(sum(paired_draws[i]) for i in rng.choices(range(n_draws), k=n_draws)) / (2 * n_draws)
        for _ in range(10000)
    ]
    pooled_successes = sum(sum(draw) for draw in paired_draws)
    support_intervals.append({
        'dose': dose,
        'per_sign': per_sign,
        'pooled': {
            'successes': pooled_successes,
            'directions': 2 * n_draws,
            'rate': pooled_successes / (2 * n_draws),
            'cluster_bootstrap': {
                'seed': 175,
                'replicates': 10000,
                'resampling_unit': 'draw, keeping both signs together',
                'percentile_method': 'linear percentiles at 2.5 and 97.5',
                'percent_1dp': display_percent((percentile(bootstrap_rates, 0.025),
                                                percentile(bootstrap_rates, 0.975))),
            },
        },
    })

    partial_rows = []
    lower_support = 0
    upper_support = 0
    for row in rows:
        if row.get('reader_state') == 'partly labelled':
            unknown = int(row.get('reader_unlabelled_clean_answers', 0))
            makers = {name: int(count) for name, count in row.get('full_name_distribution', {}).items()
                      if name != 'qwen'}
            max_known = max(makers.values(), default=0)
            possible_modal = max(max_known + unknown, unknown)
            guaranteed = max_known >= 10 and int(row['broken']) <= 8
            possible = possible_modal >= 10 and int(row['broken']) <= 8
            google = makers.get('google', 0)
            max_competitor = max((count for name, count in makers.items() if name != 'google'), default=0)
            google_guaranteed = (google >= 10 and google > max_competitor + unknown
                                 and int(row['broken']) <= 8)
            partial_rows.append({
                'seed': row['seed'],
                'sign': 'positive' if float(row['dose']) > 0 else 'negative',
                'labelled_maker_distribution': makers,
                'google_labelled_count': google,
                'unlabelled_clean_answers': unknown,
                'maximum_possible_maker_modal_count': possible_modal,
                'supported_in_every_completion': guaranteed,
                'supported_in_some_completion': possible,
                'google_led_in_every_completion': google_guaranteed,
            })
            lower_support += int(guaranteed)
            upper_support += int(possible)
        else:
            fixed = int(row.get('supported') is True)
            lower_support += fixed
            upper_support += fixed
    reported_bounds = score['dose_reports']['by_dose'][str(dose)]['counts']['supported']
    assert [lower_support, upper_support] == [reported_bounds['lower'], reported_bounds['upper']]
    support_bounds.append({
        'dose': dose,
        'rule': 'maker modal at least 10 of 44, broken at most 8 of 44',
        'partial_directions': partial_rows,
        'supported_lower_bound': lower_support,
        'supported_upper_bound': upper_support,
        'matches_score_bounds': True,
    })

selection_candidate_doses = [row['dose'] for row in claim_shares]
selected_candidate = max(claim_shares, key=lambda row: (row['supported_directions'], row['dose']))
common_draws = sorted({row['seed'] for row in directions if row.get('family') == 'f100d'})
selection_disclosure = {
    'rule': score.get('selected_rule'),
    'candidate_doses': selection_candidate_doses,
    'common_draw_count': len(common_draws),
    'selected_dose': selected_candidate['dose'],
    'selected_supported_directions': selected_candidate['supported_directions'],
    'is_in_sample_maximum': selected_candidate['supported_directions'] == max(
        row['supported_directions'] for row in claim_shares
    ),
}
assert len(selection_candidate_doses) == 3 and len(common_draws) == 20

layer_rows = {}
for row in layer_norms['rows']:
    if row.get('set') == 'f100d_d174101' and near(float(row['dose']), 0.30):
        layer_rows['flat_f100d_d174101_d0.30'] = row
    if row.get('set') == 'u100d_d174001' and near(float(row['dose']), 0.50):
        layer_rows['u100d_d174001_d0.50'] = row

flat_layer = layer_rows['flat_f100d_d174101_d0.30']
u100d_layer = layer_rows['u100d_d174001_d0.50']
scale_0225 = 0.225 / float(flat_layer['dose'])
scaled_layers = {
    str(layer): {
        'written_norm': value['written_norm'] * scale_0225,
        'written_over_residual_norm': value['written_over_residual_norm'] * scale_0225,
    }
    for layer, value in flat_layer['layers'].items()
}
layer_comparison = {
    'measurement': layer_norms['measurement'],
    'residual_norm_contexts': layer_norms['residual_norms']['contexts'],
    'flat': {
        'set': flat_layer['set'],
        'dose': flat_layer['dose'],
        'sum_written_layer_norms': flat_layer['sum_written_layer_norms'],
        'written_layers': flat_layer['written_layers'],
        'layers': flat_layer['layers'],
    },
    'u100d': {
        'set': u100d_layer['set'],
        'dose': u100d_layer['dose'],
        'sum_written_layer_norms': u100d_layer['sum_written_layer_norms'],
        'written_layers': u100d_layer['written_layers'],
        'layers': u100d_layer['layers'],
    },
    'flat_scaled_to_0.225': {
        'dose': 0.225,
        'scale_factor': scale_0225,
        'sum_written_layer_norms': flat_layer['sum_written_layer_norms'] * scale_0225,
        'layers': scaled_layers,
    },
}
layer_comparison['display_rounded_3dp'] = {
    'flat_sum_written_layer_norms': round(layer_comparison['flat']['sum_written_layer_norms'], 3),
    'u100d_sum_written_layer_norms': round(layer_comparison['u100d']['sum_written_layer_norms'], 3),
    'flat_0.225_sum_written_layer_norms': round(
        layer_comparison['flat_scaled_to_0.225']['sum_written_layer_norms'], 3
    ),
    'flat_0.225_layers': [
        {
            'layer': int(layer),
            'written_norm': round(values['written_norm'], 3),
            'written_over_residual_norm': round(values['written_over_residual_norm'], 3),
        }
        for layer, values in scaled_layers.items()
    ],
    'layers': [
        {
            'layer': int(layer),
            'flat_written_norm': round(flat_layer['layers'][layer]['written_norm'], 3),
            'flat_written_over_residual_norm': round(flat_layer['layers'][layer]['written_over_residual_norm'], 3),
            'u100d_written_norm': round(u100d_layer['layers'][layer]['written_norm'], 3),
            'u100d_written_over_residual_norm': round(u100d_layer['layers'][layer]['written_over_residual_norm'], 3),
        }
        for layer in flat_layer['layers']
    ],
}

reader_gate = score['reader_gate9']
error_cap = reader_gate['error_cap']
error_fractions = [float(value) for value in re.findall(r'"error_fraction":\s*([0-9.]+)', readers_log)]
assert len(error_fractions) >= 2
first_pass_fraction = error_fractions[0]
retry_pass_fraction = error_fractions[1]
first_pass_failures = round(first_pass_fraction * int(error_cap['total_pairs']))
amendment1 = design_text.split('## Amendment 1,', 1)[1].split('## Amendment 2,', 1)[0]
amendment2 = design_text.split('## Amendment 2,', 1)[1].split('## Amendment 3,', 1)[0]
amendment3 = design_text.split('## Amendment 3,', 1)[1]
retry_failure_match = re.search(r'(\d+) of [\d,]+ pairs failed', amendment1)
dns_retry_match = re.search(r'(\d+) pairs still carry a name\s+resolution error', amendment2)
non_dns_match = re.search(r'(\d+) pairs remain failed for reasons other than\s+DNS', amendment2)
pre_agreement_match = re.search(r'primary agreement (\d+) of (\d+)', amendment3)
assert retry_failure_match and dns_retry_match and non_dns_match and pre_agreement_match
retry_pass_failures = int(retry_failure_match.group(1))
retry_pass_attempts = round(retry_pass_failures / retry_pass_fraction)
dns_retry_targets = int(dns_retry_match.group(1))
pre_matches, pre_comparable = map(int, pre_agreement_match.groups())
dns_retry_result = re.search(r"'targets':\s*(\d+),\s*'answered_by_a_model':\s*(\d+)", readers_log)
assert dns_retry_result
dns_answered_targets = int(dns_retry_result.group(2))
final_failures = [row for row in first_reader_labels if row.get('stop_reason')]
failure_reasons = defaultdict(int)
for row in final_failures:
    failure_reasons[str(row.get('stop_reason'))] += 1
assert len(final_failures) == error_cap['failed_pairs']
assert failure_reasons == {'max_tokens': len(final_failures)}
non_dns_before_retry = int(non_dns_match.group(1))
dns_retries_ending_in_max_tokens = len(final_failures) - non_dns_before_retry
assert dns_retry_targets == dns_answered_targets == 18
assert dns_retries_ending_in_max_tokens == 1

after_primary = reader_gate['second_reader']['replacement_primary']
after_matches = int(after_primary['matching_pairs'])
after_comparable = int(after_primary['comparable_pairs'])
agreement_threshold = float(after_primary['threshold'])
pre_agreement = pre_matches / pre_comparable
after_agreement = after_matches / after_comparable
assert after_matches - pre_matches == 5
assert pre_agreement >= agreement_threshold and after_agreement >= agreement_threshold

def file_time_ct(path):
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec='seconds')


amendment3_header_match = re.search(r'^## Amendment 3, (\d{4}-\d\d-\d\d \d\d:\d\d CT)$',
                                    design_text, re.MULTILINE)
amendment3_written = amendment3_header_match.group(1) if amendment3_header_match else None
assert amendment3_written is not None
reader_gate_summary = {
    'first_pass': {
        'failed_pairs': first_pass_failures,
        'total_pairs': int(error_cap['total_pairs']),
        'error_fraction': first_pass_fraction,
        'error_percent_1dp': round(first_pass_fraction * 100, 1),
        'cause_breakdown_available_in_readers_log': False,
        'cause_breakdown_note': 'The first-pass record provides failed pair IDs and an error fraction, not per-pair causes.',
    },
    'retry_pass_at_10_01': {
        'remaining_failures_amendment1_denominator': retry_pass_failures,
        'amendment1_original_denominator': int(error_cap['total_pairs']),
        'amendment1_percent_2dp': round(100 * retry_pass_failures / int(error_cap['total_pairs']), 2),
        'readers_log_attempts': retry_pass_attempts,
        'readers_log_error_fraction': retry_pass_fraction,
        'readers_log_percent_2dp': round(100 * retry_pass_fraction, 2),
        'cause': 'name resolution per Amendment 1',
    },
    'final_first_reader_failures': {
        'failed_pairs': len(final_failures),
        'by_stop_reason': dict(failure_reasons),
        'fraction': error_cap['error_fraction'],
        'fraction_percent_2dp': round(100 * error_cap['error_fraction'], 2),
        'cap_fraction': error_cap['maximum_fraction'],
        'all_are_max_tokens': True,
    },
    'dns_and_non_dns_reconciliation': {
        'non_dns_pairs_remaining_before_amendment2_dns_retry': non_dns_before_retry,
        'dns_pairs_retried': dns_retry_targets,
        'dns_pairs_answered_by_a_model': dns_answered_targets,
        'dns_retry_rows_ending_at_max_tokens': dns_retries_ending_in_max_tokens,
        'final_failures': len(final_failures),
        'reconciles_as': f'{non_dns_before_retry} non-DNS failures + {dns_retries_ending_in_max_tokens} retried DNS row ending at max_tokens',
    },
    'amendment3': {
        'written': amendment3_written,
        'primary_agreement_before': {
            'matching_pairs': pre_matches,
            'comparable_pairs': pre_comparable,
            'percent_1dp': round(100 * pre_agreement, 1),
            'agreement_threshold': agreement_threshold,
            'agreement_component_passed': pre_agreement >= agreement_threshold,
        },
        'primary_agreement_after': {
            'matching_pairs': after_matches,
            'comparable_pairs': after_comparable,
            'percent_1dp': round(100 * after_agreement, 1),
            'agreement_threshold': agreement_threshold,
            'agreement_component_passed': after_agreement >= agreement_threshold,
        },
        'agreement_matches_added_by_retry': after_matches - pre_matches,
        'pre_retry_coverage_errors': 8,
        'gate9_passed_after_retry': bool(reader_gate['passed']),
        'retry_output_file_time_ct': file_time_ct(ROOT / 'labels/labels-haiku.jsonl'),
        'scorer_rerun_output_file_time_ct': file_time_ct(ROOT / 'score.json'),
        'final_prediction_read_status': score.get('prediction_read_status'),
    },
}

inert = score['inert_guard']
inert_summary = {
    'comparison': inert['comparison'],
    'different': inert['different'],
    'total': inert['total'],
    'share': inert['share'],
    'share_percent_1dp': round(100 * inert['share'], 1),
    'threshold': inert['threshold'],
    'inert': inert['inert'],
    'ran_only_at_d_star': True,
}
zero_rule = {
    'successes': 0,
    'draws': 20,
    'exact_one_sided_95_upper_bound': 1 - 0.05 ** (1 / 20),
    'exact_one_sided_95_upper_percent_1dp': round(100 * (1 - 0.05 ** (1 / 20)), 1),
    'bounds': 'the support probability for a sign with zero successes in 20 draws',
}

out = {
    'part': 175,
    'source_files': [
        'scored_directions.jsonl', 'stage_a.json', 'layer_norms.json', 'score.json',
        'labels/labels-ds.jsonl', 'labels/labels-haiku.jsonl', 'readers.log', '../../PART175-DESIGN.md',
    ],
    'selection_disclosure': selection_disclosure,
    'claim_shares_by_abs_dose': claim_shares,
    'support_intervals_by_abs_dose': support_intervals,
    'zero_rule': zero_rule,
    'support_bound_robustness_by_abs_dose': support_bounds,
    'layer_comparison': layer_comparison,
    'inert_guard': inert_summary,
    'reader_and_gate_summary': reader_gate_summary,
    'verification': {
        'reader_error_count_matches_score': len(final_failures) == int(error_cap['failed_pairs']),
        'all_final_reader_errors_are_max_tokens': failure_reasons == {'max_tokens': len(final_failures)},
        'computed_support_bounds_match_score': all(row['matches_score_bounds'] for row in support_bounds),
        'registered_gate9_verdict_preserved': bool(reader_gate['passed']),
        'selected_count_is_in_sample_maximum': selection_disclosure['is_in_sample_maximum'],
        'selected_rule': score.get('selected_rule'),
        'headline_support_counts_unchanged': {
            str(dose): score['dose_reports']['by_dose'][str(dose)]['counts']['supported']
            for dose in ('0.15', '0.225', '0.3')
        },
    },
}
(ROOT / 'numbers175.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
