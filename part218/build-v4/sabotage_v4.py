#!/usr/bin/env python3
"""Sabotage run for the Part 218 v4 build: break each rule once in a COPY and
show the matching test turns red.  Writes sabotage_results.json beside itself.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = Path('/home/elisepc/green/worktrees/ab-part218/part218')
OUT = Path(__file__).resolve().parent
PYTHON = '/home/elisepc/dspy-wsp/.venv/bin/python'

CASES = [
    ('run218.py', 'opening appended with no trailing space',
     '    return build_prompt(tok, question) + opening\n',
     "    return build_prompt(tok, question) + opening + ' '\n",
     'test_run218.py::test_item_input_appends_the_opening_with_no_trailing_space', True),
    ('run218.py', 'text = opening + continuation (unstripped)',
     '    return opening + continuation\n',
     '    return (opening + continuation).strip()\n',
     'test_run218.py::test_answer_text_is_exactly_opening_plus_unstripped_continuation', True),
    ('run218.py', 'item order item_index = 2p + o',
     "        'item_index': int(2 * prompt_index + opening_index),\n",
     "        'item_index': int(2 * prompt_index + (N_OPENINGS - 1 - opening_index)),\n",
     'test_run218.py::test_items_are_eighty_eight_in_the_fixed_order', True),
    ('run218.py', 'pilot batch 41 rows / 40 directions',
     'PILOT_SEEDS = FULL_SEEDS',
     'PILOT_SEEDS = tuple(range(5))',
     'test_run218.py::test_pilot_and_main_batches_are_41_rows_40_pushed_plus_zero', True),
    ('run218.py', 'dose = d x median residual norm',
     '    scale = float(dose) * float(median_norm)\n',
     '    scale = float(dose)\n',
     'test_run218.py::test_pilot_and_main_batches_are_41_rows_40_pushed_plus_zero', True),
    ('run218.py', 'gate vectors are exactly zero',
     '    return torch.zeros(int(rows), int(hidden), dtype=torch.float32)\n',
     '    return torch.ones(int(rows), int(hidden), dtype=torch.float32)\n',
     'test_run218.py::test_gate_vectors_are_all_exactly_zero', True),
    ('run218.py', 'gate mode builds no pushed vector (smoke)',
     '    vecs_cpu = gate_vectors(hidden, batch_rows)\n',
     '    vecs_cpu = gate_vectors(hidden, batch_rows) + 1.0\n',
     'test_run218.py::test_tiny_model_end_to_end_smoke_pilot_gate_and_full', False),
    ('score218.py', 'gate 60 of 88 items',
     '    passed = (len(eligible_items) >= GATE_MIN_ITEMS\n',
     '    passed = (len(eligible_items) >= GATE_MIN_ITEMS - 1\n',
     'test_score218.py::test_gate_is_sixty_of_eighty_eight_and_twenty_four_per_opening', True),
    ('score218.py', 'gate 24 of 44 under each opening (constant)',
     'GATE_MIN_PER_OPENING = 24',
     'GATE_MIN_PER_OPENING = 23',
     'test_score218.py::test_gate_is_sixty_of_eighty_eight_and_twenty_four_per_opening', True),
    ('score218.py', 'gate per-opening condition enforced',
     '    passed = (len(eligible_items) >= GATE_MIN_ITEMS\n'
     '              and all(per_opening[opening] >= GATE_MIN_PER_OPENING\n'
     '                      for opening in per_opening))\n',
     '    passed = (len(eligible_items) >= GATE_MIN_ITEMS)\n',
     'test_score218.py::test_gate_is_sixty_of_eighty_eight_and_twenty_four_per_opening', True),
    ('score218.py', 'moved needs 3 distinct prompts',
     'MOVED_MIN_PROMPTS = 3',
     'MOVED_MIN_PROMPTS = 2',
     'test_score218.py::test_moved_needs_three_distinct_prompts_not_three_items', True),
    ('score218.py', 'supported: broken on at most 10%',
     'BROKEN_FRACTION = 0.10',
     'BROKEN_FRACTION = 1.0',
     'test_score218.py::test_supported_and_led_with_ties', True),
    ('score218.py', 'led/refusal/other_maker split by opening',
     '    by_opening = {}\n    for opening in range(N_OPENINGS):\n',
     '    by_opening = {}\n    for opening in range(0):\n',
     'test_score218.py::test_supported_and_led_with_ties', True),
    ('score218.py', 'read rung 36 of 40 (pilot too)',
     'READ_RUNG_MIN = 36',
     'READ_RUNG_MIN = 9',
     'test_score218.py::test_pilot_read_rung_is_also_36_of_40', True),
    ('score218.py', '218.5 floor 84 of 88 items',
     'IDENTICAL_MIN_ITEMS = 84',
     'IDENTICAL_MIN_ITEMS = 83',
     'test_score218.py::test_218_5_eighty_four_of_eighty_eight_items_and_null', True),
    ('score218.py', '218.6 paired-item intersection only shared openings',
     '    for item in sorted(sft_eligible & dpo_eligible):\n',
     '    for item in sorted(sft_eligible | dpo_eligible):\n',
     'test_score218.py::test_218_6_paired_item_intersection_uses_only_shared_openings', True),
    ('score218.py', '218.6 intersection at least 25 prompts',
     'INTERSECTION_MIN = 25',
     'INTERSECTION_MIN = 19',
     'test_score218.py::test_218_6_not_read_for_small_intersection_or_few_other_makers', True),
    ('score218.py', '218.6 at least 20 other_maker answers pooled',
     'OTHER_MAKER_MIN = 20',
     'OTHER_MAKER_MIN = 5',
     'test_score218.py::test_218_6_not_read_for_small_intersection_or_few_other_makers', True),
    ('score218.py', '218.6 shift at least 10 points',
     'SHIFT_MIN = 0.10',
     'SHIFT_MIN = 1.5',
     'test_score218.py::test_218_6_held_when_the_stage_shift_is_clear', True),
    ('score218.py', 'registered prompt sha256',
     "PROMPTS_SHA256 = ('d84ff8497a182896bdbf6a6c3c1aaa659df4c3902f98bd2364673df7c58097f2')",
     "PROMPTS_SHA256 = ('0' * 64)",
     'test_score218.py::test_registered_thresholds_match_the_design', True),
    ('score218.py', 'eligible when ALL zero rows are self',
     "        all_self = (complete and len(counts) == 1 and majority == 'self')\n",
     "        all_self = bool(complete and majority == 'self')\n",
     'test_score218.py::test_item_is_eligible_only_when_all_zero_rows_are_self', True),
    ('run218.py', 'saved null logits are a row copy, not a batch view',
     "    return scores[-1].detach().to('cpu').clone()\n",
     "    return scores[-1].detach().to('cpu')\n",
     'test_run218.py::test_last_row_logits_copies_the_row_off_the_batch_buffer', True),
]

IGNORE = shutil.ignore_patterns('__pycache__', '.pytest_cache', 'screen', 'results',
                                '*.pyc')


def make_copy(destination: Path) -> Path:
    copy = destination / 'part218'
    shutil.copytree(SOURCE, copy, ignore=IGNORE)
    return copy


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory(prefix='part218-sabotage-') as temp:
        root = Path(temp)
        for index, case in enumerate(CASES, start=1):
            filename, rule, old, new, test, skip_smoke = case
            copy = make_copy(root / f'case{index:02d}')
            target = copy / filename
            source = target.read_text(encoding='utf-8')
            if source.count(old) != 1:
                raise SystemExit(
                    f'case {index}: break target occurs {source.count(old)} '
                    f'times in {filename}: {old!r}')
            target.write_text(source.replace(old, new), encoding='utf-8')
            env = dict(__import__('os').environ)
            env.pop('PART218_SKIP_SMOKE', None)
            if skip_smoke:
                env['PART218_SKIP_SMOKE'] = '1'
            command = [PYTHON, '-I', '-m', 'pytest', '-q', '-p',
                       'no:cacheprovider', str(copy / test)]
            completed = subprocess.run(command, cwd=str(root), env=env,
                                       capture_output=True, text=True,
                                       timeout=1800)
            red = completed.returncode != 0
            tail = (completed.stdout + completed.stderr).strip().splitlines()
            tail = '\n'.join(tail[-6:])
            results.append({
                'case': index,
                'rule': rule,
                'file': filename,
                'test': test,
                'returncode': completed.returncode,
                'turned_red': red,
                'tail': tail,
            })
            print(f'case {index:02d} {"RED " if red else "GREEN!"} {rule} '
                  f'-> {test}', flush=True)

    (OUT / 'sabotage_results.json').write_text(
        json.dumps(results, indent=1) + '\n', encoding='utf-8')
    red = sum(1 for row in results if row['turned_red'])
    print(f'\n{red}/{len(results)} sabotage cases turned their test red')
    return 0 if red == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
