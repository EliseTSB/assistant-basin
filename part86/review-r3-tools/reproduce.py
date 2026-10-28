"""Small unmodified-code reproductions for the cold review."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/elisepc/green/worktrees/ab-part67')
sys.path.insert(0, str(ROOT / 'part86'))
sys.path.insert(0, str(ROOT))
import blind86
import common86
import triage86
import verify86

OUT = Path(__file__).resolve().parent / 'repros'
OUT.mkdir(exist_ok=True)
screen = OUT / 'screen'
screen.mkdir(exist_ok=True)
rows = [dict(layer=10, kind='neuron', id=i,
             answer=f'I am a quiet traveler sailing the open sea number {i}.') for i in range(20)]
(screen / 'layer_10.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
mapping = {f'x{i}': dict(layer=10, source='real', item_index=i,
                        triage_category='changed', miss_rate_sample=False) for i in range(20)}
primary = [dict(id=item_id, persona='a quiet traveler', kind='human', coherent=True) for item_id in mapping]
picks = [dict(layer=10, neuron=i, persona=f'a deity ruling planet {i}') for i in range(20)]
picked_path = OUT / 'picked20.json'
picked_path.write_text(json.dumps(picks))
verify86.validate_picked_against_screen(verify86.load_picked(picked_path), screen, {'rows': mapping}, primary)
print('Mismatched picked persona phrases: ACCEPTED (all screen labels were a quiet traveler)')

# Real score gate on conflicting binary screen labels.
first = {str(i): dict(coherent=True, kind='human') for i in range(10)}
second = {str(i): dict(coherent=True, kind='still Qwen or an assistant') for i in range(10)}
print('Screen coherent-persona kappa:', blind86.cohen_kappa(first, second, first))

source = OUT / 'verify-source'
source.mkdir(exist_ok=True)
for n in range(20):
    records = []
    for condition, direction in [('neuron', None)] + [('random_direction', d) for d in range(10)]:
        for qi in range(3):
            for sample in range(8):
                records.append(dict(layer=10, neuron=n, condition=condition, direction=direction,
                                    question_index=qi, sample=sample, seed=verify86.seed_for(qi, sample),
                                    persona=f'persona {n}', answer='I am a quiet traveler.'))
    # The last complete file is from a different dose/configuration.
    payload = dict(status='complete', config=dict(dose=0.4 if n == 19 else 0.8,
                                                 calibration_sha256='different' if n == 19 else 'original'),
                   records=records)
    (source / f'L10_N{n}.json').write_text(json.dumps(payload))
items = OUT / 'reader' / 'items.jsonl'
key_path = OUT / 'private' / 'key.json'
verify86.prepare(source, items, key_path, force=True)
key = json.loads(key_path.read_text())
labels = [dict(id=item_id, claims_persona=(info['condition'] == 'neuron' and
                                          info['question_index'] * 8 + info['sample'] < 12))
          for item_id, info in key['rows'].items()]
secondary = [row for row in labels if row['id'] in key['second_reader_ids']]
result = verify86.score(labels, key, items, secondary)
print('Mixed verification generation configs: ACCEPTED')
print('Verification result without any screen-agreement input:', result['verdict_86.5'])
print('Verification kappa:', result['reader_agreement']['kappa'])
print('Verification preparation manifest:', key['manifest'])
(OUT / 'verification-result.json').write_text(json.dumps(result, indent=2) + '\n')

# Exercise the actual screen score CLI with full denominators and low screen agreement.
full = OUT / 'full-screen'
full.mkdir(exist_ok=True)
for layer in range(10, 27):
    config = dict(layer=layer, median=24.8, dose=0.8, actual_dtype='torch.bfloat16',
                  code_sha256=common86.generation_code_sha256(), versions=common86.dependency_versions())
    (full / f'layer_{layer}.meta.json').write_text(json.dumps(dict(status='complete', config=config)))
    with (full / f'layer_{layer}.jsonl').open('w') as f:
        for kind, count in [('neuron', 9216), ('random_direction', 512)]:
            for n in range(count):
                answer = f'I am a quiet traveler sailing the open sea number {n}.' if layer == 10 and kind == 'neuron' and n < 20 else 'I am Qwen, an AI assistant.'
                f.write(json.dumps(dict(layer=layer, kind=kind, id=n, answer=answer)) + '\n')
triage_path = OUT / 'full-triage.json'
triage_path.write_text(json.dumps(triage86.triage(full)))
full_items, full_key_path = OUT / 'full-reader' / 'items.jsonl', OUT / 'full-private' / 'key.json'
blind86.prepare(full, triage_path, full_items, full_key_path, force=True)
full_key = json.loads(full_key_path.read_text())
screen_labels = [dict(id=item_id, persona='a traveler' if info['triage_category'] == 'changed' else 'Qwen',
                      kind='human' if info['triage_category'] == 'changed' else 'still Qwen or an assistant',
                      coherent=True) for item_id, info in full_key['rows'].items()]
screen_labels2 = [dict(row, kind='still Qwen or an assistant' if row['kind'] == 'human' else 'human')
                  for row in screen_labels if row['id'] in full_key['second_reader_ids']]
first_path, second_path = OUT / 'screen-labels.jsonl', OUT / 'screen-labels2.jsonl'
for path, labels_for_path in [(first_path, screen_labels), (second_path, screen_labels2)]:
    path.write_text(''.join(json.dumps(row) + '\n' for row in labels_for_path))
screen_result_path = OUT / 'screen-result.json'
blind86.main(['score', '--screen-dir', str(full), '--items', str(full_items), '--key', str(full_key_path),
              '--labels', str(first_path), '--labels2', str(second_path), '--out', str(screen_result_path), '--force'])
screen_result = json.loads(screen_result_path.read_text())
print('Actual screen CLI agreement:', screen_result['reader_agreement'])
print('Actual screen CLI verdicts:', {name: row['status'] for name, row in screen_result['verdicts'].items()})
print('Independent verification still has no input for that screen gate:', result['verdict_86.5'])
