"""CPU regressions for Part 26 using tiny random models and the three cached tokenizers."""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import neuron_report as nr
import part20c as p20c
import part20f
import part26 as p


HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOTS = {
    'qwen': os.environ.get('QWEN35_SNAPSHOT',
        '/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
    'olmo': os.environ.get('OLMO2_SNAPSHOT',
        '/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e'),
    'smol': os.environ.get('SMOLLM3_SNAPSHOT',
        '/mnt/e/neuron-probe/hf/hub/models--HuggingFaceTB--SmolLM3-3B/snapshots/a07cc9a04f16550a088caea529712d1d335b0ac1'),
}
checks = 0
fails = 0


def check(name, ok, detail=''):
    global checks, fails
    checks += 1
    print(('  ok   ' if ok else '  FAIL ') + name + (f' ({detail})' if detail else ''), flush=True)
    fails += not bool(ok)


def tiny_model(snapshot, dirname, family):
    """Build a small random model by shrinking a pinned real config; only tokenizer files are read."""
    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if family == 'qwen':
        text = dict(cfg.text_config.to_dict())
        text.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                    linear_num_key_heads=2, linear_num_value_heads=4,
                    linear_key_head_dim=16, linear_value_head_dim=16,
                    layer_types=text['layer_types'][:4], tie_word_embeddings=True)
        text.pop('architectures', None)
        text.pop('model_type', None)
        cfg = AutoConfig.for_model('qwen3_5_text', **text)
    else:
        cfg.num_hidden_layers = 4
        cfg.hidden_size = 64
        cfg.intermediate_size = 96
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        if getattr(cfg, 'layer_types', None):
            cfg.layer_types = cfg.layer_types[:4]
        if hasattr(cfg, 'head_dim'):
            cfg.head_dim = 16
    torch.manual_seed({'qwen': 2601, 'olmo': 2602, 'smol': 2603}[family])
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if family == 'qwen':
        model.save_pretrained(dirname)
        tok.save_pretrained(dirname)
    return tok, model


def peak_rss_kib() -> int:
    """Read current resident memory for the short sampling window."""
    try:
        with open('/proc/self/status', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except OSError:
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def sampled_peak(call):
    stop = threading.Event()
    before = peak_rss_kib()
    state = {'peak': before}

    def monitor():
        while not stop.is_set():
            state['peak'] = max(state['peak'], peak_rss_kib())
            time.sleep(0.002)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        call()
    finally:
        stop.set()
        thread.join()
    return before, max(state['peak'], peak_rss_kib())


for snapshot in SNAPSHOTS.values():
    if not os.path.isfile(os.path.join(snapshot, 'config.json')):
        raise SystemExit(f'missing local snapshot config: {snapshot}')

with tempfile.TemporaryDirectory() as td:
    tiny_dirs = {name: os.path.join(td, f'tiny-{name}') for name in SNAPSHOTS}
    for path in tiny_dirs.values():
        os.makedirs(path)

    qtok = AutoTokenizer.from_pretrained(SNAPSHOTS['qwen'], local_files_only=True)
    otok = AutoTokenizer.from_pretrained(SNAPSHOTS['olmo'], local_files_only=True)
    stok, smol = tiny_model(SNAPSHOTS['smol'], tiny_dirs['smol'], 'smol')
    smol_layers = nr.find_text_layers(smol)[1]
    smol_texts = [' '.join((['A', 'quiet', 'farmer', 'repairs', 'a', 'wooden', 'fence', 'beside', 'the', 'river'] * 26)[:256])
                  for _ in range(4)]
    smol_batch = [p._tokenize_document(stok, text) for text in smol_texts]
    smol_ids, smol_mask = p._padded_inputs(smol_batch, stok, 'cpu')
    batch_shapes = []
    smol_hooks = []
    try:
        for layer in smol_layers:
            def measure(values):
                pooled = p._pool_word_tensor(values, smol_batch)
                batch_shapes.append((tuple(values.shape), tuple(pooled.shape)))
            smol_hooks.append(p._CallbackHook(layer.mlp.down_proj, measure))
        before_kib, peak_kib = sampled_peak(lambda: p._forward_decoder(smol, smol_ids, smol_mask))
    finally:
        for hook in smol_hooks:
            hook.remove()
    batch_tokens = int(smol_ids.shape[1])
    real_transient = (4 * batch_tokens * 11008 * 2 +
                      2 * 4 * batch_tokens * 11008 * 4 +
                      4 * 256 * 11008 * 4) / (1024 ** 3)
    check('tiny SmolLM3 config runs a four-document target batch',
          len(batch_shapes) == 4 and all(x[1][0:2] == (4, 256) for x in batch_shapes))
    check('tiny SmolLM3 batch peak RSS measured', peak_kib >= before_kib,
          f'baseline {before_kib / 1024:.1f} MiB; peak {peak_kib / 1024:.1f} MiB; '
          f'real Smol batch transient estimate {real_transient:.3f} GiB (batch 4, {batch_tokens} tokens, MLP width 11,008)')
    del smol, smol_layers, smol_hooks

    for family, tok in [('qwen', qtok), ('olmo', otok), ('smol', stok)]:
        text = 'Antidisestablishmentarianism is alien.\nA second word follows.'
        info = p._tokenize_document(tok, text)
        offsets = [(0, 0)] + info['offsets'] + [(0, 0)]
        token_values = np.arange(2 * (len(info['ids']) + 2), dtype=np.float32).reshape(-1, 2)
        token_values[0] = 1e9
        token_values[-1] = 2e9
        actual = p.word_activations(offsets, info['spans'], token_values)
        expected = np.stack([token_values[indices].max(axis=0)
                             for indices in p.word_token_indices(offsets, info['spans'])])
        actual_specials = [i for i, (a, b) in enumerate(offsets) if b <= a]
        assigned = {i for group in p.word_token_indices(offsets, info['spans']) for i in group}
        check(f'{family} word activation is max over overlapping token offsets',
              np.array_equal(actual, expected) and all(i not in assigned for i in actual_specials),
              f'{len(info["ids"])} tokenizer tokens; {len(actual_specials)} empty-span specials excluded')
        # Deliberate break: first overlapping token instead of the maximum must be caught.
        token_groups = p.word_token_indices(offsets, info['spans'])
        broken = np.stack([token_values[indices[0]] for indices in token_groups])
        check(f'{family} max-pooling regression catches first-token break',
              not np.array_equal(broken, actual))

    # Deliberate missing-token break: an uncovered word is refused rather than assigned a special token.
    refused_missing = False
    try:
        p.word_token_indices([(0, 0)], [(0, 4)])
    except ValueError:
        refused_missing = True
    check('empty special-token spans do not satisfy a word token requirement', refused_missing)

    rng = np.random.default_rng(26)
    source = rng.normal(size=(5, 7, 3)).astype(np.float32)  # [document, word, source]
    target = rng.normal(size=(5, 7, 4)).astype(np.float32)
    aligned = p.CorrelationAccumulator(3, 4)
    for start in (0, 2, 4):
        aligned.add(source.reshape(-1, 3)[start * 7:min(5, start + 2) * 7],
                    target.reshape(-1, 4)[start * 7:min(5, start + 2) * 7], 'cpu')
    aligned_expected = np.array([[np.corrcoef(source[:, :, i].ravel(), target[:, :, j].ravel())[0, 1]
                                  for j in range(4)] for i in range(3)])
    check('batch-accumulated aligned correlation equals numpy.corrcoef',
          np.allclose(aligned.correlations(), aligned_expected, atol=1e-5, rtol=1e-5),
          f'max error {np.max(np.abs(aligned.correlations() - aligned_expected)):.3g}')

    target_docs = np.arange(5)
    shifted_docs = p.misaligned_source_indices(target_docs, 5)
    null_source = source[shifted_docs]
    null_acc = p.CorrelationAccumulator(3, 4)
    null_acc.add(null_source.reshape(-1, 3), target.reshape(-1, 4), 'cpu')
    null_expected = np.array([[np.corrcoef(null_source[:, :, i].ravel(), target[:, :, j].ravel())[0, 1]
                               for j in range(4)] for i in range(3)])
    check('misaligned correlation pairs target d with source d+1, wrapping last to first',
          np.array_equal(shifted_docs, [1, 2, 3, 4, 0]) and
          np.allclose(null_acc.correlations(), null_expected, atol=1e-5, rtol=1e-5))
    broken_null = p.CorrelationAccumulator(3, 4)
    broken_null.add(source.reshape(-1, 3), target.reshape(-1, 4), 'cpu')
    check('misalignment regression catches same-document pairing break',
          not np.allclose(broken_null.correlations(), null_acc.correlations(), atol=1e-5, rtol=1e-5))

    check('write normalization merges space, case and byte-level marker forms',
          {p.normalize_write_token(x) for x in (' Alien', 'Alien', 'Ġalien')} == {'alien'})

    class FakeTokenizer:
        decoded = {0: ' Alien', 1: 'Alien', 2: 'Ġalien', 3: ' UFO', 4: 'Mars', 5: 'Comet'}

        def decode(self, ids):
            return self.decoded[int(ids[0])]

    fake = FakeTokenizer()
    source_sets = [{'alien', 'ufo', 'martian'}, {'alien', 'ufo', 'mars'}]
    overlap = p._writes_overlaps(fake, np.asarray([[0, 1, 2, 3, 4, 5]]), source_sets, {})
    check('normalized top-token set overlap counts distinct strings correctly',
          overlap[:, 0].tolist() == [2, 3], f'got {overlap[:, 0].tolist()}, expected [2, 3]')
    broken_normalizer_count = sum(p.normalize_write_token(fake.decode([i])) in source_sets[1]
                                  for i in [0, 1, 2, 3, 4, 5])
    check('write-overlap regression catches counting duplicate spellings as separate strings',
          broken_normalizer_count != int(overlap[1, 0]),
          f'broken list count {broken_normalizer_count}; set overlap {int(overlap[1, 0])}')

    # Part 20f's suffix begins at the fixed lead sentence. A phrase-only regression must be visible.
    fixture_text = 'filler\n\n' + p.LEAD + 'a stone from nowhere.'
    full_suffix = p.battery_span_positions(qtok, fixture_text, len('filler'))
    phrase_start = fixture_text.index('a stone')
    phrase_only = p.p20.token_span(qtok, fixture_text, (phrase_start, len(fixture_text)))
    fixture_values = np.zeros(len(qtok(fixture_text, add_special_tokens=False)['input_ids']), dtype=np.float32)
    fixture_values[full_suffix[0]] = 9.0
    fixture_values[phrase_only[0]] = 1.0
    check('battery-span regression catches excluding Part 20f fixed lead tokens',
          full_suffix[0] not in phrase_only and fixture_values[full_suffix].max() != fixture_values[phrase_only].max(),
          f'full suffix max {fixture_values[full_suffix].max():.1f}; phrase-only max {fixture_values[phrase_only].max():.1f}')

    qtok_tiny, qwen = tiny_model(SNAPSHOTS['qwen'], tiny_dirs['qwen'], 'qwen')
    qwen_tok, qwen = nr.load_model(tiny_dirs['qwen'], None, 'cpu')
    qwen.eval()
    _, qlayers, _ = nr.find_text_layers(qwen)
    down = qlayers[1].mlp.down_proj
    unit, other = 7, 8
    prompt = qwen_tok('Who made you?', return_tensors='pt').to('cpu')
    original_rows = []
    spy = down.register_forward_pre_hook(lambda _mod, args: original_rows.append(args[0].detach().float().cpu().clone()))
    clamp_hook = nr.NeuronHook(down, [unit, other])
    clamp_hook.clamp = (torch.tensor([unit]), torch.tensor([3.25]))
    try:
        with torch.no_grad():
            qwen.generate(**prompt, do_sample=False, max_new_tokens=5,
                           pad_token_id=qwen_tok.pad_token_id or qwen_tok.eos_token_id)
    finally:
        clamp_hook.clamp = None
        clamp_hook.remove()
        spy.remove()
    good_clamp = len(original_rows) == len(clamp_hook.captured) and len(original_rows) >= 2
    for before, after in zip(original_rows, clamp_hook.captured):
        good_clamp = good_clamp and torch.allclose(after[..., 0], torch.full_like(after[..., 0], 3.25))
        good_clamp = good_clamp and torch.allclose(after[..., 1], before[..., other], atol=0, rtol=0)
    check('generation clamp holds target at 3.25 at every prompt/decode position and leaves other unit unchanged',
          good_clamp, f'{len(original_rows)} hooked forward calls')

    # Deliberate break: a bad hook shifts every unit and ignores the requested clamp.
    good_method = nr.NeuronHook._hook
    bad_before, bad_after = [], []
    def broken_hook(self, _mod, args):
        changed = args[0].clone() + 0.25
        bad_after.append(changed[..., [unit, other]].detach().float().cpu())
        return (changed,) + tuple(args[1:])
    nr.NeuronHook._hook = broken_hook
    bad_spy = down.register_forward_pre_hook(lambda _mod, args: bad_before.append(args[0].detach().float().cpu().clone()))
    bad_hook = nr.NeuronHook(down, [unit, other])
    bad_hook.clamp = (torch.tensor([unit]), torch.tensor([3.25]))
    try:
        with torch.no_grad():
            qwen.generate(**prompt, do_sample=False, max_new_tokens=2,
                           pad_token_id=qwen_tok.pad_token_id or qwen_tok.eos_token_id)
    finally:
        bad_hook.remove()
        bad_spy.remove()
        nr.NeuronHook._hook = good_method
    check('clamp regression catches a hook that changes every unit and ignores the set value',
          bool(bad_after) and any(not torch.allclose(row[..., 1], before[..., other], atol=0, rtol=0)
                                  for before, row in zip(bad_before, bad_after)) and
          any(not torch.allclose(row[..., 0], torch.full_like(row[..., 0], 3.25)) for row in bad_after))

    # Score a compact nine-set fixture through both implementations on the same tiny Qwen model.
    corpus = os.path.join(td, 'corpus.jsonl')
    filler_line = 'In a quiet town a patient farmer repairs a wooden fence beside the river before returning home.'
    with open(corpus, 'w', encoding='utf-8') as f:
        for _ in range(360):
            f.write(json.dumps({'text': filler_line}, ensure_ascii=False) + '\n')
    battery_path = os.path.join(td, 'battery.json')
    battery_sets = {
        'E': ['a stone from no place on earth and unreachable by ship or plane'],
        'E-map': ['strangers whose birthplace lies on no map people have ever drawn'],
        'D': ['relatives who traveled all night by bus to attend the funeral'],
        'U': ['a lake hidden beneath miles of antarctic ice'],
        'C1': ['the rights and duties we owe to fellow beings'],
        'C2': ['neighbors who are not from here but have settled in nicely'],
        'C3': ['beings not from here who have lived on this street for decades'],
        'C3-bare': ['people not from here who have lived on this street for decades'],
        'C4': ['the oldest trees on earth'],
    }
    with open(battery_path, 'w', encoding='utf-8') as f:
        json.dump({'sets': battery_sets}, f)
    part20f_out = os.path.join(td, 'part20f')
    command = [sys.executable, os.path.join(HERE, 'part20f.py'), '--model', tiny_dirs['qwen'],
               '--layer', '1', '--neuron', '7',
               '--corpus', corpus, '--battery', battery_path, '--out', part20f_out,
               '--device', 'cpu', '--no-self-check']
    part20f_run = subprocess.run(command, capture_output=True, text=True)
    check('tiny Part 20f reference scorer completes', part20f_run.returncode == 0,
          f'exit {part20f_run.returncode}; {part20f_run.stdout.strip()[-250:]}')
    if part20f_run.returncode != 0:
        print(part20f_run.stderr[-1500:], flush=True)
    else:
        reference = json.load(open(os.path.join(part20f_out, 'battery.json'), encoding='utf-8'))
        _, qlayers, _ = nr.find_text_layers(qwen)
        candidate = {'layer': 1, 'unit': 7, 'r': 0.0, 'largest_activation': 0.0}
        ours = p.score_battery_candidates(qwen_tok, qwen, qlayers, [candidate], corpus,
                                         battery_path, 'cpu', 14.5)
        want = reference['summary']['mean_max_by_set']
        got = ours['mean_max_by_candidate']['1:7']
        match = all(np.isclose(got[name], want[name], atol=1e-6, rtol=1e-6) for name in battery_sets)
        check('Part 26 battery means equal part20f.py on tiny Qwen for all nine sets', match,
              f'max diff {max(abs(got[name] - want[name]) for name in battery_sets):.3g}; filler docs match={ours["filler_doc_indices"] == reference["filler_doc_indices"]}')
        broken_battery = {name: value + 0.5 for name, value in got.items()}
        check('battery equality regression catches a deliberately offset scorer',
              not all(np.isclose(broken_battery[name], want[name], atol=1e-6, rtol=1e-6) for name in battery_sets),
              'deliberate scorer offset: +0.5 per set')

print(f'{checks - fails} passed, {fails} failed')
print('PASS' if not fails else 'FAIL')
raise SystemExit(1 if fails else 0)
