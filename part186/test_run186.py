"""Tests for part186/run186.py with Part 79's mains and the model load replaced (no torch needed)."""
import json, sys, types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import part186.run186 as r


def fake(monkeypatch, kernel, seen):
    nr = types.ModuleType('neuron_report'); nr.LAST_LOAD = None
    def make(name):
        def main():
            seen.append(list(sys.argv))
            out = r.out_dir(sys.argv[1:]); out.mkdir(parents=True, exist_ok=True)
            (out / r.RECORD[name]).write_text(json.dumps({'medians': {}}), encoding='utf-8')
            nr.LAST_LOAD = {'attn_implementation': kernel}
        return main
    pkg = types.ModuleType('part79')
    s0 = types.ModuleType('part79.step0'); s0.main = make('step0')
    p79 = types.ModuleType('part79.part79'); p79.main = make('main')
    pkg.step0, pkg.part79 = s0, p79
    for k, v in {'neuron_report': nr, 'part79': pkg, 'part79.step0': s0, 'part79.part79': p79}.items():
        monkeypatch.setitem(sys.modules, k, v)


@pytest.mark.parametrize('which', ['step0', 'main'])
def test_passes_args_and_records_load(monkeypatch, tmp_path, which):
    seen = []; fake(monkeypatch, 'eager', seen)
    out = tmp_path / 'o'
    assert r.main([which, '--out', str(out), '--batch', '8']) == 0
    assert seen[0][1:] == ['--out', str(out), '--batch', '8']
    rec = json.loads((out / r.RECORD[which]).read_text())
    assert rec['load']['attn_implementation'] == 'eager' and rec['part186']['reruns'] == f'Part 79 {which}'
    assert sys.argv != seen[0]  # restored


def test_gate_fails_on_sdpa(monkeypatch, tmp_path):
    seen = []; fake(monkeypatch, 'sdpa', seen)
    out = tmp_path / 'o'
    assert r.main(['main', '--out', str(out), '--step0', 'x']) == 3
    assert json.loads((out / 'run.json').read_text())['load']['attn_implementation'] == 'sdpa'


def test_refuses_unknown_step():
    with pytest.raises(SystemExit):
        r.main(['pilot', '--out', 'x'])
