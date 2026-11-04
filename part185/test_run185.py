"""Tests for part185/run185.py with the part's main and the model load replaced (no torch needed)."""
import json, sys, types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import part185.run185 as r


def fake_modules(monkeypatch, tmp_path, kernel, sha):
    nr = types.ModuleType('neuron_report'); nr.LAST_LOAD = None
    def part_main(rest):
        out = r.out_dir(rest); out.mkdir(parents=True, exist_ok=True)
        (out / 'run.json').write_text(json.dumps({'calib_sha256': sha, 'arms': {}}), encoding='utf-8')
        nr.LAST_LOAD = {'attn_implementation': kernel, 'dtype': 'torch.bfloat16'}
    pkg = types.ModuleType('part84'); mod = types.ModuleType('part84.part84'); mod.main = part_main
    pkgb = types.ModuleType('part84b'); modb = types.ModuleType('part84b.part84b'); modb.main = part_main
    for k, v in {'neuron_report': nr, 'part84': pkg, 'part84.part84': mod, 'part84b': pkgb, 'part84b.part84b': modb}.items():
        monkeypatch.setitem(sys.modules, k, v)


@pytest.mark.parametrize('which', ['84', '84b'])
def test_gate_passes_and_records_load(monkeypatch, tmp_path, which):
    fake_modules(monkeypatch, tmp_path, 'eager', r.CALIB_SHA256)
    out = tmp_path / 'o'
    assert r.main([which, '--out', str(out), '--calib', 'x']) == 0
    run = json.loads((out / 'run.json').read_text())
    assert run['load']['attn_implementation'] == 'eager' and run['part185']['reruns'] == 'Part ' + which


def test_gate_fails_on_sdpa(monkeypatch, tmp_path):
    fake_modules(monkeypatch, tmp_path, 'sdpa', r.CALIB_SHA256)
    out = tmp_path / 'o'
    assert r.main(['84', '--out', str(out), '--calib', 'x']) == 3
    assert json.loads((out / 'run.json').read_text())['load']['attn_implementation'] == 'sdpa'


def test_gate_fails_on_other_calibration(monkeypatch, tmp_path):
    fake_modules(monkeypatch, tmp_path, 'eager', '0' * 64)
    assert r.main(['84b', '--out=' + str(tmp_path / 'o'), '--calib', 'x']) == 3


def test_refuses_unknown_part():
    with pytest.raises(SystemExit):
        r.main(['85', '--out', 'x'])
