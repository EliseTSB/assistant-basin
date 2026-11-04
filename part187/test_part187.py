"""Part 187: the loader Part 140 uses passes and records the eager kernel, and the wrapper refuses a non-eager load."""
import json, sys, types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'part115b'))
sys.path.insert(0, str(ROOT / 'part115'))
sys.path.insert(0, str(ROOT / 'part86'))


def test_load_direct_passes_and_records_eager(monkeypatch):
    import torch, transformers
    import neuron_report as nr
    import howmany115b
    seen = {}

    class Tok:
        pad_token_id = 0
        eos_token = '<e>'

    class Model(torch.nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
            self.config = types.SimpleNamespace(_attn_implementation=attn)

    def fake_model(model_id, **kw):
        seen.update(kw)
        return Model(kw.get('attn_implementation'))

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', classmethod(lambda cls, *a, **k: Tok()))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', classmethod(lambda cls, m, **k: fake_model(m, **k)))
    nr.LAST_LOAD = None
    howmany115b.load_direct('cpu')
    assert seen['attn_implementation'] == 'eager'
    assert nr.LAST_LOAD['attn_implementation'] == 'eager'
    howmany115b.load_direct('cpu', attn_implementation='sdpa')
    assert seen['attn_implementation'] == 'sdpa' and nr.LAST_LOAD['attn_implementation'] == 'sdpa'


@pytest.mark.parametrize('kernel,expected', [('eager', 0), ('sdpa', 3), (None, 3)])
def test_wrapper_gate(monkeypatch, tmp_path, kernel, expected):
    import neuron_report as nr
    from part140 import run140
    from part187 import run187

    def fake_main(argv):
        nr.LAST_LOAD = None if kernel is None else {'attn_implementation': kernel}
        return 0

    monkeypatch.setattr(run140, 'main', fake_main)
    rc = run187.main(['--out', str(tmp_path / 'out'), '--device', 'cpu'])
    assert rc == expected
    rec = json.loads((tmp_path / 'out' / 'load.json').read_text(encoding='utf-8'))
    assert rec['load'] == (None if kernel is None else {'attn_implementation': kernel})
