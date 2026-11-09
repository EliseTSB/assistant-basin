import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

repo = Path(__file__).resolve().parent
sys.path.insert(0, str(repo))
import run198 as R

class Stats:
    @staticmethod
    def localized(values, edited):
        valid = {layer: value for layer, value in values.items() if value is not None}
        top = max(valid.values())
        winners = [layer for layer, value in valid.items() if value == top]
        if len(winners) != 1:
            return False, None, len(winners) > 1
        return winners[0] in edited, winners[0], False

def rows(layer1):
    return {
        0: {name: 5.0 for name in ('f1', 'f2', 'pcs', 't_max', 't_word')} | {'pairs': []},
        1: {name: layer1 for name in ('f1', 'f2', 'pcs', 't_max', 't_word')} | {'pairs': []},
    }

with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / 'base-qwen35.json'
    wrong_model_baseline = rows(3.0)
    cache_path.write_text(json.dumps({
        'model': 'different-model-or-old-statistics',
        'inputs_sha256': 'stale-input-digest',
        'layers': {str(k): v for k, v in wrong_model_baseline.items()},
    }))
    expected_digest = 'current-synthetic-input-digest'
    try:
        R.load_base(cache_path, 'Qwen/Qwen3.5-0.8B', expected_digest)
        refused = False
        refusal = 'cache unexpectedly accepted'
    except RuntimeError as exc:
        refused = True
        refusal = str(exc)
    except TypeError as exc:
        refused = False
        refusal = f'expected an input-checking loader API, got {type(exc).__name__}: {exc}'
    current = {0: {name: 6.0 for name in ('f1', 'f2', 'pcs', 't_max', 't_word')} | {'pairs': []}}
    correct = R.localization(rows(10.0), current, {0}, Stats())
    stale = R.localization(wrong_model_baseline, current, {0}, Stats())
    stat_src = Path(repo / 'run198.py').read_text()
    print('torch_imported:', 'torch' in sys.modules)
    print('cache_model_marker:', json.loads(cache_path.read_text())['model'])
    print('cache_input_hash_marker:', json.loads(cache_path.read_text())['inputs_sha256'])
    print('load_base_refuses_stale_cache:', refused)
    print('refusal:', refusal)
    print('fresh_base_f1_localized:', correct['f1']['localized'])
    print('cached_base_f1_localized:', stale['f1']['localized'])
    checked_calls = stat_src.count('base_path, WSP_MODEL_ID, base_inputs_sha256')
    print('both_arm_b_call_sites_require_expected_digest:', checked_calls == 2)
    if not refused or checked_calls != 2:
        raise SystemExit('WSP base-cache refusal check failed')
