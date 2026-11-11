#!/usr/bin/env python3
"""Run the Part 201 runner's whole smoke path inside a folder that holds ONLY one launch step's --copy files.

    python -I -B part201/copy_closure_driver.py <folder>

Under -I neither this script's folder nor the working directory is importable, so the only repository modules Python
can find are the ones in <folder>. The model load is replaced by a one-layer random Qwen3.5 on the CPU; everything
else is the real runner: plan, G5 against Part 163's generator, three processes, the float32 upcast, the timing
probe. A module the step's --copy list leaves out ends this with ImportError and a non-zero exit, on the host and not
on the card. Used by part201/test_part201.py (CopyClosure). Written by Claude (Fable 5.2), 2026-11-09, after the
DeepSeek code review of a3a40b1 found tools/part_check.py missing from three steps' lists.
"""
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(folder))

import torch                      # noqa: E402
import neuron_report as nr        # noqa: E402
from part201 import run201        # noqa: E402

assert Path(run201.__file__).resolve().is_relative_to(folder), run201.__file__


class TinyTokenizer:
    pad_token_id = 0; eos_token_id = 2; padding_side = 'right'

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[-1]['content'] + ' assistant'

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[3 + (sum(map(ord, c)) % 20) for c in text.split()] for text in texts]
        width = max(map(len, rows)) if padding else len(rows[0])
        ids = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(masks, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(int(x)) for x in ids if not skip_special_tokens or int(x) != 2)


def fake_load(model_id, revision, device, attn_implementation='eager', dtype=torch.bfloat16):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(201)
    cfg = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
                            linear_key_head_dim=8, linear_value_head_dim=8, layer_types=['full_attention'],
                            eos_token_id=2, pad_token_id=0)
    model = Qwen3_5ForCausalLM(cfg).eval().to(dtype=torch.bfloat16)
    model.config._attn_implementation = 'eager'
    nr.LAST_LOAD = {'attn_implementation': 'eager'}
    return TinyTokenizer(), model


nr.load_model = fake_load
work = folder / '_closure'
(work / 'ckpt').mkdir(parents=True)
units = ([{'unit_id': f'L0N{i}', 'pair_id': f'L0D{i}', 'arm': 'neuron', 'layer': 0, 'neuron': i, 'random_index': None} for i in range(12)]
         + [{'unit_id': f'L0D{i}', 'pair_id': f'L0D{i}', 'arm': 'random', 'layer': 0, 'neuron': None, 'random_index': i} for i in range(12)])
(work / 'units.json').write_text(json.dumps(units), encoding='utf-8')
(work / 'medians.json').write_text(json.dumps({'0': 1.0}), encoding='utf-8')
code = run201.main(['--smoke', '--device', 'cpu', '--model', str(work / 'ckpt'), '--units', str(work / 'units.json'),
                    '--medians', str(work / 'medians.json'), '--out', str(work / 'out'), '--max-new-tokens', '6'])
load = json.loads((work / 'out' / 'load.json').read_text(encoding='utf-8'))
print('closure ok:', [x['process'] for x in load['loads']], 'probe', (work / 'out' / 'probe.json').exists())
raise SystemExit(code)
