#!/usr/bin/env python3
# tok_slices.py — the wikitext-103 token slices topk_named.py and heldout.py read, written as .npy so a job without
# pyarrow (the A770's Windows environment) gets exactly the same tokens (Claude, 2026-10-16, Part 11.10-11.13).
#
# Same procedure as topk_named.py and heldout.py: wikitext lines, headings and blanks dropped, each line tokenized with
# no special tokens and appended to one stream. Lines are streamed in batches and reading stops once the stream is long
# enough, so the whole corpus is never in memory.
#   desc.npy  tokens [0, desc), cut to whole 512-token chunks   (topk_named.py: 100,000 -> 99,840)
#   held.npy  tokens [start, start + held), cut the same way     (heldout.py: 1,000,000 + 300,000 -> 299,520)
# Usage: python tok_slices.py <model> <out dir> [--desc 100000] [--start 1000000] [--held 300000] [--corpus PARQUET]
import argparse, json, os
import numpy as np, pyarrow.parquet as pq
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('out')
ap.add_argument('--desc', type=int, default=100_000); ap.add_argument('--start', type=int, default=1_000_000)
ap.add_argument('--held', type=int, default=300_000)
ap.add_argument('--corpus', default=os.environ.get('WIKITEXT_PARQUET', 'data/wikitext103-train0.parquet'))
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
need = max(a.desc, a.start + a.held)
ids = []
for batch in pq.ParquetFile(a.corpus).iter_batches(columns=['text'], batch_size=2000):
    for t in batch.column('text').to_pylist():
        if t.strip() and not t.strip().startswith('='):
            ids.extend(tok(t, add_special_tokens=False)['input_ids'])
            if len(ids) >= need: break
    if len(ids) >= need: break
if len(ids) < need: raise SystemExit(f'REFUSED: corpus has {len(ids)} tokens, need {need}')
SEQ = 512
desc = np.array(ids[: a.desc], dtype=np.int64); desc = desc[: len(desc) // SEQ * SEQ]
held = np.array(ids[a.start: a.start + a.held], dtype=np.int64); held = held[: len(held) // SEQ * SEQ]
os.makedirs(a.out, exist_ok=True)
np.save(os.path.join(a.out, 'desc.npy'), desc); np.save(os.path.join(a.out, 'held.npy'), held)
json.dump({'model': a.model, 'desc_tokens': int(len(desc)), 'held_start': a.start, 'held_tokens': int(len(held))},
          open(os.path.join(a.out, 'slices.json'), 'w'))
print(f'desc {len(desc)} tokens, held {len(held)} tokens from {a.start}', flush=True)
