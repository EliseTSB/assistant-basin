# Counts scanned documents holding the known unit's words. Run from the repo with E: mounted at /mnt/e, Claude, 2026-10-31.
import re
from pathlib import Path
src=Path('part136/scan136.py').read_text(encoding='utf-8')
i=src.index('def read_rows'); j=src.index('\ndef ',i+10)
ns={}
exec("from pathlib import Path\n"+src[i:j],ns)
w=ns['read_rows']('/mnt/e/neuron-probe/corpus/wikitext103-train0.parquet','wiki',4000)
c=ns['read_rows']('/mnt/e/neuron-probe/corpus/ultrachat-test_sft.parquet','chat',4000)
rx=re.compile(r'\b(aliens?|extraterrestrials?)\b',re.I)
for name,docs in (('wiki',w),('chat',c)): print(name,len(docs),sum(bool(rx.search(d['text'])) for d in docs))
