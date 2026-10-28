#!/usr/bin/env python3
"""Inventory quantitative prose sentences before README's Results by part."""
import csv, re
from pathlib import Path
README=Path('/home/elisepc/green/worktrees/ab-part67/README.md')
OUT=Path('/home/elisepc/green/Journal/notes/reports/frontpage-audit-2026-10-27/claims.tsv')
front=README.read_text(encoding='utf-8').split('## Results by part',1)[0]
front=front.split('## Related work',1)[0]
num_re=re.compile(r'(?<![A-Za-z0-9])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|x|k|M|B)?|(?<![A-Za-z0-9])[-+]?\.\d+(?:%|x)?')
word_re=re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|sixteen|seventeen|twenty|twenty-one|twenty-four|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|half|third|quarter)\b',re.I)
part_re=re.compile(r'\bParts?\s+\d+(?:\s*(?:and|to|through|,|&|or)\s*\d+)*',re.I)
model_re=re.compile(r'\b(?:Qwen\s?\d+(?:\.\d+)?(?:-[A-Za-z0-9.]+)?|SmolLM\d+(?:-\d+[A-Za-z]*)?|OLMo(?:-\d+)?(?:\s+\d+(?:\.\d+)?(?:B|M))?|GPT-\d+(?:\.\d+)?(?:\s+[A-Za-z0-9.]+)?|L\d+\s+N\d+|L\d+\s+h\d+)\b',re.I)
model_suffix_re=re.compile(r'\b(?:4B|3B|1\.7B|0\.6B)\b',re.I)
date_re=re.compile(r'\b20\d{2}-\d{2}-\d{2}\b')
rows=[]; section='Overview'
for line in front.splitlines():
    hm=re.match(r'\s*##\s+(.*)',line)
    if hm:
        section=hm.group(1).strip(); continue
    if not line.strip() or line.strip().startswith(('### ', '---')):
        continue
    if 'Shabalin and Piotrowski' in line:
        continue
    # Each README paragraph and bullet is stored on a single source line.
    original=line.strip()
    splitline=re.sub(r'\]\([^)]*\)',']',original)
    parts=[]
    for v in part_re.findall(original):
        if v not in parts: parts.append(v)
    for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z“\"(])',splitline):
        quote=sentence.strip(' -*')
        if not quote or quote.startswith(('## ','### ')):
            continue
        probe=re.sub(r'\]\([^)]*\)',']',quote)
        # Remove Markdown citation labels, but preserve numeric intervals like [.105, .565].
        probe=re.sub(r'\[(?!\s*[-+]?(?:\d|\.\d))[^]]*\]',' ',probe)
        probe=re.sub(r'arXiv:[\w.]+',' ',probe,flags=re.I)
        probe=part_re.sub(' ',probe)
        probe=model_re.sub(' ',probe)
        probe=model_suffix_re.sub(' ',probe)
        probe=date_re.sub(' ',probe)
        vals=num_re.findall(probe); words=word_re.findall(probe)
        if not vals and not words: continue
        collected=[]
        for x in vals+words:
            if x not in collected: collected.append(x)
        rows.append((section,quote,'; '.join(collected),'; '.join(parts)))
with OUT.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n')
    w.writerow(['id','finding/section','exact quote','number(s)','part'])
    for i,(sec,q,nums,part) in enumerate(rows,1): w.writerow([f'C{i:03}',sec,q,nums,part])
print(f'wrote {len(rows)} sentence-level quantitative claims to {OUT}')
