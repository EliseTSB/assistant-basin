#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHAT_REV="b968826d9c46dd6066d109eabc6255188de91218"
MODEL="$ROOT/models/qwen3-8b-$CHAT_REV"
OUT="$ROOT/results/part132"
cd "$ROOT"
mkdir -p "$OUT/pilot"
PILOT_NEURONS="$(mktemp "$OUT/.neurons132-pilot.XXXXXX")"
trap 'rm -f "$PILOT_NEURONS"' EXIT
python - "$PILOT_NEURONS" <<'PY'
import json
from pathlib import Path
import sys
rows=json.loads(Path("part132/neurons132.json").read_text(encoding="utf-8"))
if len(rows)!=189:
    raise SystemExit(f"expected 189 Part 132 neurons, found {len(rows)}")
pilot=[]
seen=set()
for row in rows:
    layer=row["L"]
    if layer not in seen:
        pilot.append(row)
        seen.add(layer)
        if len(pilot)==4:
            break
if len(pilot)!=4:
    raise SystemExit(f"expected four pilot neurons on distinct layers, found {len(pilot)}")
Path(sys.argv[1]).write_text(json.dumps(pilot,indent=2)+"\n",encoding="utf-8")
PY
python - <<'PY'
import torch, transformers
if torch.__version__ != "2.6.0+cu124": raise SystemExit(f"expected torch 2.6.0+cu124, got {torch.__version__}")
if transformers.__version__ != "5.20.0": raise SystemExit(f"expected transformers 5.20.0, got {transformers.__version__}")
if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
PY
python -I -B part91/push91.py --model "$MODEL" --repo Qwen/Qwen3-8B --commit "$CHAT_REV" \
  --frame chat --tag inst_chat --neurons "$PILOT_NEURONS" --out "$OUT/pilot"
RATE="$(python - "$OUT/pilot/inst_chat.meta.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
rate=float(m["rows"])/float(m["seconds"])
print(f"{rate:.6f}")
print(f"Part 132 pilot throughput: {rate:.2f} answers/s",file=sys.stderr)
PY
)"
if python - "$RATE" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) < 1.5 else 1)
PY
then
  printf 'Part 132 stopped: pilot throughput %.2f answers/s is below 1.5; push91.py uses fixed BATCH=48, so no runtime batch adjustment was made.\n' "$RATE"
  exit 3
fi
python -I -B part91/push91.py --model "$MODEL" --repo Qwen/Qwen3-8B --commit "$CHAT_REV" \
  --frame chat --tag inst_chat --neurons part132/neurons132.json --out "$OUT"
printf 'Part 132 complete: 189 neurons pushed into %s.\n' "$OUT"
