#!/usr/bin/env bash
# Download the clips in vis_manifest.csv from VocalImitationSet at the pinned commit, rate-limited, straight (no proxy).
# Audio is not committed. Usage: bash vis_download.sh
set -u
COMMIT=b77114c43c52f5faa7d0bb69b22e1ebca0632201
OUT=/mnt/e/neuron-probe/vocalimitationset/audio
mkdir -p "$OUT"
tail -n +2 vis_manifest.csv | /home/elisepc/neuron-probe/.venv/bin/python -I -c "
import csv,sys,urllib.parse
for r in csv.reader(sys.stdin):
    print(r[1]+'\t'+urllib.parse.quote(r[7]))" | while IFS=$'\t' read -r local enc; do
  [ -s "$OUT/$local" ] && continue
  curl -sS --noproxy '*' --fail --retry 3 --retry-delay 5 --limit-rate 500k --max-time 300 \
    -o "$OUT/$local.part" "https://raw.githubusercontent.com/interactiveaudiolab/VocalImitationSet/$COMMIT/$enc" \
    && mv "$OUT/$local.part" "$OUT/$local" || echo "FAIL $local"
done
echo "DONE $(ls "$OUT"/*.wav | wc -l) files, $(du -sh "$OUT" | cut -f1)"
