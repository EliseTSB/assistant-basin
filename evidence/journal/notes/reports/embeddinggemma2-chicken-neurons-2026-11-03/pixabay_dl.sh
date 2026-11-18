#!/bin/bash
# pixabay_dl.sh LABELS.csv OUTDIR: download the first ~1.2 MB (well over 30 s) of every real/ai/imitation row, at
# 300 KB/s, one at a time, skipping files already present. Small files come through the mitm proxy fine; for big ones
# go direct (notes/gotchas.md, large downloads through the proxy). Columns id,label first; src looked up by id.
set -u
L=$1; OUT=$2; n=0; fail=0
while IFS=, read -r id label rest; do
  case "$label" in real|ai|imitation) ;; *) continue;; esac
  src=$(python3 -I -c "import csv,sys; print(next(r['src'] for r in csv.DictReader(open('$L')) if r['id']=='$id'))")
  f="$OUT/$id.mp3"; [ -s "$f" ] && continue
  if SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt curl -sS -m 60 --fail --limit-rate 300k -r 0-1199999 -A 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36' -H 'Referer: https://pixabay.com/' -o "$f.part" "$src"; then mv "$f.part" "$f"; n=$((n+1)); else fail=$((fail+1)); rm -f "$f.part"; echo "FAIL $id"; fi
  sleep 0.5
done < <(tail -n +2 "$L")
echo "DONE downloaded=$n failed=$fail total_on_disk=$(ls $OUT/*.mp3 2>/dev/null | wc -l) bytes=$(du -sb $OUT | cut -f1)"
