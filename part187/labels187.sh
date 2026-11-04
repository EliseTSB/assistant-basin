#!/usr/bin/env bash
# Part 187 reader gates then labels, run locally (API calls only), as Part 140's steps.txt with out results/part187/labels.
# Run 2026-11-04 from the host, started after 12:29 CT and finished by 14:08 CT (API calls only). First labelling run scored with a python lacking
# transformers, which made score140.py write a failed gate2 record; score with the project venv (results/part187/README.md).
set -u
cd /home/elisepc/green/worktrees/ab-part180b
O=results/part187/labels; B=part86/READER-BRIEF-v2.md; K=part140/known140.json; PY="python3 -I -B"
export SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt
$PY part140/label140.py --stage gate-ds --known $K --brief $B --out $O; echo "gate-ds rc=$?"
$PY part140/label140.py --stage gate-luna --known $K --brief $B --out $O; echo "gate-luna rc=$?"
cat $O/gate-ds-result.json $O/gate-luna-result.json 2>/dev/null
$PY part140/label140.py --stage ds --input results/part187/main/answers.jsonl --brief $B --out $O > $O/ds.log 2>&1 &
p1=$!
$PY part140/label140.py --stage luna --input results/part187/main/answers.jsonl --brief $B --out $O > $O/luna.log 2>&1 &
p2=$!
wait $p1; echo "ds rc=$?"; wait $p2; echo "luna rc=$?"
grep -c '"error"' $O/ds.jsonl $O/luna.jsonl
wc -l $O/ds.jsonl $O/luna.jsonl
