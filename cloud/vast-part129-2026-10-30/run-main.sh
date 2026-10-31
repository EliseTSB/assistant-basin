#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHAT_REV="b968826d9c46dd6066d109eabc6255188de91218"
BASE_REV="49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
CHAT_DIR="$ROOT/models/qwen3-8b-$CHAT_REV"
BASE_DIR="$ROOT/models/qwen3-8b-base-$BASE_REV"
NL_ROOT="$ROOT/neuron-lens"
OUT="$ROOT/results/part129"
BATCH="${1:-32}"
cd "$ROOT"
CORPUS=("$NL_ROOT/data/corpus/act-dclm.jsonl" "$NL_ROOT/data/corpus/act-zh.jsonl" "$NL_ROOT/data/corpus/act-code.jsonl" "$NL_ROOT/data/corpus/act-chat-qwen3-8b.jsonl")
python -I -B part129/collect129.py --stage main --batch "$BATCH" --chat-model "$CHAT_DIR" --base-model "$BASE_DIR" --corpus "${CORPUS[@]}" --prepared "$OUT/prepared129.pt" --out "$OUT/main"
bash cloud/vast-part129-2026-10-30/run-part132.sh
