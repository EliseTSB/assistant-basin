#!/usr/bin/env bash
# Part 62 runner. Errors preserve the rented instance and all partial artifacts for review.
set -Eeuo pipefail

on_error() {
  local status="$1"
  trap - ERR INT TERM
  local incomplete="${RETURN_INCOMPLETE_ARCHIVE:-/root/part62-return-INCOMPLETE.tgz}"
  local project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local paths=()
  for path in part62 results/part62 results/part62-analysis.json; do
    [[ -e "$project_root/$path" ]] && paths+=("$path")
  done
  if (( ${#paths[@]} )); then tar -czf "$incomplete" -C "$project_root" "${paths[@]}" || true; fi
  printf 'RUN_FAILED status=%s archive=%s\n' "$status" "$incomplete" >&2
  printf 'Inspect %s and the runner log before any instance teardown.\n' "${PART62_OUTPUT:-output path not initialized}" >&2
  exit "$status"
}
trap 'on_error "$?"' ERR

on_signal() {
  local signal="$1"
  local status=130
  [[ "$signal" == "TERM" ]] && status=143
  on_error "$status"
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PART62_OUTPUT="$PROJECT_ROOT/results/part62"
PART62_ANALYSIS="$PROJECT_ROOT/results/part62-analysis.json"
RETURN_ARCHIVE="${RETURN_ARCHIVE:-/root/part62-return.tgz}"
export PART62_OUTPUT PART62_ANALYSIS

cd "$PROJECT_ROOT"
mkdir -p /root/hf/hub
fetch_model() {
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir="/root/hf/hub"))
PY
}
fetch_model Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
fetch_model allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e
export HF_HOME=/root/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
printf 'Part 62 generation starts: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -u part62/run62.py --out "$PART62_OUTPUT" --device cuda
python3 -u part62/analyze62.py --folder "$PART62_OUTPUT" --out "$PART62_ANALYSIS"

python3 - <<'PY'
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "part62"))
import run62
manifest_path = Path(os.environ["PART62_OUTPUT"]) / "part62-manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (manifest.get("status") != "complete"
        or manifest.get("generated_answers") != run62.expected_generation_count()):
    raise SystemExit("Part 62 manifest is incomplete; leaving the instance and artifacts for inspection")
PY

tar -czf "$RETURN_ARCHIVE" -C "$PROJECT_ROOT" part62 results/part62 results/part62-analysis.json
python3 - "$RETURN_ARCHIVE" <<'PY'
import sys
import tarfile
with tarfile.open(sys.argv[1], "r:gz") as archive:
    names = set(archive.getnames())
required = {"part62/run62.py", "results/part62/part62-manifest.json",
            "results/part62-analysis.json"}
missing = sorted(required - names)
if missing:
    raise SystemExit(f"return archive is missing required files: {missing}")
PY
test -s "$RETURN_ARCHIVE"
printf 'archive verified: %s\n' "$RETURN_ARCHIVE"
trap - ERR
printf 'RETURN_READY\n'
exit 0
