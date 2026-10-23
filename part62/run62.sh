#!/usr/bin/env bash
# Part 62 runner. Errors preserve the rented instance and all partial artifacts for review.
set -Eeuo pipefail

on_error() {
  local status="$1"
  trap - ERR
  printf 'RUN_FAILED_KEEP_BOX status=%s\n' "$status" >&2
  printf 'Inspect %s and the runner log before any instance teardown.\n' "${PART62_OUTPUT:-output path not initialized}" >&2
  exit "$status"
}
trap 'on_error "$?"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PART62_OUTPUT="$PROJECT_ROOT/results/part62"
PART62_ANALYSIS="$PROJECT_ROOT/results/part62-analysis.json"
RETURN_ARCHIVE="${RETURN_ARCHIVE:-/root/part62-return.tgz}"
export PART62_OUTPUT PART62_ANALYSIS

cd "$PROJECT_ROOT"
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
