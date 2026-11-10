#!/usr/bin/env bash
# Download only the filenames already fixed in manifest203.csv.
# Run from the repository root, or pass manifest and data-directory paths.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-$ROOT/part203/manifest203.csv}"
DATA_DIR="${2:-/mnt/e/neuron-probe/part203-data/AudioWAV}"
BATCH_SIZE=25
PAUSE_SECONDS=5

mkdir -p "$DATA_DIR"
FILES_FILE="$(mktemp)"
trap 'rm -f "$FILES_FILE"' EXIT
python3 -I - "$MANIFEST" > "$FILES_FILE" <<'PY'
import csv
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8", newline="") as source:
    reader = csv.DictReader(source)
    if not reader.fieldnames or "file" not in reader.fieldnames:
        raise SystemExit("manifest must have a 'file' column")
    for row in reader:
        name = row["file"]
        if not re.fullmatch(r"\d{4}_IEO_[A-Z]{3}_[A-Z]{2}\.wav", name):
            raise SystemExit(f"unsafe or unexpected CREMA-D filename: {name!r}")
        print(name)
PY

count=0
while IFS= read -r file; do
    [ -n "$file" ] || continue
    target="$DATA_DIR/$file"
    if [ ! -s "$target" ]; then
        url="https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/$file"
        curl --noproxy '*' --limit-rate 800k --fail --location "$url" --output "$target" || {
            rm -f "$target"
            exit 1
        }
    fi
    python3 -I - "$target" <<'PY'
import sys
import wave
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as source:
    header = source.read(12)
if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
    raise SystemExit(f"invalid WAV RIFF header: {path.name}")
with wave.open(str(path), "rb") as wav:
    channels, rate, frames = wav.getnchannels(), wav.getframerate(), wav.getnframes()
print(f"{path.name}: RIFF/WAVE, {channels} channel(s), {rate} Hz, {frames} frames" +
      ("" if channels == 1 and rate == 16000 else " (rate/channel differs; recorder must resample)"))
PY
    count=$((count + 1))
    if (( count % BATCH_SIZE == 0 )); then
        sleep "$PAUSE_SECONDS"
    fi
done < "$FILES_FILE"
