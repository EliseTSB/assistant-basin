#!/usr/bin/env python3
"""Record the registered Part 196 Expresso clips on CPU with resumable rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from part196.select196 import gate1_check, read_metadata, text_key  # noqa: E402

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
STYLES = ("confused", "default", "enunciated", "happy", "laughing", "sad", "whisper")
LENGTH_MIN = 2.0
LENGTH_MAX = 10.0
N_LAYERS = 24
UNITS_PER_LAYER = 2048
N_UNITS = N_LAYERS * UNITS_PER_LAYER
EMBEDDING_SIZE = 768
DEFAULT_DATA_DIR = Path("/mnt/e/neuron-probe/expresso/read")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate recording ids")
    required = {"id", "speaker_id", "style", "text_key", "shard", "row_index", "fold"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"manifest is missing columns: {sorted(required - set(rows[0]))}")
    return rows


def manifest_sha(rows: list[dict[str, str]]) -> str:
    content = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def wav_duration(raw: bytes) -> float:
    """Read duration from a WAV header without decoding sample data."""
    with wave.open(io.BytesIO(raw), "rb") as audio:
        rate = audio.getframerate()
        frames = audio.getnframes()
        if rate <= 0:
            raise ValueError("WAV has an invalid sample rate")
        return frames / rate


def _row_group_index(parquet: Any, row_index: int) -> tuple[int, int]:
    offset = 0
    for group in range(parquet.num_row_groups):
        length = parquet.metadata.row_group(group).num_rows
        if offset <= row_index < offset + length:
            return group, row_index - offset
        offset += length
    raise IndexError(f"row index {row_index} is outside parquet file")


def iter_audio_batches(rows: list[dict[str, str]], data_dir: Path
                       ) -> Iterator[list[tuple[dict[str, str], bytes]]]:
    """Yield selected audio rows grouped by parquet row group to bound memory."""
    import pyarrow.parquet as pq

    by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard"]].append(row)
    for shard_name, shard_rows in sorted(by_shard.items()):
        shard_path = data_dir / shard_name
        parquet = pq.ParquetFile(shard_path)
        group_rows: dict[int, list[tuple[dict[str, str], int]]] = defaultdict(list)
        for row in shard_rows:
            group, offset = _row_group_index(parquet, int(row["row_index"]))
            group_rows[group].append((row, offset))
        for group, selections in sorted(group_rows.items()):
            table = parquet.read_row_group(group, columns=["audio", "id"])
            audio_values = table.column("audio").to_pylist()
            ids = table.column("id").to_pylist()
            batch: list[tuple[dict[str, str], bytes]] = []
            for row, offset in selections:
                if str(ids[offset]) != row["id"]:
                    raise ValueError(f"parquet identity mismatch for {row['id']}")
                item = audio_values[offset]
                raw = item.get("bytes") if isinstance(item, dict) else None
                if not isinstance(raw, (bytes, bytearray)):
                    raise ValueError(f"audio bytes are missing for {row['id']}")
                batch.append((row, bytes(raw)))
            yield batch
            del table, audio_values, ids


def _atomic_npz(path: Path, **items: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **items)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def duration_filter(rows: list[dict[str, str]], data_dir: Path, work_dir: Path
                    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
    lengths: dict[str, Any] = {}
    accepted = []
    for batch in iter_audio_batches(rows, data_dir):
        for row, raw in batch:
            try:
                duration = float(wav_duration(raw))
                keep = LENGTH_MIN <= duration <= LENGTH_MAX
                lengths[row["id"]] = {"duration_seconds": duration,
                                       "status": "included" if keep else "outside_2_to_10_seconds"}
                if keep:
                    accepted.append(row)
                else:
                    _atomic_npz(work_dir / "records" / f"{row['id']}.npz",
                                id=np.asarray(row["id"]), status=np.asarray("length_excluded"),
                                duration_seconds=np.asarray(duration, dtype=np.float32),
                                reason=np.asarray("outside registered 2 to 10 second range"))
            except Exception as exc:
                lengths[row["id"]] = {"duration_seconds": None,
                                       "status": "duration_error",
                                       "error": f"{type(exc).__name__}: {exc}"}
                _atomic_npz(work_dir / "records" / f"{row['id']}.npz",
                            id=np.asarray(row["id"]), status=np.asarray("duration_error"),
                            duration_seconds=np.asarray(np.nan, dtype=np.float32),
                            reason=np.asarray(lengths[row["id"]]["error"]))
    gate = gate1_check(rows, included_ids={row["id"] for row in accepted})
    gate["length_filter"] = {"minimum_seconds": LENGTH_MIN, "maximum_seconds": LENGTH_MAX,
                             "included": len(accepted), "excluded": len(rows) - len(accepted)}
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "durations.json").write_text(
        json.dumps(lengths, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (work_dir / "gate1.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return accepted, gate


def read_transcripts(rows: list[dict[str, str]], data_dir: Path) -> dict[str, str]:
    """Load selected transcript text into memory from metadata, keeping it out of the manifest."""
    wanted = {(row["id"], row["speaker_id"], row["style"]): row["text_key"] for row in rows}
    found: dict[str, str] = {}
    for metadata in read_metadata(data_dir):
        identity = (str(metadata.get("id", "")), str(metadata.get("speaker_id", "")),
                    str(metadata.get("style", "")))
        if identity not in wanted:
            continue
        text = str(metadata["text"])
        if text_key(text) != wanted[identity]:
            raise ValueError(f"transcript hash mismatch for {identity[0]}")
        found[identity[0]] = text
    missing = sorted(identity[0] for identity in set(wanted) if identity[0] not in found)
    if missing:
        raise KeyError(f"transcripts missing for {len(missing)} selected recordings")
    return found


def decode_audio_bytes(raw: bytes) -> np.ndarray:
    """Match Part 188's mono 16 kHz f32le decode, with a ten-second cap."""
    import imageio_ffmpeg
    import subprocess

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-i", "pipe:0",
                           "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                          input=raw, capture_output=True, timeout=60, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()[:80_000]


def acoustic_features(audio: np.ndarray, text: str) -> np.ndarray:
    """Return voiced fraction, F0 median/IQR, RMS mean/std, centroid mean, seconds/word."""
    import pyworld

    samples = np.asarray(audio, dtype=np.float32)
    f0, _time_axis = pyworld.harvest(np.ascontiguousarray(samples, dtype=np.float64),
                                     fs=16000, frame_period=5.0)
    voiced = f0[f0 > 0]
    voiced_fraction = float(np.mean(f0 > 0)) if f0.size else 0.0
    if voiced.size:
        f0_median = float(np.median(voiced))
        f0_iqr = float(np.percentile(voiced, 75) - np.percentile(voiced, 25))
    else:
        f0_median = f0_iqr = 0.0

    frame_length, hop = 400, 160
    if samples.size == 0:
        frames = np.zeros((1, frame_length), dtype=np.float32)
    else:
        count = max(1, int(np.ceil(max(0, samples.size - frame_length) / hop)) + 1)
        needed = (count - 1) * hop + frame_length
        padded = np.pad(samples, (0, max(0, needed - samples.size)))
        frames = np.stack([padded[i * hop:i * hop + frame_length] for i in range(count)])
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    windowed = frames * np.hanning(frame_length).astype(np.float32)[None, :]
    magnitudes = np.abs(np.fft.rfft(windowed, axis=1))
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / 16000.0)
    totals = magnitudes.sum(axis=1)
    centroid = np.divide(magnitudes @ frequencies, totals,
                         out=np.zeros_like(totals, dtype=np.float64), where=totals > 0)
    words = max(1, len(re.findall(r"\S+", text.strip())))
    duration = samples.size / 16000.0
    return np.asarray([voiced_fraction, f0_median, f0_iqr, float(rms.mean()),
                       float(rms.std()), float(centroid.mean()), duration / words],
                      dtype=np.float32)


def load_part188(model_path: Path | None = None, threads: int = 6):
    """Use Part 188's pinned model loader and hooks, on `threads` CPU threads (Part 188 measured 6)."""
    import part188.record188 as record188

    torch, model = record188.load_model(model_path)
    torch.set_num_threads(threads)
    layers, buffer, hooks = record188.install_hooks(model)
    model._part188_buffer = buffer
    return record188, torch, model, layers, buffer, hooks


def gate2_parity(record188: Any, torch: Any, model: Any, layers: Any,
                 buffer: dict[int, Any]) -> dict[str, Any]:
    """Run Part 188's eight-clip recorder parity gate before Expresso work."""
    ref_path = ROOT / "part188" / "parity_ref.npz"
    ref = np.load(ref_path, allow_pickle=False)
    expected_names = [str(value) for value in ref["filenames"]]
    audio_dir = record188.ESC_LOCAL_AUDIO
    comparisons = []
    for index, filename in enumerate(expected_names):
        path = audio_dir / filename
        if not path.is_file():
            comparisons.append({"filename": filename, "passed": False,
                                "reason": f"parity audio missing: {path}"})
            continue
        buffer.clear()
        try:
            audio = record188.decode_audio(path)
            item = {"audio": {"array": audio, "sampling_rate": 16000}}
            with torch.inference_mode():
                embedding = np.asarray(model.encode(item, normalize_embeddings=True), dtype=np.float32)
            if len(buffer) != len(layers):
                raise RuntimeError(f"expected {len(layers)} layers, captured {len(buffer)}")
            captured = torch.stack([buffer[layer] for layer in range(len(layers))])
            activation = captured.mean(1).reshape(-1).cpu().numpy().astype(np.float16)
            if activation.size != N_UNITS or embedding.size != EMBEDDING_SIZE:
                raise RuntimeError("Part 188 parity output shapes differ")
            check = record188.compare_gate1(activation, embedding,
                                            ref["act_mean"][index], ref["emb"][index])
            comparisons.append({"filename": filename, **check})
        except Exception as exc:
            comparisons.append({"filename": filename, "passed": False,
                                "reason": f"{type(exc).__name__}: {exc}"})
    passed = len(comparisons) == 8 and all(row.get("passed") is True for row in comparisons)
    return {"verdict": "passed" if passed else "failed", "clips": comparisons,
            "rules": {"activation_abs_tolerance": 0.002,
                      "fraction_of_units_required": 0.999,
                      "embedding_cosine_required": 0.9999}}


def encode_text_assets(model: Any, rows: list[dict[str, str]], texts_by_id: dict[str, str],
                       out_dir: Path) -> None:
    phrasing_path = HERE / "phrasings.json"
    phrasing_doc = json.loads(phrasing_path.read_text(encoding="utf-8"))
    phrases = [phrase for style in STYLES for phrase in phrasing_doc["styles"][style]]
    if len(phrases) != 56:
        raise ValueError(f"expected 56 SearchQuery phrasings, found {len(phrases)}")
    prompt_path = out_dir / "prompts196.npz"
    if not prompt_path.exists():
        query_embeddings = np.asarray(model.encode(phrases, prompt_name="SearchQuery",
                                                   normalize_embeddings=True, batch_size=32), dtype=np.float32)
        if query_embeddings.shape != (56, EMBEDDING_SIZE):
            raise ValueError(f"unexpected phrasing embedding shape: {query_embeddings.shape}")
        np.savez_compressed(prompt_path, styles=np.asarray(STYLES), phrases=np.asarray(phrases),
                            embeddings=query_embeddings.reshape(7, 8, -1))
    transcript_path = out_dir / "transcripts196.npz"
    if not transcript_path.exists():
        by_key = {row["text_key"]: texts_by_id[row["id"]] for row in rows}
        unique_keys = sorted(by_key)
        unique_texts = [by_key[key] for key in unique_keys]
        text_embeddings = np.asarray(model.encode(unique_texts, prompt_name="SearchDocument",
                                                  normalize_embeddings=True, batch_size=32), dtype=np.float32)
        if text_embeddings.shape != (len(unique_texts), EMBEDDING_SIZE):
            raise ValueError(f"unexpected transcript embedding shape: {text_embeddings.shape}")
        np.savez_compressed(transcript_path, text_keys=np.asarray(unique_keys),
                            embeddings=text_embeddings)


def _record_one(row: dict[str, str], raw: bytes, record_dir: Path, model: Any,
                torch: Any, layers: Any, buffer: dict[int, Any], text: str) -> None:
    path = record_dir / f"{row['id']}.npz"
    audio = decode_audio_bytes(raw)
    if audio.size == 0:
        raise ValueError("ffmpeg returned no samples")
    features = acoustic_features(audio, text)
    buffer.clear()
    item = {"audio": {"array": audio, "sampling_rate": 16000}}
    with torch.inference_mode():
        embedding = np.asarray(model.encode(item, normalize_embeddings=True), dtype=np.float32)
    if len(buffer) != len(layers):
        raise RuntimeError(f"expected {len(layers)} hooked layers; captured {len(buffer)}")
    activations = torch.stack([buffer[layer] for layer in range(len(layers))])
    seqlen = int(activations.shape[1])
    act_mean = activations.mean(1).reshape(-1).cpu().numpy().astype(np.float16)
    if act_mean.size != N_UNITS or embedding.size != EMBEDDING_SIZE:
        raise RuntimeError(f"unexpected output shapes {act_mean.shape} / {embedding.shape}")
    duration = len(audio) / 16000.0
    _atomic_npz(path, id=np.asarray(row["id"]), act_mean=act_mean, emb=embedding,
                seqlen=np.asarray(seqlen, dtype=np.int32), status=np.asarray("ok"),
                duration_seconds=np.asarray(duration, dtype=np.float32),
                acoustic=features, audio_sha256=np.asarray(hashlib.sha256(raw).hexdigest()))


def _cached_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            return str(saved["status"].item()) if "status" in saved else None
    except (OSError, ValueError, KeyError):
        return None


def run_recording(*, manifest_path: Path, data_dir: Path, work_dir: Path,
                  model_path: Path | None = None, threads: int = 6) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    record_dir = work_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    identity = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "model_path": str(model_path) if model_path else "hub",
                "manifest_sha256": manifest_sha(rows), "data_dir": str(data_dir)}
    identity_path = work_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("recording work directory identity differs; use a fresh work directory")
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    record188, torch, model, layers, buffer, hooks = load_part188(model_path, threads)
    started = time.perf_counter()
    try:
        gate2 = gate2_parity(record188, torch, model, layers, buffer)
        (work_dir / "gate2.json").write_text(json.dumps(gate2, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
        if gate2["verdict"] != "passed":
            raise RuntimeError("Gate 2 parity failed; no Expresso clip was recorded")

        accepted, gate1 = duration_filter(rows, data_dir, work_dir)
        if gate1["verdict"] != "passed":
            raise RuntimeError("Gate 1 failed after duration filtering; no Expresso clip was recorded")
        texts_by_id = read_transcripts(accepted, data_dir)

        query_path = work_dir / "prompts196.npz"
        transcript_path = work_dir / "transcripts196.npz"
        if not query_path.exists() or not transcript_path.exists():
            encode_text_assets(model, accepted, texts_by_id, work_dir)

        # Read and record in bounded row groups; no worker pool or GPU is used.
        completed = 0
        skipped = 0
        for batch in iter_audio_batches(accepted, data_dir):
            for row, raw in batch:
                record_path = record_dir / f"{row['id']}.npz"
                if _cached_status(record_path) == "ok":
                    skipped += 1
                    continue
                try:
                    _record_one(row, raw, record_dir, model, torch, layers, buffer,
                                texts_by_id[row["id"]])
                except Exception as exc:
                    _atomic_npz(record_path, id=np.asarray(row["id"]), status=np.asarray("record_error"),
                                reason=np.asarray(f"{type(exc).__name__}: {exc}"),
                                duration_seconds=np.asarray(float(wav_duration(raw)), dtype=np.float32))
                completed += 1
                if completed % 50 == 0:
                    print(f"clips_processed={completed}/{len(accepted)}", flush=True)
        return {"gate1": gate1["verdict"], "gate2": gate2["verdict"],
                "selected_after_length_filter": len(accepted), "recorded_this_run": completed,
                "resumed_ok": skipped, "elapsed_seconds": time.perf_counter() - started,
                "work_dir": str(work_dir)}
    finally:
        for hook in hooks:
            hook.remove()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest196.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/e/neuron-probe/part196-work"))
    parser.add_argument("--threads", type=int, default=6, help="CPU threads (Part 188 measured 0.68 s per clip on 6)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="local snapshot; defaults to the registered Hub revision")
    args = parser.parse_args(argv)
    try:
        result = run_recording(manifest_path=args.manifest, data_dir=args.data_dir,
                               work_dir=args.work_dir, model_path=args.model_path, threads=args.threads)
    except Exception as exc:
        print(json.dumps({"gate1": "failed" if "Gate 1" in str(exc) else "unknown",
                          "gate2": "failed" if "Gate 2" in str(exc) else "unknown",
                          "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
