#!/usr/bin/env python3
"""Record the registered Part 203 CREMA-D readouts on CPU with resumable rows.

Readout V is Venn's audio vector, implemented as read from part203/prior/venn-server.py: the WAV is decoded with
soundfile (mono 16 kHz, at most 30 s), passed as {"text": "<|audio|>", "audio": [samples]} to the media model's
encode(..., normalize_embeddings=True) with no prompt_name, and stored as float16. Readout D is Part 196's final
embedding, recorded by part196/record196.py's own code path. The recorder records both and states in its identity
whether they are the same computation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from part196.select196 import STYLES  # noqa: E402
from part196.record196 import (EMBEDDING_SIZE, MODEL_ID, MODEL_REVISION, N_UNITS,  # noqa: E402
                               _atomic_npz, _cached_status, acoustic_features,
                               encode_text_assets, gate2_parity, load_part188,
                               recording_identity, write_or_check_identity)
from part203.select203 import EMOTIONS, read_manifest  # noqa: E402

SAMPLE_RATE = 16_000
MAX_SAMPLES_V = 30 * SAMPLE_RATE      # venn-server._decode_media: 1 <= frames <= 480000
MAX_SAMPLES_D = 10 * SAMPLE_RATE      # part196.select196.clip_sample_limit()
DEFAULT_DATA_DIR = Path("/mnt/e/neuron-probe/part203-data/AudioWAV")
V_PROMPT = "<|audio|>"
V_STORAGE_DTYPE = "float16"
PHASINGS = HERE / "phrasings.json"


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    """Read the Part 203 manifest and fill in the Part 196 fields the shared recorder code needs."""
    rows = read_manifest(path)
    for row in rows:
        row["id"] = row["file"][:-4] if row["file"].lower().endswith(".wav") else row["file"]
        row["speaker_id"] = row["actor"]
        row["style"] = row["emotion"]
        row["text_key"] = hashlib.sha256(row["sentence"].encode("utf-8")).hexdigest()
        row["shard"] = ""
        row["row_index"] = ""
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate recording ids")
    return rows


def decode_audio_v(raw: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Venn's decode: soundfile, mono 16 kHz, at most 30 s; refuses anything Venn refuses."""
    import io

    import soundfile as sf

    with sf.SoundFile(io.BytesIO(raw)) as audio:
        rate = audio.samplerate
        channels = audio.channels
        frames = audio.frames
        if rate != SAMPLE_RATE or channels != 1 or not 1 <= frames <= MAX_SAMPLES_V:
            raise ValueError(f"audio must be mono 16 kHz and at most 30 s (rate={rate}, "
                             f"channels={channels}, frames={frames})")
        samples = audio.read(dtype="float32")
    if samples.ndim != 1 or not 1 <= len(samples) <= MAX_SAMPLES_V or not np.isfinite(samples).all():
        raise ValueError("decoded audio is not a finite mono array within 30 s")
    return np.ascontiguousarray(samples, dtype=np.float32), {"rate": rate, "channels": channels,
                                                             "frames": frames}


def decode_audio_d(raw: bytes) -> np.ndarray:
    """Part 196's decode (record196.decode_audio_bytes), imported so the two cannot drift."""
    from part196.record196 import decode_audio_bytes

    return decode_audio_bytes(raw)


def v_item(samples: np.ndarray) -> dict[str, Any]:
    """The exact item venn-server._decode_media returns for audio."""
    return {"text": V_PROMPT, "audio": [samples]}


def d_item(samples: np.ndarray) -> dict[str, Any]:
    """The exact item part196.record196._record_one passes to model.encode."""
    return {"audio": {"array": samples, "sampling_rate": SAMPLE_RATE}}


def embed_v(model: Any, torch: Any, samples: np.ndarray) -> np.ndarray:
    """Venn's media encode: no prompt_name, normalize_embeddings=True, stored float16."""
    with torch.inference_mode():
        vector = np.asarray(model.encode([v_item(samples)], batch_size=1,
                                         normalize_embeddings=True), dtype=np.float16)
    if vector.shape != (1, EMBEDDING_SIZE) or not np.isfinite(vector).all():
        raise ValueError(f"unexpected V embedding shape {vector.shape}")
    return vector[0]


def embed_d(model: Any, torch: Any, samples: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        vector = np.asarray(model.encode(d_item(samples), normalize_embeddings=True), dtype=np.float32)
    if vector.shape != (EMBEDDING_SIZE,) or not np.isfinite(vector).all():
        raise ValueError(f"unexpected D embedding shape {vector.shape}")
    return vector


def v_vs_d(samples_v: np.ndarray, samples_d: np.ndarray) -> dict[str, Any]:
    """State whether V's input equals D's input: same samples, and D capped at 10 s or not."""
    same_samples = bool(samples_v.shape == samples_d.shape and np.array_equal(samples_v, samples_d))
    return {"same_samples": same_samples, "v_samples": int(samples_v.size),
            "d_samples": int(samples_d.size),
            "v_cap_seconds": MAX_SAMPLES_V // SAMPLE_RATE, "d_cap_seconds": MAX_SAMPLES_D // SAMPLE_RATE,
            "v_item": {"text": V_PROMPT, "audio": "list of float32 samples"},
            "d_item": {"audio": {"array": "float32 samples", "sampling_rate": SAMPLE_RATE}},
            "v_prompt_name": None, "d_prompt_name": None,
            "v_normalize": True, "d_normalize": True}


def _record_one(row: dict[str, str], raw: bytes, record_dir: Path, model: Any, torch: Any,
                layers: Any, buffer: dict[int, Any], text: str) -> dict[str, Any]:
    """Record V, D, E's activations and H for one clip; raise on any invalid output."""
    samples_v, header = decode_audio_v(raw)
    samples_d = decode_audio_d(raw)
    if samples_d.size == 0:
        raise ValueError("ffmpeg returned no samples")
    comparison = v_vs_d(samples_v, samples_d)
    features = acoustic_features(samples_d, text)
    v_vector = embed_v(model, torch, samples_v)
    buffer.clear()
    d_vector = embed_d(model, torch, samples_d)
    if len(buffer) != len(layers):
        raise RuntimeError(f"expected {len(layers)} hooked layers; captured {len(buffer)}")
    activations = torch.stack([buffer[layer] for layer in range(len(layers))])
    seqlen = int(activations.shape[1])
    act_mean = activations.mean(1).reshape(-1).cpu().numpy().astype(np.float16)
    if act_mean.size != N_UNITS:
        raise RuntimeError(f"unexpected activation shape {act_mean.shape}")
    _atomic_npz(record_dir / f"{row['id']}.npz", id=np.asarray(row["id"]), v=v_vector, emb=d_vector,
                act_mean=act_mean, seqlen=np.asarray(seqlen, dtype=np.int32),
                status=np.asarray("ok"),
                duration_seconds=np.asarray(samples_d.size / SAMPLE_RATE, dtype=np.float32),
                v_duration_seconds=np.asarray(samples_v.size / SAMPLE_RATE, dtype=np.float32),
                acoustic=features, audio_sha256=np.asarray(hashlib.sha256(raw).hexdigest()),
                v_same_samples=np.asarray(comparison["same_samples"]))
    return comparison


def _find_wav(data_dir: Path, name: str) -> Path:
    direct = data_dir / name
    if direct.is_file():
        return direct
    matches = sorted(data_dir.glob(f"**/{name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {name} under {data_dir}, found {len(matches)}")
    return matches[0]


def run_recording(*, manifest_path: Path, data_dir: Path, work_dir: Path,
                  model_path: Path | None = None, threads: int = 6,
                  limit: int | None = None) -> dict[str, Any]:
    rows = read_manifest_rows(manifest_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    identity = recording_identity(rows=rows, data_dir=data_dir, model_path=model_path)
    identity["study"] = "part203"
    write_or_check_identity(work_dir / "identity.json", identity)
    record_dir = work_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)

    record188, torch, model, layers, buffer, hooks = load_part188(model_path, threads)
    started = time.perf_counter()
    try:
        gate2 = gate2_parity(record188, torch, model, layers, buffer)
        (work_dir / "gate2.json").write_text(json.dumps(gate2, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
        if gate2["verdict"] != "passed":
            raise RuntimeError("Gate 2 parity failed; no CREMA-D clip was recorded")

        texts = {}
        for row in rows:
            # CREMA-D sentence code -> its transcript, read from the public file list's sibling table.
            texts[row["id"]] = SENTENCES[row["sentence"]]
        encode_text_assets(model, rows, texts, work_dir)

        comparisons = []
        completed = skipped = 0
        for row in rows:
            record_path = record_dir / f"{row['id']}.npz"
            if _cached_status(record_path) == "ok":
                skipped += 1
                continue
            if limit is not None and completed >= limit:
                break
            try:
                raw = _find_wav(data_dir, row["file"]).read_bytes()
                comparisons.append(_record_one(row, raw, record_dir, model, torch, layers, buffer,
                                               texts[row["id"]]))
            except Exception as exc:
                _atomic_npz(record_path, id=np.asarray(row["id"]), status=np.asarray("record_error"),
                            reason=np.asarray(f"{type(exc).__name__}: {exc}"))
            completed += 1
            if completed % 50 == 0:
                print(f"clips_processed={completed}/{len(rows)}", flush=True)
        successful_ids = {row["id"] for row in rows
                          if _cached_status(record_dir / f"{row['id']}.npz") == "ok"}
        statuses = {}
        for row in rows:
            statuses[row["id"]] = _cached_status(record_dir / f"{row['id']}.npz") or "missing"
        coverage = {"planned": len(rows), "ok": len(successful_ids),
                    "errors": {key: value for key, value in statuses.items() if value != "ok"}}
        gate1 = gate1_vd(rows, successful_ids, work_dir)
        _write_json(work_dir / "gate1.json", gate1)
        if gate1["verdict"] != "passed":
            raise RuntimeError("Gate 1 failed on successfully recorded clips")
        return {"gate1": gate1["verdict"], "gate2": gate2["verdict"], "planned": len(rows),
                "recorded_this_run": completed, "resumed_ok": skipped,
                "successfully_recorded": len(successful_ids), "coverage": coverage,
                "v_vs_d": gate1["v_vs_d"],
                "elapsed_seconds": time.perf_counter() - started, "work_dir": str(work_dir)}
    finally:
        for hook in hooks:
            hook.remove()


def gate1_vd(rows: list[dict[str, str]], successful_ids: set[str], work_dir: Path) -> dict[str, Any]:
    """Gate 1: every planned clip recorded, and V's input equals D's input on every recorded clip."""
    from part203.select203 import EMOTIONS as _EMOTIONS  # noqa: F401

    planned = {row["id"] for row in rows}
    missing = sorted(planned - successful_ids)
    mismatched = []
    for clip_id in sorted(successful_ids):
        path = work_dir / "records" / f"{clip_id}.npz"
        with np.load(path, allow_pickle=False) as saved:
            if "v_same_samples" in saved and not bool(saved["v_same_samples"]):
                mismatched.append(clip_id)
    passed = bool(successful_ids) and not missing and not mismatched
    return {"verdict": "passed" if passed else "failed", "planned": len(planned),
            "successfully_recorded": len(successful_ids), "missing": missing[:50],
            "missing_count": len(missing), "v_d_input_mismatch": mismatched[:50],
            "v_d_input_mismatch_count": len(mismatched),
            "v_vs_d": {"v_decode": "soundfile, mono 16 kHz, at most 30 s",
                       "d_decode": "ffmpeg mono 16 kHz f32le, capped at 10 s (part196)",
                       "v_item": v_item(np.zeros(1, dtype=np.float32)),
                       "d_item": d_item(np.zeros(1, dtype=np.float32)),
                       "v_prompt_name": None, "d_prompt_name": None,
                       "v_normalize": True, "d_normalize": True,
                       "v_storage": V_STORAGE_DTYPE}}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "They have gone to the zoo.",
    "IWW": "I think I will buy the red car.",
    "TAI": "The apple is on the table.",
    "MTI": "My team is playing well.",
    "TSI": "The sky is blue.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I'm on my way to the meeting.",
    "IWL": "I would like a cup of coffee.",
    "IOM": "I'm going to the market.",
    "WSI": "We should invite them.",
    "ITH": "I think he is right.",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest203.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/e/neuron-probe/part203-work"))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="record at most this many new clips")
    args = parser.parse_args(argv)
    try:
        result = run_recording(manifest_path=args.manifest, data_dir=args.data_dir,
                               work_dir=args.work_dir, model_path=args.model_path,
                               threads=args.threads, limit=args.limit)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
