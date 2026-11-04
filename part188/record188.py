#!/usr/bin/env python3
"""Download and record Part 188 audio clips with resumable per-clip state."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
ESC_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
ESC_CSV_SHA256 = "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246"
ESC_ARCHIVE_URL = f"https://github.com/karolpiczak/ESC-50/archive/{ESC_COMMIT}.zip"
VIS_LOCAL_MANIFEST = Path("/home/elisepc/green/Journal/notes/reports/embeddinggemma2-chicken-neurons-2026-11-03/vis_manifest.csv")
VIS_LOCAL_AUDIO = Path("/mnt/e/neuron-probe/vocalimitationset/audio")
ESC_LOCAL_AUDIO = Path("/mnt/e/neuron-probe/esc50/audio")
N_UNITS = 24 * 2048
THROUGHPUT_INTERVAL = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_gate1(activation: np.ndarray, embedding: np.ndarray, reference_activation: np.ndarray,
                  reference_embedding: np.ndarray, *, atol: float = 2e-3,
                  min_fraction: float = 0.999) -> dict[str, float | bool]:
    """Apply the registered float16 activation and embedding-cosine checks."""
    actual = np.asarray(activation, dtype=np.float16).astype(np.float32)
    expected = np.asarray(reference_activation, dtype=np.float16).astype(np.float32)
    differences = np.abs(actual - expected)
    fraction = float(np.mean(differences <= atol))
    left = np.asarray(embedding, dtype=np.float64).reshape(-1)
    right = np.asarray(reference_embedding, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(np.dot(left, right) / denominator) if denominator else 0.0
    return {"activation_fraction_within_tolerance": fraction,
            "activation_max_absolute_difference": float(differences.max(initial=0.0)),
            "embedding_cosine": cosine,
            "passed": bool(fraction >= min_fraction and cosine >= 0.9999)}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def local_vis_map(manifest_path: Path) -> dict[str, str]:
    """Resolve repository paths through the earlier study's local-name manifest."""
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        return {row["path"]: row["local"] for row in csv.DictReader(stream)
                if row.get("path") and row.get("local")}


def resolve_local_audio(row: dict[str, str], vis_map: dict[str, str], *,
                        vis_audio: Path = VIS_LOCAL_AUDIO,
                        esc_audio: Path = ESC_LOCAL_AUDIO) -> Path | None:
    if row["source"] == "esc":
        candidate = esc_audio / row["filename"]
        return candidate if candidate.is_file() else None
    repo_path = row["repository_path"]
    candidates = []
    if repo_path in vis_map:
        candidates.append(vis_audio / vis_map[repo_path])
    candidates.append(vis_audio / Path(repo_path).name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _retry_download(operation: Callable[[], Any], *, retries: int = 3,
                    backoff_seconds: float = 1.0,
                    sleep: Callable[[float], None] = time.sleep) -> Any:
    """Retry a download up to three times after its initial attempt."""
    if retries < 0:
        raise ValueError("download retries must be non-negative")
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception:
            if attempt == retries:
                raise
            sleep(backoff_seconds * (2 ** attempt))
    raise AssertionError("unreachable download retry state")


def _curl_download(url: str, destination: Path, *, rate_limit: bool = False,
                   max_time: int = 300) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".part")
    args = ["curl", "-sS", "--noproxy", "*", "--fail", "--location",
            "--max-time", str(max_time), "--output", str(temp)]
    if rate_limit:
        args.extend(["--limit-rate", "500k"])
    args.append(url)
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=max_time + 30)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"curl exited {result.returncode}")
        if not temp.is_file() or temp.stat().st_size == 0:
            raise RuntimeError("download produced an empty file")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def download_vis_clip(row: dict[str, str], audio_dir: Path) -> Path:
    target = audio_dir / f"{row['id']}.wav"
    if target.is_file() and target.stat().st_size:
        return target
    _retry_download(lambda: _curl_download(row["download_url"], target,
                                           rate_limit=True, max_time=300))
    return target


def download_esc_archive(work_dir: Path) -> Path:
    archive = work_dir / "downloads" / f"esc50-{ESC_COMMIT}.zip"
    if archive.is_file() and archive.stat().st_size > 1_000_000:
        try:
            _verify_esc_archive_metadata(archive)
            return archive
        except (OSError, KeyError, zipfile.BadZipFile, RuntimeError):
            archive.unlink(missing_ok=True)
    def download_and_verify() -> Path:
        _curl_download(ESC_ARCHIVE_URL, archive, max_time=1800)
        with zipfile.ZipFile(archive) as check:
            bad = check.testzip()
            if bad:
                raise RuntimeError(f"ESC-50 archive failed CRC check at {bad}")
        _verify_esc_archive_metadata(archive)
        return archive

    try:
        return _retry_download(download_and_verify)
    except Exception:
        archive.unlink(missing_ok=True)
        raise


def _verify_esc_archive_metadata(archive: Path) -> None:
    """Refuse a pinned archive whose class table differs from the local registered data."""
    with zipfile.ZipFile(archive) as zf:
        member = next((name for name in zf.namelist() if name.endswith("/meta/esc50.csv")), None)
        if member is None:
            raise RuntimeError("pinned ESC-50 archive has no meta/esc50.csv")
        found = hashlib.sha256(zf.read(member)).hexdigest()
    if found != ESC_CSV_SHA256:
        raise RuntimeError(f"ESC-50 metadata checksum {found} differs from the registered local CSV "
                           f"{ESC_CSV_SHA256}; refuse to record with this archive")


def extract_esc_clip(archive: Path, filename: str, audio_dir: Path) -> Path:
    target = audio_dir / filename
    if target.is_file() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        suffix = f"/audio/{filename}"
        member = next((name for name in zf.namelist() if name.endswith(suffix)), None)
        if member is None:
            raise FileNotFoundError(f"{filename} is absent from pinned ESC-50 archive")
        with zf.open(member) as source, target.with_name(target.name + ".part").open("wb") as output:
            shutil.copyfileobj(source, output)
    os.replace(target.with_name(target.name + ".part"), target)
    return target


def decode_audio(path: Path) -> np.ndarray:
    """Match the earlier study: mono, 16 kHz, f32le using imageio-ffmpeg's binary."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
                           "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                          capture_output=True, timeout=60, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()[:80_000]


def load_model(model_path: Path | None):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    common = {"device": "cpu", "config_kwargs": {"vision_config": None},
              "model_kwargs": {"dtype": torch.float32}}
    if model_path:
        model = SentenceTransformer(str(model_path), **common)
    else:
        model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, **common)
    return torch, model


def install_hooks(model: Any):
    layers = model[0].model.language_model.layers
    buffer: dict[int, Any] = {}
    hooks = [layers[i].mlp.down_proj.register_forward_pre_hook(
        lambda _module, args, i=i: buffer.__setitem__(i, args[0][0].float()))
        for i in range(len(layers))]
    if len(layers) != 24:
        for hook in hooks:
            hook.remove()
        raise ValueError(f"expected 24 language-model layers; found {len(layers)}")
    return layers, buffer, hooks


def encode_prompts(model: Any, classes_path: Path, output: Path) -> None:
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    phrases = list(dict.fromkeys(item["phrase"] for item in classes["classes"] if item.get("phrase")))
    phrases.append(classes["class_free_phrase"])
    real_texts = [f"the sound of {phrase}" for phrase in phrases]
    imitation_texts = [f"a person imitating the sound of {phrase}" for phrase in phrases]
    real = np.asarray(model.encode(real_texts, prompt_name="SearchQuery", normalize_embeddings=True),
                      dtype=np.float32)
    imitation = np.asarray(model.encode(imitation_texts, prompt_name="SearchQuery", normalize_embeddings=True),
                           dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, phrases=np.asarray(phrases), real_texts=np.asarray(real_texts),
                        imitation_texts=np.asarray(imitation_texts), real_embeddings=real,
                        imitation_embeddings=imitation)


def _save_record(path: Path, *, row: dict[str, str], status: str, sha: str,
                 activation: np.ndarray | None = None, embedding: np.ndarray | None = None,
                 seqlen: int = 0, error: str = "", decode_status: str | None = None) -> None:
    activation = (np.asarray(activation, dtype=np.float16) if activation is not None
                  else np.full(N_UNITS, np.nan, dtype=np.float16))
    embedding = (np.asarray(embedding, dtype=np.float32) if embedding is not None
                 else np.full(768, np.nan, dtype=np.float32))
    if decode_status is None:
        # A record_error can still mean audio decoded successfully. Gate 2 uses
        # the separate record status and counts only the status "ok" as recorded.
        decode_status = "ok" if status in ("ok", "record_error") else (
            "not_downloaded" if status == "download_error" else status)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, id=np.asarray(row["id"]), filename=np.asarray(row["filename"]),
                            source=np.asarray(row["source"]), act_mean=activation, emb=embedding,
                            seqlen=np.asarray(seqlen, dtype=np.int32), status=np.asarray(status),
                            decode_status=np.asarray(decode_status), sha256=np.asarray(sha),
                            error=np.asarray(error))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _append_done(path: Path, row_id: str, status: str, wall_seconds: float = 0.0) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": row_id, "status": status,
                                "wall_seconds": round(float(wall_seconds), 6)},
                               sort_keys=True) + "\n")


def _record_identity_mismatch(saved: Any, row: dict[str, str]) -> str | None:
    """Return why a stored record does not belong to this manifest row."""
    if "id" not in saved or str(saved["id"].item()) != row["id"]:
        return "stored id does not match manifest"
    for field in ("filename", "repository_path"):
        if field in saved and str(saved[field].item()) != row.get(field, ""):
            return f"stored {field} does not match manifest"
    return None


def _read_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    found: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = _parse_done_line(line)
        if row is not None:
            found[row["id"]] = row
    return found


def _parse_done_line(line: str) -> dict[str, Any] | None:
    try:
        row = json.loads(line)
        return row if "id" in row else None
    except (ValueError, TypeError):
        return None


def _download_pending_rows(rows: list[dict[str, str]], record_dir: Path) -> list[dict[str, str]]:
    """A restart re-downloads missing, mismatched, or download-error clips."""
    pending = []
    for row in rows:
        path = record_dir / f"{row['id']}.npz"
        if not path.exists():
            pending.append(row)
            continue
        try:
            with np.load(path, allow_pickle=False) as saved:
                mismatch = _record_identity_mismatch(saved, row)
                status = str(saved["status"].item())
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pending.append(row)
            continue
        if mismatch or status == "download_error":
            pending.append(row)
    return pending


def _gate1_verdict(parity_rows: list[dict[str, str]], gate_rows: list[dict],
                   reference_filenames: list[str]) -> bool:
    """Require all eight registered parity records to have a completed comparison."""
    comparison_fields = {"activation_fraction_within_tolerance",
                         "activation_max_absolute_difference", "embedding_cosine"}
    expected_ids = [row["id"] for row in parity_rows]
    expected_filenames = [row["filename"] for row in parity_rows]
    if (len(parity_rows) != 8 or len(set(expected_ids)) != 8 or
            len(reference_filenames) != 8 or len(set(reference_filenames)) != 8 or
            set(expected_filenames) != set(reference_filenames) or
            len(gate_rows) != 8 or [row.get("id") for row in gate_rows] != expected_ids or
            [row.get("filename") for row in gate_rows] != expected_filenames):
        return False
    return all(row.get("passed") is True and comparison_fields <= row.keys()
               for row in gate_rows)


def _should_log_throughput(completed: int) -> bool:
    return completed > 0 and completed % THROUGHPUT_INTERVAL == 0


def _throughput_line(completed: int, planned: int, elapsed_seconds: float) -> str:
    elapsed_minutes = max(float(elapsed_seconds) / 60.0, 1e-9)
    return (f"clips_processed={completed}/{planned} "
            f"clips_per_minute={completed / elapsed_minutes:.2f}")


def _process_one(row: dict[str, str], source_path: Path, record_dir: Path,
                 model: Any, torch: Any, layer_count: int) -> tuple[str, str, str, np.ndarray | None, np.ndarray | None, int, str]:
    record_path = record_dir / f"{row['id']}.npz"
    digest = sha256_file(source_path)
    buffer: dict[int, Any] = model._part188_buffer
    status = "decode_error"
    message = "decode did not complete"
    for attempt in range(2):
        buffer.clear()
        try:
            audio = decode_audio(source_path)
            if audio.size == 0:
                raise ValueError("ffmpeg returned no samples")
        except Exception as exc:
            status = "decode_error"
            message = f"{type(exc).__name__}: {exc}"
        else:
            item = {"audio": {"array": audio, "sampling_rate": 16000}}
            try:
                with torch.inference_mode():
                    embedding = np.asarray(model.encode(item, normalize_embeddings=True), dtype=np.float32)
                if len(buffer) != layer_count:
                    raise RuntimeError(f"expected {layer_count} hooked layers; captured {len(buffer)}")
                import torch as torch_module
                layers = torch_module.stack([buffer[i] for i in range(layer_count)])
                seqlen = int(layers.shape[1])
                activation = layers.mean(1).reshape(-1).cpu().numpy().astype(np.float16)
                if activation.size != N_UNITS or embedding.size != 768:
                    raise RuntimeError(f"unexpected output shapes {activation.shape} / {embedding.shape}")
                _save_record(record_path, row=row, status="ok", sha=digest,
                             activation=activation, embedding=embedding, seqlen=seqlen)
                return row["id"], "ok", digest, activation, embedding, seqlen, ""
            except Exception as exc:
                status = "record_error"
                message = f"{type(exc).__name__}: {exc}"
        if attempt == 0:
            continue

    _save_record(record_path, row=row, status=status, sha=digest, error=message)
    return row["id"], status, digest, None, None, 0, message


def _manifest_sha(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_recording(*, manifest_path: Path, work_dir: Path, model_path: Path | None = None,
                  local: bool = False, limit: int | None = None,
                  vis_audio: Path = VIS_LOCAL_AUDIO, esc_audio: Path = ESC_LOCAL_AUDIO,
                  local_manifest: Path = VIS_LOCAL_MANIFEST) -> dict:
    rows = read_manifest(manifest_path)
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")
    work_dir.mkdir(parents=True, exist_ok=True)
    record_dir = work_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    identity = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "model_path": str(model_path) if model_path else "hub",
                "manifest_sha256": _manifest_sha(rows), "local": bool(local)}
    identity_path = work_dir / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError("recording work directory identity differs; use a fresh work directory")
    else:
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    p_rows = [row for row in rows if row["set"] == "P"]
    other_rows = [row for row in rows if row["set"] != "P"]
    if limit is not None:
        other_rows = other_rows[:limit]
    ordered = p_rows + other_rows
    started = time.perf_counter()
    retry_rows = _download_pending_rows(ordered, record_dir)
    vis_map = local_vis_map(local_manifest) if local else {}
    audio_dir = work_dir / "audio"
    esc_audio_dir = audio_dir / "esc50"
    vis_audio_dir = audio_dir / "vocalimitationset"
    source_paths: dict[str, Path] = {}
    errors: dict[str, str] = {}
    download_wall: dict[str, float] = {}
    esc_archive = None
    retry_esc_rows = [row for row in retry_rows if row["source"] == "esc"]
    if not local and retry_esc_rows:
        archive_started = time.perf_counter()
        esc_archive = download_esc_archive(work_dir)
        archive_seconds = time.perf_counter() - archive_started
        per_clip_archive_seconds = archive_seconds / len(retry_esc_rows)
        for row in retry_esc_rows:
            download_wall[row["id"]] = per_clip_archive_seconds

    # Download VIS files in parallel; each transfer is rate-limited to 500 kB/s.
    vis_rows = [row for row in retry_rows if row["source"] == "vis"]
    if not local and vis_rows:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(download_vis_clip, row, vis_audio_dir): row for row in vis_rows}
            submitted_at = {row["id"]: time.perf_counter() for row in vis_rows}
            for future in as_completed(futures):
                row = futures[future]
                download_wall[row["id"]] = time.perf_counter() - submitted_at[row["id"]]
                try:
                    source_paths[row["id"]] = future.result()
                except Exception as exc:
                    errors[row["id"]] = f"download_error: {type(exc).__name__}: {exc}"
    for row in retry_rows:
        if row["id"] in source_paths or row["id"] in errors:
            continue
        local_started = time.perf_counter()
        try:
            if local:
                path = resolve_local_audio(row, vis_map, vis_audio=vis_audio, esc_audio=esc_audio)
                if path is None:
                    raise FileNotFoundError("local audio file not found")
            elif row["source"] == "esc":
                assert esc_archive is not None
                path = extract_esc_clip(esc_archive, row["filename"], esc_audio_dir)
            else:
                path = vis_audio_dir / f"{row['id']}.wav"
            source_paths[row["id"]] = path
        except Exception as exc:
            errors[row["id"]] = f"download_error: {type(exc).__name__}: {exc}"
        finally:
            download_wall[row["id"]] = download_wall.get(row["id"], 0.0) + (
                time.perf_counter() - local_started)

    torch, model = load_model(model_path)
    layers, buffer, hooks = install_hooks(model)
    model._part188_buffer = buffer
    try:
        done_path = work_dir / "done.jsonl"
        done_rows = _read_done(done_path)
        ref_path = HERE / "parity_ref.npz"
        parity_ref = np.load(ref_path, allow_pickle=False)
        ref_index = {str(name): i for i, name in enumerate(parity_ref["filenames"])}
        gate_rows: list[dict] = []
        gate_checked = False
        progress_count = 0

        def note_clip_completed() -> None:
            nonlocal progress_count
            progress_count += 1
            if _should_log_throughput(progress_count):
                print(_throughput_line(progress_count, len(retry_rows),
                                       time.perf_counter() - started), flush=True)

        for row in ordered:
            if row["set"] != "P" and not gate_checked:
                gate = {"verdict": "passed" if _gate1_verdict(
                            p_rows, gate_rows, [str(name) for name in parity_ref["filenames"]]) else "failed",
                        "clips": gate_rows,
                        "rules": {"activation_abs_tolerance": 0.002,
                                  "fraction_of_units_required": 0.999,
                                  "embedding_cosine_required": 0.9999}}
                (work_dir / "gate1.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n",
                                                      encoding="utf-8")
                if gate["verdict"] != "passed":
                    raise RuntimeError("Gate 1 failed; stopping before any non-parity clips are recorded")
                encode_prompts(model, HERE / "classes.json", work_dir / "prompts.npz")
                gate_checked = True
            record_path = record_dir / f"{row['id']}.npz"
            if record_path.exists():
                try:
                    with np.load(record_path, allow_pickle=False) as saved:
                        mismatch = _record_identity_mismatch(saved, row)
                        if mismatch:
                            print(f"resume_mismatch id={row['id']} {mismatch}; re-recording", flush=True)
                        else:
                            saved_status = str(saved["status"].item())
                            if saved_status != "download_error":
                                if (row["id"] not in done_rows or
                                        "wall_seconds" not in done_rows[row["id"]]):
                                    _append_done(done_path, row["id"], saved_status, wall_seconds=0.0)
                                    done_rows[row["id"]] = {"id": row["id"], "status": saved_status,
                                                             "wall_seconds": 0.0}
                                if row["set"] == "P":
                                    if saved_status == "ok":
                                        gate_rows.append(_gate_row(row, saved, parity_ref, ref_index))
                                    else:
                                        gate_rows.append({"id": row["id"], "filename": row["filename"],
                                                          "passed": False, "reason": saved_status})
                                continue
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                    print(f"resume_mismatch id={row['id']} unreadable record "
                          f"({type(exc).__name__}); re-recording", flush=True)
            clip_started = time.perf_counter()
            if row["id"] in errors:
                message = errors[row["id"]]
                _save_record(record_path, row=row, status="download_error", sha="", error=message)
                wall_seconds = download_wall.get(row["id"], 0.0) + (time.perf_counter() - clip_started)
                _append_done(done_path, row["id"], "download_error", wall_seconds=wall_seconds)
                done_rows[row["id"]] = {"id": row["id"], "status": "download_error",
                                          "wall_seconds": wall_seconds}
                if row["set"] == "P":
                    gate_rows.append({"id": row["id"], "filename": row["filename"],
                                      "passed": False, "reason": message})
                note_clip_completed()
                continue
            result = _process_one(row, source_paths[row["id"]], record_dir, model, torch, len(layers))
            row_id, status, _sha, activation, embedding, _seqlen, error = result
            wall_seconds = download_wall.get(row_id, 0.0) + (time.perf_counter() - clip_started)
            _append_done(done_path, row_id, status, wall_seconds=wall_seconds)
            done_rows[row_id] = {"id": row_id, "status": status, "wall_seconds": wall_seconds}
            if row["set"] == "P":
                with np.load(record_path, allow_pickle=False) as saved:
                    gate_rows.append(_gate_row(row, saved, parity_ref, ref_index)
                                     if status == "ok" else {"id": row["id"], "filename": row["filename"],
                                                             "passed": False, "reason": error or status})
            note_clip_completed()

        reference_filenames = [str(name) for name in parity_ref["filenames"]]
        if not _gate1_verdict(p_rows, gate_rows, reference_filenames):
            gate = {"verdict": "failed", "clips": gate_rows,
                    "reason": "Gate 1 requires exactly eight recorded and compared parity clips"}
            (work_dir / "gate1.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
            raise RuntimeError("Gate 1 failed; stopping before non-parity clips are recorded")
        gate = {"verdict": "passed" if _gate1_verdict(p_rows, gate_rows, reference_filenames) else "failed",
                "clips": gate_rows,
                "rules": {"activation_abs_tolerance": 0.002,
                          "fraction_of_units_required": 0.999,
                          "embedding_cosine_required": 0.9999}}
        (work_dir / "gate1.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        if gate["verdict"] != "passed":
            raise RuntimeError("Gate 1 failed; stopping before analysis")
        if not gate_checked:
            encode_prompts(model, HERE / "classes.json", work_dir / "prompts.npz")
        statuses = Counter_statuses(record_dir, ordered)
        return {"gate1": gate["verdict"], "planned_in_this_run": len(ordered),
                "record_status_counts": statuses,
                "elapsed_seconds": time.perf_counter() - started,
                "done_file": str(done_path)}
    finally:
        for hook in hooks:
            hook.remove()


def _gate_row(row: dict[str, str], saved: Any, reference: Any,
              reference_index: dict[str, int]) -> dict:
    filename = row["filename"]
    if filename not in reference_index:
        return {"id": row["id"], "filename": filename, "passed": False,
                "reason": "reference row missing"}
    i = reference_index[filename]
    result = compare_gate1(saved["act_mean"], saved["emb"], reference["act_mean"][i], reference["emb"][i])
    return {"id": row["id"], "filename": filename, **result}


def Counter_statuses(record_dir: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    from collections import Counter

    counts = Counter()
    for row in rows:
        path = record_dir / f"{row['id']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                counts[str(saved["status"].item())] += 1
    return dict(sorted(counts.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest188.csv")
    parser.add_argument("--work-dir", type=Path, default=HERE / "work")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="local snapshot; defaults to the registered Hub revision")
    parser.add_argument("--local", action="store_true",
                        help="read ESC-50 and VocalImitationSet from the local audio folders")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N non-parity clips after Gate 1")
    parser.add_argument("--vis-audio", type=Path, default=VIS_LOCAL_AUDIO)
    parser.add_argument("--esc-audio", type=Path, default=ESC_LOCAL_AUDIO)
    parser.add_argument("--vis-local-manifest", type=Path, default=VIS_LOCAL_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_recording(manifest_path=args.manifest, work_dir=args.work_dir,
                               model_path=args.model_path, local=args.local, limit=args.limit,
                               vis_audio=args.vis_audio, esc_audio=args.esc_audio,
                               local_manifest=args.vis_local_manifest)
    except Exception as exc:
        print(json.dumps({"work_dir": str(args.work_dir),
                          "gate1": "failed" if "Gate 1 failed" in str(exc) else "unknown"},
                         sort_keys=True), file=sys.stderr, flush=True)
        return 2 if "Gate 1 failed" in str(exc) else 1
    print(json.dumps({"gate1": result["gate1"], "planned": result["planned_in_this_run"],
                      "record_status_counts": result["record_status_counts"],
                      "done_file": result["done_file"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
