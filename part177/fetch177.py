#!/usr/bin/env python3
"""Fetch Part 177's pinned inputs without overwriting changed data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = ROOT / "results" / "part177"
INPUTS = "inputs"
LOG_REPO = "RikiyaT/lm_understand-logs"
LOG_REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
DATASET_REPO = "HuggingFaceH4/MATH-500"
TOKENIZER_REPO = "Qwen/Qwen3.5-0.8B"
TOKENIZER_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def _ssl_context():
    cert_file = os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=cert_file if cert_file else None)


def _read_url(url, context=None):
    request = urllib.request.Request(url, headers={"User-Agent": "part177-fetch/1.0"})
    with urllib.request.urlopen(request, context=context or _ssl_context(), timeout=60) as response:
        return response.read()


def _json_url(url):
    return json.loads(_read_url(url).decode("utf-8"))


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _write_manifest(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _dataset_revision_and_file():
    api = f"https://huggingface.co/api/datasets/{DATASET_REPO}/revision/main"
    info = _json_url(api)
    revision = info.get("sha")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("Hugging Face dataset revision response had no commit SHA")
    siblings = info.get("siblings")
    if not isinstance(siblings, list):
        tree_url = f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/{revision}?recursive=true&expand=false"
        siblings = _json_url(tree_url)
    names = [item.get("rfilename", item.get("path", "")) for item in siblings if isinstance(item, dict)]
    candidates = [name for name in names if "test" in Path(name).name.casefold() and
                  Path(name).suffix.casefold() in {".parquet", ".jsonl", ".json"}]
    parquet = sorted(name for name in candidates if name.casefold().endswith(".parquet"))
    selected = parquet or sorted(candidates)
    if not selected:
        raise RuntimeError(f"No test parquet or JSONL file found in {DATASET_REPO}@{revision}")
    if len(selected) != 1:
        raise RuntimeError(f"Expected one test file in {DATASET_REPO}@{revision}; found {selected}")
    return revision, selected[0]


def _download(url, destination, revision, manifest_path, manifest):
    rel = destination.relative_to(manifest_path.parent).as_posix()
    previous = manifest.get("files", {}).get(rel)
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if not isinstance(previous, dict):
            raise FileExistsError(f"refusing to overwrite unmanifested input {destination}")
        if actual != previous.get("sha256"):
            raise ValueError(f"refusing to overwrite changed input {destination}: {actual} != {previous.get('sha256')}")
        if previous.get("url") == url and previous.get("revision") == revision:
            return previous
        raise ValueError(f"refusing to replace already-pinned input {destination} from revision "
                         f"{previous.get('revision')} with {revision}; use a fresh output directory")
    payload = _read_url(url)
    record = {"url": url, "revision": revision, "sha256": _sha256(payload), "bytes": len(payload)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    manifest.setdefault("files", {})[rel] = record
    _write_manifest(manifest_path, manifest)
    return record


def fetch(out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir).resolve()
    inputs = out_dir / INPUTS
    manifest_path = out_dir / "inputs-manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "part": 177, "files": {}, "revisions": {}}

    log_base = f"https://raw.githubusercontent.com/{LOG_REPO}/{LOG_REVISION}/exp8_math/eval"
    for arm in ("unedited", "p50c"):
        merged_rel = f"eval/{arm}/merged.json"
        _download(f"{log_base}/{arm}/merged.json", inputs / merged_rel,
                  LOG_REVISION, manifest_path, manifest)
        for shard in range(10):
            shard_name = f"math500_s{shard}of10"
            for filename in ("math500_per_problem.json", "math500_first_samples.jsonl.gz"):
                rel = f"eval/{arm}/parts/{shard_name}/{filename}"
                _download(f"{log_base}/{arm}/parts/{shard_name}/{filename}", inputs / rel,
                          LOG_REVISION, manifest_path, manifest)

    data_revision, data_file = _dataset_revision_and_file()
    data_url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{data_revision}/{data_file}"
    dataset_suffix = Path(data_file).suffix
    dataset_rel = f"math500-test{dataset_suffix}"
    _download(data_url, inputs / dataset_rel, data_revision, manifest_path, manifest)

    tokenizer_url = (f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/tokenizer.json")
    _download(tokenizer_url, inputs / "tokenizer.json", TOKENIZER_REVISION, manifest_path, manifest)
    manifest["revisions"] = {"upstream_logs": LOG_REVISION,
                             "math500_dataset": {"repo": DATASET_REPO, "split": "test",
                                                 "revision": data_revision, "file": data_file},
                             "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION}}
    _write_manifest(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    value = fetch(args.out_dir)
    print(json.dumps({"files": len(value.get("files", {})), "manifest": str(args.out_dir / "inputs-manifest.json")},
                     ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
