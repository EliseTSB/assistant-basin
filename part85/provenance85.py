#!/usr/bin/env python3
"""Pin Part 85 replay inputs to committed Git blobs and Hugging Face metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
KEYS = ("pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T", "pretrain_4T",
        "midtrain_13B", "midtrain_26B", "midtrain_51B", "sft", "dpo", "instruct")
DEFAULT_BASE = "allenai/OLMo-2-0425-1B"

sys.path.insert(0, str(HERE))
from download85 import expected_hash, git_blob_sha1, sha256_file  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_rel(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"unsafe repository path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe repository path {value!r}")
    return path


def git(repo_root: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(repo_root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout


def committed_blob(repo_root: Path, commit: str, path: str) -> bytes:
    safe_rel(path)
    return git(repo_root, "show", f"{commit}:{path}")


def checkpoint_table(design: bytes) -> tuple[dict, dict]:
    text = design.decode("utf-8")
    rows = re.findall(r"^\| [^|]+\| `([^`]+)`, ([0-9a-f]{12}) \|", text, re.M)
    if len(rows) != len(KEYS):
        raise ValueError(f"expected {len(KEYS)} checkpoint rows in committed design, found {len(rows)}")
    repos = {key: DEFAULT_BASE for key in KEYS}
    repos.update({"sft": DEFAULT_BASE + "-SFT", "dpo": DEFAULT_BASE + "-DPO",
                  "instruct": DEFAULT_BASE + "-Instruct"})
    return ({key: prefix for key, (_, prefix) in zip(KEYS, rows)}, repos)


def _files_from_committed_refs(repo_root: Path, commit: str, refs_bytes: bytes) -> list[dict]:
    manifest = json.loads(refs_bytes.decode("utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("committed references.json has no files list")
    out, seen = [], set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("committed references.json has a malformed file entry")
        path = safe_rel(entry.get("path"))
        rel = path.as_posix()
        digest = entry.get("sha256")
        if rel in seen or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"committed references.json has a duplicate path or invalid hash: {rel}")
        data = committed_blob(repo_root, commit, rel)
        actual = sha256_bytes(data)
        if actual != digest:
            raise ValueError(f"committed reference {rel} has sha256 {actual}, manifest pins {digest}")
        seen.add(rel)
        out.append({"path": rel, "sha256": actual})
    return sorted(out, key=lambda item: item["path"])


def _snapshot_files(snapshot: Path, repo_id: str, model_commit: str, api) -> list[dict]:
    info = api.model_info(repo_id, revision=model_commit, files_metadata=True)
    if getattr(info, "sha", None) != model_commit:
        raise ValueError(f"Hugging Face metadata is for {getattr(info, 'sha', None)!r}, not {model_commit}")
    metadata = {}
    for sibling in info.siblings:
        name = safe_rel(getattr(sibling, "rfilename", None)).as_posix()
        if name in metadata:
            raise ValueError(f"Hugging Face metadata repeats snapshot file {name}")
        algorithm, digest, size = expected_hash(sibling)
        metadata[name] = (algorithm, digest, size)

    actual = {p.relative_to(snapshot).as_posix(): p for p in snapshot.rglob("*") if p.is_file()}
    required = set(metadata) - {".gitattributes", "README.md"}
    if not required.issubset(actual) or not set(actual).issubset(metadata):
        missing, extra = sorted(required - set(actual)), sorted(set(actual) - set(metadata))
        raise ValueError(f"snapshot file listing differs from Hub metadata (missing {missing[:5]}, extra {extra[:5]})")
    files = []
    for name, path in sorted(actual.items()):
        if name not in metadata:
            raise ValueError(f"snapshot file {name} is absent from Hub metadata")
        algorithm, digest, size = metadata[name]
        if path.stat().st_size != size:
            raise ValueError(f"snapshot file {name} is {path.stat().st_size} bytes, Hub lists {size}")
        actual_sha256 = sha256_file(path)
        actual_hub_hash = actual_sha256 if algorithm == "sha256" else git_blob_sha1(path)
        if actual_hub_hash != digest:
            raise ValueError(f"snapshot file {name} {algorithm} {actual_hub_hash} differs from Hub {digest}")
        files.append({"path": name, "sha256": actual_sha256, "hub_algorithm": algorithm,
                      "hub_digest": digest, "size": size})
    return files


def build_record(*, repo_root: Path, commit: str, checkpoint: str, snapshot: Path,
                 step1_dir: Path, audit_log: Path, commits_file: Path, api=None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be one full 40-hex Git commit")
    resolved = git(repo_root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode("ascii").strip()
    if resolved != commit:
        raise ValueError("--commit must name the full commit id")
    design_bytes = committed_blob(repo_root, commit, "PART85-DESIGN.md")
    refs_bytes = committed_blob(repo_root, commit, "part85/references.json")
    prefixes, repos = checkpoint_table(design_bytes)
    if checkpoint not in prefixes:
        raise ValueError(f"unknown checkpoint {checkpoint!r}")
    local_commits = json.loads(commits_file.read_text(encoding="utf-8"))
    model_commit = local_commits.get(checkpoint)
    if not isinstance(model_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", model_commit):
        raise ValueError(f"commits.local.json has no full commit for {checkpoint}")
    if not model_commit.startswith(prefixes[checkpoint]):
        raise ValueError(f"{checkpoint} commit {model_commit} differs from design prefix {prefixes[checkpoint]}")
    if snapshot.name != model_commit:
        try:
            snap_commit = json.loads((snapshot / "config.json").read_text(encoding="utf-8")).get("_commit_hash")
        except (OSError, json.JSONDecodeError):
            snap_commit = None
        if snap_commit != model_commit:
            raise ValueError(f"snapshot {snapshot} does not identify pinned commit {model_commit}")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    snapshot_files = _snapshot_files(snapshot, repos[checkpoint], model_commit, api)

    refs_files = _files_from_committed_refs(repo_root, commit, refs_bytes)
    expected_audit = f"AUDIT85 {checkpoint} step1: PASS"
    lines = audit_log.read_text(encoding="utf-8").splitlines()
    if lines.count(expected_audit) != 1:
        raise ValueError(f"audit log must contain exactly one {expected_audit!r} line")
    step1_observe = step1_dir / "observe" / "step1.jsonl"
    step1_json = step1_dir / "step1.json"
    for path in (step1_observe, step1_json):
        if not path.is_file():
            raise FileNotFoundError(f"missing audited step-1 file {path}")
    return {
        "schema_version": 1,
        "code_commit": commit,
        "checkpoint": checkpoint,
        "repo_id": repos[checkpoint],
        "model_commit": model_commit,
        "design": {"path": "PART85-DESIGN.md", "sha256": sha256_bytes(design_bytes)},
        "references": {"path": "part85/references.json", "sha256": sha256_bytes(refs_bytes), "files": refs_files},
        "snapshot": {"repo_id": repos[checkpoint], "commit": model_commit, "files": snapshot_files},
        "step1": {"step1_jsonl_sha256": sha256_file(step1_observe), "step1_json_sha256": sha256_file(step1_json),
                  "audit_line": expected_audit},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Git checkout containing the committed Part 85 sources")
    parser.add_argument("--commit", required=True, help="full Git commit containing design and references")
    parser.add_argument("--checkpoint", required=True, choices=KEYS)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--step1-dir", required=True)
    parser.add_argument("--audit-log", required=True, help="audit85.py log containing AUDIT85 KEY step1: PASS")
    parser.add_argument("--commits-file", default=str(HERE / "commits.local.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        record = build_record(repo_root=Path(args.repo_root).resolve(), commit=args.commit,
                              checkpoint=args.checkpoint, snapshot=Path(args.snapshot),
                              step1_dir=Path(args.step1_dir), audit_log=Path(args.audit_log),
                              commits_file=Path(args.commits_file))
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"PROVENANCE85: REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"PROVENANCE85: recorded {args.checkpoint} {record['model_commit']} at {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
