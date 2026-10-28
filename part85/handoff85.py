#!/usr/bin/env python3
"""Copy, validate, and push Part 85 results between isolated cloud sessions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoints import BY_KEY, PILOT_KEYS, POST_TRAINED

RESULTS_REPO_PATH = PurePosixPath("results/part85")
REFERENCE_FILES = ("instruct/step0.json", "instruct/step1.json",
                   "neutral-token-ids.json", "pilot-score.json")
MAX_PUSH_ATTEMPTS = 3


def _git(*args, check=True, cwd=None):
    proc = subprocess.run(["git", *map(str, args)], cwd=cwd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode:
        raise RuntimeError(f"git {' '.join(map(str, args))} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def _repo_root():
    return Path(_git("rev-parse", "--show-toplevel").stdout.strip()).resolve()


def _relative_to_repo(path: Path, root: Path) -> PurePosixPath:
    resolved = path.expanduser().resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must be inside the git repository: {path}") from exc
    return PurePosixPath(rel.as_posix())


def _safe_relative(value: str) -> PurePosixPath:
    rel = PurePosixPath(value)
    if rel.is_absolute() or not rel.parts or any(part in ("..", ".") for part in rel.parts):
        raise ValueError(f"unsafe relative path in references manifest: {value!r}")
    return rel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _branch_exists(branch: str) -> bool:
    result = _git("ls-remote", "--heads", "origin", f"refs/heads/{branch}", check=False)
    if result.returncode:
        raise RuntimeError(f"could not query remote branch {branch}: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def _fetch(branch: str) -> str | None:
    if not _branch_exists(branch):
        return None
    _git("fetch", "--no-tags", "origin", f"refs/heads/{branch}")
    return _git("rev-parse", "FETCH_HEAD").stdout.strip()


def restore(branch: str, results_root: Path) -> None:
    root = _repo_root()
    commit = _fetch(branch)
    if commit is None:
        print(f"remote branch {branch} does not exist; nothing to restore")
        return
    names = _git("ls-tree", "-r", "--name-only", commit, "--", str(RESULTS_REPO_PATH)).stdout.splitlines()
    copied = 0
    for name in names:
        rel = PurePosixPath(name)
        if not rel.is_relative_to(RESULTS_REPO_PATH):
            continue
        relative_result = PurePosixPath(*rel.parts[len(RESULTS_REPO_PATH.parts):])
        destination = results_root / Path(relative_result.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_git_bytes_show(commit, name))
        copied += 1
    print(f"restored {copied} files from {branch} into {results_root}")


def _git_bytes_show(commit: str, name: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{name}"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def _root_path(results_root: Path, root: Path) -> tuple[Path, PurePosixPath]:
    path = results_root.expanduser().resolve()
    rel = _relative_to_repo(path, root)
    return path, rel


def _allowed_push_path(path: str, root_rel: PurePosixPath, checkpoint: str) -> bool:
    rel = PurePosixPath(path)
    key_root = root_rel / checkpoint
    if rel == key_root or key_root in rel.parents:
        return True
    return checkpoint == "instruct" and rel == root_rel / "neutral-token-ids.json"


def _index_blob(stage: int, path: str) -> bytes:
    result = subprocess.run(["git", "show", f":{stage}:{path}"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"cannot read rebase stage {stage} for {path}: {result.stderr.decode().strip()}")
    return result.stdout


def _merge_run_records(remote_bytes: bytes, local_bytes: bytes, path: str) -> bytes:
    remote = json.loads(remote_bytes)
    local = json.loads(local_bytes)
    merged = dict(local)
    for field in ("hook_fire_counts", "completed_cells"):
        combined = dict(remote.get(field) or {})
        for key, value in (local.get(field) or {}).items():
            if key in combined and combined[key] != value:
                raise RuntimeError(f"concurrent pushes disagree on {path} {field} entry {key}; resolve it manually")
            combined[key] = value
        merged[field] = combined
    for field in set(remote).intersection(local) - {"hook_fire_counts", "completed_cells", "updated_unix_time"}:
        if remote[field] != local[field]:
            raise RuntimeError(f"concurrent pushes disagree on {path} field {field}; resolve it manually")
    if "updated_unix_time" in remote or "updated_unix_time" in local:
        merged["updated_unix_time"] = max(remote.get("updated_unix_time", 0),
                                           local.get("updated_unix_time", 0))
    return (json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _finish_rebase(upstream: str) -> None:
    result = _git("rebase", upstream, check=False)
    if result.returncode == 0:
        return
    for _ in range(5):
        conflicts = [name for name in _git("diff", "--name-only", "--diff-filter=U", "-z").stdout.split("\0") if name]
        if not conflicts:
            raise RuntimeError(f"rebase onto {upstream} failed: {result.stderr.strip()}")
        for name in conflicts:
            ours = _index_blob(2, name)
            theirs = _index_blob(3, name)
            if ours == theirs:
                Path(name).write_bytes(ours)
            elif name.endswith("/run.json"):
                Path(name).write_bytes(_merge_run_records(ours, theirs, name))
            else:
                raise RuntimeError(f"concurrent pushes contain different bytes at {name}; resolve it manually")
            _git("add", "--", name)
        continued = _git("-c", "core.editor=true", "rebase", "--continue", check=False)
        if continued.returncode == 0:
            return
        result = continued
    raise RuntimeError(f"rebase onto {upstream} did not finish after conflict resolution")


def _rebase_preserving_restored_files(upstream: str, results_root_rel: PurePosixPath) -> None:
    """Rebase after moving restored, untracked result files out of checkout's way."""
    names = [name for name in _git("ls-files", "--others", "-z", "--", str(results_root_rel)).stdout.split("\0") if name]
    with tempfile.TemporaryDirectory(prefix="part85-rebase-") as temporary:
        saved = []
        for name in names:
            source = Path(name)
            if not source.exists() and not source.is_symlink():
                continue
            target = Path(temporary) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            saved.append((name, target))
        rebased = False
        try:
            try:
                _finish_rebase(upstream)
            except RuntimeError:
                _git("rebase", "--abort", check=False)
                raise
            rebased = True
        finally:
            for name, saved_path in saved:
                destination = Path(name)
                if not rebased:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(saved_path, destination)
                    continue
                tracked = _git("ls-files", "--error-unmatch", "--", name, check=False).returncode == 0
                if tracked:
                    if not destination.is_file() or destination.read_bytes() != saved_path.read_bytes():
                        raise RuntimeError(f"restored file {name} conflicts with the rebased branch contents")
                    saved_path.unlink()
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(saved_path, destination)


def push(checkpoint: str, branch: str, results_root: Path) -> None:
    if checkpoint not in BY_KEY:
        raise ValueError(f"unknown checkpoint {checkpoint!r}")
    root = _repo_root()
    result_path, root_rel = _root_path(results_root, root)
    allowed_dir = root_rel / checkpoint
    allowed_token_ids = root_rel / "neutral-token-ids.json"
    status = _git("diff", "--cached", "--name-only", "-z").stdout
    staged = [x for x in status.split("\0") if x]
    stray = [name for name in staged if not _allowed_push_path(name, root_rel, checkpoint)]
    if stray:
        raise RuntimeError(f"refusing push; unrelated paths are already staged: {stray}")
    if not (result_path / checkpoint).exists():
        raise FileNotFoundError(f"checkpoint results directory is missing: {result_path / checkpoint}")
    _git("add", "--", str(allowed_dir))
    if checkpoint == "instruct" and (result_path / "neutral-token-ids.json").is_file():
        _git("add", "--", str(allowed_token_ids))
    staged = [x for x in _git("diff", "--cached", "--name-only", "-z").stdout.split("\0") if x]
    stray = [name for name in staged if not _allowed_push_path(name, root_rel, checkpoint)]
    if stray:
        raise RuntimeError(f"refusing push; unrelated paths would be staged: {stray}")
    if not staged:
        print("no new checkpoint result changes to push")
        return
    _git("commit", "-m", f"Part 85 results: {checkpoint}")
    for attempt in range(1, MAX_PUSH_ATTEMPTS + 1):
        result = _git("push", "origin", f"HEAD:refs/heads/{branch}", check=False)
        if result.returncode == 0:
            print(f"pushed {checkpoint} results to {branch}")
            return
        if attempt == MAX_PUSH_ATTEMPTS:
            raise RuntimeError(f"push to {branch} rejected after {attempt} attempts: {result.stderr.strip()}")
        fetched = _fetch(branch)
        if fetched is None:
            raise RuntimeError(f"push to {branch} failed and the branch is still absent: {result.stderr.strip()}")
        _rebase_preserving_restored_files(fetched, root_rel)


def refs_manifest(results_root: Path, out: Path) -> None:
    root = results_root.expanduser()
    files = list(REFERENCE_FILES)
    score_path = root / "pilot-score.json"
    if not score_path.is_file():
        raise FileNotFoundError(f"pilot score is missing: {score_path}")
    score = json.loads(score_path.read_text(encoding="utf-8"))
    gate = score.get("pilot_gate") or {}
    expected_template = gate.get("plain_template_sha256")
    if not isinstance(expected_template, str):
        raise ValueError("pilot score is missing plain_template_sha256")
    default_template = "Question: {q}\nAnswer:"
    default_sha = hashlib.sha256(default_template.encode("utf-8")).hexdigest()
    if expected_template and expected_template != default_sha:
        template_rel = "pilot-frame-v2.txt"
        template = root / template_rel
        if not template.is_file():
            raise FileNotFoundError(f"pilot used a custom frame but its template is missing: {template}")
        if _sha256(template) != expected_template:
            raise ValueError(f"pilot frame template SHA-256 mismatch: {template}")
        files.append(template_rel)
    entries = []
    for rel in files:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"cannot make reference manifest; missing {path}")
        entries.append({"path": rel, "sha256": _sha256(path)})
    payload = {"schema_version": 1, "files": entries,
               "pilot_gate_action": gate.get("action"),
               "plain_template_sha256": expected_template}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} reference hashes to {out}")


def preflight(checkpoint: str, phase: str, results_root: Path, references: Path) -> None:
    if checkpoint not in BY_KEY:
        raise ValueError(f"unknown checkpoint {checkpoint!r}")
    root = results_root.expanduser()
    if phase == "pilot":
        required = [] if checkpoint == "instruct" else [
            "instruct/step0.json", "instruct/step1.json", "neutral-token-ids.json"]
        missing = [rel for rel in required if not (root / rel).is_file()]
        if missing:
            raise RuntimeError("pilot preflight missing required files: " + ", ".join(missing))
        print(f"pilot preflight passed for {checkpoint}")
        return
    manifest_path = references
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    if not manifest_path.is_file():
        raise RuntimeError(f"main preflight missing reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError(f"main preflight reference manifest has no files list: {manifest_path}")
    listed = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError("main preflight has malformed reference entry")
        rel = _safe_relative(entry["path"])
        listed[str(rel)] = entry.get("sha256")
    missing_required = [rel for rel in REFERENCE_FILES if rel not in listed]
    problems = [f"manifest omits required reference {rel}" for rel in missing_required]
    plain_sha = hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()
    if manifest.get("plain_template_sha256") not in (None, plain_sha) and "pilot-frame-v2.txt" not in listed:
        problems.append("manifest omits custom pilot frame template pilot-frame-v2.txt")
    score_file = root / "pilot-score.json"
    if score_file.is_file():
        try:
            gate_sha = (json.loads(score_file.read_text(encoding="utf-8")).get("pilot_gate") or {}).get(
                "plain_template_sha256")
        except json.JSONDecodeError:
            gate_sha = None
        if gate_sha != manifest.get("plain_template_sha256"):
            problems.append("manifest plain_template_sha256 differs from pilot-score.json")
        if gate_sha and gate_sha != plain_sha:
            if "pilot-frame-v2.txt" not in listed:
                problems.append("pilot-score.json used a custom frame but the manifest omits pilot-frame-v2.txt")
            elif (root / "pilot-frame-v2.txt").is_file() and _sha256(root / "pilot-frame-v2.txt") != gate_sha:
                problems.append("pilot-frame-v2.txt SHA-256 differs from pilot-score.json")
    for rel, digest in listed.items():
        path = root / Path(rel)
        if not path.is_file():
            problems.append(f"missing {path}")
        elif not isinstance(digest, str) or _sha256(path) != digest:
            problems.append(f"SHA-256 mismatch {path}")
    if problems:
        raise RuntimeError("main preflight failed: " + "; ".join(problems))
    print(f"main preflight passed for {checkpoint}; verified {len(listed)} files")


def assemble(branches: list[str], out: Path) -> None:
    root = _repo_root()
    gathered: dict[str, bytes] = {}
    owners: dict[str, str] = {}
    for branch in branches:
        suffix = branch.rsplit("/", 1)[-1]
        allowed = set(PILOT_KEYS) if suffix == "pilot" else ({suffix} if suffix in BY_KEY else set())
        if not allowed:
            raise ValueError(f"cannot infer checkpoint ownership from branch {branch!r}")
        commit = _fetch(branch)
        if commit is None:
            raise FileNotFoundError(f"remote branch does not exist: {branch}")
        names = _git("ls-tree", "-r", "--name-only", commit, "--", str(RESULTS_REPO_PATH)).stdout.splitlines()
        owned = set()
        for name in names:
            rel = PurePosixPath(name)
            if rel == RESULTS_REPO_PATH / "neutral-token-ids.json":
                if "instruct" not in allowed:
                    raise ValueError(f"branch {branch} contains neutral-token-ids.json without owning instruct")
                owned.add("neutral-token-ids.json")
                continue
            if not rel.is_relative_to(RESULTS_REPO_PATH) or len(rel.parts) < len(RESULTS_REPO_PATH.parts) + 2:
                continue
            key = rel.parts[len(RESULTS_REPO_PATH.parts)]
            if key not in BY_KEY:
                continue
            if key not in allowed:
                raise ValueError(f"branch {branch} contains files for checkpoint {key}, outside its ownership {sorted(allowed)}")
            owned.add(key)
            relative = PurePosixPath(*rel.parts[len(RESULTS_REPO_PATH.parts):]).as_posix()
            data = _git_bytes_show(commit, name)
            if relative in gathered and gathered[relative] != data:
                raise ValueError(f"branches {owners[relative]} and {branch} contain different bytes at {relative}")
            gathered[relative] = data
            owners[relative] = branch
        if not owned.intersection(allowed):
            raise ValueError(f"branch {branch} contains no results for its checkpoint ownership")
    for relative, data in gathered.items():
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"assembled {len(gathered)} files from {len(branches)} branches into {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("restore")
    p.add_argument("--branch", required=True)
    p.add_argument("--results-root", required=True, type=Path)
    p = subs.add_parser("preflight")
    p.add_argument("--checkpoint", required=True, choices=tuple(BY_KEY))
    p.add_argument("--phase", required=True, choices=("pilot", "main"))
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--references", type=Path, default=Path("part85/references.json"))
    p = subs.add_parser("refs-manifest")
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = subs.add_parser("push")
    p.add_argument("--checkpoint", required=True, choices=tuple(BY_KEY))
    p.add_argument("--branch", required=True)
    p.add_argument("--results-root", required=True, type=Path)
    p = subs.add_parser("assemble")
    p.add_argument("--branches", nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "restore":
        restore(args.branch, args.results_root)
    elif args.command == "preflight":
        preflight(args.checkpoint, args.phase, args.results_root, args.references)
    elif args.command == "refs-manifest":
        refs_manifest(args.results_root, args.out)
    elif args.command == "push":
        push(args.checkpoint, args.branch, args.results_root)
    else:
        assemble(args.branches, args.out)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"handoff85: {exc}", file=sys.stderr)
        raise SystemExit(1)
