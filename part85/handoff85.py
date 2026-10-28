#!/usr/bin/env python3
"""Single-writer, branch-owned Part 85 result handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoints import BY_KEY, PILOT_KEYS

RESULTS_REPO_PATH = PurePosixPath("results/part85")
REFERENCE_DIR = Path("part85/references")
REFERENCE_FILES = (
    "part85/references/instruct/step0.json",
    "part85/references/instruct/step1.json",
    "part85/references/neutral-token-ids.json",
    "part85/references/pilot-score.json",
)
EXPECTED_TOKEN_SHA256 = "96699282f18fc53d5b5467a51803241aab479e23523b92980bdb83546ae15d87"
MAX_PUSH_ATTEMPTS = 3


def _git(*args, check=True, cwd=None):
    proc = subprocess.run(["git", *map(str, args)], cwd=cwd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode:
        raise RuntimeError(f"git {' '.join(map(str, args))} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def _repo_root():
    return Path(_git("rev-parse", "--show-toplevel").stdout.strip()).resolve()


def code_commit() -> str:
    """Return the latest Part 85 code/reference commit, excluding result-only commits."""
    try:
        pinned = os.environ.get("PART85_CODE_COMMIT", "").strip()
        staged = Path(__file__).resolve().with_name("CODE_COMMIT")
        if not pinned and staged.is_file():
            pinned = staged.read_text(encoding="utf-8").strip()  # the staged A770 copy is not a git checkout
        if re.fullmatch(r"[0-9a-f]{40,64}", pinned):
            return pinned
        proc = _git("log", "-1", "--format=%H", "HEAD", "--", "part85", check=False)
        value = proc.stdout.strip() if isinstance(proc.stdout, str) else ""
        if not value:
            proc = _git("rev-parse", "HEAD", check=False)
            value = proc.stdout.strip() if isinstance(proc.stdout, str) else ""
        return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else "unknown"
    except (OSError, RuntimeError):
        return "unknown"


def branch_directory(branch: str, checkpoint: str | None = None) -> PurePosixPath:
    """Map the allowed branch name to its one result directory and reject mismatches."""
    match = re.fullmatch(r"part85-run/pilot-a([1-9][0-9]*)", branch)
    if match:
        if checkpoint is not None and checkpoint not in PILOT_KEYS:
            raise ValueError(f"checkpoint {checkpoint} is not owned by pilot branch {branch}")
        return PurePosixPath(f"pilot-a{match.group(1)}")
    match = re.fullmatch(r"part85-run/main-([a-zA-Z0-9_]+)", branch)
    if match and match.group(1) in BY_KEY:
        key = match.group(1)
        if checkpoint is not None and checkpoint != key:
            raise ValueError(f"branch {branch} does not match checkpoint {checkpoint}")
        return PurePosixPath("main") / key
    raise ValueError(f"branch name does not identify a Part 85 result directory: {branch!r}")


def result_dir(results_root: Path, checkpoint: str, *, branch: str | None = None,
               mode: str | None = None, attempt: int = 1) -> Path:
    if branch:
        base = branch_directory(branch, checkpoint)
    elif mode == "pilot":
        if attempt not in (1, 2):
            raise ValueError("pilot attempt must be 1 or 2")
        if checkpoint not in PILOT_KEYS:
            raise ValueError(f"checkpoint {checkpoint} is not in the registered pilot")
        base = PurePosixPath(f"pilot-a{attempt}")
    elif mode == "main":
        if checkpoint not in BY_KEY:
            raise ValueError(f"unknown checkpoint {checkpoint!r}")
        base = PurePosixPath("main") / checkpoint
    else:
        raise ValueError("result directory requires a branch or run mode")
    if base.parts[0].startswith("pilot-a"):
        base = base / checkpoint
    return Path(results_root) / Path(base.as_posix())


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
        raise ValueError(f"unsafe relative path: {value!r}")
    return rel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remote_tip(branch: str) -> str | None:
    result = _git("ls-remote", "--heads", "origin", f"refs/heads/{branch}", check=False)
    if result.returncode:
        raise RuntimeError(f"could not query remote branch {branch}: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == f"refs/heads/{branch}":
            return fields[0]
    return None


def _tree_names(commit: str) -> list[str]:
    return _git("ls-tree", "-r", "--name-only", commit, "--", str(RESULTS_REPO_PATH)).stdout.splitlines()


def _verify_owned_tree(branch: str, commit: str, *, require_results=True) -> tuple[PurePosixPath, list[str]]:
    owner = branch_directory(branch)
    expected = RESULTS_REPO_PATH / owner
    names = _tree_names(commit)
    owned_names = []
    for name in names:
        rel = PurePosixPath(name)
        if rel == expected or expected in rel.parents:
            owned_names.append(name)
        else:
            raise ValueError(f"branch {branch} contains a file outside its owned directory: {name}")
    if require_results and not owned_names:
        raise ValueError(f"branch {branch} contains no results in {expected}")
    for name in owned_names:
        if name.endswith("/run.json"):
            payload = json.loads(subprocess.run(["git", "show", f"{commit}:{name}"],
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                check=True).stdout)
            completed = payload.get("completed_cells", {})
            if not isinstance(completed, dict):
                raise ValueError(f"{name} has malformed completed_cells")
            for cell, record in completed.items():
                answer_file = record.get("answer_file") if isinstance(record, dict) else None
                if not isinstance(answer_file, str):
                    raise ValueError(f"{name} claims {cell} without an answer_file")
                answer_rel = _safe_relative(answer_file)
                if answer_rel.is_absolute() or str(answer_rel) not in names:
                    raise ValueError(f"{name} claims missing answer file {answer_file}")
                answer_path = PurePosixPath(answer_file)
                if expected != answer_path and expected not in answer_path.parents:
                    raise ValueError(f"{name} claims answer outside its owned directory: {answer_file}")
    return owner, owned_names


def restore(branch: str, results_root: Path) -> None:
    root = _repo_root()
    root_rel = _relative_to_repo(results_root, root)
    if root_rel != RESULTS_REPO_PATH:
        raise ValueError(f"restore results root must be {RESULTS_REPO_PATH}, got {root_rel}")
    tip = _remote_tip(branch)
    if tip is None:
        print(f"remote branch {branch} does not exist; nothing to restore")
        return
    _git("fetch", "--no-tags", "origin", f"refs/heads/{branch}")
    fetched = _git("rev-parse", "FETCH_HEAD").stdout.strip()
    if fetched != tip:
        raise RuntimeError(f"remote branch {branch} changed while restoring; rerun restore")
    _verify_owned_tree(branch, fetched)
    _git("checkout", "--detach", fetched)
    print(f"checked out {branch} tip {fetched}")


def push(checkpoint: str, branch: str, results_root: Path) -> None:
    if checkpoint not in BY_KEY:
        raise ValueError(f"unknown checkpoint {checkpoint!r}")
    owner = branch_directory(branch, checkpoint)
    root = _repo_root()
    result_path, root_rel = _relative_to_repo(results_root, root), None
    root_rel = result_path
    if root_rel != RESULTS_REPO_PATH:
        raise ValueError(f"push results root must be {RESULTS_REPO_PATH}, got {root_rel}")
    owned_rel = RESULTS_REPO_PATH / owner
    owned_path = root / Path(owned_rel.as_posix())
    if not owned_path.is_dir():
        raise FileNotFoundError(f"owned result directory is missing: {owned_path}")
    staged = [x for x in _git("diff", "--cached", "--name-only", "-z").stdout.split("\0") if x]
    stray = [name for name in staged if not (name == str(owned_rel) or str(owned_rel) + "/" in name)]
    if stray:
        raise RuntimeError(f"refusing push; unrelated paths are already staged: {stray}")
    _git("add", "-A", "--", str(owned_rel))
    staged = [x for x in _git("diff", "--cached", "--name-only", "-z").stdout.split("\0") if x]
    stray = [name for name in staged if not (name == str(owned_rel) or str(owned_rel) + "/" in name)]
    if stray:
        raise RuntimeError(f"refusing push; unrelated paths would be staged: {stray}")
    if staged:
        name = _git("config", "--local", "user.name", check=False)
        email = _git("config", "--local", "user.email", check=False)
        if name.returncode or not name.stdout.strip() or email.returncode or not email.stdout.strip():
            _git("config", "--local", "user.name", "Part 85 cloud run")
            _git("config", "--local", "user.email", "noreply@anthropic.com")
        _git("commit", "-m", f"Part 85 results: {owner.as_posix()} {time.time_ns()}")

    detail = "remote tip did not converge"
    for attempt in range(1, MAX_PUSH_ATTEMPTS + 1):
        try:
            head = _git("rev-parse", "HEAD").stdout.strip()
            remote = _remote_tip(branch)
            if remote == head:
                print(f"verified {branch} at {head}")
                return
            if remote is not None:
                ancestry = _git("merge-base", "--is-ancestor", remote, head, check=False)
                if ancestry.returncode != 0:
                    raise RuntimeError(f"remote {branch} tip {remote} is not an ancestor of local HEAD {head}; restore before writing")
            lease = f"refs/heads/{branch}:{remote or ''}"
            result = _git("push", f"--force-with-lease={lease}", "origin", f"HEAD:refs/heads/{branch}", check=False)
            if result.returncode == 0:
                verified = _remote_tip(branch)
                if verified == head:
                    print(f"verified {branch} at {head}")
                    return
                detail = f"push returned success but remote tip is {verified!r}, expected {head}"
            else:
                detail = result.stderr.strip() or f"git push exited {result.returncode}"
        except RuntimeError as exc:
            detail = str(exc)
        if attempt < MAX_PUSH_ATTEMPTS:
            time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"could not verify push to {branch} after {MAX_PUSH_ATTEMPTS} attempts: {detail}")


def refs_manifest(results_root: Path, out: Path, references_dir: Path = REFERENCE_DIR) -> None:
    root = Path(results_root)
    source_base = root / "main" / "instruct"
    pilot_score = root / "pilot-score.json"
    if not pilot_score.is_file():
        raise FileNotFoundError(f"accepted pilot score is missing: {pilot_score}")
    score = json.loads(pilot_score.read_text(encoding="utf-8"))
    gate = score.get("pilot_gate") or {}
    expected_template = gate.get("plain_template_sha256")
    if not isinstance(expected_template, str):
        raise ValueError("pilot score is missing plain_template_sha256")
    default_sha = hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()
    sources = {
        "instruct/step0.json": source_base / "step0.json",
        "instruct/step1.json": source_base / "step1.json",
        "neutral-token-ids.json": root / "neutral-token-ids.json",
        "pilot-score.json": pilot_score,
    }
    if expected_template != default_sha:
        custom = root / "pilot-a2" / "pilot-frame-v2.txt"
        if not custom.is_file():
            raise FileNotFoundError(f"pilot used a custom frame but its attempt-2 template is missing: {custom}")
        if _sha256(custom) != expected_template:
            raise ValueError(f"attempt-2 template SHA-256 mismatch: {custom}")
        sources["pilot-frame-v2.txt"] = custom
    files = []
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"cannot publish references; missing {source}")
        relative = Path("part85/references") / name
        destination = references_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        files.append({"path": relative.as_posix(), "sha256": _sha256(destination)})
    payload = {"schema_version": 2, "files": files,
               "pilot_gate_action": gate.get("action"),
               "plain_template_sha256": expected_template,
               "pilot_attempt": gate.get("attempt")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"copied and hashed {len(files)} pinned references into {references_dir}")


def _validate_step_file(path: Path, checkpoint: str, name: str) -> str | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = payload.get("checkpoint") or {}
    if identity.get("key") != checkpoint:
        raise ValueError(f"{path} identifies checkpoint {identity.get('key')!r}, expected {checkpoint!r}")
    if not str(identity.get("commit", "")).startswith(BY_KEY[checkpoint].commit_prefix):
        raise ValueError(f"{path} commit does not match registered checkpoint {checkpoint}")
    if not isinstance(payload.get("code_commit"), str) or not payload["code_commit"]:
        raise ValueError(f"{path} lacks code_commit provenance")
    return name


def preflight(checkpoint: str, phase: str, results_root: Path, references: Path, attempt: int = 1) -> None:
    if checkpoint not in BY_KEY:
        raise ValueError(f"unknown checkpoint {checkpoint!r}")
    root = Path(results_root)
    if phase == "pilot":
        if attempt not in (1, 2):
            raise ValueError("pilot attempt must be 1 or 2")
        pilot_root = root / f"pilot-a{attempt}"
        if checkpoint == "pretrain_4T":
            required = [pilot_root / "instruct" / "step0.json", pilot_root / "instruct" / "step1.json", pilot_root / "neutral-token-ids.json"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError("pilot preflight missing required files: " + ", ".join(missing))
            _validate_step_file(required[0], "instruct", "step0")
            _validate_step_file(required[1], "instruct", "step1")
            token = json.loads(required[2].read_text(encoding="utf-8"))
            if token.get("sha256") != EXPECTED_TOKEN_SHA256:
                raise RuntimeError("pilot neutral-token-ids.json has the wrong pinned sha256")
        print(f"pilot attempt {attempt} preflight passed for {checkpoint}")
        return

    manifest_path = references if references.is_absolute() else Path.cwd() / references
    if not manifest_path.is_file():
        raise RuntimeError(f"main preflight missing reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError(f"main preflight reference manifest has no files list: {manifest_path}")
    listed = {}
    problems = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            problems.append("malformed reference manifest entry")
            continue
        rel = _safe_relative(entry["path"])
        listed[str(rel)] = entry.get("sha256")
    for required in REFERENCE_FILES:
        if required not in listed:
            problems.append(f"manifest omits required reference {required}")
    score_path = Path.cwd() / "part85/references/pilot-score.json"
    score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.is_file() else {}
    gate = score.get("pilot_gate") or {}
    gate_sha = gate.get("plain_template_sha256")
    manifest_sha = manifest.get("plain_template_sha256")
    if gate_sha != manifest_sha:
        problems.append("manifest template SHA differs from pinned pilot-score.json")
    default_sha = hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()
    custom_rel = "part85/references/pilot-frame-v2.txt"
    if gate_sha != default_sha and custom_rel not in listed:
        problems.append("manifest omits attempt-2 template used by the pilot")
    for rel, digest in listed.items():
        path = Path.cwd() / rel
        if not path.is_file():
            problems.append(f"missing reference {path}")
        elif not isinstance(digest, str) or _sha256(path) != digest:
            problems.append(f"SHA-256 mismatch {path}")
    tokens_path = Path.cwd() / "part85/references/neutral-token-ids.json"
    if tokens_path.is_file():
        token = json.loads(tokens_path.read_text(encoding="utf-8"))
        if token.get("sha256") != EXPECTED_TOKEN_SHA256:
            problems.append("pinned neutral-token-ids.json has the wrong expected sha256")

    own_dir = root / "main" / checkpoint
    for name in ("step0", "step1"):
        try:
            _validate_step_file(own_dir / f"{name}.json", checkpoint, name)
        except (ValueError, json.JSONDecodeError) as exc:
            problems.append(str(exc))
    if problems:
        raise RuntimeError("main preflight failed: " + "; ".join(problems))
    print(f"main preflight passed for {checkpoint}; verified {len(listed)} references; existing own step files checked")


def assemble(branches: list[str], out: Path) -> None:
    root = _repo_root()
    gathered: dict[str, bytes] = {}
    for branch in branches:
        remote_before = _remote_tip(branch)
        if remote_before is None:
            raise FileNotFoundError(f"remote branch does not exist: {branch}")
        fetched = _git("fetch", "--no-tags", "origin", f"refs/heads/{branch}", check=False)
        if fetched.returncode:
            raise RuntimeError(f"could not fetch branch {branch}: {fetched.stderr.strip()}")
        tip = _git("rev-parse", "FETCH_HEAD").stdout.strip()
        if tip != remote_before or _remote_tip(branch) != tip:
            raise RuntimeError(f"remote branch {branch} changed while assembling; rerun assemble")
        owner, names = _verify_owned_tree(branch, tip)
        expected = RESULTS_REPO_PATH / owner
        for name in names:
            rel = PurePosixPath(name)
            relative = PurePosixPath(*rel.parts[len(RESULTS_REPO_PATH.parts):]).as_posix()
            data = subprocess.run(["git", "show", f"{tip}:{name}"], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, check=True).stdout
            gathered[relative] = data
    out_root = Path(out)
    for relative, data in gathered.items():
        target = out_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"assembled {len(gathered)} owned files from {len(branches)} branches into {out_root}")


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
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--references", type=Path, default=Path("part85/references.json"))
    p = subs.add_parser("refs-manifest")
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--references-dir", type=Path, default=REFERENCE_DIR)
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
        preflight(args.checkpoint, args.phase, args.results_root, args.references, args.attempt)
    elif args.command == "refs-manifest":
        refs_manifest(args.results_root, args.out, args.references_dir)
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
