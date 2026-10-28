#!/usr/bin/env python3
"""download85.py -- fetch Part 85's eleven non-cached checkpoints into the Hugging Face hub cache on E:, rate-capped.

Written by Claude (Opus 5.5), 2026-10-28, for decision part85-after-r4-2026-10-28 option 1 (d) and decision
part85-checkpoint-download-2026-10-28 (Elise: "The download is fine"; overnight, 5 MB/s). Run it with the Windows venv
(C:\\Users\\alexa\\jlens-tools\\.venv\\Scripts\\python.exe): two large downloads from WSL have coincided with the Gmail relay
losing the network, and Terra and JJ's channel share this line.

For every registered checkpoint except Instruct (already cached), it resolves the pinned commit (part85/checkpoints.py,
prefix checked), lists the commit's files with the Hub's metadata (files_metadata=True), and writes each file to
<cache-dir>/models--<org>--<name>/snapshots/<full commit>/<file>: the layout huggingface_hub itself writes when the
filesystem has no symlinks, so from_pretrained(repo, revision=<commit>) finds it offline, and part85's --snapshot-dir
accepts the directory because its name is the commit. Each file downloads to "<file>.incomplete" with an HTTP Range
resume from whatever is already there, at most --rate bytes per second, and is moved into place only after its hash
matches the Hub's: the LFS sha256 for weights, the git blob sha1 for small files. A mismatch REFUSES: the bad bytes
are kept as "<file>.bad-<sha prefix>" and the command exits 1 before touching the next file. A file already in place is
re-hashed, not trusted.

Usage: download85.py [--cache-dir E:\\neuron-probe\\hf\\hub] [--rate 5000000] [--only KEY ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

from checkpoints import CHECKPOINTS, LOCAL_COMMITS, resolve_checkpoint  # noqa: E402

DEFAULT_CACHE = r"E:\neuron-probe\hf\hub"
DEFAULT_RATE = 5_000_000
CACHED = ("instruct",)
CHUNK = 1 << 20


class HashMismatch(RuntimeError):
    pass


def repo_folder(cache_dir: Path, repo_id: str) -> Path:
    return Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_hash(sibling) -> tuple[str, str, int]:
    """(algorithm, hex digest, size) the Hub publishes for one file."""
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None:
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        if sha:
            return "sha256", sha, int(size if size is not None else sibling.size)
    blob = getattr(sibling, "blob_id", None)
    if not blob:
        raise RuntimeError(f"the Hub gave no hash for {sibling.rfilename}")
    return "git-sha1", blob, int(sibling.size)


def verify(path: Path, algorithm: str, digest: str) -> bool:
    return (sha256_file(path) if algorithm == "sha256" else git_blob_sha1(path)) == digest


class RateLimiter:
    """Never more than `rate` bytes in any prefix of the transfer, measured from the first byte."""

    def __init__(self, rate: int, clock=time.monotonic, sleep=time.sleep):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate, self.clock, self.sleep = rate, clock, sleep
        self.start = None
        self.sent = 0

    def take(self, n: int):
        if self.start is None:
            self.start = self.clock()
        self.sent += n
        due = self.start + self.sent / self.rate
        wait = due - self.clock()
        if wait > 0:
            self.sleep(wait)


def fetch(url: str, dest: Path, size: int, limiter: RateLimiter, opener=urllib.request.urlopen) -> Path:
    """Download url into dest + '.incomplete', resuming from its current length. Returns the incomplete path."""
    part = dest.with_name(dest.name + ".incomplete")
    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with opener(req) as resp, open(part, "ab") as out:
            status = getattr(resp, "status", 200)
            if have and status != 206:
                raise RuntimeError(f"{url}: server ignored the resume Range (status {status})")
            while True:
                chunk = resp.read(min(CHUNK, limiter.rate))
                if not chunk:
                    break
                out.write(chunk)
                limiter.take(len(chunk))
            out.flush()
            os.fsync(out.fileno())
    if part.stat().st_size != size:
        raise RuntimeError(f"{url}: got {part.stat().st_size} bytes, the Hub lists {size}")
    return part


def place(part: Path, dest: Path, algorithm: str, digest: str):
    if not verify(part, algorithm, digest):
        actual = sha256_file(part) if algorithm == "sha256" else git_blob_sha1(part)
        bad = dest.with_name(f"{dest.name}.bad-{actual[:12]}")
        os.replace(part, bad)
        raise HashMismatch(f"{dest.name}: {algorithm} {actual} is not the Hub's {digest}; kept as {bad.name}")
    os.replace(part, dest)


def plan(keys=None, api=None):
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    out = []
    for cp in CHECKPOINTS:
        if cp.key in CACHED or (keys and cp.key not in keys):
            continue
        resolved = resolve_checkpoint(cp, api=api)
        info = api.model_info(cp.repo_id, revision=resolved["commit"], files_metadata=True)
        if getattr(info, "sha", None) != resolved["commit"]:
            raise RuntimeError(f"{cp.key}: file listing is for {getattr(info, 'sha', None)}, not {resolved['commit']}")
        files = []
        for sib in info.siblings:
            if sib.rfilename in (".gitattributes", "README.md"):
                continue
            algorithm, digest, size = expected_hash(sib)
            files.append({"name": sib.rfilename, "algorithm": algorithm, "digest": digest, "size": size})
        out.append({"key": cp.key, "repo_id": cp.repo_id, "commit": resolved["commit"], "files": files})
    return out


def download(entries, cache_dir: Path, rate: int, opener=urllib.request.urlopen, clock=time.monotonic,
             sleep=time.sleep, log=print):
    limiter = RateLimiter(rate, clock, sleep)
    for entry in entries:
        snap = repo_folder(cache_dir, entry["repo_id"]) / "snapshots" / entry["commit"]
        for f in entry["files"]:
            dest = snap / f["name"]
            if dest.exists():
                if dest.stat().st_size == f["size"] and verify(dest, f["algorithm"], f["digest"]):
                    log(f"ok   {entry['key']} {f['name']} (already in place, hash checked)")
                    continue
                raise HashMismatch(f"{dest} is in place but does not match the Hub; move it aside by hand")
            url = f"https://huggingface.co/{entry['repo_id']}/resolve/{entry['commit']}/{f['name']}"
            started = clock()
            part = fetch(url, dest, f["size"], limiter, opener)
            place(part, dest, f["algorithm"], f["digest"])
            log(f"got  {entry['key']} {f['name']} {f['size']:,} B in {clock() - started:.0f} s")
        log(f"DONE {entry['key']} {entry['commit']} -> {snap}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--rate", type=int, default=DEFAULT_RATE, help="bytes per second (default 5,000,000)")
    ap.add_argument("--only", nargs="*", help="checkpoint keys to fetch (default: all eleven non-cached)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    entries = plan(args.only)
    total = sum(f["size"] for e in entries for f in e["files"])
    for e in entries:
        weights = sum(f["size"] for f in e["files"])
        print(f"{e['key']:14s} {e['commit'][:12]} {len(e['files'])} files {weights / 1e9:.2f} GB")
    print(f"TOTAL {total:,} bytes ({total / 1e9:.1f} GB); at {args.rate:,} B/s about {total / args.rate / 3600:.1f} h")
    if args.dry_run:
        print("DRY RUN: nothing downloaded")
        return 0
    try:
        download(entries, Path(args.cache_dir), args.rate)
    except HashMismatch as exc:
        print(f"REFUSED: {exc}")
        return 1
    write_local_commits()
    print(f"DOWNLOAD85: all files in place and hash-checked; commits pinned in {LOCAL_COMMITS.name}")
    return 0


def write_local_commits(api=None, path=LOCAL_COMMITS):
    """All twelve full commits, resolved online and prefix-checked, for resolve_checkpoint in the offline runner."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    commits = {cp.key: resolve_checkpoint(cp, api=api)["commit"] for cp in CHECKPOINTS}
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(json.dumps(commits, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return commits


if __name__ == "__main__":
    sys.exit(main())
