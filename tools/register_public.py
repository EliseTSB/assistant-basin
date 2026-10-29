#!/usr/bin/env python3
"""Publish a part's registration (design, code, predictions section) on top of the public main WITHOUT publishing
unreviewed local results, then merge it back so later pushes stay fast-forwards.

Why it exists (2026-10-29): the rule is "predictions pushed before each run", but the local branch often carries results
still waiting on a cold check. Pushing the branch would publish them early. This builds a commit on origin/main whose
tree is origin/main plus only the named files, with PREDICTIONS.md = origin/main's copy plus the part's section copied
from the working tree, pushes it, then merges origin/main into the branch with -X ours (the local PREDICTIONS.md is a
superset, so "ours" is right; the other files are identical on both sides).

What it got wrong the first time, by hand: (1) a three-way cherry-pick of the registration onto origin/main conflicts,
because the new section is appended next to rows the local results commit filled in; hence the temp-index build.
(2) The section must be cut from its own "## Part N (registered" heading to the end of the file, so register parts in
order and push each before writing the next. (3) Without the merge-back, the next push of the branch is refused as a
non-fast-forward.

Usage: python3 tools/register_public.py --part 115 --message "Register Part 115 ..." FILE [FILE ...]
Needs SSL_CERT_FILE and GIT_SSL_CAINFO pointed at the mitm bundle on this host. Written by Claude (Opus 5.5).
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"


def git(*args, env=None, inp=None):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, env=env, input=inp).stdout.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True)
    p.add_argument("--message", required=True)
    p.add_argument("--remote", default="origin")
    p.add_argument("files", nargs="+")
    a = p.parse_args()
    git("fetch", "-q", a.remote, "main")
    base = git("rev-parse", f"{a.remote}/main")
    cur = Path("PREDICTIONS.md").read_text(encoding="utf-8")
    head = f"\n## Part {a.part} (registered"
    if cur.count(head) != 1:
        raise SystemExit(f"expected one '{head.strip()}' heading in PREDICTIONS.md, found {cur.count(head)}")
    section = cur[cur.index(head):]
    pub = git("show", f"{base}:PREDICTIONS.md")
    pred = pub.rstrip("\n") + "\n" + section
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(d) / "index")}
        git("read-tree", base, env=env)
        for f in a.files:
            git("update-index", "--add", "--cacheinfo", f"100644,{git('hash-object', '-w', f)},{f}", env=env)
        blob = git("hash-object", "-w", "--stdin", inp=pred)
        git("update-index", "--cacheinfo", f"100644,{blob},PREDICTIONS.md", env=env)
        tree = git("write-tree", env=env)
    commit = git("commit-tree", tree, "-p", base, inp=a.message + TRAILER)
    git("push", "-q", a.remote, f"{commit}:refs/heads/main")
    git("fetch", "-q", a.remote, "main")
    git("merge", "-X", "ours", "--no-edit", "-m", f"Merge the public Part {a.part} registration into the working branch" + TRAILER,
        f"{a.remote}/main")
    print(f"registered Part {a.part}: public main {commit[:12]} (base {base[:12]}); merged back")


if __name__ == "__main__":
    main()
