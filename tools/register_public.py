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
(2) The section is cut from its own "## Part N (registered" heading to the next "## Part" heading (see (5)). (3) Without the merge-back, the next push of the branch is refused as a
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
    # (5) 2026-10-29, Part 112's amendment: the section ran to the end of the file, so registering an older part's
    # amendment republished every later part's section, including verdict cells not yet cold-checked. The section now
    # stops at the next "## Part" heading, and a part whose section is already public leaves PREDICTIONS.md alone.
    start = cur.index(head)
    nxt = cur.find("\n## Part ", start + 1)
    section = cur[start:] if nxt < 0 else cur[start:nxt + 1]
    pub = git("show", f"{base}:PREDICTIONS.md")
    pred = pub if head in pub else pub.rstrip("\n") + "\n" + section
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
    # (4) 2026-10-29, Part 116: registering files that are still untracked, with PREDICTIONS.md modified, made the
    # merge refuse ("untracked working tree files would be overwritten"), and the branch had to be repaired by hand.
    # Untracked registered files identical to the pushed blobs are removed (the merge brings them back), and a
    # modified PREDICTIONS.md is set aside and restored after the merge (it is a superset of the pushed copy).
    # (6) 2026-10-29, Part 112's amendment: a registered file that is tracked and modified (identical to the pushed
    # blob) made the merge refuse, and the untracked files already removed were then missing. Every registered file
    # identical to the pushed blob is now set aside (a tracked one checked out from HEAD, an untracked one moved to a
    # temporary folder), and all of it is put back if the merge fails.
    tracked = set(git("ls-files").splitlines())
    aside = Path(tempfile.mkdtemp(prefix="register-aside-"))
    moved, reset = [], []
    for f in a.files:
        if not Path(f).exists() or git("show", f"{commit}:{f}") != Path(f).read_text(encoding="utf-8").strip():
            continue
        if f in tracked:
            if git("diff", "--name-only", "--", f):
                (aside / f).parent.mkdir(parents=True, exist_ok=True)
                (aside / f).write_bytes(Path(f).read_bytes())
                git("checkout", "--", f)
                reset.append(f)
        else:
            (aside / f).parent.mkdir(parents=True, exist_ok=True)
            os.replace(f, aside / f)
            moved.append(f)
    saved = Path("PREDICTIONS.md").read_text(encoding="utf-8") if git("diff", "--name-only", "--", "PREDICTIONS.md") else None
    if saved is not None:
        git("checkout", "--", "PREDICTIONS.md")
    try:
        git("merge", "-X", "ours", "--no-edit", "-m", f"Merge the public Part {a.part} registration into the working branch" + TRAILER,
            f"{a.remote}/main")
    except subprocess.CalledProcessError as e:
        for f in moved + reset:
            Path(f).parent.mkdir(parents=True, exist_ok=True)
            Path(f).write_bytes((aside / f).read_bytes())
        raise SystemExit(f"pushed {commit[:12]} but the merge-back failed; the working files are restored: {e.stderr}")
    finally:
        if saved is not None:
            Path("PREDICTIONS.md").write_text(saved, encoding="utf-8")
    print(f"registered Part {a.part}: public main {commit[:12]} (base {base[:12]}); merged back")


if __name__ == "__main__":
    main()
