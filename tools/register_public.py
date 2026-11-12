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
(2) The section is cut from its own "## Part N (registered" heading to the next "## Part" heading (see (5)). (4, 2026-11-11, Part 203) Every file was written as 100644, so an executable script lost its mode on main and the
merge-back failed on a mode-only add/add conflict; the mode now follows the working file. (3) Without the merge-back, the next push of the branch is refused as a
non-fast-forward.

Usage: python3 tools/register_public.py --part 115 --message "Register Part 115 ..." FILE [FILE ...]
Needs SSL_CERT_FILE and GIT_SSL_CAINFO pointed at the mitm bundle on this host. Written by Claude (Opus 5.5).
"""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"


def git_bytes(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def git(*args, env=None, inp=None):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, env=env, input=inp).stdout.strip()


def _section(text, head):
    """(start, end, text) of the section from its heading to the next "## Part" heading or the end."""
    start = text.index(head)
    nxt = text.find("\n## Part ", start + 1)
    end = len(text) if nxt < 0 else nxt + 1
    return start, end, text[start:end]


def _row_id(line, part):
    return line.split("|")[1].strip() if line.startswith(f"| {part}.") else None


def _with_new_rows(pub, head, section, part):
    """(6) 2026-10-30, Part 118's Amendment 2: (5) kept origin/main's copy whole, so the rows an amendment adds to an
    already-public section were never published. Rows the public section lacks are now appended after its last row,
    but only while their result and verdict cells are empty or "pending"; rows it already has keep the public text,
    so local verdicts stay local."""
    start, end, psec = _section(pub, head)
    have = {_row_id(line, part) for line in psec.splitlines()} - {None}
    new = []
    for line in section.splitlines():
        rid = _row_id(line, part)
        if rid and rid not in have:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Tables with an Author column have one result cell at the end; older ones have Result and Verdict.
            header = next((l for l in section.splitlines() if l.startswith("| #")), "")
            result_cells = 2 if "Verdict" in header else 1
            if len(cells) < 4 or any(c not in ("", "pending") for c in cells[-result_cells:]):
                raise SystemExit(f"row {rid} is new to the public section but its result or verdict is filled")
            new.append(line + "\n")
    if not new:
        return pub
    lines = psec.splitlines(keepends=True)
    last = max((i for i, line in enumerate(lines) if _row_id(line, part)), default=len(lines) - 1)
    lines[last + 1:last + 1] = new
    return pub[:start] + "".join(lines) + pub[end:]


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
    section = _section(cur, head)[2]
    pub = git("show", f"{base}:PREDICTIONS.md") + "\n"   # git() strips the file's final newline
    pred = pub.rstrip("\n") + "\n" + section if head not in pub else _with_new_rows(pub, head, section, a.part)
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(d) / "index")}
        git("read-tree", base, env=env)
        for f in a.files:
            mode = "100755" if os.access(f, os.X_OK) else "100644"  # 2026-11-11: was always 100644 (dropped download.sh's x bit)
            git("update-index", "--add", "--cacheinfo", f"{mode},{git('hash-object', '-w', f)},{f}", env=env)
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
        # (7) 2026-11-04, Part 188: this comparison decoded the pushed blob as text, so a binary registered file
        # (part188/parity_ref.npz) raised UnicodeDecodeError AFTER the push, and the merge-back had to be done by
        # hand. Blobs are now compared as bytes, exactly.
        if not Path(f).exists() or git_bytes("show", f"{commit}:{f}") != Path(f).read_bytes():
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
    # (8) 2026-11-08, Part 196's Amendment 3: the restored working PREDICTIONS.md is a superset of the PUSHED copy but
    # not of origin/main when main gained another part's section after this branch was cut (Part 194's), so the
    # restore dropped that section, and the next push of the branch published the loss (7bab68d; restored 0441905).
    # Every part section on origin/main that the working copy lacks is now put back in main's order, and a missing
    # front-page entry in README.md stops the script by name.
    readded = _restore_main_sections(commit)
    lost = _missing_front_page_entries(commit)
    if readded:
        print(f"re-added from public main to the working PREDICTIONS.md (commit it with your next change): {readded}")
    if lost:
        raise SystemExit(f"pushed {commit[:12]} and merged back, but README.md lacks public main's entries for {lost}; "
                         "copy them from origin/main before any push")
    print(f"registered Part {a.part}: public main {commit[:12]} (base {base[:12]}); merged back")


_PART_HEAD = re.compile(r"^## Part (\S+) \(", re.M)


def _restore_main_sections(ref):
    """Insert into the working PREDICTIONS.md every "## Part N (" section that ref has and it lacks."""
    remote = git_bytes("show", f"{ref}:PREDICTIONS.md").decode("utf-8")
    path = Path("PREDICTIONS.md")
    local = path.read_text(encoding="utf-8")
    order = _PART_HEAD.findall(remote)
    missing = [n for n in order if n not in set(_PART_HEAD.findall(local))]
    for n in missing:
        m = re.search(rf"^## Part {re.escape(n)} \(", remote, re.M)
        nxt = remote.find("\n## Part ", m.start() + 1)
        section = remote[m.start():(len(remote) if nxt < 0 else nxt + 1)]
        if not section.endswith("\n"):
            section += "\n"
        later = [k for k in order[order.index(n) + 1:] if re.search(rf"^## Part {re.escape(k)} \(", local, re.M)]
        if later:
            at = re.search(rf"^## Part {re.escape(later[0])} \(", local, re.M).start()
            local = local[:at] + section + "\n" + local[at:]
        else:
            local = local.rstrip("\n") + "\n\n" + section
    if missing:
        path.write_text(local, encoding="utf-8")
    return missing


def _missing_front_page_entries(ref):
    try:
        remote = git_bytes("show", f"{ref}:README.md").decode("utf-8")
    except subprocess.CalledProcessError:
        return []
    local = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""
    entry = re.compile(r"^\*\*\[Part (\S+)\]", re.M)
    return sorted(set(entry.findall(remote)) - set(entry.findall(local)))


if __name__ == "__main__":
    main()
