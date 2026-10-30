#!/usr/bin/env python3
"""part_check.py: the parts of PROCEDURE.md a script can check.

  part_check.py prelaunch NN   before any run: the design follows templates/PART-DESIGN.md, its size is multiplied
                               out, a run over about 5 minutes carries a measured optimization pass, and the design and
                               its predictions are on public main
  part_check.py publish NN     before the results merge to main: every registered prediction has a verdict in
                               results/partNN/README.md, the README records its cold check, every path it cites exists
                               and is committed, and it names no local path

NN is the part id as file names write it (115, 109b). Exit 0 = every check passed, 1 = a check failed (each failure is
printed), 2 = usage error. --repo picks the checkout (default: the one this file is in). --no-fetch skips
`git fetch origin main` (offline, tests).

Written by Claude (Opus 5.5), 2026-10-29, after Elise asked for "a checklist or procedure or something for each test
where you optimize it and follow any other useful advice we find broadly applicable over this work". A design without
the line "Procedure: v1" predates the procedure: both stages refuse it unless --legacy-ok is given, so a new design
that forgets the line does not pass silently. Things a script cannot check (whether a control is matched on the
right property, whether a sentence claims more than its table) stay in PROCEDURE.md and the cold check.
"""
import argparse, re, subprocess, sys
from pathlib import Path

SECTIONS = ["Question", "Prior art and existing data", "Method", "Controls", "Size", "Optimization",
            "Predictions", "Reading rules", "Limits"]
PLACEHOLDER_TIME = re.compile(r"\b\d{1,2}:(?:\dx|xx)\b|\bHH:MM\b|\?\?:\?\?", re.I)
VERDICT = re.compile(r"\b(held|failed|mixed|partly held|partial|open|untestable|not tested|not testable|withdrawn|"
                     r"not read|not scored|inconclusive|void)\b", re.I)
PATH_EXT = (".md", ".py", ".json", ".jsonl", ".txt", ".sh", ".csv", ".png", ".svg", ".html", ".npz", ".pt", ".gz",
            ".log", ".yaml", ".yml", ".tsv")
LOCAL_PATH = re.compile(r"(?<![\w/.])(?:/home/|/mnt/|/tmp/|/etc/|/opt/|/Users/|/var/|/root/|~/|[A-Za-z]:\\)")
# A measured throughput: a number with a rate (per second or minute), or a count done in a stated time.
RATE = re.compile(r"\d[\d.,]*\s*[A-Za-z ]{0,40}?(?:/\s?(?:s|sec|second|min|minute|h|hour)\b|per (?:second|minute|hour|s|min)\b)"
                  r"|\d[\d.,]*\s+\w+(?: \w+){0,3} in (?:about )?\d[\d.,]*\s*(?:s|sec|seconds|min|minutes|h|hours)\b", re.I)
GENERATION_UNITS = re.compile(r"generation|answer|rollout|completion", re.I)


class Report:
    def __init__(self, title):
        self.title, self.rows = title, []

    def ok(self, msg):
        self.rows.append(("ok", msg))

    def fail(self, msg):
        self.rows.append(("FAIL", msg))

    def check(self, cond, ok_msg, fail_msg):
        (self.ok if cond else self.fail)(ok_msg if cond else fail_msg)
        return cond

    @property
    def failed(self):
        return [m for s, m in self.rows if s == "FAIL"]

    def print(self):
        print(self.title)
        for s, m in self.rows:
            print(f"  {s:4}  {m}")
        n = len(self.failed)
        print(f"{'FAILED' if n else 'PASSED'}: {n} of {len(self.rows)} checks failed" if n else
              f"PASSED: all {len(self.rows)} checks")


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def find_design(repo, pid):
    want = f"PART{pid.upper()}-DESIGN.md"
    for p in repo.glob("PART*-DESIGN.md"):
        if p.name.upper() == want.upper():
            return p
    return None


def sections(text):
    """{heading: body} for every '## ' heading; the body runs to the next '## ' heading."""
    out, cur, fence = {}, None, None
    for line in text.splitlines():
        f = re.match(r"^\s*(```|~~~)", line)
        if f:
            fence = None if fence == f.group(1) else (fence or f.group(1))
        m = None if fence or f else re.match(r"^## (.+?)\s*$", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def predictions_section(repo_text, pid):
    """Every PREDICTIONS.md section whose heading is '## Part <pid>' (not '## Part <pid>b'), joined."""
    parts, keep = [], False
    head = re.compile(rf"^## Part {re.escape(pid)}\b", re.I)
    for line in repo_text.splitlines():
        if line.startswith("## "):
            keep = bool(head.match(line))
        if keep:
            parts.append(line)
    return "\n".join(parts)


def pred_ids(text, pid):
    return sorted(set(m.group(0) for m in re.finditer(rf"(?<![\w.]){re.escape(pid)}\.[A-Za-z0-9]+\b", text, re.I)),
                  key=lambda s: (len(s), s))


def parse_size(body):
    total = re.search(r"Total:\s*(\d[\d,]*(?:\.\d+)?)\s*(million|thousand)?\s+([A-Za-z][\w -]*)", body, re.I)
    mins = re.search(r"about\s+([\d.,]+)\s*(minutes?|mins?|hours?|h)\b", body, re.I)
    count, unit = None, ""
    if total:
        count = float(total.group(1).replace(",", "")) * {"million": 1e6, "thousand": 1e3}.get((total.group(2) or "").lower(), 1)
        count, unit = int(round(count)), total.group(3)
    minutes = None
    if mins:
        v = float(mins.group(1).replace(",", ""))
        minutes = v * 60 if mins.group(2).lower().startswith("h") else v
    return count, unit, minutes


def prelaunch(repo, pid, fetch=True, legacy_ok=False):
    r = Report(f"part_check prelaunch {pid} ({repo})")
    design = find_design(repo, pid)
    if not r.check(design is not None, f"design {design.name if design else ''}",
                   f"no PART{pid.upper()}-DESIGN.md (start from templates/PART-DESIGN.md)"):
        return r
    text = design.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:15])
    v1 = re.search(r"^Procedure: v1\b", head, re.M)
    if not v1:
        r.check(legacy_ok, "legacy design (no 'Procedure: v1'); checks skipped by --legacy-ok",
                "no 'Procedure: v1' line in the first 15 lines: a new design starts from templates/PART-DESIGN.md "
                "(an older design launches only with --legacy-ok)")
        return r
    r.check(re.search(r"^Registered \d{4}-\d{2}-\d{2} \d{2}:\d{2} CT", head, re.M) is not None,
            "Registered line with date and time", "no 'Registered YYYY-MM-DD HH:MM CT' line near the top")
    r.check("{{" not in text, "no unfilled template text", "unfilled template text: " +
            "; ".join(sorted(set(re.findall(r"\{\{[^}\n]{0,40}", text)))[:3]))
    sec = sections(text)
    missing = [s for s in SECTIONS if not sec.get(s)]
    r.check(not missing, f"all {len(SECTIONS)} sections present and filled",
            "missing or empty sections: " + ", ".join(missing))
    controls = sec.get("Controls", "")
    r.check(re.search(r"random|matched|Not used:", controls, re.I) is not None,
            "Controls name the random or matched controls (or say which are not used)",
            "Controls: no random or matched control and no 'Not used:' line")
    r.check(re.search(r"gate|positive control", controls, re.I) is not None,
            "Controls name a gate or positive control", "Controls: no 'Gate:' line (or 'No gate: <reason>')")
    count, unit, minutes = parse_size(sec.get("Size", ""))
    size_ok = r.check(count is not None and minutes is not None and count > 0 and minutes > 0,
                      f"Size: Total {count:,} {unit.strip()}, about {minutes:g} minutes" if count is not None
                      and minutes is not None else "",
                      "Size: no 'Total: N <unit>, about M minutes' line with N and M above zero")
    r.check(re.search(r"smallest effect", sec.get("Size", ""), re.I) is not None,
            "Size names the smallest effect worth knowing", "Size: no 'Smallest effect worth knowing' line")
    pilot = re.search(r"^\s*-?\s*Pilot:\s*(.*)$", sec.get("Size", ""), re.M | re.I)
    big = size_ok and (minutes > 60 or (count > 1000 and GENERATION_UNITS.search(unit) is not None))
    if not pilot:
        r.fail("Size: no 'Pilot:' line")
    elif big:
        r.check(not re.match(r"not needed", pilot.group(1), re.I), "Size: a pilot is planned for a big run",
                "Size: over 1,000 generations or one GPU-hour, but the pilot says 'not needed'")
    else:
        r.ok("Size: Pilot line present")
    opt = sec.get("Optimization", "")
    if size_ok and minutes > 5:
        r.check(re.search(r"Bound resource:\s*\S", opt) is not None, "Optimization names the bound resource",
                "Optimization: no 'Bound resource:' line")
        meas = re.search(r"Measured:\s*(.*)", opt)
        r.check(meas is not None and RATE.search(meas.group(1)) is not None and not
                re.match(r"\s*(n/?a|not measured|none)", meas.group(1), re.I),
                "Optimization: a measured throughput", "Optimization: a run over 5 minutes needs 'Measured:' with a "
                "rate on real rows (e.g. '2.0 answers/s at batch 64', or '6,736 rows in 145 s')")
        r.check(re.search(r"\b(taken|declined)\b", opt, re.I) is not None, "Optimization: levers taken or declined",
                "Optimization: no lever marked 'taken' or 'declined'")
    elif size_ok:
        r.ok("Optimization: run under 5 minutes, no measurement required")
    ids = pred_ids(sec.get("Predictions", ""), pid)
    r.check(bool(ids), f"Predictions: {len(ids)} numbered ({', '.join(ids)})",
            f"Predictions: no numbered predictions like {pid}.1")
    preds_path = repo / "PREDICTIONS.md"
    psec = predictions_section(preds_path.read_text(encoding="utf-8"), pid) if preds_path.exists() else ""
    r.check(bool(psec), "PREDICTIONS.md has the part's section", f"PREDICTIONS.md: no '## Part {pid}' section")
    absent = sorted(set(ids) - set(pred_ids(psec, pid)))
    r.check(not absent, "every prediction id is in PREDICTIONS.md", "not in PREDICTIONS.md: " + ", ".join(absent))
    stale = sorted(set(PLACEHOLDER_TIME.findall(text + "\n" + psec)))
    r.check(not stale, "no typed placeholder times", "placeholder times (use `date` or a message id): " +
            ", ".join(stale))
    if fetch:
        f = git(repo, "fetch", "-q", "origin", "main", check=False)
        if f.returncode:
            r.fail("git fetch origin main failed: " + f.stderr.strip()[:120])
    rel = design.relative_to(repo).as_posix()
    reg = git(repo, "show", f"origin/main:{rel}", check=False)
    if r.check(reg.returncode == 0, f"registered: {rel} is on origin/main",
               f"not registered: {rel} is not on origin/main (tools/register_public.py)"):
        r.check(reg.stdout == text, "registered: the design matches its copy on origin/main",
                "the design differs from its registered copy on origin/main: register the amendment before launching "
                f"(git diff origin/main -- {rel})")
    pub = git(repo, "show", "origin/main:PREDICTIONS.md", check=False)
    pub_sec = predictions_section(pub.stdout, pid) if pub.returncode == 0 else ""
    pub_missing = sorted(set(ids) - set(pred_ids(pub_sec, pid)))
    if r.check(bool(pub_sec) and not pub_missing, "registered: the predictions are on origin/main",
               "not registered: origin/main's PREDICTIONS.md lacks " + (", ".join(pub_missing) or f"Part {pid}")):
        r.check(pub_sec.strip() == psec.strip(), "registered: the part's PREDICTIONS.md section matches origin/main",
                "the part's PREDICTIONS.md section differs from origin/main: register the change before launching")
    return r


def cited_paths(text):
    """Path-like tokens: every whitespace-separated token inside backticks (so a backticked command's file arguments
    count), plus bare results/... and partNN/... paths in plain text."""
    toks = []
    for m in re.finditer(r"`([^`]+)`", text):
        toks += m.group(1).split()
    toks += re.findall(r"(?<![\w/`.-])((?:results|part\d+[a-z]?|tools|templates)/[\w./-]*[\w/])", text)
    for tok in toks:
        tok = tok.strip("\"'").rstrip(".,;:)")
        if not tok or not re.search(r"\w", tok) or tok.startswith(("http", "-")) or any(c in tok for c in "*{}<>$|=\\"):
            continue
        if "/" in tok or tok.endswith(PATH_EXT):
            yield tok


def verdict_rows(lines, pred_id):
    """Table rows whose first cell is exactly the id; returns their last non-empty cells."""
    out = []
    for ln in lines:
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip().strip("*").strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0].lower() == pred_id.lower():
            filled = [c for c in cells[1:] if c]
            out.append(filled[-1] if filled else "")
    return out


def publish(repo, pid, legacy_ok=False):
    r = Report(f"part_check publish {pid} ({repo})")
    design = find_design(repo, pid)
    v1 = design is not None and re.search(r"^Procedure: v1\b", "\n".join(
        design.read_text(encoding="utf-8").splitlines()[:15]), re.M)
    if not v1:
        r.check(legacy_ok, "legacy part; checks skipped by --legacy-ok",
                "the design has no 'Procedure: v1' line (an older part publishes only with --legacy-ok)")
        return r
    rdir = repo / "results" / f"part{pid.lower()}"
    readme = rdir / "README.md"
    if not r.check(readme.exists(), f"{readme.relative_to(repo)} exists", f"no {readme.relative_to(repo)} "
                   "(start from templates/RESULTS-README.md)"):
        return r
    text = readme.read_text(encoding="utf-8")
    r.check("{{" not in text, "no unfilled template text", "unfilled template text in the README")
    preds_path = repo / "PREDICTIONS.md"
    psec = predictions_section(preds_path.read_text(encoding="utf-8"), pid) if preds_path.exists() else ""
    ids = pred_ids(psec, pid)
    r.check(bool(ids), f"{len(ids)} registered predictions", f"PREDICTIONS.md: no ids under '## Part {pid}'")
    lines = text.splitlines()
    no_verdict = [i for i in ids if not any(VERDICT.search(cell) for cell in verdict_rows(lines, i))]
    r.check(not no_verdict, "every registered prediction has a verdict in the last cell of its table row",
            "no verdict (held, failed, mixed, untestable...) in the last cell of a table row whose first cell is: " +
            ", ".join(no_verdict))
    cold = re.search(r"^Cold check:\s*(.*)$", text, re.M | re.I)
    r.check(cold is not None and cold.group(1).strip() != "" and not
            re.match(r"(pending|todo|tbd|none|not yet)\b", cold.group(1).strip(), re.I),
            "a 'Cold check:' line recording a finished check",
            "no finished 'Cold check:' line (reader, date, findings, commit; 'pending' does not count)")
    local = sorted(set(m.group(0) for m in LOCAL_PATH.finditer(text)))
    stale = sorted(set(PLACEHOLDER_TIME.findall(text)))
    r.check(not stale, "no typed placeholder times", "placeholder times: " + ", ".join(stale))
    tracked = set(git(repo, "ls-files").stdout.splitlines())
    missing, uncommitted = [], []
    for tok in sorted(set(cited_paths(text))):
        if LOCAL_PATH.match(tok):
            continue
        if tok.startswith("/") or ".." in Path(tok).parts:
            local.append(tok)
            continue
        cands = [repo / tok, rdir / tok]
        hit = next((c for c in cands if c.exists()), None)
        if hit is None:
            named = [p for p in rdir.rglob(Path(tok).name)] if "/" not in tok else []
            if named:
                continue
            missing.append(tok)
            continue
        try:
            rel = hit.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            local.append(tok)
            continue
        if hit.is_dir():
            if not any(t.startswith(rel.rstrip("/") + "/") for t in tracked):
                uncommitted.append(tok)
        elif rel not in tracked and git(repo, "check-ignore", "-q", rel, check=False).returncode != 0:
            uncommitted.append(tok)
    r.check(not local, "no local or out-of-repo paths", "local or out-of-repo paths in a public README: " +
            ", ".join(sorted(set(local))[:8]))
    r.check(not missing, "every cited path exists", "cited but missing: " + ", ".join(missing[:8]))
    r.check(not uncommitted, "every cited path is committed (or gitignored and local by design)",
            "cited but not committed: " + ", ".join(uncommitted[:8]))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["prelaunch", "publish"])
    ap.add_argument("part")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--legacy-ok", action="store_true")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    if not re.fullmatch(r"\d+[a-z]?", a.part, re.I):
        print(f"part id must look like 115 or 109b, not {a.part!r}", file=sys.stderr)
        return 2
    rep = prelaunch(repo, a.part, fetch=not a.no_fetch, legacy_ok=a.legacy_ok) if a.stage == "prelaunch" else \
        publish(repo, a.part, legacy_ok=a.legacy_ok)
    rep.print()
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
