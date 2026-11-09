"""loop_onset_envelope.py: what each published loop contrast could have been if the faulty kernel only ADDED loops.

Reads the loop-onset-*.jsonl outputs of loop_onset.py. In each arm, loops established before t0 are kernel-invariant;
the arm's exposed loops (established at or after t0) are, under the fabrication-only assumption, somewhere between none
and all of them genuine. So the correct-kernel count of an arm is early + x with x in [0, exposed]. Each contrast is
reported at its published value, at the early-only point, and as the interval over every such allocation in both arms.
The assumption excludes the other direction: a correct kernel producing MORE late loops than the faulty one did.
Run with any python3 from this directory: python3 loop_onset_envelope.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def arm(part, match):
    loops = early = 0
    for line in (HERE / f"loop-onset-{part}.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "cell" in row and match(row["cell"]):
            loops += row["loops"]
            early += row["before"]
    return early, loops  # correct-kernel count lies in [early, loops] under fabrication only


def cut(none, ban):
    """Relative cut (none - ban) / none, published, early-only and its interval."""
    (e0, p0), (e1, p1) = none, ban
    pub = (p0 - p1) / p0
    early = (e0 - e1) / e0
    lo = (e0 - p1) / e0          # baseline at its smallest, banned arm at its largest
    hi = (p0 - e1) / p0          # baseline at its largest, banned arm at its smallest
    return pub, early, lo, hi


def diff(base, alt, n=None):
    """Difference alt - base in loops (or rate over n rows)."""
    (e0, p0), (e1, p1) = base, alt
    s = (lambda v: v / n) if n else (lambda v: v)
    return s(p1 - p0), s(e1 - e0), s(e1 - p0), s(p1 - e0)


def low_k(part, ban):
    return arm(part, lambda c: c.get("stage") == "battery" and c.get("arm") in ("n+", "k4") and c.get("ban") == ban)


def p170(dose, ban):
    return arm(170, lambda c: c.get("stage") == "push" and c.get("dose") == dose and c.get("ban") == ban)


def show(label, vals, pct):
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:+.0f}")
    pub, early, lo, hi = vals
    print(f"{label}: published {f(pub)} | early-only {f(early)} | fabrication-only interval {f(lo)} to {f(hi)}")


if __name__ == "__main__":
    show("168 own-word ban cut (n+ and k4)", cut(low_k(168, "none"), low_k(168, "own")), True)
    show("169 own-word ban cut (n+ and k4)", cut(low_k(169, "none"), low_k(169, "own")), True)
    show("169 plain50 minus none, loops", diff(low_k(169, "none"), low_k(169, "plain50")), False)
    show("169 free minus none, loops", diff(low_k(169, "none"), low_k(169, "free")), False)
    for dose in (0.25, 0.5, 1.0):
        show(f"170 dose {dose} b50 minus b0, rate over 384", diff(p170(dose, "b0"), p170(dose, "b50"), 384), True)
