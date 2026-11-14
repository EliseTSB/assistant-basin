#!/usr/bin/env python3
"""checkpoints211.py: the checkpoint table of PART211-DESIGN.md (Method) and the host-side pinning of each revision.

Six Pythia-160m runs x twelve steps (design: Method, Pythia-160m; the weight arm is the two corrected weight-seed
runs) and OLMo 2 1B's fourteen (design: Method, OLMo 2 1B). Every entry names a repository and a hub branch; `pin()` asks the Hugging Face API for each branch's full commit
hash and writes part211/checkpoints.json. `pin()` needs the network and is run on the host; here it is tested against a
fake HTTP function (test_part211.py). The OLMo twelve come from part85/checkpoints.py, which is imported, not copied.

Pythia's branch names are `step<N>`; the run name is the `step<N>` of the model card's own run (the final step is
143000 for the base run and for every seed run).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import part85.checkpoints as P85  # noqa: E402

STEPS = (0, 128, 512, 1000, 2000, 4000, 8000, 10000, 16000, 32000, 64000, 143000)
FINAL_STEP = STEPS[-1]
THREAD_STEP = 10000            # the thread's final step (design: Pythia-160m)
OLMO_EXTRA = (("olmo_step0", "pretraining step 0 (stage 1, 0 tokens)", "stage1-step0-tokens0B"),
              ("olmo_step300", "pretraining step 300 (stage 1, 1B tokens)", "stage1-step300-tokens1B"))
OLMO_REPO = "allenai/OLMo-2-0425-1B"

# run key -> (hub repository, seed factor, description). R0 first; W runs vary initial weights, D runs data order.
# Run counts are derived from this table (runs_of), never typed.
PYTHIA_RUNS = (
    ("R0", "EleutherAI/pythia-160m", "reference", "seed 1234"),
    # design (2026-11-13 change): the uncorrected weight-seed releases do not vary initialisation (their card says a
    # training error continued them from the original Pythia weights); the weight arm is the two corrected runs.
    ("W1", "EleutherAI/pythia-160m-weight-seed1-corrected", "weights", "data order fixed, initial weights varied"),
    ("W2", "EleutherAI/pythia-160m-weight-seed2-corrected", "weights", "data order fixed, initial weights varied"),
    ("D1", "EleutherAI/pythia-160m-data-seed1", "data", "initial weights fixed, data order varied"),
    ("D2", "EleutherAI/pythia-160m-data-seed2", "data", "initial weights fixed, data order varied"),
    ("D3", "EleutherAI/pythia-160m-data-seed3", "data", "initial weights fixed, data order varied"),
)
PYTHIA_FAMILY = "pythia"
OLMO_FAMILY = "olmo"
# design: Storage and memory. Pythia: each run's step 0, step 10000 and step 143000 keep their attention rows on disk.
PYTHIA_REFERENCE_STEPS = (0, THREAD_STEP, FINAL_STEP)
# design: Storage and memory. OLMo: step 0, pretraining 3T, 4T, mid-training end (51B) and Instruct.
OLMO_REFERENCE_KEYS = ("olmo_step0", "pretrain_3T", "pretrain_4T", "midtrain_51B", "instruct")


def runs_of(family: str = PYTHIA_FAMILY) -> tuple[str, ...]:
    """The run keys of a family, in table order (R0 first). Every counting rule reads its run count from here."""
    seen = []
    for cp in CHECKPOINTS:
        if cp["family"] == family and cp["run"] not in seen:
            seen.append(cp["run"])
    return tuple(seen)


def runs_with_factor(factor: str, family: str = PYTHIA_FAMILY) -> tuple[str, ...]:
    """The runs of one seed factor ('weights', 'data', 'reference')."""
    return tuple(r for r in runs_of(family) if BY_RUN[r]["seed_factor"] == factor)


def pythia_key(run: str, step: int) -> str:
    return f"{run}_step{step}"


def checkpoint_table() -> list[dict]:
    """Every checkpoint, in reading order: the six Pythia runs (R0 first) x twelve steps, then OLMo's fourteen."""
    out = []
    for run, repo_id, factor, desc in PYTHIA_RUNS:
        for step in STEPS:
            out.append({"key": pythia_key(run, step), "family": PYTHIA_FAMILY, "run": run, "repo_id": repo_id,
                        "branch": f"step{step}", "step": step, "seed_factor": factor, "description": desc,
                        "reference": step in PYTHIA_REFERENCE_STEPS, "commit_prefix": None})
    # design: OLMo 2 1B, 14 checkpoints: step 0, step 300, then part85's twelve at their registered commit prefixes
    out.append({"key": OLMO_EXTRA[0][0], "family": OLMO_FAMILY, "run": "O0", "repo_id": OLMO_REPO,
                "branch": OLMO_EXTRA[0][2], "step": 0, "seed_factor": None, "description": OLMO_EXTRA[0][1],
                "reference": True, "commit_prefix": None})
    out.append({"key": OLMO_EXTRA[1][0], "family": OLMO_FAMILY, "run": "O0", "repo_id": OLMO_REPO,
                "branch": OLMO_EXTRA[1][2], "step": 300, "seed_factor": None, "description": OLMO_EXTRA[1][1],
                "reference": False, "commit_prefix": None})
    for cp in P85.CHECKPOINTS:
        out.append({"key": f"olmo_{cp.key}", "family": OLMO_FAMILY, "run": "O0", "repo_id": cp.repo_id,
                    "branch": cp.branch, "step": None, "seed_factor": None, "description": cp.stage,
                    "reference": cp.key in OLMO_REFERENCE_KEYS, "commit_prefix": cp.commit_prefix})
    return out


CHECKPOINTS = checkpoint_table()
BY_KEY = {cp["key"]: cp for cp in CHECKPOINTS}
BY_RUN = {run: next(cp for cp in CHECKPOINTS if cp["run"] == run) for run in runs_of() + runs_of(OLMO_FAMILY)}
PYTHIA_KEYS = tuple(cp["key"] for cp in CHECKPOINTS if cp["family"] == PYTHIA_FAMILY)
OLMO_KEYS = tuple(cp["key"] for cp in CHECKPOINTS if cp["family"] == OLMO_FAMILY)
OUT_PATH = HERE / "checkpoints.json"


def resolve_commit(cp: dict, api) -> str:
    """The full commit hash for one checkpoint's branch, from `api.model_info(repo_id, revision=branch)`.

    `api` is anything with that method (huggingface_hub.HfApi on the host, a fake in the tests). A registered commit
    prefix (part85's OLMo twelve) must match; a returned value that is not a 40-character hash is refused.
    """
    info = api.model_info(cp["repo_id"], revision=cp["branch"])
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise RuntimeError(f"Hugging Face returned no full commit for {cp['key']}@{cp['branch']}: {sha!r}")
    prefix = cp.get("commit_prefix")
    if prefix and not sha.startswith(prefix):
        raise RuntimeError(f"{cp['key']} resolved to {sha}, which does not start with the registered prefix {prefix}")
    return sha


def pin(api=None, path=None) -> dict:
    """Ask the Hugging Face API for every revision's commit hash and write checkpoints.json. Host only (network)."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    out = {"schema": 1, "pinned": {}, "checkpoints": []}
    for cp in CHECKPOINTS:
        commit = resolve_commit(cp, api)
        out["pinned"][cp["key"]] = commit
        out["checkpoints"].append({**cp, "commit": commit})
    path = Path(path or OUT_PATH)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return out


def load_pinned(path=None) -> dict:
    """The pinned table; raises if a checkpoint is missing, so a run cannot start half-pinned."""
    path = Path(path or OUT_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `python -I -B part211/checkpoints211.py pin` on the host first")
    pinned = json.loads(path.read_text(encoding="utf-8"))["pinned"]
    missing = [cp["key"] for cp in CHECKPOINTS if cp["key"] not in pinned]
    if missing:
        raise RuntimeError(f"{path} lacks commits for {len(missing)} checkpoints: {missing[:5]}")
    return pinned


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["pin", "list"])
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args(argv)
    if a.command == "pin":
        out = pin(path=a.out)
        print(f"pinned {len(out['pinned'])} checkpoints -> {a.out}")
    else:
        print(json.dumps(CHECKPOINTS, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
