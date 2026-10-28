#!/usr/bin/env python3
"""Part 85's registered OLMo 2 1B checkpoint ladder."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    key: str
    stage: str
    repo_id: str
    branch: str
    commit_prefix: str


BASE = "allenai/OLMo-2-0425-1B"
CHECKPOINTS = (
    Checkpoint("pretrain_21B", "pretraining, 21B tokens", BASE, "stage1-step10000-tokens21B", "22f7d18a8b40"),
    Checkpoint("pretrain_210B", "pretraining, 210B tokens", BASE, "stage1-step100000-tokens210B", "eb19ccabc193"),
    Checkpoint("pretrain_1T", "pretraining, 1.0T tokens", BASE, "stage1-step480000-tokens1007B", "eda09f034651"),
    Checkpoint("pretrain_2T", "pretraining, 2.0T tokens", BASE, "stage1-step950000-tokens1993B", "3e09b708b744"),
    Checkpoint("pretrain_3T", "pretraining, 3.0T tokens", BASE, "stage1-step1430000-tokens2999B", "22738a97eabc"),
    Checkpoint("pretrain_4T", "end of pretraining, 4.0T tokens", BASE, "stage1-step1907359-tokens4001B", "9d3e43659f00"),
    Checkpoint("midtrain_13B", "mid-training, 13B tokens", BASE, "stage2-ingredient3-step6000-tokens13B", "a7a789f6b27b"),
    Checkpoint("midtrain_26B", "mid-training, 26B tokens", BASE, "stage2-ingredient3-step12000-tokens26B", "1c3599f7b404"),
    Checkpoint("midtrain_51B", "mid-training end (SFT parent)", BASE, "stage2-ingredient3-step23852-tokens51B", "f83cd641d4f0"),
    Checkpoint("sft", "SFT", BASE + "-SFT", "main", "0d85a3d03787"),
    Checkpoint("dpo", "DPO", BASE + "-DPO", "main", "c4b0485961ab"),
    Checkpoint("instruct", "Instruct", BASE + "-Instruct", "main", "48d788eca847"),
)
BY_KEY = {cp.key: cp for cp in CHECKPOINTS}
LOCAL_COMMITS = Path(__file__).resolve().with_name("commits.local.json")
POST_TRAINED = ("sft", "dpo", "instruct")
PILOT_KEYS = ("pretrain_4T", "instruct")


def resolve_checkpoint(checkpoint: Checkpoint | str, api=None) -> dict:
    """Resolve a branch to its immutable full commit and enforce the registered prefix."""
    cp = BY_KEY[checkpoint] if isinstance(checkpoint, str) else checkpoint
    if api is None and LOCAL_COMMITS.is_file():
        # Written by download85.py from the Hub at download time; the A770 job runner is offline.
        sha = json.loads(LOCAL_COMMITS.read_text(encoding="utf-8")).get(cp.key)
        if not isinstance(sha, str) or len(sha) != 40 or not sha.startswith(cp.commit_prefix):
            raise RuntimeError(f"{LOCAL_COMMITS.name} has no full commit with prefix {cp.commit_prefix} for {cp.key}")
        return {**asdict(cp), "commit": sha}
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.model_info(cp.repo_id, revision=cp.branch)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or len(sha) != 40:
        raise RuntimeError(f"Hugging Face returned no full commit for {cp.key}@{cp.branch}: {sha!r}")
    if not sha.startswith(cp.commit_prefix):
        raise RuntimeError(
            f"{cp.key} resolved to {sha}, which does not start with design prefix {cp.commit_prefix}"
        )
    return {**asdict(cp), "commit": sha}


def resolve_all(keys=None, api=None) -> list[dict]:
    selected = list(CHECKPOINTS if keys is None else (BY_KEY[k] for k in keys))
    return [resolve_checkpoint(cp, api=api) for cp in selected]


def checkpoint_tag(checkpoint: Checkpoint | dict | str) -> str:
    if isinstance(checkpoint, str):
        return BY_KEY[checkpoint].key
    return checkpoint.key if isinstance(checkpoint, Checkpoint) else checkpoint["key"]


def validate_snapshot_commit(snapshot_dir: str | Path, commit: str) -> Path:
    """Require local snapshot provenance from its HF cache name or config metadata.

    Hugging Face cache snapshots are stored as ``snapshots/<full-commit>``.
    For a copied/plain directory, config.json must retain its ``_commit_hash``.
    A directory without either signal is rejected rather than guessed.
    """
    root = Path(snapshot_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"snapshot directory does not exist: {root}")
    if root.name == commit:
        return root
    config = root / "config.json"
    if config.is_file():
        try:
            metadata = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read snapshot config provenance at {config}: {exc}") from exc
        recorded = metadata.get("_commit_hash")
        if recorded == commit:
            return root
    raise ValueError(f"snapshot directory {root} does not identify pinned commit {commit}; "
                     "use an HF snapshots/<commit> path or a plain directory with config.json _commit_hash")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="*", choices=tuple(BY_KEY), help="resolve only these checkpoints")
    args = parser.parse_args(argv)
    print(json.dumps(resolve_all(args.keys), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
