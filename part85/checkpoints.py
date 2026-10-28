#!/usr/bin/env python3
"""Part 85's registered OLMo 2 1B checkpoint ladder."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


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
POST_TRAINED = ("sft", "dpo", "instruct")
PILOT_KEYS = ("pretrain_4T", "instruct")


def resolve_checkpoint(checkpoint: Checkpoint | str, api=None) -> dict:
    """Resolve a branch to its immutable full commit and enforce the registered prefix."""
    cp = BY_KEY[checkpoint] if isinstance(checkpoint, str) else checkpoint
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="*", choices=tuple(BY_KEY), help="resolve only these checkpoints")
    args = parser.parse_args(argv)
    print(json.dumps(resolve_all(args.keys), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
