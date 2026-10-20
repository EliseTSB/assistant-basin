#!/usr/bin/env python3
"""Unrun Part 31 judge integration stub.

Input is produced by python part31.py judge-prep. This file intentionally
does not call judge_client.call_prompt: a researcher must review the prepared
sentences and explicitly complete the prompt, route, model, and label handling.
"""
from __future__ import annotations

import argparse


def judge_one(sentence: str, *, route: str = "cliproxy", model: str = "gpt-6-luna") -> str:
    """Placeholder following judge_client.call_prompt conventions; never calls it."""
    raise NotImplementedError(
        "Unrun stub only. Review judge-prep.jsonl, then explicitly implement and authorize a judge call."
    )
    # Example integration after human review:
    # from judge_client import call_prompt
    # result = call_prompt(prompt, model=model, route=route, session="part31-persona-judge")
    # return result["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="out/part31/judge_prep.jsonl")
    parser.add_argument("--output", default="out/part31/judge_labels.jsonl")
    parser.parse_args()
    raise SystemExit("This is an unrun stub. It makes no paid or local judge API calls.")


if __name__ == "__main__":
    main()
