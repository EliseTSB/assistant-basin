#!/usr/bin/env python3
"""Download the pinned Qwen3.5-0.8B snapshot into the Windows job HF cache. (Design: Method)"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The Arc job exports offline mode globally; this downloader explicitly needs Hub access.
os.environ["HF_HUB_OFFLINE"] = "0"

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# Part 85's LOCAL-RUN.md / job.py sets HF_HOME=E:\neuron-probe\hf and the shared hub is its /hub child.
DEFAULT_HF_HOME = r"E:\neuron-probe\hf"


def download(hf_home=None):
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    home = str(hf_home or os.environ.get("HF_HOME") or DEFAULT_HF_HOME)
    cache_dir = str(Path(home) / "hub")
    retries = 0
    while True:
        try:
            path = snapshot_download(repo_id=MODEL_ID, revision=REVISION, cache_dir=cache_dir)
            break
        except Exception as exc:
            chain, current = [], exc
            while current is not None:
                chain.append(current)
                current = current.__cause__ or current.__context__
            reset = any(isinstance(error, ConnectionResetError) or
                        "connection reset" in str(error).casefold() for error in chain)
            if not reset or retries >= 3:
                raise
            delay = min(2 ** retries, 8)
            retries += 1
            print(f"connection reset while downloading; retry {retries}/3 after {delay}s",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
    return {"model": MODEL_ID, "revision": REVISION, "hf_home": home,
            "cache_dir": cache_dir, "snapshot": str(Path(path).resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", help="override the job folder's HF_HOME")
    args = parser.parse_args(argv)
    result = download(args.hf_home)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    print(result["snapshot"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
