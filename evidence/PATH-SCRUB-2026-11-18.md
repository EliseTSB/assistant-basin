# Account folder name removed from paths (2026-11-18)

On Elise's request, the Windows account folder name in recorded paths (`C:\Users\...`, `/mnt/c/Users/...`) was
replaced with `<user>` in 94 files: run scripts, logs, result files and notes. Nothing else in those files changed.

Three of those files are named by SHA-256 in other records, so their current bytes no longer match the recorded hash:

| file | recorded in |
|---|---|
| `part86/screen86.py` | `results/part86/pilot-l22-cbbfcae/manifest.json` and the `*.meta.json` files beside it |
| `results/part16/search.json` | `results/part31-full-*/run.json` |
| `results/part30/split.json` | `results/part30/adapter-*.json` |

The original bytes, which do match those hashes, are in this repository's history at commit `dfefb7bc` (the commit
before the change): `git show dfefb7bc:<path> | sha256sum`.
