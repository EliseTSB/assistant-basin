"""Submit the full atlas labels one chunk at a time, with no retry and no recovery path (Claude, 2026-10-22).

Why this exists: the round-3 reviews of atlas_label.py's multi-chunk `submit` (reviews/cheap-part56-fold2-*-r3.md in the Journal repo)
found that the --recover path could pay for a chunk twice. This wrapper makes each chunk its own fresh `submit` into its own
output directory, runs them strictly in order, and STOPS on the first failure. A chunk whose directory already exists is never
submitted again: it is skipped with its recorded batch id, or the run refuses and says why. A crash between the POST and the
manifest write therefore leaves a directory with no batch id: the operator (a person) lists the batches in the Console and decides.
Usage: atlas_label_chunks.py submit --records A.jsonl B.jsonl ... --out DIR [--chunk-size 10000] [--cap-usd 2.5] [--dry-run]
       atlas_label_chunks.py status --out DIR            (batch id and state per chunk)
       atlas_label_chunks.py collect --out DIR           (collect every ended chunk; never resubmits)
Needs ANTHROPIC_CREDITS_KEY in the environment (never printed)."""
import argparse, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL = os.environ.get("ATLAS_LABEL_PY", str(HERE / "atlas_label.py"))
PY = sys.executable


def chunk_files(records, out, size):
    """Stream the record files into numbered chunk files; return their paths. Deterministic by order and size."""
    out.mkdir(parents=True, exist_ok=True)
    paths, cur, n, idx = [], None, 0, 0
    for rec in records:
        with open(rec, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if cur is None or n >= size:
                    if cur:
                        cur.close()
                    idx += 1
                    p = out / f"chunk-{idx:03d}.jsonl"
                    if not p.exists():
                        cur = open(p, "w", encoding="utf-8")
                    else:
                        cur = open(os.devnull, "w")  # existing chunk file: keep it, do not rewrite
                    paths.append(p)
                    n = 0
                cur.write(line if line.endswith("\n") else line + "\n")
                n += 1
    if cur:
        cur.close()
    return paths


def chunk_dir(out, p):
    return out / ("label-" + p.stem)


def batch_id_of(d):
    m = d / "batches.json"
    if not m.exists():
        return None
    ids = [b.get("batch_id") for b in json.loads(m.read_text()).get("batches", [])]
    return ids[0] if ids and ids[0] else None


def submit(args):
    out = Path(args.out)
    paths = chunk_files(args.records, out, args.chunk_size)
    print(f"{len(paths)} chunks of up to {args.chunk_size} records")
    for p in paths:
        d = chunk_dir(out, p)
        if d.exists():
            bid = batch_id_of(d)
            if bid:
                print(f"{p.stem}: already submitted as {bid}; skipping")
                continue
            print(f"{p.stem}: directory {d} exists with NO batch id (a dry run, or a crash in the POST window).")
            print("  Not submitting again. List the batches in the Console, then either delete this directory (if no batch was created) or write its batch id into batches.json by hand.")
            return 3
        cmd = [PY, LABEL, "submit", "--records", str(p), "--out", str(d), "--spend-cap-usd", str(args.cap_usd)]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"{p.stem}: submitting ({' '.join(cmd[2:4])}...)", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"{p.stem}: submit exited {r.returncode}; STOPPING. Nothing after this chunk was sent. Do not rerun until a person has checked the Console.")
            return 2
        print(f"{p.stem}: batch {batch_id_of(d)}")
    return 0


def each(out, fn):
    out = Path(out)
    for d in sorted(out.glob("label-chunk-*")):
        fn(d)


def status(args):
    def one(d):
        bid = batch_id_of(d)
        if not bid:
            print(f"{d.name}: no batch id")
            return
        r = subprocess.run([PY, LABEL, "status", bid], capture_output=True, text=True)
        print(f"{d.name}: {bid} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-120:]}")
    each(args.out, one)
    return 0


def collect(args):
    def one(d):
        bid = batch_id_of(d)
        if not bid:
            print(f"{d.name}: no batch id; skipped")
            return
        r = subprocess.run([PY, LABEL, "status", bid], capture_output=True, text=True)
        if not r.stdout.strip().startswith("ended"):
            print(f"{d.name}: not ended: {r.stdout.strip()[:80]}")
            return
        if list(d.glob("atlas-labels*.jsonl")):
            print(f"{d.name}: already collected")
            return
        s = subprocess.run([PY, LABEL, "collect", "--out", str(d)], capture_output=True, text=True)
        print(f"{d.name}: {s.stdout.strip().splitlines()[0] if s.stdout.strip() else s.stderr.strip()[-120:]}")
    each(args.out, one)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--records", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--chunk-size", type=int, default=10000)
    s.add_argument("--cap-usd", type=float, default=2.5)
    s.add_argument("--dry-run", action="store_true")
    for name in ("status", "collect"):
        t = sub.add_parser(name)
        t.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"submit": submit, "status": status, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
