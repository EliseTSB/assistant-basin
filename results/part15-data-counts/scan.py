#!/usr/bin/env python3
"""assistant-basin: identity-string counts across OLMo 2/3 training stages.

Two modes:
  parquet — full exact scans of the datasets-server parquet conversion (small sets)
  raw     — bounded streaming samples from the released source files (big mixes)

Writes one JSON per scan unit under out/. The CSV is built from those JSONs.
"""
import json, re, sys, time, argparse
from pathlib import Path

sys.path.insert(0, "/tmp/opencode/assistant-basin/trainspotting")

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "out").mkdir(exist_ok=True)
LOG = open(OUT / "scan.log", "a")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()

APO = "['’]"
STRINGS = [
    ("I am ChatGPT", "i am chatgpt"),
    ("I'm ChatGPT", f"i{APO}m chatgpt"),
    ("as ChatGPT", "as chatgpt"),
    ("developed by OpenAI", "developed by openai"),
    ("I am Claude", "i am claude"),
    ("I'm Claude", f"i{APO}m claude"),
    ("created by Anthropic", "created by anthropic"),
    ("I am OLMo", "i am olmo"),
    ("made by Ai2 / Allen Institute", "(made by ai2|allen institute)"),
    ("as an AI language model", "as an ai language model"),
    ("I am an AI assistant", "i am an ai assistant"),
]
UNION = "(?:" + "|".join(f"(?:{p})" for _, p in STRINGS) + ")"
PY_PATTERNS = [(label, re.compile(p, re.IGNORECASE)) for label, p in STRINGS]
PY_UNION = re.compile(UNION, re.IGNORECASE)

def empty_counts():
    return {label: 0 for label, _ in STRINGS}

def sql_lit(s):
    return "'" + s.replace("'", "''") + "'"

# ---------------- parquet mode ----------------

PARQUET_PLAN = [
    ("olmo2", "sft", "allenai/tulu-3-sft-olmo-2-mixture", "sft"),
    ("olmo2", "dpo", "allenai/olmo-2-1124-7b-preference-mix", "dpo"),
    ("olmo3", "pretrain", "allenai/dolma3_mix-6T-1025-7B", "text"),
    ("olmo3", "sft", "allenai/Dolci-Instruct-SFT", "sft"),
    ("olmo3", "dpo", "allenai/Dolci-Instruct-DPO", "dpo"),
]

def exprs_for(kind, cols):
    if kind == "text":
        return {"any": "text", "resp": None, "chosen": None, "rejected": None}
    if kind == "sft":
        any_e = "array_to_string(list_transform(messages, m -> coalesce(m.content, '')), chr(10))"
        resp_e = ("array_to_string(list_transform(list_filter(messages, "
                  "m -> lower(coalesce(m.role, '')) = 'assistant'), m -> coalesce(m.content, '')), chr(10))")
        return {"any": any_e, "resp": resp_e, "chosen": None, "rejected": None}
    if kind == "dpo":
        ch = "array_to_string(list_transform(chosen, m -> coalesce(m.content, '')), chr(10))"
        rj = "array_to_string(list_transform(rejected, m -> coalesce(m.content, '')), chr(10))"
        prompt = "coalesce(prompt, '')" if "prompt" in cols else "''"
        return {"any": f"concat({prompt}, chr(10), {ch}, chr(10), {rj})", "resp": f"concat({ch}, chr(10), {rj})",
                "chosen": ch, "rejected": rj}
    raise ValueError(kind)

def scan_parquet(only=None):
    from trainspotting import grep
    con = grep.connect()
    for family, stage, dataset, kind in PARQUET_PLAN:
        key = dataset.split("/")[-1]
        if only and only not in key:
            continue
        outfile = OUT / "out" / f"{family}.{stage}.{key}.json"
        if outfile.exists():
            log(f"skip {key} (done)")
            continue
        log(f"START parquet scan: {dataset} ({kind})")
        try:
            listing = grep.parquet_listing(dataset)
            urls = listing["urls"]
            cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_lit(urls[0])})").fetchall()]
            ex = exprs_for(kind, cols)
            agg = {"docs": 0, "chars": 0, "counts": empty_counts(), "union": 0,
                   "resp_chars": 0, "resp_counts": empty_counts(), "resp_union": 0,
                   "chosen_union": 0, "rejected_union": 0}
            errors = []
            for i, url in enumerate(urls):
                t0 = time.time()
                selects = [f"count(*) AS n", f"sum(length({ex['any']})) AS chars"]
                for label, p in STRINGS:
                    selects.append(f"count(*) FILTER (WHERE regexp_matches({ex['any']}, {sql_lit('(?i)' + p)})) AS c_{len(selects)}")
                selects.append(f"count(*) FILTER (WHERE regexp_matches({ex['any']}, {sql_lit('(?i)' + UNION)})) AS cu")
                if ex["resp"]:
                    selects.append(f"sum(length({ex['resp']})) AS rchars")
                    for label, p in STRINGS:
                        selects.append(f"count(*) FILTER (WHERE regexp_matches({ex['resp']}, {sql_lit('(?i)' + p)})) AS r_{len(selects)}")
                    selects.append(f"count(*) FILTER (WHERE regexp_matches({ex['resp']}, {sql_lit('(?i)' + UNION)})) AS ru")
                if ex["chosen"]:
                    selects.append(f"count(*) FILTER (WHERE regexp_matches({ex['chosen']}, {sql_lit('(?i)' + UNION)})) AS chu")
                    selects.append(f"count(*) FILTER (WHERE regexp_matches({ex['rejected']}, {sql_lit('(?i)' + UNION)})) AS rju")
                q = f"SELECT {', '.join(selects)} FROM read_parquet({sql_lit(url)})"
                row = con.execute(q).fetchone()
                agg["docs"] += row[0] or 0
                agg["chars"] += row[1] or 0
                for j, (label, _) in enumerate(STRINGS):
                    agg["counts"][label] += row[2 + j] or 0
                idx = 2 + len(STRINGS)
                agg["union"] += row[idx] or 0
                idx += 1
                if ex["resp"]:
                    agg["resp_chars"] += row[idx] or 0; idx += 1
                    for j, (label, _) in enumerate(STRINGS):
                        agg["resp_counts"][label] += row[idx + j] or 0
                    idx += len(STRINGS)
                    agg["resp_union"] += row[idx] or 0; idx += 1
                if ex["chosen"]:
                    agg["chosen_union"] += row[idx] or 0; idx += 1
                    agg["rejected_union"] += row[idx] or 0; idx += 1
                log(f"  file {i+1}/{len(urls)}: {row[0]:,} rows ({time.time()-t0:.0f}s)")
            payload = {"family": family, "stage": stage, "dataset": dataset, "kind": kind,
                       "method": "parquet-full", "revision": listing["revision"],
                       "files": len(urls), "partial": listing["partial"], **agg, "errors": errors}
            outfile.write_text(json.dumps(payload, indent=1))
            log(f"DONE {key}: {agg['docs']:,} docs, union={agg['union']:,}")
        except Exception as e:
            log(f"ERROR {key}: {type(e).__name__}: {str(e)[:300]}")

# ---------------- raw mode ----------------

RAW_PLAN = {
    "allenai/olmo-mix-1124": {"family": "olmo2", "stage": "pretrain", "sources": {
        "dclm": ("data/dclm", 3, 8000), "starcoder": ("data/starcoder", 3, 8000),
        "pes2o": ("data/pes2o", 3, 8000), "algebraic-stack": ("data/algebraic-stack", 3, 8000),
        "arxiv": ("data/arxiv", 3, 8000), "open-web-math": ("data/open-web-math", 2, 8000),
        "wiki": ("data/wiki", 2, 8000)}},
    "allenai/dolmino-mix-1124": {"family": "olmo2", "stage": "midtrain", "sources": {
        "dclm": ("data/dclm", 3, 8000), "flan": ("data/flan", 2, 8000),
        "math": ("data/math", 3, 8000), "pes2o": ("data/pes2o", 2, 8000),
        "stackexchange": ("data/stackexchange", 2, 8000), "wiki": ("data/wiki", 1, 8000)}},
    "allenai/dolma3_dolmino_mix-100B-1025": {"family": "olmo3", "stage": "midtrain", "sources": {
        "code-meta-reasoning": ("data/code-meta-reasoning", 1, 5000),
        "cranecode": ("data/cranecode", 1, 5000), "cranemath": ("data/cranemath", 1, 5000),
        "dolmino-math": ("data/dolmino-math", 1, 5000), "dolmino_1-flan": ("data/dolmino_1-flan", 1, 5000),
        "gemini-reasoning-traces": ("data/gemini-reasoning-traces", 1, 5000),
        "general_reasoning_mix": ("data/general_reasoning_mix", 1, 5000),
        "megamatt": ("data/megamatt", 1, 5000), "nemotron-synth-qa": ("data/nemotron-synth-qa", 1, 5000),
        "openthoughts2-reasoning-traces": ("data/openthoughts2-reasoning-traces", 1, 5000),
        "program_verifiable": ("data/program_verifiable", 1, 5000),
        "qwq-reasoning-traces": ("data/qwq-reasoning-traces", 1, 5000),
        "stem-heavy-crawl": ("data/stem-heavy-crawl", 1, 5000),
        "tulu-3-sft": ("data/tulu-3-sft", 1, 5000),
        "wiki_to_rcqa-part1": ("data/wiki_to_rcqa-part1", 1, 5000),
        "cc_finance": ("data/common_crawl-high-quality_19_finance_and_business", 1, 5000),
        "cc_software": ("data/common_crawl-high-quality_20_software", 1, 5000)}},
}

import urllib.request
def hf_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "assistant-basin/0.1"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())

def walk_files(dataset, root, max_files=60, max_depth=8):
    """Collect up to max_files file paths under root (depth-first, breadth-limited)."""
    files = []
    stack = [(root, 0)]
    seen = 0
    while stack and len(files) < max_files and seen < 400:
        path, depth = stack.pop(0)
        seen += 1
        try:
            t = hf_get(f"https://huggingface.co/api/datasets/{dataset}/tree/main/{path}?limit=200")
        except Exception:
            continue
        for e in t:
            if e.get("type") == "file":
                files.append(e["path"])
                if len(files) >= max_files:
                    break
            elif e.get("type") == "directory" and depth < max_depth:
                stack.append((e["path"], depth + 1))
    return sorted(files)

def pick_spread(files, k):
    if len(files) <= k:
        return files
    idxs = [round(i * (len(files) - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    return [files[i] for i in dict.fromkeys(idxs)]

def count_doc(text):
    counts = {}
    hit_any = False
    for label, pat in PY_PATTERNS:
        if pat.search(text):
            counts[label] = 1
            hit_any = True
    return counts, hit_any

def sample_raw(only=None):
    from datasets import load_dataset
    for dataset, spec in RAW_PLAN.items():
        family, stage = spec["family"], spec["stage"]
        key = dataset.split("/")[-1]
        for src, (root, nfiles, per_file) in spec["sources"].items():
            if only:
                full = f"{key}/{src}"
                if ("/" in only and only not in full) or ("/" not in only and only not in key):
                    continue
            outfile = OUT / "out" / f"{family}.{stage}.{key}.{src}.json"
            if outfile.exists():
                log(f"skip {key}/{src} (done)")
                continue
            log(f"START raw sample: {key}/{src} ({root})")
            try:
                files = walk_files(dataset, root)
                if not files:
                    log(f"  no files under {root}")
                    continue
                picked = pick_spread(files, nfiles)
                agg = {"docs": 0, "chars": 0, "counts": empty_counts(), "union": 0}
                errors = []
                for path in picked:
                    ext = path.rsplit(".", 1)[-1].lower()
                    loader = "parquet" if ext == "parquet" else "json"
                    t0 = time.time()
                    try:
                        d = load_dataset(loader, data_files=f"hf://datasets/{dataset}/{path}", split="train", streaming=True)
                        n = 0
                        for row in d:
                            text = row.get("text")
                            if text is None:
                                for alt in ("content", "raw_content", "document"):
                                    if isinstance(row.get(alt), str):
                                        text = row[alt]; break
                            if not isinstance(text, str):
                                text = ""
                            agg["chars"] += len(text)
                            counts, hit = count_doc(text)
                            for label in counts:
                                agg["counts"][label] += 1
                            if hit:
                                agg["union"] += 1
                            n += 1
                            if n >= per_file:
                                break
                        agg["docs"] += n
                        log(f"  {path.split('/')[-1][:60]}: {n:,} docs ({time.time()-t0:.0f}s)")
                    except Exception as e:
                        errors.append(f"{path}: {type(e).__name__}: {str(e)[:150]}")
                        log(f"  ERR {path.split('/')[-1][:50]}: {str(e)[:120]}")
                payload = {"family": family, "stage": stage, "dataset": dataset, "source": src,
                           "kind": "text", "method": "raw-sample", "files_sampled": picked,
                           "total_files_seen": len(files), "per_file_limit": per_file, **agg, "errors": errors}
                outfile.write_text(json.dumps(payload, indent=1))
                log(f"DONE {key}/{src}: {agg['docs']:,} docs, union={agg['union']:,}")
            except Exception as e:
                log(f"ERROR {key}/{src}: {type(e).__name__}: {str(e)[:300]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parquet", "raw"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.mode == "parquet":
        scan_parquet(args.only)
    else:
        sample_raw(args.only)
    log("MODE DONE: " + args.mode)
