#!/usr/bin/env python3
"""audit_c3_label.py — Dual blind-reader labeling harness for Stage C3 (WSP v1).

Imports the frozen blind-reader prompt, JSON parser, and agreement metrics from `c3_scoring.py`,
strips all arm/dose/ban/split metadata via deterministic seeded blinding, runs the 14-item disjoint
`READER_GOLD_14` pre-check first, and collects independent binary judgments (`pirate_persona`, `coherent`)
from two blind readers:
  - `ds`:   `deepseek-v4.1-flash` via `ocgo`
  - `luna`: `gpt-6-luna` via `cliproxy`

Resumable via per-reader JSONL logs (`labels-ds.jsonl`, `labels-luna.jsonl`).
Supports a pluggable `reader_fn` for 0-cost offline unit testing (stamped `mock_readers: true` so the
production scorer refuses mock labels unless explicitly enabled in tests).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import c3_scoring  # noqa: E402

READER_LABELS_SCHEMA_VERSION = "olmo3-stage-c3-reader-labels-v1"
BLIND_SHUFFLE_SEED = 2026112699

READERS: dict[str, dict[str, Any]] = {
    "ds": {"model": "deepseek-v4.1-flash", "route": "ocgo", "session": "olmo3-c3-audit-ds", "workers": 2},
    "luna": {"model": "gpt-6-luna", "route": "cliproxy", "session": "olmo3-c3-audit-luna", "workers": 3},
}
MAX_TOKENS = 4000  # DeepSeek thinks first; lower cap truncates before JSON (Part 218 lesson)
ATTEMPTS = 3

ReaderCallable = Callable[[str, str, str], tuple[dict[str, bool] | None, str | None]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_or_gz(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def build_blind_prompt(prompt: str, answer: str) -> str:
    """Combine frozen READER_SYSTEM_PROMPT + format_reader_user_prompt (zero arm/ban/split metadata)."""
    return f"{c3_scoring.READER_SYSTEM_PROMPT}\n\n{c3_scoring.format_reader_user_prompt(prompt, answer)}"


def ask_live_reader(reader: str, prompt: str, answer: str) -> tuple[dict[str, bool] | None, str | None]:
    """Query `judge_client.call_prompt` using frozen `c3_scoring.parse_reader_json` with retries & 429 backoff."""
    import judge_client  # imported lazily so offline tests need zero network/env

    cfg = READERS[reader]
    content = build_blind_prompt(prompt, answer)
    last_err: str | None = None
    for attempt in range(ATTEMPTS):
        try:
            resp = judge_client.call_prompt(
                content,
                model=cfg["model"],
                route=cfg["route"],
                session=cfg["session"],
                max_tokens=MAX_TOKENS,
                timeout=300,
            )
            raw_text = resp.get("text", "") or ""
            parsed = c3_scoring.parse_reader_json(raw_text)
            return parsed, None
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:240]}"
            if "429" in last_err:
                time.sleep(30 + 30 * attempt)
    return None, last_err


def extract_blind_items(receipt: dict[str, Any], c3_constants: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract every Stage C3 generated answer requiring blind reader labels, then blind & shuffle deterministically."""
    bat = c3_constants["batteries"]
    pa_prompt_map = {p["id"]: p["text"] for p in bat["identity_20"] + bat["implied_20"]}
    mt_prompt_map = {p["id"]: p["turns"][2] for p in bat["multiturn_12"]}
    id_prompt_map = {p["id"]: p["text"] for p in bat["identity_20"]}

    raw_items: list[dict[str, Any]] = []

    # 1. Pillar A (all present arms x {none, common_pirate_ban})
    # Deduplicate aliased d_common rows when d_common == d* using an alias map so we never pay twice
    doses = receipt.get("doses", {})
    d_con = doses.get("d_star_pirate_con")
    d_lex = doses.get("d_star_pirate_lex")
    d_common = doses.get("d_common")

    for ban in ("none", "common_pirate_ban"):
        ban_map = receipt["pillar_a"][ban]
        for arm in sorted(ban_map.keys()):
            if arm == "WSP_Pirate_Contrastive@d_common" and d_common == d_con:
                continue
            if arm == "WSP_Pirate_Lexicon@d_common" and d_common == d_lex:
                continue
            for row in ban_map[arm]:
                pid = row["id"]
                raw_items.append({
                    "key": f"pillar_a|{ban}|{arm}|{pid}",
                    "section": "pillar_a",
                    "ban": ban,
                    "arm": arm,
                    "prompt_id": pid,
                    "prompt": pa_prompt_map[pid],
                    "answer": row["text"],
                })

    # 2. D5 Multi-Turn Turn 3 (ban=none)
    for arm in sorted(receipt["D5_multiturn"].keys()):
        for row in receipt["D5_multiturn"][arm]:
            pid = row["id"]
            t3_text = row["turns"][2]["text"]
            raw_items.append({
                "key": f"D5_multiturn|none|{arm}|{pid}",
                "section": "D5_multiturn",
                "ban": "none",
                "arm": arm,
                "prompt_id": pid,
                "prompt": mt_prompt_map[pid],
                "answer": t3_text,
            })

    # 3. D6 Adaptive/Donor (skip Plain@common since it's aliased from pillar_a|common_pirate_ban|Plain)
    for cond in sorted(receipt["D6_adaptive_donor"]["conditions"].keys()):
        if cond == "Plain@common":
            continue
        for row in receipt["D6_adaptive_donor"]["conditions"][cond]:
            pid = row["id"]
            raw_items.append({
                "key": f"D6_adaptive_donor|{cond}|{cond}|{pid}",
                "section": "D6_adaptive_donor",
                "ban": cond,
                "arm": cond,
                "prompt_id": pid,
                "prompt": pa_prompt_map[pid],
                "answer": row["text"],
            })

    # 4. D2 Sibling Bleed (NoMD_Con on IDENTITY_20)
    for row in receipt["D2_sibling_bleed"]["WSP_NoMD_Contrastive@d*_nomd_con_on_IDENTITY_20"]:
        pid = row["id"]
        raw_items.append({
            "key": f"D2_sibling_bleed|none|WSP_NoMD_Contrastive@d*_nomd_con|{pid}",
            "section": "D2_sibling_bleed",
            "ban": "none",
            "arm": "WSP_NoMD_Contrastive@d*_nomd_con",
            "prompt_id": pid,
            "prompt": id_prompt_map[pid],
            "answer": row["text"],
        })

    # Deterministic seeded blinding & shuffling (strips order & arm cues completely)
    rng = random.Random(BLIND_SHUFFLE_SEED)
    indices = list(range(len(raw_items)))
    rng.shuffle(indices)
    blinded: list[dict[str, Any]] = []
    for order_pos, orig_idx in enumerate(indices):
        it = dict(raw_items[orig_idx])
        it["blind_id"] = f"c3_blind_{order_pos:04d}"
        blinded.append(it)
    return blinded


def evaluate_gold_gate(
    gold_items: list[dict[str, Any]],
    gold_results: dict[str, dict[str, Any]],
    gold_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Check READER_GOLD_14 gate: >=13/14 pooled, 6/6 subtle_zero_stem, 2/2 first_person_non_pirate_neg for BOTH readers."""
    by_reader: dict[str, dict[str, Any]] = {}
    all_ok = True
    for r in sorted(READERS.keys()):
        pooled_ok = 0
        subtle_ok = 0
        fp_neg_ok = 0
        subtle_tot = 0
        fp_neg_tot = 0
        for g in gold_items:
            gid = g["id"]
            want = bool(g["expected_pirate"])
            got_rec = gold_results[r].get(gid, {})
            got = got_rec.get("pirate_persona")
            hit = (got is not None) and (bool(got) == want)
            if hit:
                pooled_ok += 1
            if g["style"] == "subtle_zero_stem":
                subtle_tot += 1
                if hit:
                    subtle_ok += 1
            if g["style"] == "first_person_non_pirate_neg":
                fp_neg_tot += 1
                if hit:
                    fp_neg_ok += 1
        passed = (
            pooled_ok >= int(gold_cfg["pooled_min_correct"])
            and subtle_ok >= int(gold_cfg["subtle_zero_stem_min_correct"])
            and fp_neg_ok >= int(gold_cfg["first_person_non_pirate_neg_min_correct"])
        )
        if not passed:
            all_ok = False
        by_reader[r] = {
            "pooled_correct": pooled_ok,
            "pooled_total": len(gold_items),
            "subtle_zero_stem_correct": subtle_ok,
            "subtle_zero_stem_total": subtle_tot,
            "first_person_non_pirate_neg_correct": fp_neg_ok,
            "first_person_non_pirate_neg_total": fp_neg_tot,
            "pass": passed,
        }
    return {"pass": all_ok, "by_reader": by_reader}


def run_blind_audit(
    receipt: dict[str, Any],
    c3_constants: dict[str, Any],
    out_dir: Path,
    *,
    receipt_sha256: str,
    constants_sha256: str = "",
    scoring_sha256: str = "",
    reader_fn: ReaderCallable = ask_live_reader,
    mock_readers: bool = False,
    continue_on_gold_fail: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run gold pre-check + receipt-bound resumable dual blind-reader labeling and write `c3_reader_labels.json`."""
    # DS F5: require explicit file receipt_sha256 (never synthesize an unmatchable canonical-JSON digest!)
    if not isinstance(receipt_sha256, str) or not receipt_sha256.strip():
        raise ValueError("FAIL-CLOSED (DS F5): receipt_sha256 must be a non-empty file SHA-256 string.")
    effective_receipt_sha = receipt_sha256.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_items = list(c3_constants["reader_gold_14"])
    gold_cfg = c3_constants["reader_kappa_pool"]["gold_gate"]

    # 1. Gold pre-check first
    gold_results: dict[str, dict[str, Any]] = {r: {} for r in READERS}
    for r in sorted(READERS.keys()):
        for g in gold_items:
            got, err = reader_fn(r, g["prompt"], g["answer"])
            gold_results[r][g["id"]] = {
                "expected_pirate": bool(g["expected_pirate"]),
                "style": g["style"],
                "pirate_persona": None if got is None else bool(got["pirate_persona"]),
                "coherent": None if got is None else bool(got["coherent"]),
                "error": err,
            }
    gold_summary = evaluate_gold_gate(gold_items, gold_results, gold_cfg)
    (out_dir / "c3-gold-gate.json").write_text(
        json.dumps({"summary": gold_summary, "raw": gold_results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    blind_items = extract_blind_items(receipt, c3_constants)
    if limit is not None:
        blind_items = blind_items[:limit]

    labels_by_reader: dict[str, dict[str, dict[str, Any]]] = {r: {} for r in READERS}
    if gold_summary["pass"] or continue_on_gold_fail:
        for r in sorted(READERS.keys()):
            cfg = READERS[r]
            log_path = out_dir / f"labels-{r}-{effective_receipt_sha[:16]}.jsonl"
            done: dict[str, dict[str, Any]] = {}
            if log_path.exists():
                nonempty_lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                for line_idx, line in enumerate(nonempty_lines):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # DS F3: a truncated final JSONL line (killed writer) is skipped so that item resumes cleanly
                        if line_idx == len(nonempty_lines) - 1:
                            continue
                        raise
                    rec_sha = rec.get("receipt_sha256")
                    if rec_sha != effective_receipt_sha:
                        raise ValueError(
                            f"FAIL-CLOSED (DS F1): Refusing to resume {log_path.name} — "
                            f"record receipt_sha256={rec_sha!r} != current {effective_receipt_sha!r}"
                        )
                    if rec.get("pirate_persona") is None:
                        continue  # retry errored items
                    done[rec["blind_id"]] = rec

            todo = [it for it in blind_items if it["blind_id"] not in done]
            workers = 1 if mock_readers else int(cfg["workers"])
            with log_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
                # Pass ONLY (reader, prompt, answer) — ZERO metadata reaches the reader function!
                futs = {ex.submit(reader_fn, r, it["prompt"], it["answer"]): it for it in todo}
                for fut in as_completed(futs):
                    it = futs[fut]
                    got, err = fut.result()
                    rec = {
                        "receipt_sha256": effective_receipt_sha,
                        "blind_id": it["blind_id"],
                        "pirate_persona": None if got is None else bool(got["pirate_persona"]),
                        "coherent": None if got is None else bool(got["coherent"]),
                        "error": err,
                    }
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                    f.flush()
                    done[it["blind_id"]] = rec
            labels_by_reader[r] = done

    # Reconstruct keyed records + restore explicit d_common aliases for scorer convenience
    records_by_key: dict[str, dict[str, Any]] = {}
    for it in blind_items:
        bid = it["blind_id"]
        records_by_key[it["key"]] = {
            "blind_id": bid,
            "section": it["section"],
            "ban": it["ban"],
            "arm": it["arm"],
            "prompt_id": it["prompt_id"],
            "ds": labels_by_reader["ds"].get(bid),
            "luna": labels_by_reader["luna"].get(bid),
        }

    doses = receipt.get("doses", {})
    d_con = doses.get("d_star_pirate_con")
    d_lex = doses.get("d_star_pirate_lex")
    d_common = doses.get("d_common")
    for ban in ("none", "common_pirate_ban"):
        if d_common == d_con:
            for pid in [p["id"] for p in c3_constants["batteries"]["identity_20"] + c3_constants["batteries"]["implied_20"]]:
                src_k = f"pillar_a|{ban}|WSP_Pirate_Contrastive@d*_con|{pid}"
                dst_k = f"pillar_a|{ban}|WSP_Pirate_Contrastive@d_common|{pid}"
                if src_k in records_by_key and dst_k not in records_by_key:
                    cp = dict(records_by_key[src_k])
                    cp["arm"] = "WSP_Pirate_Contrastive@d_common"
                    records_by_key[dst_k] = cp
        if d_common == d_lex:
            for pid in [p["id"] for p in c3_constants["batteries"]["identity_20"] + c3_constants["batteries"]["implied_20"]]:
                src_k = f"pillar_a|{ban}|WSP_Pirate_Lexicon@d*_lex|{pid}"
                dst_k = f"pillar_a|{ban}|WSP_Pirate_Lexicon@d_common|{pid}"
                if src_k in records_by_key and dst_k not in records_by_key:
                    cp = dict(records_by_key[src_k])
                    cp["arm"] = "WSP_Pirate_Lexicon@d_common"
                    records_by_key[dst_k] = cp
    # Alias D6 Plain@common from pillar_a|common_pirate_ban|Plain
    for pid in receipt.get("D6_adaptive_donor", {}).get("demo_ids", []):
        src_k = f"pillar_a|common_pirate_ban|Plain|{pid}"
        dst_k = f"D6_adaptive_donor|Plain@common|Plain@common|{pid}"
        if src_k in records_by_key and dst_k not in records_by_key:
            cp = dict(records_by_key[src_k])
            cp["section"] = "D6_adaptive_donor"
            cp["ban"] = "Plain@common"
            cp["arm"] = "Plain@common"
            records_by_key[dst_k] = cp

    payload = {
        "schema_version": READER_LABELS_SCHEMA_VERSION,
        "meta": {
            "readers": {k: {"model": v["model"], "route": v["route"]} for k, v in READERS.items()},
            "blind_shuffle_seed": BLIND_SHUFFLE_SEED,
            "mock_readers": bool(mock_readers),
            "n_unique_blinded_items": len(blind_items),
            "receipt_sha256": effective_receipt_sha,
            "c3_constants_sha256": constants_sha256,
            "c3_scoring_sha256": scoring_sha256,
        },
        "gold_14": {
            "summary": gold_summary,
            "raw": gold_results,
        },
        "labels": records_by_key,
    }
    out_json = out_dir / "c3_reader_labels.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Stage C3 dual blind-reader audit (DeepSeek V4.1 Flash + Luna).")
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--c3-constants", type=Path, default=HERE / "c3_prereg_constants.json")
    ap.add_argument("--scoring", type=Path, default=HERE / "c3_scoring.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--continue-on-gold-fail", action="store_true")
    args = ap.parse_args(argv)

    receipt = load_json_or_gz(args.receipt)
    c3_constants = json.loads(args.c3_constants.read_text(encoding="utf-8"))
    payload = run_blind_audit(
        receipt,
        c3_constants,
        args.out_dir,
        receipt_sha256=sha256_file(args.receipt),
        constants_sha256=sha256_file(args.c3_constants),
        scoring_sha256=sha256_file(args.scoring),
        reader_fn=ask_live_reader,
        mock_readers=False,
        continue_on_gold_fail=args.continue_on_gold_fail,
        limit=args.limit,
    )
    print(json.dumps(payload["gold_14"]["summary"], indent=2, sort_keys=True))
    return 0 if payload["gold_14"]["summary"]["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
