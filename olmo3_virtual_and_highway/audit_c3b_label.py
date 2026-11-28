#!/usr/bin/env python3
"""audit_c3b_label.py — Pre-GPU Preflight Canary & Dual Blind-Reader Harness for Stage C3b (WSP v2).

Features:
  1. `--preflight-only` mode (`C3b.R_preflight`): runs the 2-item synthetic canary
     `READER_PREFLIGHT_CANARY_2` on BOTH `deepseek-v4.1-flash` (`ocgo`) and `gpt-6-luna` (`cliproxy`)
     BEFORE L4 GPU launch, verifying parseable JSON AND exact expected labels (`pirate_persona` & `coherent`),
     and writes `c3b-preflight-canary-both.json` (also supports split per-reader preflights merged into
     `c3b-preflight-canary-both.json`).
  2. Fail-closed preflight enforcement (Round-1 Must-Fix 2 / DeepSeek F2): `run_blind_audit` requires a
     passing `c3b-preflight-canary-both.json` older than the assembled labels file, embeds the full canary
     record + SHA-256 + timestamps into `c3b_reader_labels.json`, and refuses to assemble labels without it.
  3. Full blind audit over brand-new cue-free `READER_GOLD_16_V2` (`>= 14/16` pooled, `>= 5/6` subtle,
     `>= 3/4` first-person non-pirate negatives) + all blinded Stage C3b items (`528` pooled + `36` D6_heldout
     + `20` D2_sibling_bleed + `24` D_key_ablation Pirate kSparse32 + optional `24` D_above_cap Pirate).
  4. Supports `--readers {both,ds,luna}` and `--merge-only` for key-isolated shells, strictly receipt-SHA-256
     bound resumable JSONL logs, zero metadata leaked to readers, and `mock_readers: true` stamping.
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

READER_LABELS_SCHEMA_VERSION = "olmo3-stage-c3b-reader-labels-v1"
PREFLIGHT_SCHEMA_VERSION = "olmo3-stage-c3b-reader-preflight-v1"
BLIND_SHUFFLE_SEED = 2026112799

READERS: dict[str, dict[str, Any]] = {
    "ds": {"model": "deepseek-v4.1-flash", "route": "ocgo", "session": "olmo3-c3b-audit-ds", "workers": 2},
    "luna": {"model": "gpt-6-luna", "route": "cliproxy", "session": "olmo3-c3b-audit-luna", "workers": 3},
}
MAX_TOKENS = 4000
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


def verify_preflight_canary_doc(
    pf_doc: dict[str, Any],
    c3b_constants: dict[str, Any],
    *,
    allow_mock_readers: bool = False,
) -> tuple[bool, str | None]:
    """Re-verify a `c3b-preflight-canary-both.json` document from raw per-item outputs (never trust summary bool)."""
    if not isinstance(pf_doc, dict) or pf_doc.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        return False, "preflight_canary_schema_mismatch"
    if pf_doc.get("mock_readers", False) and not allow_mock_readers:
        return False, "preflight_canary_mock_refused"
    canary_items = list(c3b_constants["reader_kappa_pool"]["preflight_canary_2"])
    by_reader = pf_doc.get("by_reader", {})
    if not isinstance(by_reader, dict):
        return False, "preflight_canary_missing_by_reader"
    for r in ("ds", "luna"):
        r_blk = by_reader.get(r)
        if not isinstance(r_blk, dict):
            return False, f"preflight_canary_missing_reader_{r}"
        items = r_blk.get("items", [])
        if not isinstance(items, list) or len(items) != len(canary_items):
            return False, f"preflight_canary_bad_items_{r}"
        by_id = {it.get("id"): it for it in items if isinstance(it, dict)}
        for exp in canary_items:
            got = by_id.get(exp["id"])
            if not got or got.get("error") is not None:
                return False, f"preflight_canary_error_{r}_{exp['id']}"
            if got.get("got_pirate") is not bool(exp["expected_pirate"]):
                return False, f"preflight_canary_wrong_pirate_{r}_{exp['id']}"
            if got.get("got_coherent") is not bool(exp["expected_coherent"]):
                return False, f"preflight_canary_wrong_coherent_{r}_{exp['id']}"
    return True, None


def run_preflight_canary(
    c3b_constants: dict[str, Any],
    *,
    readers: list[str] | None = None,
    reader_fn: ReaderCallable = ask_live_reader,
    out_dir: Path | None = None,
    mock_readers: bool = False,
    constants_sha256: str = "",
    scoring_sha256: str = "",
) -> dict[str, Any]:
    """Run mandatory pre-GPU `C3b.R_preflight` 2-item canary checking parseable JSON AND exact expected labels.

    If `out_dir` is provided and single-reader preflights for both `ds` and `luna` exist, automatically
    assembles `c3b-preflight-canary-both.json` as well.
    """
    target_readers = list(readers) if readers is not None else sorted(READERS.keys())
    canary_items = list(c3b_constants["reader_kappa_pool"]["preflight_canary_2"])
    by_reader: dict[str, Any] = {}
    all_pass = True

    for r in target_readers:
        items_out: list[dict[str, Any]] = []
        r_ok = True
        for item in canary_items:
            got, err = reader_fn(r, item["prompt"], item["answer"])
            exp_p = bool(item["expected_pirate"])
            exp_c = bool(item["expected_coherent"])
            exact_ok = (
                err is None
                and got is not None
                and (bool(got.get("pirate_persona")) == exp_p)
                and (bool(got.get("coherent")) == exp_c)
            )
            if not exact_ok:
                r_ok = False
            items_out.append({
                "id": item["id"],
                "expected_pirate": exp_p,
                "expected_coherent": exp_c,
                "got_pirate": None if got is None else bool(got["pirate_persona"]),
                "got_coherent": None if got is None else bool(got["coherent"]),
                "error": err,
                "pass": exact_ok,
            })
        if not r_ok:
            all_pass = False
        by_reader[r] = {
            "model": READERS[r]["model"],
            "route": READERS[r]["route"],
            "pass": r_ok,
            "items": items_out,
        }

    now_ts = time.time()
    payload = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "recorded_epoch": now_ts,
        "mock_readers": bool(mock_readers),
        "c3b_constants_sha256": constants_sha256,
        "c3_scoring_sha256": scoring_sha256,
        "readers_checked": target_readers,
        "pass": all_pass,
        "by_reader": by_reader,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in target_readers:
            single_doc = {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "recorded_epoch": now_ts,
                "mock_readers": bool(mock_readers),
                "c3b_constants_sha256": constants_sha256,
                "c3_scoring_sha256": scoring_sha256,
                "readers_checked": [r],
                "pass": by_reader[r]["pass"],
                "by_reader": {r: by_reader[r]},
            }
            (out_dir / f"c3b-preflight-canary-{r}.json").write_text(
                json.dumps(single_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if set(target_readers) == set(READERS.keys()):
            (out_dir / "c3b-preflight-canary-both.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            # Auto-merge into c3b-preflight-canary-both.json if both single-reader files are now present
            p_ds = out_dir / "c3b-preflight-canary-ds.json"
            p_lu = out_dir / "c3b-preflight-canary-luna.json"
            if p_ds.exists() and p_lu.exists():
                d_ds = json.loads(p_ds.read_text(encoding="utf-8"))
                d_lu = json.loads(p_lu.read_text(encoding="utf-8"))
                merged_by = {
                    "ds": d_ds["by_reader"]["ds"],
                    "luna": d_lu["by_reader"]["luna"],
                }
                merged_doc = {
                    "schema_version": PREFLIGHT_SCHEMA_VERSION,
                    "recorded_epoch": max(float(d_ds.get("recorded_epoch", now_ts)), float(d_lu.get("recorded_epoch", now_ts))),
                    "mock_readers": bool(d_ds.get("mock_readers", False) or d_lu.get("mock_readers", False)),
                    "c3b_constants_sha256": constants_sha256 or d_ds.get("c3b_constants_sha256", ""),
                    "c3_scoring_sha256": scoring_sha256 or d_ds.get("c3_scoring_sha256", ""),
                    "readers_checked": ["ds", "luna"],
                    "pass": bool(merged_by["ds"]["pass"] and merged_by["luna"]["pass"]),
                    "by_reader": merged_by,
                }
                (out_dir / "c3b-preflight-canary-both.json").write_text(
                    json.dumps(merged_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
    return payload


def extract_blind_items(receipt: dict[str, Any], c3b_constants: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract every Stage C3b generated answer requiring blind reader labels, then blind & shuffle deterministically."""
    bat = c3b_constants["batteries"]
    pa_prompt_map = {p["id"]: p["text"] for p in bat["identity_20"] + bat["implied_20"]}
    mt_prompt_map = {p["id"]: p["turns"][2] for p in bat["multiturn_12"]}
    id_prompt_map = {p["id"]: p["text"] for p in bat["identity_20"]}

    raw_items: list[dict[str, Any]] = []

    # 1. Pillar A (core_arms_6 x 40 x 2 bans = 480 items)
    for ban in ("none", "common_pirate_ban"):
        ban_map = receipt["pillar_a"][ban]
        for arm in sorted(ban_map.keys()):
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

    # 2. D5 Multi-Turn Turn 3 (multiturn_d5_arms_4 x 12, ban=none = 48 items -> 528 pooled exact!)
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

    # 3. D6_heldout (3 non-aliased conditions x 12 = 36 items; Plain@common aliased from pillar_a)
    for cond in sorted(receipt["D6_heldout"]["conditions"].keys()):
        if cond == "Plain@common":
            continue
        for row in receipt["D6_heldout"]["conditions"][cond]:
            pid = row["id"]
            raw_items.append({
                "key": f"D6_heldout|{cond}|{cond}|{pid}",
                "section": "D6_heldout",
                "ban": cond,
                "arm": cond,
                "prompt_id": pid,
                "prompt": pa_prompt_map[pid],
                "answer": row["text"],
            })

    # 4. D2 Sibling Bleed (WSP_NoMD_12L on IDENTITY_20 = 20 items)
    for row in receipt["D2_sibling_bleed"]["WSP_NoMD_12L@d*_nomd_on_IDENTITY_20"]:
        pid = row["id"]
        raw_items.append({
            "key": f"D2_sibling_bleed|none|WSP_NoMD_12L@d*_nomd|{pid}",
            "section": "D2_sibling_bleed",
            "ban": "none",
            "arm": "WSP_NoMD_12L@d*_nomd",
            "prompt_id": pid,
            "prompt": id_prompt_map[pid],
            "answer": row["text"],
        })

    # 5. Conditional D_above_cap Pirate slice (if triggered: 12 x 2 bans = 24 items)
    dac_p = receipt.get("D_above_cap", {}).get("WSP_Pirate_12L", {})
    if dac_p.get("triggered"):
        for ban in ("none", "common_pirate_ban"):
            for row in dac_p["generations"][ban]:
                pid = row["id"]
                raw_items.append({
                    "key": f"D_above_cap|{ban}|WSP_Pirate_12L@d_fail|{pid}",
                    "section": "D_above_cap",
                    "ban": ban,
                    "arm": "WSP_Pirate_12L@d_fail",
                    "prompt_id": pid,
                    "prompt": pa_prompt_map[pid],
                    "answer": row["text"],
                })

    # 6. Exploratory D_key_ablation Pirate kSparse32 slice (12 x 2 bans = 24 items)
    dk_p = receipt.get("D_key_ablation", {}).get("WSP_Pirate_12L_kSparse32", {})
    if "generations" in dk_p:
        for ban in ("none", "common_pirate_ban"):
            for row in dk_p["generations"][ban]:
                pid = row["id"]
                raw_items.append({
                    "key": f"D_key_ablation|{ban}|WSP_Pirate_12L_kSparse32|{pid}",
                    "section": "D_key_ablation",
                    "ban": ban,
                    "arm": "WSP_Pirate_12L_kSparse32",
                    "prompt_id": pid,
                    "prompt": pa_prompt_map[pid],
                    "answer": row["text"],
                })

    # Deterministic seeded blinding & shuffling (strips order, arm, ban, and split cues completely)
    rng = random.Random(BLIND_SHUFFLE_SEED)
    indices = list(range(len(raw_items)))
    rng.shuffle(indices)
    blinded: list[dict[str, Any]] = []
    for order_pos, orig_idx in enumerate(indices):
        it = dict(raw_items[orig_idx])
        it["blind_id"] = f"c3b_blind_{order_pos:04d}"
        blinded.append(it)
    return blinded


def evaluate_gold_gate(
    gold_items: list[dict[str, Any]],
    gold_results: dict[str, dict[str, Any]],
    gold_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Check READER_GOLD_16_V2 gate: >=14/16 pooled, >=5/6 subtle_zero_stem, >=3/4 first_person_non_pirate_neg."""
    by_reader: dict[str, dict[str, Any]] = {}
    all_ok = True
    for r in sorted(READERS.keys()):
        r_map = gold_results.get(r, {})
        pooled_ok = 0
        subtle_ok = 0
        fp_neg_ok = 0
        subtle_tot = 0
        fp_neg_tot = 0
        for g in gold_items:
            gid = g["id"]
            want = bool(g["expected_pirate"])
            got_rec = r_map.get(gid, {})
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


def _load_resumable_jsonl(log_path: Path, effective_receipt_sha: str) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not log_path.exists():
        return done
    nonempty_lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line_idx, line in enumerate(nonempty_lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if line_idx == len(nonempty_lines) - 1:
                continue  # truncated final JSONL line resumes cleanly
            raise
        rec_sha = rec.get("receipt_sha256")
        if rec_sha != effective_receipt_sha:
            raise ValueError(
                f"FAIL-CLOSED: Refusing to resume {log_path.name} — "
                f"record receipt_sha256={rec_sha!r} != current {effective_receipt_sha!r}"
            )
        if rec.get("pirate_persona") is None:
            continue  # retry errored items
        done[rec["blind_id"]] = rec
    return done


def run_blind_audit(
    receipt: dict[str, Any],
    c3b_constants: dict[str, Any],
    out_dir: Path,
    *,
    receipt_sha256: str,
    constants_sha256: str = "",
    scoring_sha256: str = "",
    preflight_canary_path: Path | None = None,
    readers: list[str] | None = None,
    merge_only: bool = False,
    reader_fn: ReaderCallable = ask_live_reader,
    mock_readers: bool = False,
    continue_on_gold_fail: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run READER_GOLD_16_V2 + receipt-bound resumable dual blind-reader labeling -> `c3b_reader_labels.json`.

    Enforces `C3b.R_preflight` fail-closed: requires a passing `c3b-preflight-canary-both.json`
    recorded before assembling labels (refuses missing, failing, or mock-in-live preflights).
    """
    if not isinstance(receipt_sha256, str) or not receipt_sha256.strip():
        raise ValueError("FAIL-CLOSED: receipt_sha256 must be a non-empty file SHA-256 string.")
    effective_receipt_sha = receipt_sha256.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_readers = list(readers) if readers is not None else sorted(READERS.keys())

    # Fail-closed C3b.R_preflight canary verification (Round-1 Must-Fix 2 / DeepSeek F2)
    pf_path = preflight_canary_path if preflight_canary_path is not None else (out_dir / "c3b-preflight-canary-both.json")
    if not pf_path.exists():
        raise ValueError(
            f"FAIL-CLOSED (C3b.R_preflight): Missing required passing preflight canary file: {pf_path}. "
            "Run `audit_c3b_label.py --preflight-only` on both readers before blind labeling."
        )
    pf_doc = json.loads(pf_path.read_text(encoding="utf-8"))
    pf_ok, pf_reason = verify_preflight_canary_doc(pf_doc, c3b_constants, allow_mock_readers=mock_readers)
    if not pf_ok:
        raise ValueError(f"FAIL-CLOSED (C3b.R_preflight): Invalid/failed preflight canary {pf_path}: {pf_reason}")
    pf_sha256 = sha256_file(pf_path)
    pf_mtime = float(pf_path.stat().st_mtime)

    gold_items = list(c3b_constants["reader_gold_16_v2"])
    gold_cfg = c3b_constants["reader_kappa_pool"]["gold_gate"]

    # 1. Load any existing per-reader gold files, then run gold for target_readers unless merge_only
    gold_results: dict[str, dict[str, Any]] = {r: {} for r in READERS}
    for r in sorted(READERS.keys()):
        gpath = out_dir / f"c3b-gold-16-{r}.json"
        if gpath.exists():
            gold_results[r] = json.loads(gpath.read_text(encoding="utf-8"))

    if not merge_only:
        for r in target_readers:
            r_gold: dict[str, Any] = {}
            for g in gold_items:
                got, err = reader_fn(r, g["prompt"], g["answer"])
                r_gold[g["id"]] = {
                    "expected_pirate": bool(g["expected_pirate"]),
                    "style": g["style"],
                    "pirate_persona": None if got is None else bool(got["pirate_persona"]),
                    "coherent": None if got is None else bool(got["coherent"]),
                    "error": err,
                }
            gold_results[r] = r_gold
            (out_dir / f"c3b-gold-16-{r}.json").write_text(
                json.dumps(r_gold, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    gold_summary = evaluate_gold_gate(gold_items, gold_results, gold_cfg)
    (out_dir / "c3b-gold-gate.json").write_text(
        json.dumps({"summary": gold_summary, "raw": gold_results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    blind_items = extract_blind_items(receipt, c3b_constants)
    if limit is not None:
        blind_items = blind_items[:limit]

    target_gold_ok = all(bool(gold_summary["by_reader"][tr]["pass"]) for tr in target_readers)
    labels_by_reader: dict[str, dict[str, dict[str, Any]]] = {r: {} for r in READERS}
    for r in sorted(READERS.keys()):
        log_path = out_dir / f"labels-{r}-{effective_receipt_sha[:16]}.jsonl"
        done = _load_resumable_jsonl(log_path, effective_receipt_sha)

        should_run = (not merge_only) and (r in target_readers) and (target_gold_ok or continue_on_gold_fail)
        if should_run:
            cfg = READERS[r]
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

    # Reconstruct keyed records + restore D6 Plain@common alias from pillar_a|common_pirate_ban|Plain
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

    for pid in receipt.get("D6_heldout", {}).get("demo_ids", []):
        src_k = f"pillar_a|common_pirate_ban|Plain|{pid}"
        dst_k = f"D6_heldout|Plain@common|Plain@common|{pid}"
        if src_k in records_by_key and dst_k not in records_by_key:
            cp = dict(records_by_key[src_k])
            cp["section"] = "D6_heldout"
            cp["ban"] = "Plain@common"
            cp["arm"] = "Plain@common"
            records_by_key[dst_k] = cp

    labels_written_epoch = time.time()
    payload = {
        "schema_version": READER_LABELS_SCHEMA_VERSION,
        "meta": {
            "readers": {k: {"model": v["model"], "route": v["route"]} for k, v in READERS.items()},
            "blind_shuffle_seed": BLIND_SHUFFLE_SEED,
            "mock_readers": bool(mock_readers),
            "n_unique_blinded_items": len(blind_items),
            "receipt_sha256": effective_receipt_sha,
            "c3b_constants_sha256": constants_sha256,
            "c3_scoring_sha256": scoring_sha256,
            "preflight_canary_pass": True,
            "preflight_canary_sha256": pf_sha256,
            "preflight_canary_mtime": pf_mtime,
            "labels_written_epoch": labels_written_epoch,
            "preflight_canary_older_than_labels": bool(pf_mtime <= labels_written_epoch + 1e-6),
        },
        "preflight_canary": pf_doc,
        "gold_16_v2": {
            "summary": gold_summary,
            "raw": gold_results,
        },
        "labels": records_by_key,
    }
    out_json = out_dir / "c3b_reader_labels.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage C3b pre-GPU preflight canary & dual blind-reader audit.")
    ap.add_argument("--c3b-constants", type=Path, default=HERE / "c3b_prereg_constants.json")
    ap.add_argument("--scoring", type=Path, default=HERE / "c3_scoring.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--readers", choices=["both", "ds", "luna"], default="both")
    ap.add_argument("--preflight-only", action="store_true", help="Run C3b.R_preflight 2-item canary only (pre-GPU).")
    ap.add_argument("--preflight-canary", type=Path, default=None, help="Path to passing c3b-preflight-canary-both.json.")
    ap.add_argument("--receipt", type=Path, default=None, help="Stage C3b receipt (.json or .json.gz).")
    ap.add_argument("--merge-only", action="store_true", help="Assemble c3b_reader_labels.json from existing JSONL logs.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--continue-on-gold-fail", action="store_true")
    args = ap.parse_args(argv)

    c3b_constants = json.loads(args.c3b_constants.read_text(encoding="utf-8"))
    target_readers = sorted(READERS.keys()) if args.readers == "both" else [args.readers]
    const_sha = sha256_file(args.c3b_constants)
    score_sha = sha256_file(args.scoring)

    if args.preflight_only:
        res = run_preflight_canary(
            c3b_constants,
            readers=target_readers,
            reader_fn=ask_live_reader,
            out_dir=args.out_dir,
            mock_readers=False,
            constants_sha256=const_sha,
            scoring_sha256=score_sha,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res["pass"] else 2

    if args.receipt is None:
        ap.error("--receipt is required unless --preflight-only is specified.")

    receipt = load_json_or_gz(args.receipt)
    payload = run_blind_audit(
        receipt,
        c3b_constants,
        args.out_dir,
        receipt_sha256=sha256_file(args.receipt),
        constants_sha256=const_sha,
        scoring_sha256=score_sha,
        preflight_canary_path=args.preflight_canary,
        readers=target_readers,
        merge_only=args.merge_only,
        reader_fn=ask_live_reader,
        mock_readers=False,
        continue_on_gold_fail=args.continue_on_gold_fail,
        limit=args.limit,
    )
    print(json.dumps(payload["gold_16_v2"]["summary"], indent=2, sort_keys=True))
    return 0 if payload["gold_16_v2"]["summary"]["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
