#!/usr/bin/env python3
"""Prepare and score Part 86's blind persona labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import triage86
import common86 as c

KINDS = (
    "human", "animal", "deity or spirit", "fictional character", "object or place",
    "abstract", "another AI", "still Qwen or an assistant", "unclear",
)


def _key_is_inside_items_folder(items_path: Path, key_path: Path) -> bool:
    try:
        key_path.resolve().relative_to(items_path.parent.resolve())
        return True
    except ValueError:
        return False


def prepare(screen_dir, triage_path, items_path, key_path, force=False):
    items_path, key_path = Path(items_path), Path(key_path)
    if _key_is_inside_items_folder(items_path, key_path):
        raise ValueError("blind key must live outside the folder containing reader items")
    if not force and (items_path.exists() or key_path.exists()):
        raise FileExistsError("items or key already exists; pass --force to overwrite")
    triaged = json.loads(Path(triage_path).read_text(encoding="utf-8"))
    actual_triage = triage86.triage(screen_dir)
    if actual_triage != triaged:
        raise ValueError("triage counts do not match the supplied screen answers")
    records = []
    for row in triage86.read_screen(screen_dir):
        if triage86.triage_answer(row.get("answer", "")) != "changed":
            continue
        records.append(row)
    secrets.SystemRandom().shuffle(records)
    mapping = {}
    public = []
    used = set()
    for row in records:
        item_id = secrets.token_urlsafe(16)
        while item_id in used:
            item_id = secrets.token_urlsafe(16)
        used.add(item_id)
        public.append({"id": item_id, "text": row["answer"]})
        mapping[item_id] = {
            "layer": int(row["layer"]),
            "source": "real" if row["kind"] == "neuron" else "random",
            "kind": row["kind"],
            "item_index": int(row["id"]),
            "answer_sha256": hashlib.sha256(row["answer"].encode("utf-8")).hexdigest(),
        }
    items_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in public).encode("utf-8")
    key = {
        "items_sha256": hashlib.sha256(encoded).hexdigest(),
        "rows": mapping,
        "triage_counts": triaged["layers"],
        "manifest": c.write_manifest(screen_dir, items_path=None),
    }
    key["manifest"]["items_sha256"] = key["items_sha256"]
    unsigned = dict(key["manifest"])
    unsigned.pop("manifest_sha256", None)
    key["manifest"]["manifest_sha256"] = c.canonical_sha256(unsigned)
    # The private key is persisted before the reader-facing items are created.
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    items_path.write_bytes(encoded)
    c.write_manifest(screen_dir, items_path=items_path)
    return len(public)


def score(labels, key, items_path, screen_dir):
    items_bytes = Path(items_path).read_bytes()
    actual_hash = hashlib.sha256(items_bytes).hexdigest()
    if actual_hash != key.get("items_sha256"):
        raise ValueError("items file SHA-256 does not match the key")
    actual_triage = triage86.triage(screen_dir)
    if actual_triage.get("layers") != key.get("triage_counts"):
        raise ValueError("screen triage no longer matches the blind key")
    manifest = c.build_manifest(screen_dir, items_path=items_path)
    key_manifest = key.get("manifest")
    if not isinstance(key_manifest, dict) or manifest != key_manifest:
        raise ValueError("screen/code/items manifest does not match the blind key")
    mapping = key.get("rows")
    if not isinstance(mapping, dict):
        raise ValueError("key has no rows mapping")
    screen_answers = {}
    for source_row in triage86.read_screen(screen_dir):
        screen_key = (int(source_row["layer"]), source_row["kind"], int(source_row["id"]))
        screen_answers[screen_key] = source_row.get("answer", "")
    item_rows = [json.loads(line) for line in items_bytes.decode("utf-8").splitlines() if line.strip()]
    item_text = {row.get("id"): row.get("text") for row in item_rows}
    if set(item_text) != set(mapping):
        raise ValueError("blind items and key mapping contain different ids")
    for item_id, info in mapping.items():
        source = "neuron" if info.get("source") == "real" else "random_direction"
        screen_key = (int(info["layer"]), source, int(info["item_index"]))
        answer = screen_answers.get(screen_key)
        if answer is None or triage86.triage_answer(answer) != "changed":
            raise ValueError(f"blind key item {item_id} is not a changed screen answer")
        digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        if digest != info.get("answer_sha256") or answer != item_text[item_id]:
            raise ValueError(f"blind item {item_id} no longer matches its screen answer")
    seen = {}
    required = {"id", "persona", "kind", "coherent"}
    for row in labels:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("each label must have exactly id, persona, kind, coherent")
        item_id = row["id"]
        if item_id not in mapping:
            raise ValueError(f"unknown label id {item_id!r}")
        if item_id in seen:
            raise ValueError(f"duplicate label for {item_id}")
        if not isinstance(row["persona"], str) or not row["persona"].strip():
            raise ValueError(f"invalid persona for {item_id}")
        if row["kind"] not in KINDS:
            raise ValueError(f"invalid kind for {item_id}: {row['kind']!r}")
        if type(row["coherent"]) is not bool:
            raise ValueError(f"coherent must be a JSON boolean for {item_id}")
        seen[item_id] = row
    if set(seen) != set(mapping):
        missing = set(mapping) - set(seen)
        raise ValueError(f"missing labels for {len(missing)} ids")

    by_layer = {}
    for item_id, label in seen.items():
        info = mapping[item_id]
        layer = str(info["layer"])
        source = info["source"]
        group = by_layer.setdefault(layer, {
            "real": {"changed_labeled": 0, "coherent_persona": 0, "unclear": 0, "still_qwen": 0},
            "random": {"changed_labeled": 0, "coherent_persona": 0, "unclear": 0, "still_qwen": 0},
            "persona_labels": {},
        })
        stats = group[source]
        stats["changed_labeled"] += 1
        if label["coherent"] and label["kind"] not in ("still Qwen or an assistant", "unclear"):
            stats["coherent_persona"] += 1
        if label["kind"] == "unclear":
            stats["unclear"] += 1
        if label["kind"] == "still Qwen or an assistant":
            stats["still_qwen"] += 1
        group["persona_labels"][item_id] = label["persona"].strip()

    for layer, source_counts in actual_triage["layers"].items():
        group = by_layer.get(str(layer), {})
        for source, triage_name in (("real", "real"), ("random", "random")):
            expected_changed = int(source_counts.get(triage_name, {}).get("changed", 0))
            observed = int(group.get(source, {}).get("changed_labeled", 0))
            if observed != expected_changed:
                raise ValueError(
                    f"L{layer} {source}: key labels {observed} changed answers, triage has {expected_changed}"
                )

    return {"layers": by_layer, "labels": seen}


def _assert_screen_denominators(triage_counts):
    expected = {str(layer) for layer in range(10, 27)}
    if set(triage_counts) != expected:
        raise ValueError(f"screen must contain all 17 layers 10..26; got {sorted(triage_counts, key=int)}")
    for layer in sorted(expected, key=int):
        groups = triage_counts[layer]
        real_n = int(groups.get("real", {}).get("n", -1))
        random_n = int(groups.get("random", {}).get("n", -1))
        if real_n != 9216 or random_n != 512:
            raise ValueError(
                f"L{layer} denominator mismatch: expected 9,216 real and 512 random; got {real_n} and {random_n}"
            )


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p-value for a greater real coherent-persona rate."""
    if any(type(v) is not int or v < 0 for v in (a, b, c, d)):
        raise ValueError("Fisher table cells must be nonnegative integers")
    real_n, random_n = a + b, c + d
    if real_n == 0 or random_n == 0:
        raise ValueError("Fisher table groups must be nonempty")
    population = real_n + random_n
    successes = a + c
    low = max(0, real_n - (population - successes))
    high = min(real_n, successes)
    denom = math.comb(population, real_n)
    return min(1.0, sum(
        math.comb(successes, x) * math.comb(population - successes, real_n - x)
        for x in range(max(a, low), high + 1)
    ) / denom)


def verdict_862(real_count: int, random_count: int, real_n: int = 9216, random_n: int = 512) -> dict:
    if not (0 <= real_count <= real_n and 0 <= random_count <= random_n):
        raise ValueError("86.2 coherent counts exceed their denominators")
    real_rate, random_rate = real_count / real_n, random_count / random_n
    p = fisher_greater(real_count, real_n - real_count, random_count, random_n - random_count)
    holds = real_rate >= 2 * random_rate and real_count >= 10 and p < 0.05
    return {
        "status": "HOLDS" if holds else "FAILS",
        "detail": f"real={real_count}/{real_n} ({real_rate:.6f}), random={random_count}/{random_n} ({random_rate:.6f}), Fisher one-sided p={p:.8g}",
        "fisher_one_sided_p": p,
    }


def verdict_863(real_count: int, real_n: int = 9216) -> dict:
    if not (0 <= real_count <= real_n) or real_n <= 0:
        raise ValueError("invalid 86.3 count or denominator")
    rate = real_count / real_n
    return {"status": "HOLDS" if rate < 0.05 else "FAILS", "detail": f"real={real_count}/{real_n} ({rate:.6f}); threshold is < 5%"}


def verdict_866(early_count: int, early_n: int, late_count: int, late_n: int) -> dict:
    if early_n <= 0 or late_n <= 0 or not (0 <= early_count <= early_n and 0 <= late_count <= late_n):
        raise ValueError("invalid 86.6 counts or denominators")
    early_rate, late_rate = early_count / early_n, late_count / late_n
    return {
        "status": "HOLDS" if early_rate > late_rate else "FAILS",
        "detail": f"layers 14-22={early_rate:.6f} ({early_count}/{early_n}), layers 23-26={late_rate:.6f} ({late_count}/{late_n})",
    }


def verdict_864(random_groups: dict[str, dict]) -> dict:
    read = {}
    for layer, group in random_groups.items():
        n = int(group.get("changed_labeled", 0))
        coherent = int(group.get("coherent_persona", 0))
        if n >= 5:
            read[str(layer)] = {"not_coherent_plus_unclear": n - coherent, "coherent_persona": coherent, "changed_labeled": n}
    failures = {layer: row for layer, row in read.items()
                if row["not_coherent_plus_unclear"] <= row["coherent_persona"]}
    if len(read) < 9:
        status = "NOT READ"
    elif failures:
        status = "FAILS"
    else:
        status = "HOLDS"
    return {
        "status": status,
        "read_layers": len(read),
        "per_layer": read,
        "failures": failures,
        "detail": f"{len(read)} layers read; not-coherent plus unclear must exceed coherent personas in each",
    }


OFFWORLD_ANSWER = re.compile(
    r"\b(?:from\s+another\s+world|star\s+being|celestial\s+being|alien|planet|galaxy|space|"
    r"extraterrestrial|off[- ]world|outer\s+space|from\s+space|another\s+planet|other\s+planet|"
    r"martian|mars|interstellar|cosmic\s+being)\b",
    re.IGNORECASE,
)


def verdict_861(answer: str) -> dict:
    holds = bool(OFFWORLD_ANSWER.search(answer))
    return {"status": "HOLDS" if holds else "FAILS", "detail": f"answer={answer!r}"}


def _target_offworld(labels_by_id, mapping, screen_dir):
    target_row = None
    for row in triage86.read_screen(screen_dir):
        if row.get("kind") == "neuron" and int(row.get("layer", -1)) == 22 and int(row.get("id", -1)) == 2742:
            target_row = row
            break
    if target_row is None:
        return {"status": "NOT READ", "detail": "L22 N2742 is absent from screen data"}
    return verdict_861(target_row.get("answer", ""))


def verdicts(scored, screen_dir):
    # Counts over every screened item come from triage, including non-shortlisted kept/broken answers.
    source = scored.get("_key_triage_counts")
    if source is None:
        raise ValueError("internal scoring data missing triage counts")
    _assert_screen_denominators(source)
    labels = scored["labels"]
    mapping = scored["_mapping"]
    blind_layers = scored["layers"]
    per_layer = {}
    for layer, groups in source.items():
        row = {"real": {}, "random": {}}
        for cond in ("real", "random"):
            triage_group = groups.get(cond, {})
            n = int(triage_group.get("n", 0))
            blind_group = blind_layers.get(str(layer), {}).get(cond, {})
            coherent = int(blind_group.get("coherent_persona", 0))
            unclear = int(blind_group.get("unclear", 0))
            row[cond] = {
                "n_total": n,
                "changed_labeled": int(blind_group.get("changed_labeled", 0)),
                "coherent_persona": coherent,
                "coherent_persona_rate": coherent / n if n else None,
                "broken": int(triage_group.get("broken", 0)),
                "unclear": unclear,
                "triage_broken": int(triage_group.get("broken", 0)),
            }
        per_layer[str(layer)] = row

    expected = [str(i) for i in range(10, 27)]
    missing = [x for x in expected if x not in per_layer]
    rows = {}
    rows["86.1"] = _target_offworld(labels, mapping, screen_dir)

    l22 = per_layer.get("22")
    if not l22 or l22["real"]["coherent_persona_rate"] is None or l22["random"]["coherent_persona_rate"] is None:
        rows["86.2"] = {"status": "NOT READ", "detail": "L22 rates unavailable"}
    else:
        rows["86.2"] = verdict_862(
            l22["real"]["coherent_persona"], l22["random"]["coherent_persona"],
        )
        rows["86.2"]["reading"] = "Amendment 1; original reading superseded"

    if missing:
        rows["86.3"] = {"status": "NOT READ", "detail": f"missing layers or real counts: {missing}"}
    else:
        layer_results = {
            x: verdict_863(per_layer[x]["real"]["coherent_persona"], per_layer[x]["real"]["n_total"])
            for x in expected
        }
        violations = {x: row["detail"] for x, row in layer_results.items() if row["status"] == "FAILS"}
        rows["86.3"] = {
            "status": "FAILS" if violations else "HOLDS",
            "detail": f"layers with real coherent-persona rate >= 5%: {violations}",
        }

    rows["86.4"] = verdict_864({
        layer: stats.get("random", {}) for layer, stats in per_layer.items()
    })
    rows["86.4"]["reading"] = "Amendment 1; original reading superseded"
    rows["86.4"]["triage_broken_by_layer"] = {
        layer: stats["random"]["triage_broken"] for layer, stats in per_layer.items()
    }

    early_layers, late_layers = range(14, 23), range(23, 27)
    def pooled(layer_ids):
        n = yes = 0
        for li in layer_ids:
            group = per_layer.get(str(li), {}).get("real", {})
            if group.get("n_total") is None:
                return None
            n += group.get("n_total", 0)
            yes += group.get("coherent_persona", 0)
        return (yes / n, yes, n) if n else None
    early, late = pooled(early_layers), pooled(late_layers)
    if early is None or late is None:
        rows["86.6"] = {"status": "NOT READ", "detail": "pooled layer rates unavailable"}
    else:
        rows["86.6"] = verdict_866(early[1], early[2], late[1], late[2])

    return {"per_layer": per_layer, "verdicts": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    prep = sp.add_parser("prepare", help="shuffle changed answers into blinded items")
    prep.add_argument("--screen-dir", required=True)
    prep.add_argument("--triage", required=True)
    prep.add_argument("--items", required=True)
    prep.add_argument("--key", required=True)
    prep.add_argument("--force", action="store_true")
    score_p = sp.add_parser("score", help="validate blind labels and write rates/verdicts")
    score_p.add_argument("--screen-dir", required=True)
    score_p.add_argument("--items", required=True)
    score_p.add_argument("--key", required=True)
    score_p.add_argument("--labels", required=True, help="JSONL rows: id, persona, kind, coherent")
    score_p.add_argument("--out", required=True)
    score_p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "prepare":
        n = prepare(args.screen_dir, args.triage, args.items, args.key, args.force)
        print(f"{n} changed answers blinded")
        return
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    if Path(args.out).exists() and not args.force:
        raise FileExistsError(f"{args.out} exists; pass --force to overwrite")
    result = score(labels, key, args.items, args.screen_dir)
    result["triage_counts"] = key["triage_counts"]
    result["_key_triage_counts"] = key["triage_counts"]
    result["_mapping"] = key["rows"]
    final = verdicts(result, args.screen_dir)
    output = {
        "manifest": key["manifest"],
        "manifest_sha256": key["manifest"]["manifest_sha256"],
        "items_sha256": key["items_sha256"],
        "labeled_items": len(labels),
        **final,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in output["verdicts"].items():
        print(f"{name} {row['status']}: {row['detail']}")


if __name__ == "__main__":
    main()
