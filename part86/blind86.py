#!/usr/bin/env python3
"""Prepare and score Part 86's blind persona labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import triage86

KINDS = (
    "human", "animal", "deity or spirit", "fictional character", "object or place",
    "abstract", "another AI", "still Qwen or an assistant", "unclear",
)


def prepare(screen_dir, triage_path, items_path, key_path, force=False):
    items_path, key_path = Path(items_path), Path(key_path)
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
        }
    items_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in public).encode("utf-8")
    items_path.write_bytes(encoded)
    key = {
        "items_sha256": hashlib.sha256(encoded).hexdigest(),
        "rows": mapping,
        "triage_counts": triaged["layers"],
    }
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(public)


def score(labels, key, items_path):
    items_bytes = Path(items_path).read_bytes()
    actual_hash = hashlib.sha256(items_bytes).hexdigest()
    if actual_hash != key.get("items_sha256"):
        raise ValueError("items file SHA-256 does not match the key")
    mapping = key.get("rows")
    if not isinstance(mapping, dict):
        raise ValueError("key has no rows mapping")
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

    return {"layers": by_layer, "labels": seen}


def _target_offworld(labels_by_id, mapping, screen_dir):
    import re
    target_row = None
    for row in triage86.read_screen(screen_dir):
        if row.get("kind") == "neuron" and int(row.get("layer", -1)) == 22 and int(row.get("id", -1)) == 2742:
            target_row = row
            break
    if target_row is None:
        return {"status": "NOT READ", "detail": "L22 N2742 is absent from screen data"}
    if triage86.triage_answer(target_row.get("answer", "")) != "changed":
        return {"status": "NOT READ", "detail": f"L22 N2742 triaged {triage86.triage_answer(target_row.get('answer', ''))}; it was not blind-labeled"}
    item_id = next((i for i, info in mapping.items() if info.get("layer") == 22 and info.get("kind") == "neuron" and info.get("item_index") == 2742), None)
    if item_id is None or item_id not in labels_by_id:
        return {"status": "NOT READ", "detail": "L22 N2742 has no blind label"}
    phrase = labels_by_id[item_id]["persona"].casefold()
    pattern = re.compile(r"\b(?:alien|extraterrestrial|off[- ]world|outer space|from space|another planet|other planet|martian|mars|interstellar|cosmic being)\b", re.I)
    holds = bool(pattern.search(phrase))
    return {"status": "HOLDS" if holds else "FAILS", "detail": f"L22 N2742 persona={labels_by_id[item_id]['persona']!r}"}


def verdicts(scored, screen_dir):
    all_counts = scored["triage_counts"] if "triage_counts" in scored else None
    # Counts over every screened item come from triage, including non-shortlisted kept/broken answers.
    source = scored.get("_key_triage_counts")
    if source is None:
        raise ValueError("internal scoring data missing triage counts")
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
        real = l22["real"]["coherent_persona_rate"]
        random = l22["random"]["coherent_persona_rate"]
        holds = real > 0 and real >= 2 * random
        rows["86.2"] = {"status": "HOLDS" if holds else "FAILS", "detail": f"L22 real={real:.6f}, random={random:.6f}"}

    if missing or any(per_layer[x]["real"]["n_total"] == 0 for x in expected):
        rows["86.3"] = {"status": "NOT READ", "detail": f"missing layers or real counts: {missing}"}
    else:
        violations = {x: per_layer[x]["real"]["coherent_persona_rate"] for x in expected if per_layer[x]["real"]["coherent_persona_rate"] >= 0.05}
        rows["86.3"] = {"status": "FAILS" if violations else "HOLDS", "detail": f"layers with rate >= 5%: {violations}"}

    if missing or any(per_layer[x]["random"]["n_total"] == 0 for x in expected):
        rows["86.4"] = {"status": "NOT READ", "detail": f"missing layers or random counts: {missing}"}
    else:
        failures = {}
        for x in expected:
            g = per_layer[x]["random"]
            if g["broken"] + g["unclear"] <= g["coherent_persona"]:
                failures[x] = {"broken_plus_unclear": g["broken"] + g["unclear"], "coherent_persona": g["coherent_persona"]}
        rows["86.4"] = {"status": "FAILS" if failures else "HOLDS", "detail": f"layers not exceeding coherent-persona count: {failures}"}

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
        rows["86.6"] = {"status": "HOLDS" if early[0] > late[0] else "FAILS", "detail": f"layers 14-22={early[0]:.6f} ({early[1]}/{early[2]}), layers 23-26={late[0]:.6f} ({late[1]}/{late[2]})"}

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
    args = ap.parse_args(argv)
    if args.command == "prepare":
        n = prepare(args.screen_dir, args.triage, args.items, args.key, args.force)
        print(f"{n} changed answers blinded")
        return
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = score(labels, key, args.items)
    result["triage_counts"] = key["triage_counts"]
    result["_key_triage_counts"] = key["triage_counts"]
    result["_mapping"] = key["rows"]
    final = verdicts(result, args.screen_dir)
    output = {
        "items_sha256": key["items_sha256"],
        "labeled_items": len(labels),
        **final,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in output["verdicts"].items():
        print(f"{name} {row['status']}: {row['detail']}")


if __name__ == "__main__":
    main()
