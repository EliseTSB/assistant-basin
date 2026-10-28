#!/usr/bin/env python3
"""Prepare and score Part 86's blind persona labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import random
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


def _reader_chunk_dir(items_path: Path, reader_name: str) -> Path:
    return items_path.parent / f"{items_path.stem}-{reader_name}-chunks"


def _write_chunks(path: Path, items: list[dict], chunk_size: int = 500) -> None:
    if not 1 <= chunk_size <= 500:
        raise ValueError("reader chunks must contain between 1 and 500 items")
    path.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(items), chunk_size), 1):
        rows = items[start:start + chunk_size]
        encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        (path / f"chunk-{index:04d}.jsonl").write_text(encoded, encoding="utf-8")


def prepare(screen_dir, triage_path, items_path, key_path, force=False, chunk_size=500):
    items_path, key_path = Path(items_path), Path(key_path)
    if _key_is_inside_items_folder(items_path, key_path):
        raise ValueError("blind key must live outside the folder containing reader items")
    primary_chunks = _reader_chunk_dir(items_path, "reader1")
    second_chunks = _reader_chunk_dir(items_path, "reader2")
    if not force and (items_path.exists() or key_path.exists() or primary_chunks.exists() or second_chunks.exists()):
        raise FileExistsError("items or key already exists; pass --force to overwrite")
    if force:
        import shutil
        for path in (primary_chunks, second_chunks):
            if path.exists():
                shutil.rmtree(path)
    triaged = json.loads(Path(triage_path).read_text(encoding="utf-8"))
    actual_triage = triage86.triage(screen_dir)
    if actual_triage != triaged:
        raise ValueError("triage counts do not match the supplied screen answers")

    changed_records = []
    miss_candidates = {(source, category): [] for source in ("real", "random") for category in ("kept", "broken")}
    for row in triage86.read_screen(screen_dir):
        source = "real" if row["kind"] == "neuron" else "random"
        category = triage86.triage_answer(row.get("answer", ""))
        if category == "changed":
            changed_records.append((row, category, False))
        else:
            miss_candidates[(source, category)].append(row)
    chooser = random.Random(c.MISS_SAMPLE_SEED)
    miss_records = []
    for source in ("real", "random"):
        for category in ("kept", "broken"):
            candidates = miss_candidates[(source, category)]
            chooser.shuffle(candidates)
            miss_records.extend((row, category, True) for row in candidates[:200])
    records = changed_records + miss_records
    secrets.SystemRandom().shuffle(records)
    mapping = {}
    public = []
    used = set()
    for row, category, sampled in records:
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
            "triage_category": category,
            "miss_rate_sample": sampled,
        }

    # Select by hidden screen identity, not random opaque ids, so seed 8687 is reproducible.
    stable_ids = sorted(mapping, key=lambda item_id: (
        int(mapping[item_id]["layer"]), mapping[item_id]["source"],
        int(mapping[item_id]["item_index"]), mapping[item_id]["triage_category"],
    ))
    second_count = max(1, int(round(len(stable_ids) * 0.10))) if stable_ids else 0
    second_ids = set(random.Random(c.SECOND_READER_SEED).sample(stable_ids, second_count))
    items_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in public).encode("utf-8")
    key = {
        "items_sha256": hashlib.sha256(encoded).hexdigest(),
        "rows": mapping,
        "triage_counts": triaged["layers"],
        "second_reader_ids": sorted(second_ids),
    }
    # Keep the private key and reader chunks outside the screen generation path.
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    items_path.write_bytes(encoded)
    key["manifest"] = c.write_manifest(screen_dir, items_path=items_path)
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    public_by_id = {row["id"]: row for row in public}
    _write_chunks(primary_chunks, public, chunk_size)
    _write_chunks(second_chunks, [public_by_id[item_id] for item_id in sorted(second_ids)], chunk_size)
    return len(public)


def _label_class(label: dict) -> str:
    """One mapping: false coherent -> not_coherent; else unclear/still-Qwen kinds are separate; all else is persona."""
    if not label["coherent"]:
        return "not_coherent"
    if label["kind"] == "unclear":
        return "unclear"
    if label["kind"] == "still Qwen or an assistant":
        return "coherent_still_qwen"
    return "coherent_persona"


def _validate_labels(labels, mapping, expected_ids=None):
    seen = {}
    required = {"id", "persona", "kind", "coherent"}
    for row in labels:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("each label must have exactly id, persona, kind, coherent")
        item_id = row["id"]
        if item_id not in mapping:
            raise ValueError(f"unknown label id {item_id!r}")
        if expected_ids is not None and item_id not in expected_ids:
            raise ValueError(f"label id {item_id!r} was not selected for this reader")
        if item_id in seen:
            raise ValueError(f"duplicate label for {item_id}")
        if not isinstance(row["persona"], str) or not row["persona"].strip():
            raise ValueError(f"invalid persona for {item_id}")
        if row["kind"] not in KINDS:
            raise ValueError(f"invalid kind for {item_id}: {row['kind']!r}")
        if type(row["coherent"]) is not bool:
            raise ValueError(f"coherent must be a JSON boolean for {item_id}")
        seen[item_id] = row
    wanted = set(mapping) if expected_ids is None else set(expected_ids)
    if set(seen) != wanted:
        missing = wanted - set(seen)
        raise ValueError(f"missing labels for {len(missing)} ids")
    return seen


def cohen_kappa(labels_a, labels_b, item_ids) -> float:
    """Cohen's kappa for the binary coherent-persona classification."""
    ids = list(item_ids)
    if not ids:
        raise ValueError("kappa needs at least one double-read item")
    left = [int(_label_class(labels_a[item_id]) == "coherent_persona") for item_id in ids]
    right = [int(_label_class(labels_b[item_id]) == "coherent_persona") for item_id in ids]
    observed = sum(a == b for a, b in zip(left, right)) / len(ids)
    p_left = sum(left) / len(ids)
    p_right = sum(right) / len(ids)
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def label_counts(labels, key, triage_counts):
    """Count the golden-verdict fields from labels, hidden key metadata, and screen triage counts."""
    mapping = key.get("rows")
    if not isinstance(mapping, dict):
        raise ValueError("key has no rows mapping")
    seen = _validate_labels(labels, mapping)
    layers = {}
    for layer, groups in triage_counts.items():
        layers[str(layer)] = {
            "real": {"n": int(groups.get("real", {}).get("n", -1)), "coherent_persona": 0},
            "random": {
                "n": int(groups.get("random", {}).get("n", -1)), "changed_labeled": 0,
                "not_coherent": 0, "unclear": 0, "coherent_persona": 0,
                "coherent_still_qwen": 0,
                "triage_broken": int(groups.get("random", {}).get("broken", 0)),
            },
        }
    miss_sample = {
        source: {category: {"sampled": 0, "coherent_persona": 0, "rate": None}
                 for category in ("kept", "broken")}
        for source in ("real", "random")
    }
    observed_changed = {}
    for item_id, label in seen.items():
        info = mapping[item_id]
        layer = str(int(info["layer"]))
        source = info.get("source")
        if source not in ("real", "random") or layer not in layers:
            raise ValueError(f"invalid blind-key source or layer for {item_id}")
        category = info.get("triage_category", "changed")
        label_class = _label_class(label)
        if category == "changed" and not info.get("miss_rate_sample", False):
            group = layers[layer][source]
            if source == "random":
                group["changed_labeled"] += 1
                group[label_class] += 1
            elif label_class == "coherent_persona":
                group["coherent_persona"] += 1
            observed_changed[(layer, source)] = observed_changed.get((layer, source), 0) + 1
        else:
            if category not in ("kept", "broken") or not info.get("miss_rate_sample", False):
                raise ValueError(f"unexpected triage category {category!r} for blind item {item_id}")
            sample = miss_sample[source][category]
            sample["sampled"] += 1
            sample["coherent_persona"] += int(label_class == "coherent_persona")
    for layer, groups in triage_counts.items():
        for source in ("real", "random"):
            expected_changed = int(groups.get(source, {}).get("changed", 0))
            observed = observed_changed.get((str(layer), source), 0)
            if observed != expected_changed:
                raise ValueError(f"L{layer} {source}: labeled {observed} changed answers, triage has {expected_changed}")
    for categories in miss_sample.values():
        for sample in categories.values():
            sample["rate"] = (
                sample["coherent_persona"] / sample["sampled"] if sample["sampled"] else None
            )
    return {"layers": layers, "triage_miss_rates": miss_sample}


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
        if answer is None:
            raise ValueError(f"blind key item {item_id} is absent from screen answers")
        actual_category = triage86.triage_answer(answer)
        expected_category = info.get("triage_category", "changed")
        if actual_category != expected_category:
            raise ValueError(f"blind key item {item_id} triage category differs from its screen answer")
        if expected_category == "changed" and info.get("miss_rate_sample", False):
            raise ValueError(f"changed item {item_id} cannot be a triage miss-rate sample")
        if expected_category != "changed" and not info.get("miss_rate_sample", False):
            raise ValueError(f"non-changed item {item_id} must be a triage miss-rate sample")
        digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        if digest != info.get("answer_sha256") or answer != item_text[item_id]:
            raise ValueError(f"blind item {item_id} no longer matches its screen answer")
    seen = _validate_labels(labels, mapping)
    counts = label_counts(labels, key, actual_triage["layers"])
    return {**counts, "labels": seen, "manifest": manifest,
            "scorer_manifest": {"code_sha256": c.scorer_code_sha256(),
                                "screen_manifest_sha256": manifest["manifest_sha256"]}}


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
    minimum_count_ok = real_count >= 10
    holds = real_rate >= 2 * random_rate and minimum_count_ok and p < 0.05
    return {
        "status": "HOLDS" if holds else "FAILS",
        "detail": f"real={real_count}/{real_n} ({real_rate:.6f}), random={random_count}/{random_n} ({random_rate:.6f}), Fisher one-sided p={p:.8g}",
        "fisher_one_sided_p": p,
        "minimum_real_count_met": minimum_count_ok,
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
            not_coherent = int(group.get("not_coherent", 0))
            unclear = int(group.get("unclear", 0))
            read[str(layer)] = {
                "not_coherent_plus_unclear": not_coherent + unclear,
                "not_coherent": not_coherent,
                "unclear": unclear,
                "coherent_persona": coherent,
                "coherent_still_qwen": int(group.get("coherent_still_qwen", 0)),
                "changed_labeled": n,
            }
    failures = {layer: row for layer, row in read.items()
                if row["not_coherent_plus_unclear"] <= row["coherent_persona"]}
    if failures:
        status = "FAILS"
    elif len(read) >= 9:
        status = "HOLDS"
    else:
        status = "NOT READ"
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


def _target_offworld(screen_dir):
    target_row = None
    for row in triage86.read_screen(screen_dir):
        if row.get("kind") == "neuron" and int(row.get("layer", -1)) == 22 and int(row.get("id", -1)) == 2742:
            target_row = row
            break
    if target_row is None:
        return {
            "status": "NOT READ", "reading": "hand-read",
            "automatic_match": {"status": "NOT READ", "authoritative": False},
            "detail": "L22 N2742 is absent from screen data",
        }
    automatic = verdict_861(target_row.get("answer", ""))
    return {
        "status": "NOT READ",
        "reading": "hand-read",
        "hand_read_required": True,
        "answer": target_row.get("answer", ""),
        "automatic_match": {**automatic, "authoritative": False},
        "detail": "Read the raw answer by hand; the automatic match is a non-authoritative convenience.",
    }


def verdicts_from_counts(counts):
    """Apply the registered rules to counts only; this function has no file or model dependencies."""
    layers = counts.get("layers", counts)
    if not isinstance(layers, dict):
        raise ValueError("verdict counts must contain a layer mapping")
    normalized = {}
    for layer, groups in layers.items():
        layer = str(int(layer))
        if not isinstance(groups, dict) or not all(name in groups for name in ("real", "random")):
            raise ValueError(f"L{layer} counts must contain real and random groups")
        real, random_group = groups["real"], groups["random"]
        for group_name, group, expected_n in (("real", real, 9216), ("random", random_group, 512)):
            if not isinstance(group, dict) or type(group.get("n")) is not int or group["n"] != expected_n:
                got = group.get("n") if isinstance(group, dict) else None
                raise ValueError(f"L{layer} {group_name} denominator must be {expected_n}; got {got}")
        for field in ("coherent_persona",):
            if type(real.get(field)) is not int or not 0 <= real[field] <= real["n"]:
                raise ValueError(f"L{layer} real {field} must be an integer within its denominator")
        required = ("changed_labeled", "not_coherent", "unclear", "coherent_persona", "coherent_still_qwen")
        for field in required:
            if type(random_group.get(field)) is not int or random_group[field] < 0:
                raise ValueError(f"L{layer} random {field} must be a nonnegative integer")
        if random_group["changed_labeled"] > random_group["n"]:
            raise ValueError(f"L{layer} random changed labels exceed denominator")
        if sum(random_group[field] for field in required[1:]) != random_group["changed_labeled"]:
            raise ValueError(f"L{layer} random label categories do not sum to changed_labeled")
        normalized[layer] = {"real": real, "random": random_group}

    expected_layers = [str(i) for i in range(10, 27)]
    missing = [layer for layer in expected_layers if layer not in normalized]
    rows = {}
    l22 = normalized.get("22")
    if l22 is None:
        rows["86.2"] = {"status": "NOT READ", "detail": "L22 counts are missing"}
    else:
        rows["86.2"] = verdict_862(
            l22["real"]["coherent_persona"], l22["random"]["coherent_persona"],
            l22["real"]["n"], l22["random"]["n"],
        )

    if missing:
        rows["86.3"] = {"status": "NOT READ", "detail": f"missing layers: {missing}"}
    else:
        violations = {
            layer: normalized[layer]["real"]["coherent_persona"] / normalized[layer]["real"]["n"]
            for layer in expected_layers
            if normalized[layer]["real"]["coherent_persona"] / normalized[layer]["real"]["n"] >= 0.05
        }
        rows["86.3"] = {
            "status": "FAILS" if violations else "HOLDS",
            "detail": f"layers at or above 5%: {violations}",
        }

    rows["86.4"] = verdict_864({layer: group["random"] for layer, group in normalized.items()})
    rows["86.4"]["triage_broken_by_layer"] = {
        layer: int(group["random"].get("triage_broken", 0)) for layer, group in normalized.items()
    }
    if missing:
        rows["86.4"]["missing_layers"] = missing

    early_ids, late_ids = [str(i) for i in range(14, 23)], [str(i) for i in range(23, 27)]
    if any(layer not in normalized for layer in early_ids + late_ids):
        rows["86.6"] = {"status": "NOT READ", "detail": "one or more pooled layers are missing"}
    else:
        early_count = sum(normalized[layer]["real"]["coherent_persona"] for layer in early_ids)
        early_n = sum(normalized[layer]["real"]["n"] for layer in early_ids)
        late_count = sum(normalized[layer]["real"]["coherent_persona"] for layer in late_ids)
        late_n = sum(normalized[layer]["real"]["n"] for layer in late_ids)
        rows["86.6"] = verdict_866(early_count, early_n, late_count, late_n)
    return {"verdicts": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    prep = sp.add_parser("prepare", help="shuffle changed answers into blinded items")
    prep.add_argument("--screen-dir", required=True)
    prep.add_argument("--triage", required=True)
    prep.add_argument("--items", required=True)
    prep.add_argument("--key", required=True)
    prep.add_argument("--chunk-size", type=int, default=500)
    prep.add_argument("--force", action="store_true")
    score_p = sp.add_parser("score", help="validate blind labels and write rates/verdicts")
    score_p.add_argument("--screen-dir", required=True)
    score_p.add_argument("--items", required=True)
    score_p.add_argument("--key", required=True)
    score_p.add_argument("--labels", required=True, help="JSONL rows: id, persona, kind, coherent")
    score_p.add_argument("--labels2", help="second-reader JSONL for the seed-8687 overlap sample")
    score_p.add_argument("--selected-set", help="picked20 JSON used to bind agreement to verification provenance")
    score_p.add_argument("--out", required=True)
    score_p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "prepare":
        n = prepare(args.screen_dir, args.triage, args.items, args.key, args.force, args.chunk_size)
        print(f"{n} changed and triage-miss sample answers blinded; reader chunks are at most {args.chunk_size}")
        return
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    labels = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    score_path = Path(args.out)
    artifact_path = Path(str(score_path) + ".manifest.json")
    if (score_path.exists() or artifact_path.exists()) and not args.force:
        raise FileExistsError(f"{args.out} exists; pass --force to overwrite")
    result = score(labels, key, args.items, args.screen_dir)
    _assert_screen_denominators(key["triage_counts"])
    final = verdicts_from_counts(result)
    verdict_rows = final["verdicts"]
    verdict_rows["86.1"] = _target_offworld(args.screen_dir)
    second_ids = set(key.get("second_reader_ids", []))
    if args.labels2:
        labels2 = [json.loads(line) for line in Path(args.labels2).read_text(encoding="utf-8").splitlines() if line.strip()]
        # The registered second reading is the seed-8687 10% sample; a full second reading is also accepted.
        # (Claude 2026-10-28: this used to validate against every row, so the 10% file was always refused.)
        label2_ids = {row.get("id") for row in labels2 if isinstance(row, dict)}
        if label2_ids == second_ids:
            second_seen = _validate_labels(labels2, key["rows"], expected_ids=second_ids)
        elif label2_ids == set(key["rows"]):
            second_seen = _validate_labels(labels2, key["rows"])
        else:
            raise ValueError("second-reader labels must cover the seed-8687 sample or the full item set")
        scope = "full second reading" if set(second_seen) == set(key["rows"]) else "seed-8687 10% sample"
        kappa_ids = set(second_seen)
        kappa = cohen_kappa(result["labels"], second_seen, kappa_ids)
        agreement = {"method": "Cohen's kappa", "items": len(kappa_ids), "scope": scope,
                     "kappa": kappa, "threshold": 0.6}
    else:
        agreement = {"method": "Cohen's kappa", "items": 0, "kappa": None, "threshold": 0.6,
                     "status": "NOT READ: second-reader labels are required"}
    if not c.kappa_reads(agreement["kappa"]):
        reason = "second-reader agreement is missing" if agreement["kappa"] is None else f"Cohen's kappa {agreement['kappa']:.4f} is below 0.6"
        for name in ("86.2", "86.3", "86.4", "86.6"):
            verdict_rows[name] = {"status": "NOT READ", "detail": reason}
    selected_set_sha256 = c.sha256_file(Path(args.selected_set)) if args.selected_set else None
    score_manifest = {
        "code_sha256": c.scorer_code_sha256(),
        "screen_manifest_sha256": result["manifest"]["manifest_sha256"],
        "items_sha256": result["manifest"]["items_sha256"],
        "labels_sha256": c.sha256_file(Path(args.labels)),
        "primary_labels_sha256": c.sha256_file(Path(args.labels)),
        "selected_set_sha256": selected_set_sha256,
        "labels2_sha256": c.sha256_file(Path(args.labels2)) if args.labels2 else None,
        "kappa": agreement["kappa"],
    }
    score_manifest["manifest_sha256"] = c.canonical_sha256(score_manifest)
    output = {
        "manifest": result["manifest"],
        "score_manifest": score_manifest,
        "items_sha256": key["items_sha256"],
        "labeled_items": len(labels),
        "reader_agreement": agreement,
        "triage_miss_rates": result["triage_miss_rates"],
        "counts": result["layers"],
        **final,
    }
    c.atomic_json(score_path, output)
    artifact_manifest = {
        "score_file": score_path.name,
        "score_file_sha256": c.sha256_file(score_path),
        "score_manifest_sha256": score_manifest["manifest_sha256"],
        "screen_manifest_sha256": score_manifest["screen_manifest_sha256"],
        "primary_labels_sha256": score_manifest["primary_labels_sha256"],
        "selected_set_sha256": score_manifest["selected_set_sha256"],
    }
    artifact_manifest["manifest_sha256"] = c.canonical_sha256(artifact_manifest)
    c.atomic_json(artifact_path, artifact_manifest)
    if agreement["kappa"] is not None:
        print(f"reader agreement Cohen's kappa={agreement['kappa']:.4f} on {agreement['items']} items", flush=True)
    else:
        print("reader agreement NOT READ: second-reader labels are required", flush=True)
    for source, categories in output["triage_miss_rates"].items():
        for category, stats in categories.items():
            print(f"triage miss sample {source} {category}: {stats['coherent_persona']}/{stats['sampled']} "
                  f"({stats['rate'] if stats['rate'] is not None else 'n/a'})", flush=True)
    for name, row in output["verdicts"].items():
        print(f"{name} {row['status']}: {row['detail']}")


if __name__ == "__main__":
    main()
