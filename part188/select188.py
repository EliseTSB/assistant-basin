#!/usr/bin/env python3
"""Build the immutable Part 188 clip manifest from the pinned dataset metadata."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIS_COMMIT = "b77114c43c52f5faa7d0bb69b22e1ebca0632201"
VIS_META = Path("/mnt/e/neuron-probe/vocalimitationset/meta")
ESC_ROOT = Path("/mnt/e/neuron-probe/esc50")
ACTS_PATH = Path("/home/elisepc/green/Journal/notes/reports/embeddinggemma2-chicken-neurons-2026-11-03/acts.npz")
ESC_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
ESC_REPO = "https://github.com/karolpiczak/ESC-50"
VIS_REPO = "https://raw.githubusercontent.com/interactiveaudiolab/VocalImitationSet"
FIELDS = ["id", "set", "analysis_set", "source", "repository_path", "filename",
          "download_url", "class_key", "group", "phrase", "fold", "participant_id",
          "participant_group", "esc_category", "esc_fold", "parity"]


def _category_path(row: dict[str, str]) -> str:
    return " / ".join(row[f"category_d{i}"].strip() for i in range(2, 6)
                      if row.get(f"category_d{i}", "").strip())


def _sha_id(prefix: str, key: str) -> str:
    return prefix + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def assign_folds(weights: dict[str, int], seed: int = 188) -> dict[str, str]:
    """Shuffle sorted groups, stable-sort by size, then greedily balance A/B."""
    groups = sorted(weights)
    np.random.default_rng(seed).shuffle(groups)
    groups.sort(key=lambda group: weights[group], reverse=True)
    loads = {"A": 0, "B": 0}
    assigned: dict[str, str] = {}
    for group in groups:
        fold = "A" if loads["A"] <= loads["B"] else "B"
        assigned[group] = fold
        loads[fold] += int(weights[group])
    return assigned


def _read_vis(meta_dir: Path, classes: dict) -> tuple[list[dict], list[dict], list[dict]]:
    tree = json.loads((meta_dir / "tree.json").read_text(encoding="utf-8"))
    if tree.get("sha") != VIS_COMMIT:
        raise ValueError(f"VocalImitationSet tree is {tree.get('sha')!r}, expected {VIS_COMMIT}")
    blobs = sorted(item["path"] for item in tree["tree"]
                   if item.get("type") == "blob" and item.get("path", "").endswith(".wav"))
    file_paths = defaultdict(list)
    for path in blobs:
        file_paths[Path(path).name].append(path)
    with (meta_dir / "vocal_imitations.txt").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    class_map = {item["vis"]: item for item in classes["classes"] if item.get("vis")}
    excluded = set(classes.get("excluded_vis", []))
    id_to_path = {}
    imitations_by_class = defaultdict(list)
    meta_by_filename = {}
    for row in records:
        path = _category_path(row)
        cid = int(row["category_id"])
        id_to_path[cid] = path
        meta_by_filename[row["imitation_filename"]] = row
        if row.get("included", "").lower() != "true":
            continue
        if path in excluded:
            continue
        if path in class_map:
            imitations_by_class[path].append(row)

    rows: list[dict] = []
    for category, meta in imitations_by_class.items():
        spec = class_map[category]
        group = spec["group"]
        for item in sorted(meta, key=lambda r: r["imitation_filename"]):
            paths = file_paths.get(item["imitation_filename"], [])
            repo_path = next((p for p in paths if p.startswith("vocal_imitations/included/")), None)
            if repo_path is None:
                raise ValueError(f"included imitation absent from tree: {item['imitation_filename']}")
            rows.append({"set": "C_I" if group == "Chicken" else "I",
                         "analysis_set": "C_I" if group == "Chicken" else "I",
                         "source": "vis", "repository_path": repo_path, "filename": "",
                         "download_url": f"{VIS_REPO}/{VIS_COMMIT}/{urllib.parse.quote(repo_path, safe='/')}" ,
                         "class_key": category, "group": group, "phrase": spec["phrase"],
                         "fold": "", "participant_id": item.get("participant_id", ""),
                         "participant_group": item.get("participant_id", ""),
                         "esc_category": "", "esc_fold": "", "parity": "0"})

    # Non-animal classes are dataset classes whose top-level label is not Animal.
    nonanimal = defaultdict(list)
    all_paths = {_category_path(row): row for row in records}
    for item in records:
        if item.get("included", "").lower() != "true":
            continue
        category = _category_path(item)
        if (item.get("category_d1") == "Animal" or category in excluded or category in class_map or
                "cowbell" in category.lower()):
            continue
        if item["imitation_filename"] not in file_paths:
            continue
        nonanimal[category].append(item)
    rng = np.random.default_rng(188)
    for category in sorted(nonanimal):
        candidates = sorted(nonanimal[category], key=lambda r: r["imitation_filename"])
        picked = candidates[int(rng.integers(len(candidates)))]
        spec = {"group": "", "phrase": _phrase_from_path(category)}
        repo_path = next(p for p in file_paths[picked["imitation_filename"]]
                         if p.startswith("vocal_imitations/included/"))
        rows.append({"set": "N", "analysis_set": "N", "source": "vis",
                     "repository_path": repo_path, "filename": "",
                     "download_url": f"{VIS_REPO}/{VIS_COMMIT}/{urllib.parse.quote(repo_path, safe='/')}",
                     "class_key": category, "group": "", "phrase": spec["phrase"], "fold": "",
                     "participant_id": picked.get("participant_id", ""),
                     "participant_group": picked.get("participant_id", ""),
                     "esc_category": "", "esc_fold": "", "parity": "0"})

    # All original recordings, reference and non-reference, for retained VIS animal classes.
    for repo_path in blobs:
        if not repo_path.startswith("original_recordings/"):
            continue
        parts = repo_path.split("/")
        if len(parts) < 3:
            continue
        category_id = None
        if parts[1] == "reference":
            match = re.match(r"(\d{3})", Path(repo_path).name)
            category_id = int(match.group(1)) if match else None
        elif parts[1] == "non_reference" and len(parts) >= 4:
            match = re.match(r"(\d{3})", parts[2])
            category_id = int(match.group(1)) if match else None
        if category_id is None:
            continue
        category = id_to_path.get(category_id)
        if not category or category in excluded or category not in class_map:
            continue
        spec = class_map[category]
        group = spec["group"]
        base_set = "C_R" if group == "Chicken" else "R"
        rows.append({"set": base_set, "analysis_set": base_set, "source": "vis",
                     "repository_path": repo_path, "filename": "",
                     "download_url": f"{VIS_REPO}/{VIS_COMMIT}/{urllib.parse.quote(repo_path, safe='/')}",
                     "class_key": category, "group": group, "phrase": spec["phrase"],
                     "fold": "", "participant_id": "", "participant_group": "",
                     "esc_category": "", "esc_fold": "", "parity": "0"})
    return rows, records, blobs


def _phrase_from_path(category: str) -> str:
    leaf = category.rsplit(" / ", 1)[-1]
    return re.sub(r"[_-]+", " ", leaf).strip().lower()


def _read_esc(esc_root: Path, classes: dict) -> tuple[list[dict], list[dict]]:
    with (esc_root / "esc50.csv").open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    esc_specs = {item["esc"]: item for item in classes["classes"] if item.get("esc")}
    humans = set(classes["esc_human"])
    output = []
    for item in sorted(all_rows, key=lambda r: r["filename"]):
        category = item["category"]
        if category in esc_specs:
            spec = esc_specs[category]
            group = spec["group"]
            base_set = "C_R" if group == "Chicken" else "R"
            phrase = spec["phrase"]
        elif category in humans:
            spec = None
            group = ""
            base_set = "H"
            phrase = _phrase_from_path(category)
        else:
            spec = None
            group = ""
            base_set = "E"
            phrase = _phrase_from_path(category)
        output.append({"set": base_set, "analysis_set": base_set, "source": "esc",
                       "repository_path": "", "filename": item["filename"],
                       "download_url": f"{ESC_REPO}/archive/{ESC_COMMIT}.zip",
                       "class_key": category, "group": group, "phrase": phrase, "fold": "",
                       "participant_id": "", "participant_group": "", "esc_category": category,
                       "esc_fold": item["fold"], "parity": "0"})
    return output, all_rows


def choose_parity(rows: list[dict], acts_path: Path) -> list[str]:
    with np.load(acts_path, allow_pickle=False) as acts:
        ref = {str(key): (str(kind), int(fold)) for key, kind, fold in zip(acts["key"], acts["kind"], acts["fold"])
               if str(kind) == "esc"}
    esc_by_name = {r["filename"]: r for r in rows if r["source"] == "esc"}
    candidates = sorted(name for name, (_, _fold) in ref.items()
                       if name in esc_by_name and int(next(x["esc_fold"] for x in rows
                                                           if x["source"] == "esc" and x["filename"] == name)) == 1)
    selected: list[str] = []
    categories: set[str] = set()
    for name in candidates:
        category = esc_by_name[name]["esc_category"]
        if category in categories:
            continue
        selected.append(name)
        categories.add(category)
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError(f"found only {len(selected)} eligible parity clips in 8 distinct categories")
    return selected


def build_manifest(meta_dir: Path = VIS_META, esc_root: Path = ESC_ROOT,
                   acts_path: Path = ACTS_PATH) -> tuple[list[dict], dict]:
    classes = json.loads((HERE / "classes.json").read_text(encoding="utf-8"))
    vis_rows, _meta, _blobs = _read_vis(meta_dir, classes)
    esc_rows, esc_csv = _read_esc(esc_root, classes)
    rows = vis_rows + esc_rows

    # Group weights include every real and imitation clip, including mapped ESC-50 animals.
    weights = defaultdict(int)
    for row in rows:
        if row["group"] and row["group"] != "Chicken" and row["analysis_set"] in ("R", "I"):
            weights[row["group"]] += 1
    folds = assign_folds(weights)
    folds["Chicken"] = "C"
    for row in rows:
        if row["group"] == "Chicken":
            row["fold"] = "C"
        elif row["group"] in folds and row["analysis_set"] in ("R", "I"):
            row["fold"] = folds[row["group"]]

    parity_names = choose_parity(rows, acts_path)
    parity_set = set(parity_names)
    for row in rows:
        if row["source"] == "esc" and row["filename"] in parity_set:
            # P is the unique manifest set; analysis_set preserves the planned R/H/E membership.
            row["set"] = "P"
            row["parity"] = "1"

    # Stable IDs are derived from source identity, never from source listing order.
    for row in rows:
        key = row["repository_path"] if row["source"] == "vis" else row["filename"]
        row["id"] = _sha_id("V" if row["source"] == "vis" else "E", key)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("clip IDs are not unique")
    rows.sort(key=lambda r: (0 if r["parity"] == "1" else 1,
                             0 if r["source"] == "vis" else 1,
                             r["repository_path"] or r["filename"]))
    identities = [(r["source"], r["repository_path"] or r["filename"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("manifest contains duplicate source clips")
    for row in rows:
        for field in FIELDS:
            row.setdefault(field, "")
    # Counts include the primary analysis membership and the P overlay separately.
    set_counts = Counter(r["analysis_set"] for r in rows)
    manifest_counts = Counter(r["set"] for r in rows)
    fold_counts = {fold: Counter(r["analysis_set"] for r in rows if r["fold"] == fold)
                   for fold in ("A", "B")}
    fold_counts["C"] = Counter(r["analysis_set"] for r in rows if r["analysis_set"] in ("C_R", "C_I"))
    fold_counts["unfolded"] = Counter(r["analysis_set"] for r in rows if not r["fold"])
    folds_doc = {"seed": 188, "weights": dict(sorted(weights.items())), "assignments": dict(sorted(folds.items())),
                 "esc_commit": ESC_COMMIT,
                 "esc_csv_sha256": hashlib.sha256((esc_root / "esc50.csv").read_bytes()).hexdigest(),
                 "loads": {fold: sum(weights[g] for g, assigned in folds.items() if assigned == fold)
                           for fold in ("A", "B")},
                 "counts_by_analysis_set": {fold: dict(sorted(counts.items()))
                                            for fold, counts in fold_counts.items()},
                 "manifest_set_counts": dict(sorted(manifest_counts.items())),
                 "analysis_set_counts": dict(sorted(set_counts.items())),
                 "parity_filenames": parity_names,
                 "esc_metadata_rows": len(esc_csv)}
    return rows, folds_doc


def write_parity_reference(rows: list[dict], acts_path: Path, output: Path) -> None:
    """Save the eight selected rows exactly as stored by the earlier recording."""
    parity = [row for row in rows if row["parity"] == "1"]
    with np.load(acts_path, allow_pickle=False) as acts:
        lookup = {str(key): i for i, (kind, key) in enumerate(zip(acts["kind"], acts["key"]))
                  if str(kind) == "esc"}
        missing = [row["filename"] for row in parity if row["filename"] not in lookup]
        if missing:
            raise ValueError(f"parity clips missing from earlier acts.npz: {missing}")
        indexes = [lookup[row["filename"]] for row in parity]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output,
                            ids=np.asarray([row["id"] for row in parity]),
                            filenames=np.asarray([row["filename"] for row in parity]),
                            act_mean=acts["act_mean"][indexes].astype(np.float16, copy=True),
                            emb=acts["emb"][indexes].astype(np.float32, copy=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-dir", type=Path, default=VIS_META)
    parser.add_argument("--esc-root", type=Path, default=ESC_ROOT)
    parser.add_argument("--acts", type=Path, default=ACTS_PATH)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    rows, folds_doc = build_manifest(args.meta_dir, args.esc_root, args.acts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "manifest188.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "folds.json").write_text(json.dumps(folds_doc, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
    write_parity_reference(rows, args.acts, args.out_dir / "parity_ref.npz")
    print("manifest sets (analysis membership):", json.dumps(folds_doc["analysis_set_counts"], sort_keys=True))
    print("manifest sets (exclusive row set):", json.dumps(folds_doc["manifest_set_counts"], sort_keys=True))
    print("fold loads:", json.dumps(folds_doc["loads"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
