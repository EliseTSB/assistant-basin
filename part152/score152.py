#!/usr/bin/env python3
"""Score Part 152's registered gates, predictions, and within-group shuffle controls."""
import argparse
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORD = re.compile(r"[A-Za-z]+")
ALIEN = ("alien", "extraterrestrial", "ufo", "martian")
STOP = {"language", "model", "large", "assistant", "that", "with", "from", "this", "your", "about",
        "have", "what", "developed", "created", "trained"}
CONTROL = "L22N2742"
AND_OR = "L19N4979"
AWARD_UNIT = "L24N8542"


def load_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def j_words(row):
    """Apply the registered J-list cleaning: trim, lowercase, letters only, length >= 4."""
    cleaned = set()
    for value in row.get("j", []):
        word = value.strip().lower()
        if len(word) >= 4 and re.fullmatch(r"[a-z]+", word):
            cleaned.add(word)
    return sorted(cleaned)


def persona_words(persona):
    return sorted({w.lower() for w in WORD.findall(persona or "") if len(w) >= 4 and w.lower() not in STOP})


def hit(entry, words):
    tokens = [w.lower() for w in WORD.findall(entry["mark"] + " " + entry["after"])]
    return any(token.startswith(prefix) for token in tokens for prefix in words)


def _prefix_trie(word_lists):
    """Map token prefixes to list indices without scanning every list for every window."""
    root = {}
    for idx, words in enumerate(word_lists):
        for word in words:
            node = root
            for char in word:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(idx)
    return root


def _matching_lists(token, trie):
    node = trie
    matches = []
    for char in token.lower():
        node = node.get(char)
        if node is None:
            break
        matches.extend(node.get(None, ()))
    return matches


def sparse_incidence(entries_by_unit, word_lists):
    """Sparse row maps: each unit's window hit count against only matching word lists."""
    trie = _prefix_trie(word_lists)
    rows = []
    for entries in entries_by_unit:
        counts = {}
        for entry in entries:
            indices = set()
            for token in WORD.findall(entry["mark"] + " " + entry["after"]):
                indices.update(_matching_lists(token, trie))
            for idx in indices:
                counts[idx] = counts.get(idx, 0) + 1
        rows.append(counts)
    return rows


def shuffle_counts(rows, seed, shuffles, direct=None):
    """Permute lists within one group; return per-shuffle group sums and verify identity."""
    import numpy as np
    n = len(rows)
    identity = sum(rows[i].get(i, 0) for i in range(n))
    if direct is None:
        direct = identity
    if identity != direct:
        raise AssertionError("identity shuffle differs from direct own-hit count")
    rng = np.random.default_rng(seed)
    totals = np.empty(shuffles, dtype=np.int64)
    for s in range(shuffles):
        perm = rng.permutation(n)
        totals[s] = sum(rows[i].get(int(perm[i]), 0) for i in range(n))
    return direct, totals


def layer_shuffle_counts(entries_by_unit, words_by_unit, layer_keys, seed, shuffles):
    """Combine sparse null counts after independently permuting lists within each layer."""
    import numpy as np
    groups = {}
    for i, layer in enumerate(layer_keys):
        groups.setdefault(layer, []).append(i)
    direct = _direct_own_hits(entries_by_unit, words_by_unit)
    totals = np.zeros(shuffles, dtype=np.int64)
    rng = np.random.default_rng(seed)
    own_hits = [0] * len(entries_by_unit)
    for indices in groups.values():
        entries = [entries_by_unit[i] for i in indices]
        lists = [words_by_unit[i] for i in indices]
        rows = sparse_incidence(entries, lists)
        identity = sum(rows[i].get(i, 0) for i in range(len(indices)))
        expected = sum(sum(hit(entry, lists[i]) for entry in entries[i]) for i in range(len(indices)))
        if identity != expected:
            raise AssertionError("identity layer shuffle differs from direct own-hit count")
        for local, original in enumerate(indices):
            own_hits[original] = rows[local].get(local, 0)
        for shuffle in range(shuffles):
            perm = rng.permutation(len(indices))
            totals[shuffle] += sum(rows[i].get(int(perm[i]), 0) for i in range(len(indices)))
    if sum(own_hits) != direct:
        raise AssertionError("layer identity total differs from direct own-hit count")
    return direct, totals, own_hits


def _part151b_rules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from part151b.score151b import marks_and_or, mentions_award
    return marks_and_or, mentions_award


def _direct_own_hits(entries_by_unit, words_by_unit):
    return sum(sum(hit(entry, words) for entry in entries)
               for entries, words in zip(entries_by_unit, words_by_unit))


def _hit_location(entry, words):
    """Return the earliest text location with an own-list prefix hit, if any."""
    mark_tokens = [w.lower() for w in WORD.findall(entry["mark"])]
    after_tokens = [w.lower() for w in WORD.findall(entry["after"])]
    matches = lambda tokens: any(token.startswith(prefix) for token in tokens for prefix in words)
    if matches(mark_tokens):
        return "marked_token"
    if matches(after_tokens[:1]):
        return "first_word_after"
    if matches(after_tokens[1:]):
        return "later_after"
    return None


def _mean_shuffle(entries_by_unit, words_by_unit, layer_keys, seed, shuffles):
    """Return the own-list mean, within-layer shuffled mean and upper-tail p."""
    import numpy as np
    layers = {}
    for i, layer in enumerate(layer_keys):
        layers.setdefault(layer, []).append(i)
    direct = _direct_own_hits(entries_by_unit, words_by_unit)
    rng = np.random.default_rng(seed)
    totals = np.zeros(shuffles, dtype=np.int64)
    for indices in layers.values():
        rows = sparse_incidence([entries_by_unit[i] for i in indices],
                                [words_by_unit[i] for i in indices])
        identity = sum(rows[i].get(i, 0) for i in range(len(indices)))
        expected = sum(sum(hit(entry, words_by_unit[original])
                           for entry in entries_by_unit[original]) for original in indices)
        if identity != expected:
            raise AssertionError("identity layer shuffle differs from direct own-hit count")
        for shuffle in range(shuffles):
            perm = rng.permutation(len(indices))
            totals[shuffle] += sum(rows[i].get(int(perm[i]), 0) for i in range(len(indices)))
    real_mean = direct / len(entries_by_unit) if entries_by_unit else 0.0
    null_means = totals / max(1, len(entries_by_unit))
    return {"n": len(entries_by_unit), "real_mean": real_mean,
            "shuffle_mean": float(null_means.mean()) if len(null_means) else 0.0,
            "p": (1 + int(np.count_nonzero(null_means >= real_mean))) / (shuffles + 1)}


def prediction_152_4(slot_mean, concept_mean):
    return slot_mean < 0.5 * concept_mean


def score(windows, lens, units112, kinds112, shuffles=1000):
    import numpy as np
    units = windows["units"]
    marks_and_or, mentions_award = _part151b_rules()

    # Connect lens keys to each page unit and to Part 112's blind kind/persona.
    lens_by_name = {f"L{int(k.split(':')[0])}N{int(k.split(':')[1])}": value for k, value in lens.items()}
    atlas = {(int(row["layer"]), int(row["neuron"])): row for row in units112}
    sorted_kinds = kinds112.get("units", {})
    records = {}
    for name in units:
        match = re.fullmatch(r"L(\d+)N(\d+)", name)
        key = (int(match.group(1)), int(match.group(2)))
        row = atlas.get(key)
        kind_row = sorted_kinds.get(row["id"], {}) if row else {}
        records[name] = {"j": j_words(lens_by_name.get(name, {})),
                         "persona": persona_words(row.get("persona", "") if row else ""),
                         "kind": kind_row.get("kind")}

    # Gates are fixed before looking at any scored results.
    alien = sum(any(re.search(rf"\b{term}", e["before"] + e["mark"] + e["after"], re.IGNORECASE)
                    for term in ALIEN) for e in units.get(CONTROL, {}).get("top", []))
    and_or_count = sum(marks_and_or(e) for e in units.get(AND_OR, {}).get("top", []))
    award_count = sum(mentions_award(e) for e in units.get(AWARD_UNIT, {}).get("top", []))
    gates = {"gate1_docs": windows.get("docs") == 10000,
             "gate2_full_lists": bool(units) and all(len(v.get("top", [])) == 20 and len(v.get("bottom", [])) == 20
                                                     for v in units.values()),
             "gate3_alien_control": alien >= 5,
             "gate4_part151b_controls": and_or_count >= 12 and award_count >= 12}

    j_names = [n for n in units if len(records[n]["j"]) >= 3 and len(units[n].get("top", [])) == 20]
    j_left_out = sum(len(records[n]["j"]) < 3 for n in units)
    j_words_by = [records[n]["j"] for n in j_names]
    j_entries = [units[n]["top"] for n in j_names]
    j_layers = [int(name[1:name.index("N")]) for name in j_names]
    j_real_total, j_shuffle_totals, j_hits = layer_shuffle_counts(
        j_entries, j_words_by, j_layers, 152, shuffles)
    j_real_mean = j_real_total / len(j_names) if j_names else 0.0
    j_shuffle_means = j_shuffle_totals / max(1, len(j_names))
    j_shuffle_mean = float(j_shuffle_means.mean()) if len(j_shuffle_means) else 0.0
    j_p = (1 + int(np.count_nonzero(j_shuffle_means >= j_real_mean))) / (shuffles + 1)
    j_3_share = sum(v >= 3 for v in j_hits) / len(j_names) if j_names else 0.0

    # Exploratory analyses were chosen after seeing the registered verdicts.
    locations = {"marked_token": 0, "first_word_after": 0, "later_after": 0}
    zero_hit_units = 0
    for entries, words in zip(j_entries, j_words_by):
        own_hits = 0
        for entry in entries:
            location = _hit_location(entry, words)
            if location:
                locations[location] += 1
                own_hits += 1
        zero_hit_units += own_hits == 0

    bottom_entries = [units[n]["bottom"] for n in j_names]
    bottom_result = _mean_shuffle(bottom_entries, j_words_by, j_layers, 153, shuffles)
    by_layer = {}
    for layer in sorted(set(j_layers)):
        indices = [i for i, value in enumerate(j_layers) if value == layer]
        if len(indices) >= 30:
            layer_result = _mean_shuffle([j_entries[i] for i in indices],
                                         [j_words_by[i] for i in indices],
                                         [layer] * len(indices), 152, shuffles)
            by_layer[str(layer)] = layer_result

    persona_results = {}
    persona_left_out = {kind: sum(records[n]["kind"] == kind and not records[n]["persona"]
                                  for n in units) for kind in ("concept", "slot")}
    for kind in ("concept", "slot"):
        group = [n for n in units if records[n]["kind"] == kind and records[n]["persona"]
                 and len(units[n].get("top", [])) == 20]
        words_by = [records[n]["persona"] for n in group]
        entries_by = [units[n]["top"] for n in group]
        rows = sparse_incidence(entries_by, words_by)
        direct = _direct_own_hits(entries_by, words_by)
        real_total, null_totals = shuffle_counts(rows, 152, shuffles, direct=direct)
        real_mean = real_total / len(group) if group else 0.0
        null_means = null_totals / max(1, len(group))
        null_mean = float(null_means.mean()) if len(null_means) else 0.0
        p = (1 + int(np.count_nonzero(null_means >= real_mean))) / (shuffles + 1)
        persona_results[kind] = {"n": len(group), "real_mean": real_mean,
                                 "shuffle_mean": null_mean, "p": p}

    concept_mean = persona_results["concept"]["real_mean"]
    slot_mean = persona_results["slot"]["real_mean"]
    held = {"152.1": j_real_mean >= 2 * j_shuffle_mean and j_p < 0.01,
            "152.2": j_3_share >= 0.10,
            "152.3": concept_mean >= 2 * persona_results["concept"]["shuffle_mean"]
                     and persona_results["concept"]["p"] < 0.01,
            "152.4": prediction_152_4(slot_mean, concept_mean)}
    predictions = {name: ("held" if value else "failed") if all(gates.values()) else "untestable"
                  for name, value in held.items()}

    page = {}
    for name, value in units.items():
        page[name] = {}
        for side, limit in (("top", 5), ("bottom", 3)):
            rows = sorted(value.get(side, []), key=lambda e: e["act"], reverse=(side == "top"))[:limit]
            page[name][side] = [{"act": e["act"], "doc": e["doc"], "pos": e["pos"],
                                 "before": e["before"][-60:], "mark": e["mark"], "after": e["after"][:40]}
                                for e in rows]

    # Added after the cold check, so exploratory: hits by where they fall, and persona words not in the J list.
    def _windowed(entries, keep):
        return [[keep(e) for e in unit] for unit in entries]
    mark_only = _windowed(j_entries, lambda e: {**e, "after": ""})
    mark_before = _windowed(j_entries, lambda e: {**e, "mark": e["before"] + " " + e["mark"], "after": ""})
    after_cold = {}
    for label, entries in (("mark_only", mark_only), ("mark_plus_before", mark_before)):
        total, shuffles_totals, per_unit = layer_shuffle_counts(entries, j_words_by, j_layers, 154, shuffles)
        after_cold[label] = {"real_mean": total / len(j_names), "shuffle_mean": float((shuffles_totals / len(j_names)).mean()),
                             "share_units_at_least_3": sum(v >= 3 for v in per_unit) / len(j_names)}
    for kind in ("concept", "slot"):
        group = [n for n in units if records[n]["kind"] == kind and records[n]["persona"] and len(units[n].get("top", [])) == 20]
        pairs = [(n, [w for w in records[n]["persona"] if w not in records[n]["j"]]) for n in group]
        pairs = [(n, w) for n, w in pairs if w]
        rows = sparse_incidence([units[n]["top"] for n, _ in pairs], [w for _, w in pairs])
        real, null = shuffle_counts(rows, 155, shuffles)
        after_cold[f"persona_not_in_j_{kind}"] = {"n": len(pairs), "real_mean": real / len(pairs),
                                                  "shuffle_mean": float((null / len(pairs)).mean())}

    result = {"gates": gates, "gate_controls": {"alien_control_top": alien,
              "L19N4979_and_or_top": and_or_count, "L24N8542_award_top": award_count},
              "predictions": predictions,
              "units_left_out": {"J_list_fewer_than_3_words": j_left_out,
                                 "persona_missing_words_by_kind": persona_left_out},
              "152.1": {"n": len(j_names), "real_mean": j_real_mean,
                        "shuffle_mean": j_shuffle_mean, "p": j_p},
              "152.2": {"n": len(j_names), "units_with_at_least_3_own_hits": sum(v >= 3 for v in j_hits),
                        "share": j_3_share},
              "152.3": persona_results["concept"],
              "152.4": {"slot_mean": slot_mean, "concept_mean": concept_mean,
                        "slot_n": persona_results["slot"]["n"], "concept_n": persona_results["concept"]["n"]},
              "exploratory": {"after_cold_check": after_cold,
                              "top_hit_location_counts": locations,
                              "bottom": bottom_result,
                              "top_by_layer_n_at_least_30": by_layer,
                              "zero_own_hit_units": zero_hit_units,
                              "zero_own_hit_share": zero_hit_units / len(j_names) if j_names else 0.0},
              "identity_shuffle_check": "passed", "shuffles": shuffles, "seed": 152}
    return result, page


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--lens", default=str(HERE / "lens.json"))
    ap.add_argument("--units", default=str(ROOT / "results/part112/units.json"))
    ap.add_argument("--kinds", default=str(ROOT / "results/part112/score.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffles", type=int, default=1000)
    a = ap.parse_args(argv)
    windows = load_json(a.windows)
    lens = load_json(a.lens)
    units112 = load_json(a.units)
    kinds112 = load_json(a.kinds)
    result, page = score(windows, lens, units112, kinds112, a.shuffles)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out.parent / "page-windows.json").write_text(json.dumps(page, ensure_ascii=False, separators=(",", ":")) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not all(result["gates"].values()):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
