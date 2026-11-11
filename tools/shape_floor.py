#!/usr/bin/env python3
"""Sensitivity of a part's verdicts to batch-shape label noise.

A part that reads generated text has a "shape replicate": some rows decoded twice, at two batch shapes, with the
part's labels computed on both answers. This tool takes the part's rows (with labels), those replicate pairs and the
part's scorer, redraws the labels many times and rescores. It reports, for each verdict, the share of redraws in
which the verdict comes out as observed.

How a redraw moves a row. A row whose joint label vector is s gets a transition drawn from the replicate pairs that
started at s: with probability changed(s)/n(s) it changes, and then its new vector is drawn from the changed pairs'
destinations. So labels that moved together in the replicate move together here, and each direction keeps its own
count and denominator. The assumption is that the replicate's transitions, given the starting vector, carry over to
the target rows under the same pair of shapes. Labels do not fully determine how a row reacts to a shape change.

Scenarios, all reported:
  independent                 every row draws on its own.
  shared-change-0.5, -1       with that probability (a mixture probability, not an estimated correlation), the rows
                              of one cluster (--cluster-key) share ONE uniform number for the event "this row
                              changes"; destinations given a change are still drawn per row. Sharing the change
                              event, and not a rank over destinations, makes the coupling independent of the order
                              in which label vectors happen to be listed.
  upper-marginal-independent  independent per-label flips at each direction's upper 95% Clopper-Pearson bound. A
                              stress on the marginal rates, not an upper bound on how unstable a verdict can be.

What is not measured is said, not filled in:
  - A row whose starting vector no replicate pair has ("unsupported") falls back to independent per-label flips at
    the conditional point rates. Each verdict is also scored with those rows held as observed; if the two figures
    differ by more than two standard errors of their paired per-redraw difference (both come from the same redraws),
    the verdict depends on unsupported rows and is reported so.
  - A label value that no replicate pair started with has an unknown flip rate. It is held, and counted as
    unmeasured. Zero observed transitions are not evidence that a transition cannot happen.
  - With one replicate pair for a vector, that vector's redraw is deterministic. The support table shows every n.
  - The shared-change coupling does not reach unsupported rows: they draw on their own in every scenario, so a
    cluster that holds one can still split under shared-change-1.

The output is a sensitivity conditional on the scenario and on the replicate's pair of shapes. It is not a
confidence level and not a guarantee.

Rows:  [{"id": ..., "cell": ..., "labels": {name: bool, ...}, <cluster key>: ...}, ...]
Pairs: [{"labels_a": {...}, "labels_b": {...}, <cluster key>: ...}, ...]   (labels_a = the shape the rows were run at)
Scorer: a function from the list of rows to {verdict name: bool or str}. It must not modify the rows.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCENARIOS = (
    ("independent", 0.0, "every row draws its transition on its own"),
    ("shared-change-0.5", 0.5, "with probability 0.5 a cluster's rows share one uniform number for the change event"),
    ("shared-change-1", 1.0, "a cluster's rows always share one uniform number for the change event"),
    ("upper-marginal-independent", None, "independent per-label flips at each direction's upper 95% bound"),
)
DISPLAY_THRESHOLD = 0.95


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))


def bisect(fn, target, lo=0.0, hi=1.0):
    for _ in range(70):
        mid = (lo + hi) / 2
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    if n < 0 or k < 0 or k > n:
        raise ValueError("invalid binomial count")
    if n == 0:
        return [None, None]
    lower = 0.0 if k == 0 else bisect(lambda p: 1 - binom_cdf(k - 1, n, p), alpha / 2)
    upper = 1.0 if k == n else bisect(lambda p: 1 - binom_cdf(k, n, p), 1 - alpha / 2)
    return [lower, upper]


def paired_difference(full_only, held_only, draws):
    """Difference between two shares measured on the SAME redraws, with its Monte Carlo standard error.

    full_only and held_only count the redraws in which only one of the two indicators was true. The per-draw
    difference is +1, -1 or 0, so its mean is (full_only - held_only) / draws and its standard error is the sample
    standard deviation of those differences over sqrt(draws). Treating the two shares as independent samples would
    leave out their covariance and overstate the error.
    """
    if draws < 1:
        raise ValueError("draws must be positive")
    mean = (full_only - held_only) / draws
    if draws < 2:
        return mean, 0.0
    variance = max(0.0, ((full_only + held_only) / draws - mean * mean) * draws / (draws - 1))
    return mean, math.sqrt(variance / draws)


def load_scorer(spec):
    if ":" not in spec:
        raise ValueError("scorer must be module:function")
    mod, fn = spec.split(":", 1)
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(mod), fn)


def _vector(labels):
    """A hashable, order-free form of a label dictionary."""
    return tuple(sorted((name, bool(value)) for name, value in labels.items()))


def direction_counts(label_names, pairs):
    """Per label and starting value: how many replicate pairs started there and how many flipped."""
    out = {}
    for name in label_names:
        out[name] = {}
        for start, direction in ((True, "true_to_false"), (False, "false_to_true")):
            usable = [p for p in pairs if name in p.get("labels_a", {}) and name in p.get("labels_b", {})
                      and bool(p["labels_a"][name]) == start]
            flips = sum(bool(p["labels_b"][name]) != start for p in usable)
            n = len(usable)
            out[name][direction] = {"count": flips, "denominator": n, "point_rate": flips / n if n else None,
                                    "95%_clopper_pearson": clopper_pearson(flips, n)}
    return out


def support_table(rows, pairs, cluster_key=None):
    """Per starting label vector: how many target rows have it, and what the replicate holds for it."""
    by_start = defaultdict(list)
    for pair in pairs:
        by_start[_vector(pair.get("labels_a", {}))].append(pair)
    targets = defaultdict(list)
    for row in rows:
        targets[_vector(row.get("labels", {}))].append(row)
    table = []
    for start, members in targets.items():
        have = by_start.get(start, [])
        destinations = Counter(_vector(p.get("labels_b", {})) for p in have)
        changed = sum(count for dest, count in destinations.items() if dest != start)
        clusters = None
        if cluster_key and have and all(cluster_key in p for p in have):
            clusters = len({p[cluster_key] for p in have})
        table.append({
            "start": dict(start),
            "target_rows": len(members),
            "target_rows_by_cell": dict(Counter(str(r.get("cell")) for r in members)),
            "replicate_pairs": len(have),
            "changed_pairs": changed,
            "destinations": [{"labels_b": dict(dest), "pairs": count}
                             for dest, count in sorted(destinations.items(), key=lambda item: (-item[1], item[0]))],
            "distinct_pair_clusters": clusters,
            "supported": bool(have),
        })
    table.sort(key=lambda entry: (-entry["target_rows"], sorted(entry["start"].items())))
    return table


def simulate(rows, pairs, scorer, draws=1000, seed=0, cluster_key=None):
    if draws < 1:
        raise ValueError("draws must be positive")
    if cluster_key and any(cluster_key not in row for row in rows):
        raise ValueError(f"cluster key {cluster_key!r} is missing from a row")
    observed = scorer(rows)
    if not isinstance(observed, dict) or any(not isinstance(v, (bool, str)) for v in observed.values()):
        raise ValueError("scorer must return a dictionary of bool or str verdicts")

    label_names = sorted({name for row in rows for name in row.get("labels", {})})
    directions = direction_counts(label_names, pairs)
    point = {name: {True: directions[name]["true_to_false"]["point_rate"],
                    False: directions[name]["false_to_true"]["point_rate"]} for name in label_names}
    upper = {name: {True: directions[name]["true_to_false"]["95%_clopper_pearson"][1],
                    False: directions[name]["false_to_true"]["95%_clopper_pearson"][1]} for name in label_names}

    by_start = defaultdict(list)
    for pair in pairs:
        by_start[_vector(pair.get("labels_a", {}))].append(pair)
    # For each supported starting vector: the chance of a change, and the changed pairs' destinations.
    change_rate, change_to = {}, {}
    for start, have in by_start.items():
        moved = sorted((_vector(p.get("labels_b", {})) for p in have if _vector(p.get("labels_b", {})) != start))
        change_rate[start] = len(moved) / len(have)
        change_to[start] = [dict(dest) for dest in moved]

    starts = [_vector(row.get("labels", {})) for row in rows]
    unsupported = [i for i, start in enumerate(starts) if start not in by_start]
    unsupported_set = set(unsupported)
    key = cluster_key or "id"
    cluster_of = [row.get(key, row.get("id", i)) for i, row in enumerate(rows)]
    cluster_ids = sorted(set(cluster_of), key=str)

    cells = sorted({str(row.get("cell")) for row in rows})
    cell_n = {cell: sum(str(row.get("cell")) == cell for row in rows) for cell in cells}
    observed_counts = {(name, cell): sum(bool(row.get("labels", {}).get(name, False)) for row in rows
                                         if str(row.get("cell")) == cell) for name in label_names for cell in cells}

    def marginal_flips(row, rates, rng, tally):
        """Independent per-label flips; a label value with no measured rate is held and counted."""
        labels = dict(row.get("labels", {}))
        for name, value in row.get("labels", {}).items():
            rate = rates.get(name, {}).get(bool(value))
            if rate is None:
                tally["unmeasured_label_holds"] += 1
            elif rng.random() < rate:
                labels[name] = not bool(value)
        return dict(row, labels=labels)

    scenarios = {}
    for name, share, description in SCENARIOS:
        rng = random.Random(seed)
        same, same_held = Counter(), Counter()
        full_only, held_only = Counter(), Counter()
        tally = Counter()
        count_draws = defaultdict(list)
        for _ in range(draws):
            redrawn = list(rows)
            if share is None:  # marginal stress: every row, every label, at the upper bounds
                for i, row in enumerate(rows):
                    redrawn[i] = marginal_flips(row, upper, rng, tally)
                held = None
            else:
                shared = {}
                for cluster in cluster_ids:
                    if rng.random() < share:
                        shared[cluster] = rng.random()
                for i, row in enumerate(rows):
                    start = starts[i]
                    if i in unsupported_set:
                        redrawn[i] = marginal_flips(row, point, rng, tally)
                        continue
                    u = shared[cluster_of[i]] if cluster_of[i] in shared else rng.random()
                    if u < change_rate[start]:
                        redrawn[i] = dict(row, labels=dict(rng.choice(change_to[start])))
                held = None
                if unsupported:
                    held = list(redrawn)
                    for i in unsupported:
                        held[i] = rows[i]
            result = scorer(redrawn)
            for verdict, value in observed.items():
                same[verdict] += int(result.get(verdict) == value)
            if held is not None:
                result_held = scorer(held)
                for verdict, value in observed.items():
                    as_full, as_held = result.get(verdict) == value, result_held.get(verdict) == value
                    same_held[verdict] += int(as_held)
                    full_only[verdict] += int(as_full and not as_held)
                    held_only[verdict] += int(as_held and not as_full)
            per_cell = Counter()
            for row in redrawn:
                cell = str(row.get("cell"))
                for label in label_names:
                    per_cell[(label, cell)] += int(bool(row.get("labels", {}).get(label, False)))
            for pair_key in observed_counts:
                count_draws[pair_key].append(per_cell[pair_key])

        verdicts = {}
        for verdict, value in observed.items():
            share_same = same[verdict] / draws
            se = math.sqrt(share_same * (1 - share_same) / draws)
            entry = {"observed": value, "sensitivity": share_same, "mc_se": se,
                     "within_2_mc_se_of_0.95": abs(share_same - DISPLAY_THRESHOLD) <= 2 * se}
            if share is not None:
                if unsupported:
                    # Both figures come from the same redraws, so their difference is judged against the standard
                    # error of the paired per-draw differences, not against two independent samples' errors.
                    difference, paired_se = paired_difference(full_only[verdict], held_only[verdict], draws)
                    entry["sensitivity_unsupported_rows_held"] = same_held[verdict] / draws
                    entry["difference_from_unsupported_rows_held"] = difference
                    entry["paired_mc_se_of_that_difference"] = paired_se
                    entry["redraws_as_observed_only_with_fallback"] = full_only[verdict]
                    entry["redraws_as_observed_only_with_rows_held"] = held_only[verdict]
                    entry["depends_on_unsupported_rows"] = abs(difference) > 2 * paired_se
                else:
                    entry["sensitivity_unsupported_rows_held"] = share_same
                    entry["difference_from_unsupported_rows_held"] = 0.0
                    entry["paired_mc_se_of_that_difference"] = 0.0
                    entry["depends_on_unsupported_rows"] = False
            verdicts[verdict] = entry
        counts = {}
        for (label, cell), values in count_draws.items():
            counts.setdefault(label, {})[cell] = {
                "observed_count": observed_counts[(label, cell)], "simulated_mean": statistics.mean(values),
                "simulated_sd": statistics.stdev(values) if len(values) > 1 else 0.0, "n": cell_n[cell]}
        scenarios[name] = {"description": description, "verdicts": verdicts, "label_counts": counts,
                           "unmeasured_label_holds_per_draw": tally["unmeasured_label_holds"] / draws}

    return {
        "draws": draws, "seed": seed, "cluster_key": key, "rows": len(rows), "replicate_pairs": len(pairs),
        "interpretation": "sensitivity conditional on the scenario and on the replicate's pair of shapes; "
                          "not a confidence level and not a guarantee",
        "direction_counts_and_intervals": directions,
        "support": support_table(rows, pairs, cluster_key),
        "unsupported": {"rows": len(unsupported),
                        "by_cell": dict(Counter(str(rows[i].get("cell")) for i in unsupported)),
                        "model": "independent per-label flips at the conditional point rates; "
                                 "a label value no replicate pair started with is held and counted as unmeasured"},
        "scenarios": scenarios,
    }


def render(result, title="Shape-floor sensitivity"):
    """The result as markdown: verdicts by scenario, what the replicate supports, and the direction counts."""
    names = [name for name, _, _ in SCENARIOS]
    lines = [f"# {title}", "",
             f"Rows: {result['rows']}; replicate pairs: {result['replicate_pairs']}; redraws: {result['draws']} "
             f"(seed {result['seed']}); cluster key: `{result['cluster_key']}`.", "",
             "Each figure is the share of redraws in which the verdict comes out as observed: a sensitivity "
             "conditional on the named scenario and on the replicate's pair of shapes, not a confidence level and "
             "not a guarantee. `~` marks a figure within two Monte Carlo standard errors of 0.95; `(u)` marks a "
             "verdict that depends on rows the replicate does not support.", "",
             "| Verdict | Observed | " + " | ".join(names) + " |", "|---|---|" + "---:|" * len(names)]
    for verdict in result["scenarios"][names[0]]["verdicts"]:
        cells = []
        for name in names:
            entry = result["scenarios"][name]["verdicts"][verdict]
            mark = ("~" if entry["within_2_mc_se_of_0.95"] else "") + (" (u)" if entry.get("depends_on_unsupported_rows") else "")
            cells.append(f"{entry['sensitivity']:.3f}{mark}")
        observed = result["scenarios"][names[0]]["verdicts"][verdict]["observed"]
        lines.append(f"| {verdict} | {observed} | " + " | ".join(cells) + " |")
    unsupported = result["unsupported"]
    lines += ["", "## What the replicate supports", "",
              f"Unsupported rows (starting vector in no replicate pair): {unsupported['rows']} "
              f"{unsupported['by_cell'] if unsupported['rows'] else ''}".rstrip() + ". They use: " + unsupported["model"] + ".", "",
              "| Starting labels (true ones) | Target rows | By cell | Replicate pairs | Changed | Destinations (pairs) | Distinct clusters |",
              "|---|---:|---|---:|---:|---|---:|"]
    for entry in result["support"]:
        start = ", ".join(sorted(k for k, v in entry["start"].items() if v)) or "none"
        by_cell = ", ".join(f"{cell} {n}" for cell, n in sorted(entry["target_rows_by_cell"].items()))
        dests = "; ".join((", ".join(sorted(k for k, v in d["labels_b"].items() if v)) or "none") + f" ({d['pairs']})"
                          for d in entry["destinations"]) or "no pair"
        clusters = "n/a" if entry["distinct_pair_clusters"] is None else str(entry["distinct_pair_clusters"])
        lines.append(f"| {start} | {entry['target_rows']} | {by_cell} | {entry['replicate_pairs']} | "
                     f"{entry['changed_pairs']} | {dests} | {clusters} |")
    lines += ["", "## Replicate flips by label and direction", "",
              "| Label | Direction | Flips / started there | Upper 95% bound |", "|---|---|---:|---:|"]
    for label, both in result["direction_counts_and_intervals"].items():
        for direction, entry in both.items():
            bound = entry["95%_clopper_pearson"][1]
            lines.append(f"| {label} | {direction} | {entry['count']} / {entry['denominator']} | "
                         + ("unmeasured" if bound is None else f"{bound:.3f}") + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--scorer", required=True, help="module:function")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster-key")
    parser.add_argument("--markdown", action="store_true", help="print the markdown report instead of JSON")
    args = parser.parse_args(argv)
    try:
        rows = json.loads(Path(args.rows).read_text(encoding="utf-8"))
        pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not isinstance(pairs, list):
            raise ValueError("rows and pairs must be JSON lists")
        result = simulate(rows, pairs, load_scorer(args.scorer), args.draws, args.seed, args.cluster_key)
    except (ValueError, KeyError, TypeError, AttributeError, ImportError, json.JSONDecodeError, OSError) as exc:
        print(f"bad input: {exc}", file=sys.stderr)
        return 2
    print(render(result) if args.markdown else json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
