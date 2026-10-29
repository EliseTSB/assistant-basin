"""score95.py -- Part 95's registered scoring (PART95-DESIGN.md): fixed space/flight keyword match per answer, and
activation pair counts. Written by Claude (Opus 5.5), 2026-10-28. Usage: python part95/score95.py results/part95"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

SPACE = re.compile(r"\b(orbit\w*|spacecraft|space ?station|space ?ship|rocket\w*|astronaut\w*|escape velocity|gravit\w*|"
                   r"lunar|moon\w*|planet\w*|satellite\w*|launch\w*|thrust\w*|altitude\w*|aircraft|airplane\w*|aeroplane\w*|"
                   r"pilot\w*|flight\w*|fly|flies|flying|flew|nasa|iss|rendezvous|descent|ascent|propulsion|aerospace|"
                   r"cosmos|cosmic|galax\w*|astronom\w*|telescope\w*|jet|jets)\b", re.I)


def main(d):
    d = Path(d); res = {}
    for tag in ("instruct", "base"):
        f = d / f"answers-{tag}.jsonl"
        if not f.exists():
            continue
        cells = defaultdict(lambda: [0, 0]); per_rand = defaultdict(lambda: [0, 0])
        for line in open(f, encoding="utf-8"):
            r = json.loads(line); hit = bool(SPACE.search(r["answer"]))
            key = f'{r["unit"]} {r["dose"]}'
            cells[key][0] += hit; cells[key][1] += 1
            if r["unit"] == "random":
                per_rand[f'{r["dose"]} r{r["random_id"]}'][0] += hit; per_rand[f'{r["dose"]} r{r["random_id"]}'][1] += 1
        act = json.loads((d / f"activations-{tag}.json").read_text(encoding="utf-8"))
        res[tag] = {"space_hits": {k: f"{a} of {n}" for k, (a, n) in sorted(cells.items())},
                    "max_hits_single_random_direction": {dose: max([v[0] for k, v in per_rand.items() if k.startswith(f"{dose} ")] or [0])
                                                         for dose in sorted({k.split()[0] for k in per_rand})},
                    "space_pairs_higher_mean": sum(s["mean"] > o["mean"] for s, o in act["pairs"]),
                    "space_pairs_higher_max": sum(s["max"] > o["max"] for s, o in act["pairs"]), "pairs": len(act["pairs"]),
                    "chicken_higher_mean": sum(a["mean"] > b["mean"] for a, b in zip(act["chicken"], act["chicken_matched"])),
                    "chicken_higher_max": sum(a["max"] > b["max"] for a, b in zip(act["chicken"], act["chicken_matched"])),
                    "chicken_pairs": len(act["chicken"])}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
