import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-48b")
import part48 as p48
import part48b as p48b

pair = p48b.PAIRS[0]
rows = []
for item, score in ((pair[0], 0.2), (pair[1], -0.1)):
    for template in range(len(p48.TEMPLATES)):
        rows.append({
            "kind": "twin", "item": item, "template": template,
            "score": score, "letter_matched_score": score / 2,
            "continuation_mass": 0.5, "letter_matched_continuation_mass": 0.25,
            "token_count": 2, "mean_surprisal": 1.0 + score,
            "first_token_unigram_prior": {}, "onset_bigram_logprob": -1.0,
        })
summary = p48b.summarize(rows, pairs=(pair,))
print("summary keys:", sorted(summary))
print("per-model scores:", sorted(summary["scores"]))
print("per-model median:", summary["scores"]["part48_continuations"]["all_pairs"]["median_d"])
print("pooled field present:", any("pool" in key.lower() for key in summary))
