# Part 116 number ledger

The audit reads the raw Part 116 activity and direct-write rows, Part 115’s gradient rows, the three `.pt` ranking files with `torch.load(..., map_location="cpu")`, and Part 114’s score. It checks that the `.pt` top-500 orders match the candidate rows and that the recomputed verdicts match `results/part116/score.json`.

Run from the repository root. Each command below is a one-liner; the output shown is from the script’s printed report.

| Claim | One-liner | Output |
| --- | --- | --- |
| Held-out count, tested k values, random-set count | `MPLCONFIGDIR=/tmp/part116-mplconfig python results/part116/plot116.py` | `heldout_phrasings=16`; `k_grid=1,2,5,10,20,50,100,200,500`; `matched_sets_per_ranking_per_k=10` |
| Runtime gates for both rankings | `MPLCONFIGDIR=/tmp/part116-mplconfig python results/part116/plot116.py` | All-ones mask bit-identical: `True`; none-row mean absolute difference from Part 114: `0.045811` for activity and direct write |
| Flip counts at every k and maximum matched-set flips | `MPLCONFIGDIR=/tmp/part116-mplconfig python results/part116/plot116.py` | `activity=0,0,0,0,0,0,0,0,11`; `direct-write=0,0,0,0,0,4,16,16,16`; `gradient=0,0,0,0,0,0,3,11,16`; `matched_random_max=0,0,0,0,0,0,0,0,0` |
| Mean log-odds changes and first tested k with at least 8 flips | `MPLCONFIGDIR=/tmp/part116-mplconfig python results/part116/plot116.py` | Activity deltas: `-1.8936,-1.8930,-2.9162,-2.9948,-1.8210,-3.4056,-4.1841,-5.5465,-8.2096`; direct-write: `-0.5005,-0.9528,-3.5589,-4.2154,-5.3883,-7.4923,-9.9690,-12.6248,-18.8525`; gradient: `-0.7237,-1.2009,-2.6335,-3.8226,-5.1809,-6.9567,-8.8064,-9.8324,-11.0588`; activity first k `500`; direct-write first k `100` |
| Pairwise top-200 overlaps | `MPLCONFIGDIR=/tmp/part116-mplconfig python results/part116/plot116.py` | Activity/gradient `39`; direct-write/gradient `36`; activity/direct-write `34` |
| Highest activity unit and gradient rank; Part 114 largest single effect | `MPLCONFIGDIR=/tmp/part116-mplconfig python results/part116/plot116.py` | `L28 N2283`; gradient rank `11`; Part 114 largest single effect: `L28 N2283 -1.8982` mean log-odds change |
| First words in direct-write k=100 continuations and Tongyi Wanxiang count | `MPLCONFIGDIR=/tmp/part116-mplconfig python results/part116/plot116.py` | `Claude 5, Google 3, Jiang 1, Kimi 1, Tongyi 6`; “Tongyi Wanxiang” `0` |
| Recomputed registered prediction verdicts | `MPLCONFIGDIR=/tmp/part116-mplconfig python results/part116/plot116.py` | `116.1 held, 116.2 held, 116.3 held, 116.4 held, 116.5 held` |
| Selection, held-out, ordinary prompt counts and k grid | `python -c "import sys;sys.path[0:0]=['part114','part115'];import qneuron114,howmany115;print(len(qneuron114.SELECTION),len(qneuron114.HELDOUT),len(howmany115.ORDINARY),howmany115.KS)"` | `8 16 4 [1, 2, 5, 10, 20, 50, 100, 200, 500]` |
| Greedy continuation length | `rg -n 'max_new_tokens=12' part115/howmany115.py` | The candidate generation call uses `max_new_tokens=12`. |
| Registration timestamp | `git log -1 --format=%ci 29e1b14` | `2026-10-29 19:23:49 -0500` |

The prediction text and confidence labels are copied from `PART116-DESIGN.md`. The score values above are computed from result rows, not copied from Claude’s initial read.
