# Part 200 report (computed by `part200/report200.py`)

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| R1 and R3 | passed | load record and observed call shapes |
| R2 | passed | units hash matches design: True; questions identical: True |
| Part 163 gate1_recipe | passed | `{"minimum_rate": 0.7, "n_by_arm": {"neuron": 100, "random": 100}}` |
| Part 163 gate2_no_push | passed | `{"identity_qwen_hits": 8, "identity_total": 8, "maker_alibaba_or_tongyi_hits": 8, "maker_total": 8, "minimum_each": 7}` |
| Part 163 gate3_rows | passed | `{"expected_rows": 3416, "rows": 3416, "unique_rows": 3416, "nonempty": 3416, "duplicates": 0, "units_per_arm": {"neuron": 100, "random": 100}}` |

## Battery broken and at-cap counts

| Arm | Published broken / 1,600 | Rerun broken / 1,600 | Published at cap / 1,600 | Rerun at cap / 1,600 |
| --- | ---: | ---: | ---: | ---: |
| neuron | 343 / 1600 | 387 / 1600 | 124 / 1600 | 1020 / 1600 |
| random | 83 / 1600 | 103 / 1600 | 218 / 1600 | 912 / 1600 |

Lengths are reported before cross-run rate comparisons. Token counts by arm and run:

| Run | Arm | Mean tokens | Median tokens | At cap | Under cap | Broken at cap | Broken under cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| published | neuron | 55.1 | 54.0 | 124 | 1476 | 25 | 318 |
| published | random | 59.2 | 60.0 | 218 | 1382 | 11 | 72 |
| rerun | neuron | 68.8 | 80.0 | 1020 | 580 | 366 | 21 |
| rerun | random | 68.3 | 80.0 | 912 | 688 | 98 | 5 |

Broken counts rose by 44 neuron (343 to 387) and 20 random (83 to 103). Their composition shifted toward answers at the cap: broken-at-cap counts changed from 25/11 to 366/98 (neuron/random), while under-cap broken counts changed from 318/72 to 21/5. The design attributes cross-run changes jointly to kernel and shape; these records do not separate them.

### Gate 1 changed share

Published changed share was 0.88 neuron and 0.88 random; rerun was 0.75 and 0.71, against the 0.70 floor.

## Part 163 registered verdicts: published beside rerun

| Prediction | Published verdict and estimate (95% interval) | Rerun verdict and estimate (95% interval) |
| --- | --- | --- |
| 163.1 | held (1.061; 0.985 to 1.148) | held (1.047; 0.973 to 1.128) |
| 163.2 | held (1.024; 0.970 to 1.077) | held (1.042; 0.990 to 1.101) |
| 163.3 | held (neuron 0.850 (0.803 to 0.893), random 0.801 (0.749 to 0.850)) | held (neuron 0.853 (0.805 to 0.897), random 0.815 (0.765 to 0.862)) |
| 163.4 | held (name gone at least half: neuron 91, random 86; both gone: neuron 90, random 83) | held (name gone at least half: neuron 93, random 89; both gone: neuron 92, random 88) |
| 163.5 | failed (neuron 0.214 (0.168 to 0.266), random 0.052 (0.037 to 0.069)) | failed (neuron 0.242 (0.193 to 0.294), random 0.064 (0.048 to 0.083)) |

## Part 161 rule on maker phrasings

| Arm | Published Qwen / eligible | Rerun Qwen / eligible | Published rate | Rerun rate |
| --- | ---: | ---: | ---: | ---: |
| neuron | 19 / 539 | 16 / 514 | 0.035 | 0.031 |
| random | 57 / 661 | 40 / 632 | 0.086 | 0.063 |

## Registered Part 200 predictions

| # | Prediction | Value decided on | Result |
| --- | --- | --- | --- |
| 200.1 | Gates R1 to R3 and Part 163's three gates pass. (medium) | `{"gates_r1_r3_passed": true, "gate_r2_passed": true, "part163_gates": {"gate1_recipe": true, "gate2_no_push": true, "gate3_rows": true}, "part163_gates_passed": true, "result": "held"}` | held |
| 200.2 | 163.1, 163.2, 163.3 and 163.4 hold and 163.5 fails, as published. (medium: 163.1 and 163.2 cannot fail unless a random rate falls below 2/3, and 163.3's bar is far below the published rates; 163.4 is one unit from flipping in each arm, while 163.5's neuron and random intervals are 0.193 to 0.294 and 0.048 to 0.083.) | `{"verdicts": {"163.1": "held", "163.2": "held", "163.3": "held", "163.4": "held", "163.5": "failed"}, "expected": {"163.1": "held", "163.2": "held", "163.3": "held", "163.4": "held", "163.5": "failed"}, "result": "held"}` | held |
| 200.3 | The neuron arm's broken count over its 1,600 battery answers is at least twice the random arm's (published 343 and 83). (medium: Part 199 kept its between-arm direction on eager while its rates moved.) | `{"neuron_broken": 387, "random_broken": 103, "published": {"neuron": 343, "random": 83}, "ratio": 3.7572815533980584, "minimum_ratio": 2.0}` | held |
| 200.4 | In each arm at least 480 of 1,600 battery answers (30%) reach the 80-token cap (published 124 neuron, 218 random). (medium: on SDPA almost every Part 199 answer stopped early; on eager most ran to the cap.) | `{"at_cap": {"neuron": 1020, "random": 912}, "published": {"neuron": 124, "random": 218}, "minimum": 480, "of": 1600}` | held |
| 200.5 | Each arm's broken count is higher on eager than published. (low: Part 199's loop rates rose on eager, but Part 163's triage keeps any answer naming a model or identity, and a kernel can remove loops as well as add them.) | `{"broken": {"neuron": 387, "random": 103}, "published": {"neuron": 343, "random": 83}, "higher_than_published": {"neuron": true, "random": true}}` | held |
| 200.6 | Under Part 161's rule on maker phrasings, the neuron arm's pooled Qwen rate stays below the random arm's (published 19 of 539 against 57 of 661). (low: the published rates are small and the comparison was unregistered.) | `{"neuron": {"qwen": 16, "eligible_maker_answers": 514, "rate": 0.0311284046692607}, "random": {"qwen": 40, "eligible_maker_answers": 632, "rate": 0.06329113924050633}, "published": {"neuron": {"qwen": 19, "eligible_maker_answers": 539}, "random": {"qwen": 57, "eligible_maker_answers": 661}}}` | held |
| 200.7 | At most 20 of the 200 batch-1 answers differ from the same units' chunked answers. (low: rows are pushed independently, but eager batch 8 left the batch-1 path at a near-tie on 1 of 8 kernel-check prompts.) | `{"changed": 76, "of": 200, "maximum": 20, "per_arm": {"neuron": {"rows": 100, "changed": 39}, "random": {"rows": 100, "changed": 37}}}` | failed |

## Exploratory: compare.json

```json
{
  "mid_phrase_under_cap": {
    "published": {
      "neuron": {
        "under_cap": 1476,
        "mid_phrase_among_under_cap": 510
      },
      "random": {
        "under_cap": 1382,
        "mid_phrase_among_under_cap": 409
      }
    },
    "rerun": {
      "neuron": {
        "under_cap": 580,
        "mid_phrase_among_under_cap": 2
      },
      "random": {
        "under_cap": 688,
        "mid_phrase_among_under_cap": 1
      }
    }
  },
  "identical_battery_answers": {
    "shared_rows": 3200,
    "identical": 390,
    "neuron": {
      "rows": 1600,
      "identical": 186
    },
    "random": {
      "rows": 1600,
      "identical": 204
    }
  },
  "identical_no_push_answers": {
    "shared": 16,
    "identical": 4
  },
  "per_layer_broken": {
    "published": {
      "neuron": {
        "19": {
          "answers": 32,
          "broken": 0
        },
        "20": {
          "answers": 16,
          "broken": 8
        },
        "21": {
          "answers": 192,
          "broken": 46
        },
        "22": {
          "answers": 272,
          "broken": 68
        },
        "23": {
          "answers": 320,
          "broken": 80
        },
        "24": {
          "answers": 480,
          "broken": 92
        },
        "25": {
          "answers": 208,
          "broken": 27
        },
        "26": {
          "answers": 80,
          "broken": 22
        }
      },
      "random": {
        "19": {
          "answers": 32,
          "broken": 1
        },
        "20": {
          "answers": 16,
          "broken": 0
        },
        "21": {
          "answers": 192,
          "broken": 3
        },
        "22": {
          "answers": 272,
          "broken": 10
        },
        "23": {
          "answers": 320,
          "broken": 19
        },
        "24": {
          "answers": 480,
          "broken": 33
        },
        "25": {
          "answers": 208,
          "broken": 15
        },
        "26": {
          "answers": 80,
          "broken": 2
        }
      }
    },
    "rerun": {
      "neuron": {
        "19": {
          "answers": 32,
          "broken": 0
        },
        "20": {
          "answers": 16,
          "broken": 7
        },
        "21": {
          "answers": 192,
          "broken": 55
        },
        "22": {
          "answers": 272,
          "broken": 79
        },
        "23": {
          "answers": 320,
          "broken": 87
        },
        "24": {
          "answers": 480,
          "broken": 105
        },
        "25": {
          "answers": 208,
          "broken": 29
        },
        "26": {
          "answers": 80,
          "broken": 25
        }
      },
      "random": {
        "19": {
          "answers": 32,
          "broken": 0
        },
        "20": {
          "answers": 16,
          "broken": 0
        },
        "21": {
          "answers": 192,
          "broken": 1
        },
        "22": {
          "answers": 272,
          "broken": 15
        },
        "23": {
          "answers": 320,
          "broken": 26
        },
        "24": {
          "answers": 480,
          "broken": 36
        },
        "25": {
          "answers": 208,
          "broken": 20
        },
        "26": {
          "answers": 80,
          "broken": 5
        }
      }
    }
  }
}
```

## Exploratory: batch-1 first differences

Tokenizer: qwen3.5-4b-local-cache. First positions include the rendered prompt (17 prompt tokens); decoded answers were re-tokenized because token IDs were not stored.

| Arm | Changed | At/before 64 keys | After 64 keys | Median absolute first-difference key |
| --- | ---: | ---: | ---: | ---: |
| neuron | 39 | 38 | 1 | 32.0 |
| random | 37 | 35 | 2 | 38.0 |

Among changed rows, Part 163 label flips: name gone 2, broken 6, Qwen under the Part 161 predicate 5. Exploratory rows and excerpts are in report200.json.

### Exploratory checks

Re-tokenized decoded texts reproduce the adjusted generator token count in 3178 of 3416 rows. Among under-cap battery answers, 1213 of 1268 re-tokenize to `token_count − 1`; `token_count` includes the stop token under the cap.
The first-difference convention uses 17 prompt tokens plus the 0-based answer index. Counts at or before thresholds 63, 64, and 65 are 63: 73, 64: 73, 65: 73; position uncertainty is about ±1 token.
The load record labels this Part 199; its `repair.chunk_size` is 64, the batch size passed to startup repair, while the call records show pushed generation chunks of 8. The run averaged 0.89 answers per second against the 0.17 guard; `summary.seconds` records 3836.6 seconds, while load-record timestamps span 4670 seconds. The roughly 833-second difference is most likely the untimed batch-1 stage: `run200.py` calls it after `run163.run` returns. This is an inference from the code.
