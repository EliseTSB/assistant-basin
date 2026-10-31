# What do the 3,253 neurons on the screen page read, and does it match what they write?

This table shows top and bottom reading windows and J-list matches for 3,253 Qwen3.5-4B neurons selected from the Part 86 page, across 10,000 Pile documents cut to 256 tokens (2,235,883 tokens), with one maximum per document per unit and no push.

## Gates

| Gate | Result | Part 151b comparison |
|---|---:|---:|
| 1. Documents | Passed: 10,000 | — |
| 2. Windows | Passed: 20 top and 20 bottom per unit | — |
| 3. Alien control, L22 N2742 | 12 of 20 | 11 of 20 |
| 4. L19 N4979 marks “and” or “or” | 18 of 20 | 17 of 20 |
| 4. L24 N8542 award rule | 15 of 20 | 17 of 20 |

All four gates passed. The pilot's check also passed: 18 stored Part 151 window activations were reproduced with no mismatch on the first 200 documents.

## Predictions

The wording and confidence are registered in `PART152-DESIGN.md`. Counts and verdicts come from `score.json`.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 152.1 | Across all units, the mean number of top windows hitting a unit's own J list is at least twice the mean under within-layer shuffles, p below 0.01. (medium) | 2.262 against 0.110; p = 0.001 | held |
| 152.2 | At least 10% of units hit their own J list on at least 3 of 20 top windows. (medium) | 719 of 2,409 (29.8%) | held |
| 152.3 | Among Part 112's concept units, the mean number of top windows hitting a unit's persona words is at least twice the mean under within-kind shuffles, p below 0.01. (medium) | 2.105 against 0.186; p = 0.001; n = 247 | held |
| 152.4 | Among Part 112's slot units, that mean is less than half the concept units' mean. (medium) | Slot 0.440 (n = 184); concept 2.105 (n = 247) | held |

"All units" in 152.1 and 152.2 means the 2,409 with usable lists: of 3,253 units, 844 are left out of those two because their lists have fewer than 3 usable words. Persona lists were missing for 10 concept units and 5 slot units. The page's neurons were selected for changing “Who are you?” when pushed, so this set is not a random sample of neurons.

Part 151b's five units are a small piece of this set, and 151b.1 failed on those five. Both results stand on different sets and scopes. The design's reading rule is that if 152.1 fails, the reading that neurons here read what they write is withdrawn at this size. It held here under the registered threshold, and the sections below limit what that means.

## What the hits are

Most hits are not on the fired token. The J lists hit at the marked token in 1,516 windows, at the first word after it in 2,509, and later in 1,423, counting each window once at the earliest place. So 72% of hits lie after the mark: the unit fired before the word appeared. That fits predicting the next word at least as well as reading a word, and these counts cannot separate them. "Read" in the title means where a unit fires, not that it has understood what it has read.

Counting only the marked token, the mean is 0.629 against 0.012 under shuffles, and 7.8% of units reach 3 of 20, below 152.2's 10% line. Counting the marked token and the 30 tokens before it, the mean is 1.536 against 0.218, and 20.8% reach 3 of 20. These narrower rules were chosen after the cold check, so they are exploratory.

Persona words can overlap the J list, so 152.3 and 152.4 share evidence with 152.1. With shared words removed, concept units hit 0.565 against 0.156 under shuffles, n = 232, and slot units 0.261 against 0.146, n = 184. The concept advantage over shuffles falls from 11.3 times to 3.6 times, and slot divided by concept rises from 0.21 to 0.46. The gap between the two kinds survives, smaller.

The J lists come from Neuronpedia's Jacobian lens for this model, fitted on WikiText-103, as Part 42 records. They are not computed from the weights alone and not from these documents. The script that listed them for the page is `part152/lens_all.py`, and its output is `part152/lens.json`.

## Exploratory results

These analyses were chosen after seeing the verdicts. They do not change the registered scores.

Across the 2,409 units in 152.1 and their top windows, each own-list hit window is counted once at its earliest matching position. There were 1,516 hits on the marked token alone, 2,509 on the first word after it alone, and 1,423 later in the after-text only. A hit on the first word after the mark means the unit fired just before that word appeared in the text. This does not separate reading from predicting the next word.

On the same 2,409 units' bottom windows, the mean own-list hit count was 0.845 per unit, against a within-layer shuffle mean of 0.098; p = 0.001 with 1,000 shuffles and seed 153. On top windows, 1,086 of 2,409 units had zero own-list hits (45.1%).

Top-window means by layer are shown for layers with at least 30 included units. Shuffle means use within-layer list pairings.

| Layer | Units | Own-list mean | Shuffle mean |
|---:|---:|---:|---:|
| 17 | 36 | 0.972 | 0.272 |
| 19 | 48 | 0.688 | 0.102 |
| 20 | 110 | 1.945 | 0.154 |
| 21 | 277 | 2.419 | 0.115 |
| 22 | 391 | 2.596 | 0.117 |
| 23 | 488 | 2.115 | 0.096 |
| 24 | 582 | 1.957 | 0.083 |
| 25 | 263 | 2.916 | 0.095 |
| 26 | 152 | 3.243 | 0.144 |

### Example units with the most own-list hits

Each unit below had 20 of 20 top windows hit its own J list. The J words shown are the usable words after the design's cleaning rule. The two windows are shown verbatim; `⏎` marks line breaks and `␍` marks carriage returns. The marked token is between double square brackets.

**L25 N7945 — class, classe, classes**

```
 sqlalchemy.types import TypeDecorator, CHAR⏎⏎DBSession = scoped_session(sessionmaker())⏎Base = declarative_base()⏎⏎#### New SQLAlchemy-Type #####################[[⏎]]class NumpyType (sa.types.TypeDecorator):⏎   
⏎import { Component, Prop } from "vue-property-decorator";⏎import { ColumnDefinition } from a shortened import path;⏎⏎@Component⏎export[[ default]] class ClassesForCollapsedColumnMixin extends Vue {⏎    @
```

**L24 N4564 — mental, mente, mind, minded, minds, mindset**

```
. It’s been a long campaign and while many people have made their firm decision, there are still a lot of people who may not have made up[[ their]] minds. To those I say read and read some more,
 he gave me that hurt look and came back to ‘hmpf’ at my feet, watching both Henry and Rick carefully just in case I changed[[ my]] mind. Indeed, I would
```

**L23 N3224 — dari, desde, from**

```
 by simply doing this.get('store') inside any route or controller.⏎For example here is a code extract where the default store get's registered ([[taken]] from the source)⏎Ember.onLoad('Ember
 your safety and your permission to walk see⏎conditions.Credits and list definitions are listed here⏎Credits⏎⏎The anglicised name Letterbreckaun[[ comes]] from Leitir Bhriocáin [TR], '
```

**L22 N8247 — aneous, aneously, anes, grat, instant, moment**

```
 them without any trouble.⏎⏎ASROCK P4VM8 SATA DRIVER DOWNLOAD⏎⏎Tell us some more! Nike – Jul 10, at[[ Instant]] Repair Help over the Phone 6ya. Hi all,
 Various Peep Shows⏎ Priscilla Frank (January 25, 2014). Annie Lapin's Newest Painting Exhibition Combines[[ Instant]] Attraction and a Slow Burn. Huffington Post.⏎⏎External
```

## Limits and context

This is one corpus sample, one model, and 256 tokens per document. Each unit contributes one maximum per document. A prefix rule can match unrelated words. Each J list holds 12 words, and persona word lists can be short. These results describe the selected page set. They do not establish what a unit is for or claim a mechanism.

The closest published work named in the design is Neuron Lens (stdlogits), which has read and write catalogs for Qwen3-8B. Part 128 joined that catalog to its persona-neuron sort: on the fire side, 6 of 11 carrier units matched, against 0 of 16 once units for DeepSeek and 1 of 16 for Luna. That is a different model and a different measure.

The collection took 1,170.0 seconds (19.5 minutes), as recorded in `windows.json.gz`.

## Files

- `windows.json.gz` holds the full top and bottom windows and run metadata.
- `page-windows.json` holds the reduced windows selected for the page.
- `score.json` holds gates, predictions, and exploratory counts. The scoring code is `part152/score152.py`.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; seven findings folded. It recomputed all four predictions with its own code and 1,000 shuffles and found no leak favouring the true pairing. With the 50 most-shared words and overlapping lists removed, 152.1 stays at 2.108 against 0.093. It showed that most hits lie after the mark and that 152.3 leans on words shared with the J lists, which the section above reports. The pilot's 11 MB output is on the job host and not in this repository.
