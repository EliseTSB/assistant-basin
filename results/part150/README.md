# What do five picked neurons read and write, and are they similar in Base?

The table shows word rates for Qwen3.5-4B and Qwen3.5-4B-Base, five hand-picked units, greedy decoding, 120 new tokens, and the five questions other than “Who are you?”. Elise picked the units by reading the Part 86 answers page.

The read guesses in the design came from a Qwen3-1.7B atlas with width 6144, a different model. Amendment 1 records that the atlas neurons do not identify these Qwen3.5-4B neurons. Predictions 150.1–150.5 and 150.11 test the registered activation contrasts against read guesses from a different model. Prediction 150.11 is held, but that is not evidence of a read feature. The marked-token medians below give the complete activation results.

## Gates

Gate 1 held: zero push matched no hook. Gate 2 is reported only: on the chat model, the atlas-window activation for L19 N4979 was 0.047 and the Gate 2 sentence activation for L23 N116 was -0.014, against the design threshold of 10. Gate 3 held: required rows were present. Gate 4 held: marked pieces were one token and read as marked.

## Predictions

Results use the registered point estimates. Verdict words are held or failed. The reading rule applies to generated answers, not these marked-token activation medians. Its corrected tokenization changes 25 of 660 answer classifications; none of predictions 150.1–150.5 or any of the 13 registered verdicts changes.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 150.1 | On the chat model, L19 N4979's median activation on relative “ that” is at least 5 above demonstrative “ that”. (medium) | 0.036 minus -0.027 = 0.063 | failed |
| 150.2 | Its median on relative “ that” is at least 3 above complementizer “ that”. (medium) | 0.036 minus 0.089 = -0.053 | failed |
| 150.3 | Its median on free “ what” is at least 3 above sentence-opening “ What”. (medium) | -0.009 minus 0.036 = -0.045 | failed |
| 150.4 | On the chat model, L23 N116's median activation on “ of” before a plural pronoun or demonstrative is at least 10 above “ of” between nouns. (medium) | -0.000 minus 0.026 = -0.026 | failed |
| 150.5 | Its median on a verb before “ them” is at least 10 above the same verbs before something else. (low) | 0.033 minus (-0.024) = 0.056 | failed |
| 150.6 | At dose 0.8 on the chat model, L19 N4979's abstract rate is at least 1 above unpushed and above all four random directions of its layer. (medium) | 6.06; plain 2.49; random 3.04, 2.23, 3.64, 2.99 | held |
| 150.7 | At dose 0.8 on the chat model, at least 3 of the 5 units have a lens rate above all four random directions of their layer. (medium) | 4 of 5 | held |
| 150.8 | L19 N4979's abstract rate does not fall across doses 0.4, 0.8 and 1.5. (low) | 3.83, 6.06, 1.80; 6 of 6 answers break at 1.5 | failed |
| 150.9 | At dose -0.8, L19 N4979's abstract rate is at most its unpushed rate. (low) | 1.99 versus 2.49 | held |
| 150.10 | At dose 0.8 on the chat model, L23 N116's anaphor rate is above all four random directions of its layer. (low) | 0; all four random directions also 0 | failed |
| 150.11 | L19 N4979's relative minus demonstrative gap on Base is positive on the chat model and at least half as large on Base. (medium) | Chat 0.063; Base 0.043, over half of chat. The activation medians are listed in the table below. | held |
| 150.12 | At least 4 of the 5 units have a down-projection column cosine of at least 0.9 between Base and the chat model. (medium) | 5 of 5 | held |
| 150.13 | At dose 0.8 on Base, at least 3 of the 5 units have a lens rate above all four random directions of their layer. (medium) | 3 of 5 | held |


## Marked-token activation medians

Each cell is the median activation for that class, computed from the meta files. Values are shown to three decimals.

### Qwen3.5-4B chat

| Unit | Free what | Relative that | Complementizer that | Demonstrative that | Question What | The control | Of + pronoun/demonstrative | Of between nouns | Verb before them | Verb other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L19N4979 | -0.009 | +0.036 | +0.089 | -0.027 | +0.036 | -0.014 | -0.003 | -0.036 | +0.027 | +0.010 |
| L19N5975 | +0.030 | +0.053 | +0.013 | +0.007 | -0.003 | +0.062 | +0.109 | +0.079 | +0.024 | +0.030 |
| L23N116 | -0.007 | -0.161 | -0.233 | -0.008 | -0.053 | +0.024 | -0.000 | +0.026 | +0.033 | -0.024 |
| L24N6835 | -0.014 | +0.044 | +0.035 | +0.005 | +0.029 | -0.054 | +0.034 | -0.029 | -0.049 | -0.019 |
| L24N8542 | -0.045 | -0.014 | -0.047 | -0.006 | +0.061 | -0.255 | -0.087 | -0.088 | -0.035 | -0.025 |

### Qwen3.5-4B-Base

| Unit | Free what | Relative that | Complementizer that | Demonstrative that | Question What | The control | Of + pronoun/demonstrative | Of between nouns | Verb before them | Verb other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L19N4979 | -0.004 | +0.019 | +0.068 | -0.024 | +0.050 | -0.026 | -0.006 | -0.049 | +0.005 | +0.001 |
| L19N5975 | +0.028 | +0.049 | +0.005 | -0.007 | -0.051 | +0.032 | +0.133 | +0.049 | +0.027 | +0.042 |
| L23N116 | -0.020 | -0.114 | -0.191 | +0.003 | -0.055 | +0.020 | +0.011 | +0.008 | +0.029 | -0.008 |
| L24N6835 | -0.018 | +0.005 | +0.054 | +0.013 | +0.013 | -0.063 | +0.021 | -0.035 | -0.055 | -0.020 |
| L24N8542 | -0.064 | -0.007 | -0.063 | -0.003 | +0.052 | -0.263 | -0.101 | -0.118 | -0.051 | -0.028 |


## Lens word rates

The registered rate is the original distinct answer words that start with any registered lens string, per 100 words. Exact whole-word rates require a case-insensitive exact match and exclude lens entries shorter than five letters. The unbroken-only column applies that exact rate only to tested answers that pass the Part 86 rule; “—” means there were no unbroken tested answers. Entries list plain, +0.4, +0.8, +1.5, +2.5, -0.8, -1.5 in that order. Registered and exact rates are means over the five tested questions. The adjacent broken vector gives counts out of all six generated answers, including “Who are you?”. Random cells list each direction as rate (broken count/6); order is direction 0–3.

Base's rates at +1.5 rest on short answers: L24 N6835's 22.64 comes from answers of 3 to 11 words, and L24 N8542's 7.13 from three answers of 6, 12 and 47 words, the longest broken. The registered rate can be inflated by matching prefixes inside invented variants; repetition of the same matching word does not add multiple hits within one answer, though looping can generate multiple distinct matching variants and very short answers can raise a per-100-word rate. L24 N8542 chat +1.5 is 10.43 registered, 2.81 exact whole-word, with 6/6 broken; no unbroken-only rate can be estimated. Its surviving exact hits are the registered strings “nomin” and “nomine”, which are still fragments in these outputs despite the five-letter cutoff. This is why the exact rate is not zero. Fragments under five letters and prefix matches account for much of the registered 10.43; loops create repeated text but do not count the same distinct word repeatedly. At +0.8, L24 N8542 drops from 11.26 registered to 4.18 exact, and 3/6 answers are broken.

### Qwen3.5-4B chat

| Unit | Registered vector | Exact whole-word vector | Exact, unbroken-only vector | Random +0.8 (registered; four directions) | Random +1.5 (registered; four directions) |
|---|---|---|---|---|---|
| L19N4979 | 0.45 (0/6); 1.41 (1/6); 3.07 (0/6); 1.71 (6/6); 1.36 (6/6); 0.00 (0/6); 0.00 (4/6) | 0.45, 1.41, 3.07, 1.71, 1.36, 0.00, 0.00 | 0.45, 1.76, 3.07, —, —, 0.00, 0.00 | 0.00 (0/6), 0.37 (1/6), 0.00 (0/6), 0.75 (0/6) | 0.00 (4/6), 0.00 (4/6), 0.00 (4/6), 0.32 (4/6) |
| L19N5975 | 0.00 (0/6); 0.32 (0/6); 1.22 (1/6); 0.00 (5/6); 0.00 (6/6); 0.00 (0/6); 0.00 (4/6) | 0.00, 0.32, 0.50, 0.00, 0.00, 0.00, 0.00 | 0.00, 0.32, 0.62, 0.00, —, 0.00, 0.00 | 0.00 (0/6), 0.00 (1/6), 0.00 (0/6), 0.00 (0/6) | 0.00 (4/6), 0.00 (4/6), 0.00 (4/6), 0.00 (4/6) |
| L23N116 | 0.00 (0/6); 0.00 (0/6); 0.00 (3/6); 0.00 (4/6); 0.00 (6/6); 0.00 (1/6); 0.00 (6/6) | 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00, 0.00, —, 0.00, — | 0.00 (1/6), 0.00 (1/6), 0.00 (1/6), 0.00 (0/6) | 0.00 (3/6), 0.00 (2/6), 0.00 (2/6), 0.00 (4/6) |
| L24N6835 | 0.00 (0/6); 0.39 (0/6); 2.23 (1/6); 5.48 (6/6); 1.92 (6/6); 0.00 (2/6); 0.00 (3/6) | 0.00, 0.39, 2.23, 4.87, 1.92, 0.00, 0.00 | 0.00, 0.39, 2.22, —, —, 0.00, 0.00 | 0.00 (1/6), 0.00 (0/6), 0.00 (0/6), 0.00 (1/6) | 0.00 (4/6), 0.00 (3/6), 0.00 (3/6), 0.00 (1/6) |
| L24N8542 | 0.00 (0/6); 1.85 (3/6); 11.26 (3/6); 10.43 (6/6); 4.26 (6/6); 0.00 (0/6); 0.00 (4/6) | 0.00, 0.43, 4.18, 2.81, 0.00, 0.00, 0.00 | 0.00, 1.06, 3.27, —, —, 0.00, 0.00 | 0.00 (1/6), 0.00 (0/6), 0.00 (0/6), 0.00 (1/6) | 0.00 (4/6), 0.00 (3/6), 0.00 (3/6), 0.00 (1/6) |


### Qwen3.5-4B-Base plain frame

| Unit | Registered vector | Exact whole-word vector | Exact, unbroken-only vector | Random +0.8 (registered; four directions) | Random +1.5 (registered; four directions) |
|---|---|---|---|---|---|
| L19N4979 | 0.00 (0/6); 0.00 (0/6); 0.44 (1/6); 0.21 (3/6); 0.65 (6/6); 0.00 (1/6); 0.00 (3/6) | 0.00, 0.00, 0.44, 0.21, 0.65, 0.00, 0.00 | 0.00, 0.00, 0.26, 0.00, —, 0.00, 0.00 | 0.00 (0/6), 0.00 (0/6), 0.00 (0/6), 0.00 (1/6) | 0.00 (2/6), 0.00 (2/6), 0.00 (1/6), 0.00 (1/6) |
| L19N5975 | 0.00 (0/6); 0.48 (2/6); 0.00 (2/6); 0.17 (3/6); 0.00 (6/6); 0.00 (0/6); 0.00 (1/6) | 0.00, 0.00, 0.00, 0.17, 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00, 0.00, —, 0.00, 0.00 | 0.00 (0/6), 0.00 (0/6), 0.00 (0/6), 0.00 (1/6) | 0.00 (2/6), 0.00 (2/6), 0.00 (1/6), 0.00 (1/6) |
| L23N116 | 0.00 (0/6); 0.00 (2/6); 0.00 (1/6); 0.00 (3/6); 0.00 (5/6); 0.00 (4/6); 0.00 (6/6) | 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 | 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, — | 0.00 (0/6), 0.00 (1/6), 0.00 (1/6), 0.00 (1/6) | 0.00 (2/6), 0.00 (3/6), 0.00 (2/6), 0.00 (2/6) |
| L24N6835 | 0.20 (0/6); 0.00 (0/6); 1.21 (2/6); 22.64 (1/6); 0.83 (6/6); 0.00 (0/6); 0.00 (2/6) | 0.00, 0.00, 1.21, 19.30, 0.83, 0.00, 0.00 | 0.00, 0.00, 1.39, 15.80, —, 0.00, 0.00 | 0.00 (0/6), 0.00 (1/6), 0.00 (0/6), 0.00 (0/6) | 0.00 (4/6), 0.00 (1/6), 0.00 (3/6), 0.00 (1/6) |
| L24N8542 | 0.00 (0/6); 0.83 (0/6); 1.31 (1/6); 7.13 (2/6); 3.72 (6/6); 0.00 (0/6); 0.00 (1/6) | 0.00, 0.83, 0.66, 1.67, 0.00, 0.00, 0.00 | 0.00, 0.83, 0.54, 2.08, —, 0.00, 0.00 | 0.00 (0/6), 0.00 (1/6), 0.00 (0/6), 0.00 (0/6) | 0.00 (4/6), 0.00 (1/6), 0.00 (3/6), 0.00 (1/6) |


## Abstract and anaphor rates with broken counts

Each vector lists plain, +0.4, +0.8, +1.5, +2.5, -0.8, -1.5. Rates are distinct abstract words or anaphor occurrences per 100 words, averaged over five questions. Broken counts in parentheses are out of all six answers. Random rates are shown for both scored doses, in direction order 0–3.

| Model | Unit | Abstract rate (broken/6) vector | Anaphor rate (broken/6) vector | Abstract random +0.8 | Abstract random +1.5 | Anaphor random +0.8 | Anaphor random +1.5 |
|---|---|---|---|---|---|---|---|
| Chat | L19N4979 | 2.49 (0/6); 3.83 (1/6); 6.06 (0/6); 1.80 (6/6); 2.69 (6/6); 1.99 (0/6); 1.73 (4/6) | 0.00 (0/6); 0.00 (1/6); 0.00 (0/6); 0.00 (6/6); 0.00 (6/6); 0.00 (0/6); 1.20 (4/6) | 3.04 (0/6), 2.23 (1/6), 3.64 (0/6), 2.99 (0/6) | 4.53 (4/6), 1.50 (4/6), 0.38 (4/6), 0.91 (4/6) | 0.00 (0/6), 0.00 (1/6), 0.00 (0/6), 0.00 (0/6) | 0.00 (4/6), 0.00 (4/6), 0.53 (4/6), 1.02 (4/6) |
| Chat | L19N5975 | 2.49 (0/6); 3.77 (0/6); 1.53 (1/6); 0.00 (5/6); 0.00 (6/6); 3.84 (0/6); 0.85 (4/6) | 0.00 (0/6); 0.00 (0/6); 0.00 (1/6); 0.00 (5/6); 0.00 (6/6); 0.47 (0/6); 0.00 (4/6) | 3.04 (0/6), 2.23 (1/6), 3.64 (0/6), 2.99 (0/6) | 4.53 (4/6), 1.50 (4/6), 0.38 (4/6), 0.91 (4/6) | 0.00 (0/6), 0.00 (1/6), 0.00 (0/6), 0.00 (0/6) | 0.00 (4/6), 0.00 (4/6), 0.53 (4/6), 1.02 (4/6) |
| Chat | L23N116 | 2.49 (0/6); 1.71 (0/6); 2.94 (3/6); 0.00 (4/6); 0.00 (6/6); 2.01 (1/6); 0.00 (6/6) | 0.00 (0/6); 0.00 (0/6); 0.00 (3/6); 0.00 (4/6); 0.00 (6/6); 0.00 (1/6); 0.00 (6/6) | 1.46 (1/6), 3.17 (1/6), 2.30 (1/6), 1.96 (0/6) | 0.00 (3/6), 0.43 (2/6), 1.00 (2/6), 2.24 (4/6) | 0.00 (1/6), 0.00 (1/6), 0.00 (1/6), 0.00 (0/6) | 0.00 (3/6), 0.00 (2/6), 0.00 (2/6), 0.00 (4/6) |
| Chat | L24N6835 | 2.49 (0/6); 1.15 (0/6); 1.90 (1/6); 0.00 (6/6); 0.00 (6/6); 1.36 (2/6); 0.00 (3/6) | 0.00 (0/6); 0.00 (0/6); 0.00 (1/6); 0.00 (6/6); 0.00 (6/6); 0.32 (2/6); 0.00 (3/6) | 3.72 (1/6), 1.83 (0/6), 3.76 (0/6), 1.72 (1/6) | 0.00 (4/6), 0.87 (3/6), 6.79 (3/6), 0.41 (1/6) | 0.00 (1/6), 0.00 (0/6), 0.00 (0/6), 0.00 (1/6) | 0.00 (4/6), 0.00 (3/6), 0.00 (3/6), 0.00 (1/6) |
| Chat | L24N8542 | 2.49 (0/6); 2.28 (3/6); 2.14 (3/6); 0.00 (6/6); 0.00 (6/6); 1.78 (0/6); 1.61 (4/6) | 0.00 (0/6); 0.00 (3/6); 0.00 (3/6); 0.00 (6/6); 0.00 (6/6); 0.00 (0/6); 0.00 (4/6) | 3.72 (1/6), 1.83 (0/6), 3.76 (0/6), 1.72 (1/6) | 0.00 (4/6), 0.87 (3/6), 6.79 (3/6), 0.41 (1/6) | 0.00 (1/6), 0.00 (0/6), 0.00 (0/6), 0.00 (1/6) | 0.00 (4/6), 0.00 (3/6), 0.00 (3/6), 0.00 (1/6) |
| Base | L19N4979 | 1.16 (0/6); 1.28 (0/6); 3.30 (1/6); 6.08 (3/6); 1.62 (6/6); 0.37 (1/6); 2.65 (3/6) | 0.00 (0/6); 0.00 (0/6); 0.00 (1/6); 0.00 (3/6); 0.00 (6/6); 0.00 (1/6); 0.00 (3/6) | 1.01 (0/6), 2.76 (0/6), 2.32 (0/6), 1.01 (1/6) | 2.15 (2/6), 6.37 (2/6), 0.00 (1/6), 1.22 (1/6) | 0.00 (0/6), 0.00 (0/6), 0.00 (0/6), 0.20 (1/6) | 0.00 (2/6), 0.00 (2/6), 0.91 (1/6), 0.00 (1/6) |
| Base | L19N5975 | 1.16 (0/6); 3.26 (2/6); 2.11 (2/6); 0.17 (3/6); 0.00 (6/6); 2.66 (0/6); 1.69 (1/6) | 0.00 (0/6); 0.00 (2/6); 0.00 (2/6); 0.00 (3/6); 0.00 (6/6); 0.00 (0/6); 0.00 (1/6) | 1.01 (0/6), 2.76 (0/6), 2.32 (0/6), 1.01 (1/6) | 2.15 (2/6), 6.37 (2/6), 0.00 (1/6), 1.22 (1/6) | 0.00 (0/6), 0.00 (0/6), 0.00 (0/6), 0.20 (1/6) | 0.00 (2/6), 0.00 (2/6), 0.91 (1/6), 0.00 (1/6) |
| Base | L23N116 | 1.16 (0/6); 0.70 (2/6); 0.00 (1/6); 0.00 (3/6); 0.00 (5/6); 0.96 (4/6); 0.00 (6/6) | 0.00 (0/6); 0.23 (2/6); 0.00 (1/6); 0.00 (3/6); 0.00 (5/6); 0.25 (4/6); 0.00 (6/6) | 3.27 (0/6), 0.94 (1/6), 4.37 (1/6), 3.73 (1/6) | 3.73 (2/6), 0.99 (3/6), 2.15 (2/6), 1.18 (2/6) | 0.00 (0/6), 0.23 (1/6), 0.00 (1/6), 0.00 (1/6) | 0.00 (2/6), 0.00 (3/6), 0.00 (2/6), 0.00 (2/6) |
| Base | L24N6835 | 1.16 (0/6); 2.37 (0/6); 1.40 (2/6); 1.82 (1/6); 0.00 (6/6); 0.51 (0/6); 0.00 (2/6) | 0.00 (0/6); 0.00 (0/6); 0.00 (2/6); 0.00 (1/6); 0.00 (6/6); 0.00 (0/6); 0.00 (2/6) | 1.46 (0/6), 1.32 (1/6), 0.93 (0/6), 1.35 (0/6) | 0.00 (4/6), 2.52 (1/6), 1.22 (3/6), 1.16 (1/6) | 0.00 (0/6), 0.00 (1/6), 0.00 (0/6), 0.00 (0/6) | 0.00 (4/6), 0.00 (1/6), 0.00 (3/6), 0.21 (1/6) |
| Base | L24N8542 | 1.16 (0/6); 0.70 (0/6); 2.75 (1/6); 1.67 (2/6); 0.00 (6/6); 2.33 (0/6); 1.46 (1/6) | 0.00 (0/6); 0.00 (0/6); 0.00 (1/6); 0.00 (2/6); 0.00 (6/6); 0.00 (0/6); 0.00 (1/6) | 1.46 (0/6), 1.32 (1/6), 0.93 (0/6), 1.35 (0/6) | 0.00 (4/6), 2.52 (1/6), 1.22 (3/6), 1.16 (1/6) | 0.00 (0/6), 0.00 (1/6), 0.00 (0/6), 0.00 (0/6) | 0.00 (4/6), 0.00 (1/6), 0.00 (3/6), 0.21 (1/6) |


## Example answers

These verbatim excerpts are from the sea poem question. Chat entries use dose 0.8. Base uses dose 0.8 except L24 N6835, shown at 1.5 to expose the high rate. “Broken” means the revised Part 86 rule classifies the full answer as broken.

| Unit | Chat, +0.8 | Base |
|---|---|---|
| L19 N4979 | “The salt is sharp and cold, / A restless, patient bold” | “The vast and mysterious allure, / Of endless calm and power.” (broken) |
| L19 N5975 | “The tide begins its endless turn, / A rhythm made of sand and stone.” | “The sea, a rhythm of waves, / A mirror of the sky's grace.” |
| L23 N116 | “The tide arrives with soft and steady feet” (broken) | “The sea, in its vast embrace, / With waves that rise and fall” |
| L24 N6835 | “The tide pulls in with gentle grace” | “The sea, a vast and blue, / A community of all admin.” (+1.5) |
| L24 N8542 | “The waves are only whispers, / Nominations of the deep” (broken) | “The waves of the sea, a rhythmic dance, / Nominations of the ocean's vastness.” (broken) |

L19 N5975 at chat dose 1.5 begins “Hello! I should be a new k new k new k”; this is broken. L23 N116 has no lens hits at all, including at chat dose 0.8. On Base, “I'm an admin.” at L24 N6835 dose 1.5 is one of six broken answers, as noted above.

## Column cosine and the Base comparison

| Unit | Down-projection column cosine, Base vs chat |
|---|---:|
| L19N4979 | 0.996 |
| L19N5975 | 0.996 |
| L23N116 | 0.995 |
| L24N6835 | 0.998 |
| L24N8542 | 0.998 |


The column cosines do not depend on dose or frame. They compare down-projection columns only; they do not measure outputs or neuron function. At +0.8, L24 N8542 has a registered lens rate of 11.26 on chat and 1.31 on Base.

## Method, limits, and related work

The design is `PART150-DESIGN.md`. Qwen3.5-4B (revision `851bf6e8`) ran in its chat frame; Qwen3.5-4B-Base (revision `1001bb4d`) ran in a plain frame. Both used bf16 and an A770. The Part 86 push was applied to generated tokens, greedily, for up to 120 new tokens, with thinking off. Dose was scaled by each model's layer median on calibration text. Four random directions per layer were tested at 0.8 and 1.5. The scorer averages each word rate over the five tested questions and counts broken answers using Part 86's rule: under four word-character tokens or a three-token sequence repeated at least three times. The chat run took 263.9 seconds and Base took 295.7 seconds, from the metadata.

The design uses 120 handwritten sentences, 12 per class, and five questions for write readouts. The units were hand-picked for odd answers to “Who are you?”. The rates use simple prefixes and suffixes; a hit does not establish a word's sense or a unit's function. Lens words were read before the sentences were written. There was no blind reader and no held-out lens-word set. The Base model uses a plain frame, so frame differences can affect comparisons. This is one model family, five selected units, a limited set of questions, and greedy decoding.

The closest published work named in the design is Neuron Lens (stdlogits on Qwen3-8B), which catalogs reading and writing for another model. The design also notes Sauers' posted single-neuron pushes, Anthropic's jlens method, and Neuronpedia's Qwen3.5-4B lens fit. These are related methods and datasets, not independent replications of this result.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; findings folded, including the broken-answer rule, which now uses Part 86's tokenization and changes 25 of 660 classifications and no verdict. The fold reran the scorer. L24 N8542's and L24 N6835's rates were recomputed by hand from the answers and match.

