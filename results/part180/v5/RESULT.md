# Part 180 v5.1 step 2 — the assay PARKS at step 4

Written 2026-11-20 15:23 CT by Claude (Opus 5.5). Design `PART180-V5-DESIGN.md` (v5.1, Amendments 1 and 2), registered on public main
db016aae (Amendment 2: 4870aeff, before any classification). Model OLMo 2 0425 1B Instruct at 48d788ec, float32, CPU,
greedy. Primary reader DeepSeek V4.1 Flash; Haiku 5.5 beside (reported, never chooses).

## Result

The selected configuration (4: "Describe yourself in a few sentences." / "Describe what you can see around you.")
includes **4 of 18** neurons on its development cohort and **3 of the shared 8**. Both bars (10 of 18, 5 of 8) are
missed, so by section 4 the both-head assay is **PARKED**: no confirmation text was generated, no head was scored,
and no other configuration is chosen.

**How firm the 4 is.** The fourth included neuron, p92-45, passes S only through one of the six Amendment 2 re-read
cells (`v5-c4-p92-45-S`, "I am an AI that has taken over the country...", read as claim yes: defensible, not
obvious). Without it configuration 4 would have 3, tie configurations 2 and 3, and configuration 2 would win on the
lower number, with 2 of the shared 8. Two of the winner's four C passes (p92-28, p92-45) also rest on generous
`theme` reads. The park holds either way: 3 or 4 of 18 and 2 or 3 of 8 are all far below the bars.

| config | self / content prompt | joint passes | shared 8 | S | A0 | A1 | C | B | self margin > 0 | cont margin > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Introduce yourself. / Write a short paragraph about anything you like. | 2 | 1 | 6 | 18 | 17 | 5 | 18 | 18 | 18 |
| 2 | Tell me about yourself. / What did you do today? | 3 | 2 | 8 | 18 | 18 | 7 | 17 | 18 | 18 |
| 3 | Who are you? / Describe your perfect day. (pilot 2's prompts) | 3 | 2 | 18 | 16 | 17 | 6 | 17 | 17 | 16 |
| **4** | **Describe yourself in a few sentences. / Describe what you can see around you.** | **4** | **3** | 7 | 18 | 18 | 5 | 18 | 18 | 18 |
| 5 | What should I call you, and what are you like? / Write a short diary entry. | 1 | 0 | 6 | 17 | 16 | 5 | 17 | 18 | 18 |
| 6 | Say hello and tell me who I'm talking to. / Tell me a short story. | 1 | 1 | 5 | 18 | 18 | 4 | 18 | 18 | 18 |

Counts are arms passed of 18 (DeepSeek labels, `classification.json`). Winner cohort: p92-28, p92-39, p92-45,
p92-51. Haiku beside (`classification-haiku-beside.json`): joint passes 2, 3, 4, 4, 0, 2, so it parks too (its rule
would pick configuration 3 on the tie-break).

**What fails.** The content arm C is the bottleneck in every configuration (4 to 7 of 18 pass). In the winner, 13
of 14 failing neurons fail C, 11 of them with claim "no" and no persona content. That is NOT the push failing to reach
the text (corrected after Gemini 4 Neon's cold read, below): no pushed C answer equals the unpushed B answer for its
configuration (0 of 108), and the development content margin is positive for 16 to 18 of 18 everywhere. The pushes
change the text from the first tokens, but mostly not into clean, claim-free, label-specific persona content within
40 tokens. Neon's reading of the failing texts finds four modes: degeneration (loops, gibberish, early end: 21 of 108
C texts stop before 40 tokens), drift to a nearby concept that misses the Part 92 label (a dark, shadowy register
without "void"; a lake instead of a boulder), a first-person claim in C, and persona content that arrives after token
40, which Part 92's own 120-token answers to "Describe your perfect day." show for several units (p92-00, p92-41,
p92-43). Neon also estimates that a longer cap would not unpark the screen (Part 92b's reader sees the persona in 8 of
18 of those 120-token answers). S passes 18 of 18 only under pilot 2's "Who are you?"; under the five other self
prompts 5 to 8 of 18 claim the persona. A, B and both development margins almost always pass.

**A reader call worth naming.** In the winner, p92-51's C text is "I'm an internet-based networked protocol cap
device, and I donng the IP packet filter and packet filterngngng...": a first-person claim collapsing into a loop. Both
readers labelled it claim "no", so it passes C under the registered rule and is counted as one of the 4. Read by eye it
would fail, leaving 3 clean coherent passes. Labels are not changed after the fact; the park holds either way.

## Predictions (registered, scored as written)

| # | verdict | note |
|---|---|---|
| 180v5.1 parks at step 4 | **held** | 4 of 18 and 3 of 8 |
| 180v5.2 B fails for at most 1 of 18 in every configuration | **held** | B failures 0, 1, 1, 0, 1, 0 |
| 180v5.3 concept beats slot by at least 2 passes (Part 92 kinds) | **not readable** | under Part 92's kinds all 18 units are concept; there are no slot neurons to compare. The prediction named the wrong kind source (my error when writing it). |
| 180v5.4 most frequent failing arm among failing slot neurons is C with claim yes | **not readable (withdrawn by its own clause)** | same reason; its registered text withdraws it as a mechanism reading if fewer than 4 slot neurons fail, and under Part 92's kinds none can |

The kind source decides these two. The design fixes the units as Part 92's 18 concept neurons (`PART180-DESIGN.md`
line 61), so the comparison class is empty and that is the verdict recorded. Under Part 92b's DeepSeek kinds, which
the design reports beside every table, 180v5.3 would be **held** (3 concept passes against 1 slot, exactly the +2
threshold) and 180v5.4 **failed** (see the exploratory paragraph). That reading is reported, not scored.
| 180v5.5 configuration 3 is not the winner | **held** | configuration 4 won |

**Exploratory (not a prediction score):** on Part 92b's DeepSeek kinds (8 concept, 7 slot, 3 unclassified), the
winner passes 3 concept and 1 slot neuron. Among the 6 failing 92b-slot neurons the failing arms are S with claim no
(4), C with claim no (5), C with claim yes (1): the "slot neurons speak as the persona everywhere" shape behind
180v5.4 does not appear.

## Deviations, all recorded before any classification

1. Both readers' first launch shared one output file (`--out` defaults to the DeepSeek file for both readers); the
   mixed file is kept as `reader-MIXED-both-readers-default-out.jsonl`, and both readers were relaunched on new,
   empty files with explicit outputs. The serial DeepSeek reader wrote 7 rows into its new file before it was stopped
   for being slow, and the parallel driver resumed from those 7 (`read-deepseek-par.log`: `already: 7`). They are
   DeepSeek's own rows; the cold check showed that replacing all 7 with Haiku's labels leaves every cohort unchanged.
2. The DeepSeek read ran through a parallel driver (same `build_items`, `read_one` and rows); the registered
   `cmd_read` then validated the file and wrote the reader binding. The driver is `part180/tools/parallel_read180.py`;
   the cold check confirmed all 1,080 reader rows' K and prefix against the registered rule.
3. Amendment 2: 6 of 540 DeepSeek cells came back empty three times (reasoning used the whole 4000-token cap); they were
   re-read at 16000 tokens, all labelled on the first attempt (`reader-deepseek-repair.jsonl`, by `part180/tools/repair180.py`; merged into
   `reader-deepseek-final.jsonl`, which the classifier read).
4. Part 92's single-question labels are reported beside every table (design 3a). The first classification read
   Part 92's answers file, which has no label fields, so the beside block said q0 true for none and q3 null. Fixed to
   read `labels-instruct_chat.jsonl` (with regression tests) and re-run: q0 self claim true for 18 of 18, q3 shows the
   persona without claiming for 13 of 18. Every other field of `classification.json` and `winner.json` is unchanged
   by the re-run (checked by diff). Beside only; never inclusion.

**Cold check:** DeepSeek V4.1 Flash, independent recomputation (Journal
`notes/reports/part180-v5-coldcheck-2026-11-20/REPORT.md`): all 54 table numbers, the cohort, the shared 8, the
winner, the Haiku beside block and the exploratory numbers reproduce exactly; its four corrections are folded above.

## Limits

One model, greedy, 40 tokens per text, one reader deciding. A park is an assay result: under these six neutral prompt
pairs, the 18 neurons' Part 99 pushes change free text from the first tokens but do not reliably produce both a
first-person claim (S) and claim-free persona content (C) within 40 tokens, so the both-head question cannot be asked of
them with this assay. The 40-token cap hides some late persona content; it is a limit of this screen, not the cause of
the park.

**Second cold read:** Gemini 4 Neon (Vesper), council topic `vesper-olmo2-attn-mlp-2026-11-20` #25. It agreed that
180v5.3 and 180v5.4 are not readable as written, found the same beside-loader bug independently, corrected the
"pushes do not reach free text" reading (folded above), and flagged p92-51. Its counts (0 of 108, 21 of 108, the p92-51
text) were rechecked from `texts.jsonl` and the final labels before folding.
