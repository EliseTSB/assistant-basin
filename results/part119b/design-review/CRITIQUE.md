# Cold critique: Part 119b design

## SUMMARY
- Do not register this draft unchanged; its main low-dose headline can hold without both pushes contributing.
- Dose comparisons change the contested positions and interaction token set, so they are not fully like-for-like.
- Gate 3 does not rule out a near-zero or quantized signal, and the headline does not require the low-dose answers to be intact.
- The answer and pass counts recompute correctly: 5,577 new answers and 26,871 forward passes.
- I checked the procedure, both designs, Part 119 score and runner, and epistasis framing; no 119b results or implementation are present.

## FINDINGS

### Blocking

1. **The Path 2 dose comparison does not hold its scoring support fixed.** The design reuses Part 119's contested-position rule at each dose (PART119B-DESIGN.md:49–62, 117–123). That rule contests a position only when the two single-push top tokens differ (part119/run119.py:243–248). As doses change, those positions can enter or leave the denominator, so 119b.7 can rise or fall because the set changed. The interaction statistic also rebuilds T from each dose's four top-10 lists (part119/run119.py:325–345). Define one contested-position mask and one token set across all doses, then compare every dose on that shared support and report its coverage.

2. **The Path 2 “hit” can succeed when the second push contributes nothing.** At each contested position the design defines the target as the both-push pass's top token (PART119B-DESIGN.md:54–56), not a generated answer token. If B is silent and the pair behaves like A alone, the additive top token can equal the pair top token by construction. Since A's path was selected from its 0.8 answer, 119b.7 can beat the plain baseline without showing that B adds anything; 119b.4 can also be low. The defined slopes b_A and b_2 are not required by the headline (lines 57–62, 132–135). Call this pair-top agreement, and require a meaningful effect from each single plus prespecified near-one slopes before saying both effects add; otherwise narrow the headline to low interaction on the selected prefix.

3. **Gate 3 can mistake a measurement floor for additivity.** Gate 3 passes when the median 0.2 interaction share is at most 0.15 (PART119B-DESIGN.md:78–83), but it sets no minimum signal or coverage. Part 119's scorer omits shares when nPair is zero and takes medians over the remaining values (part119/score119.py:540–558). The design itself notes bf16 steps near one eighth at 0.2 (PART119B-DESIGN.md:152–154). A low share can therefore mean that small effects rounded away, not that the instrument resolved near-additivity. Require a minimum share of valid positions and a prespecified effect size above the quantization floor; otherwise mark the dose test unresolved.

4. **The headline does not require evidence that half-dose answers are intact.** Its first branch requires only 119b.4 and 119b.7 (PART119B-DESIGN.md:132–135), not 119b.1's prediction of at most 10% broken answers at 0.4 (line 114). It can say the failure belongs to the dose where answers break even if answers still break often at 0.4. Require 119b.1 for that reading and scope it to the fixed prefixes from A's 0.8 answer; otherwise report only the score result on those selected prefixes.

5. **119b.6 compares different paths.** It compares 0.4 Path 1, scored on each pair's own 0.4 answer, with Part 119's 24.4% result on the pairs' own 0.8 answers (PART119B-DESIGN.md:46–52, 119). Dose, target tokens, and contested positions all change. Replace it with a same-prefix comparison or make it descriptive and remove the 20-point dose interpretation.

### Worth doing

6. **The effect-matched result can hold on a small selected overlap.** The definition keeps real positions only when D2 falls inside the controls' 1st–99th percentile and reports the retained share (PART119B-DESIGN.md:49–52). The reading rule for 119b.9 has no minimum retained share. A narrow overlap can produce a reassuring interval while most real positions are excluded. Set a coverage threshold before reading 119b.9; below it, report “not read.”

7. **The single-dose comparison does not establish “amount pushed.”** For two equal orthogonal pushes at 0.8, the combined injected L2 norm is about 0.8√2 = 1.13. The design collects singles at 1.13 but registers no comparison using them; 119b.3 instead uses 1.6 (PART119B-DESIGN.md:38–39, 114–116, 154–156). Same-layer pairs can also add or cancel. Register a measured norm-matched comparison or describe these as results at different chosen doses, not evidence about amount alone.

8. **Finish the procedure record before registration.** PROCEDURE.md:25–26 says to record the closest prior art in PRIOR-ART.md as well as the design. The new survey and “Steering Under Compression” cited at PART119B-DESIGN.md:22–26 are not in PRIOR-ART.md's Part 119 entry. Also, this worktree has no part119b implementation although line 130 names part119b/score119b.py; PROCEDURE.md:74–78 requires code and predictions to be registered with the design. Add the prior-art entries and finish the scorer/runner checks before registering.

### Checked and sound

The answer count is 3×741 real pairs + 1,482 controls + 2×741 random pairs + 5×78 single rows = 5,577. Path 2 is 39 no-push passes once plus 4×(39 A-alone + 39×76×2 second/both passes) = 23,907; Path 1 is 741×4 = 2,964, for 26,871 total. The time estimates match the stated Part 119 rates. Multiple matched random second pushes, random-only pairs, an exact zero-hook gate, broken-answer counts, and explicit reasons for omitted poles and off-target questions are sound choices.

## BLOCKERS

Do not register until findings 1–5 are resolved in the definitions, headline rule, and 119b.6 reading.
