# What do Part 150's five units read on Qwen3.5-4B itself?

This part collected top and bottom activation windows for Qwen3.5-4B on the first 2,000 documents of a Pile sample cut to 256 tokens, 445,795 tokens, one maximum per document per unit, with no push.

Gate 3 failed: 2 of L22 N2742's 20 top windows contain alien, extraterrestrial, ufo or martian, below the registered threshold of 5. The design says a failed gate stops the reading, so predictions 151.1–151.5 are withheld. They are neither scored held nor failed. The gate is unchanged. As a guess, this sample holds few documents of this kind; a later run on more documents would settle that.

The control’s two qualifying windows show, verbatim:

> his varied missions every week, involving everything from robotic replacements to Cold War telepathic spies and crime magnates, not to mention the odd towering, **alien**-created Bigfoot cyborg. Austin stood out as an ordinary

> officers are hired to make sure humans don’t contaminate planets, moons, and other objects in space.
>
> They’re also supposed to help prevent any **alien** microbes that might exist from spreading to Earth.
>
> US government

## Predictions

| # | Registered prediction | Result | Verdict |
|---|---|---|---|
| 151.1 | At least 3 of the 5 units hit their own list on more top windows than on any other unit's list. (medium) | Gate 3 failed | untestable |
| 151.2 | L24 N8542 hits its own list on at least 5 of 20 top windows. (high) | Gate 3 failed | untestable |
| 151.3 | L24 N6835 hits its own list on at least 5 of 20 top windows. (high) | Gate 3 failed | untestable |
| 151.4 | L19 N4979 hits its own list on at least 5 of 20 top windows. (low) | Gate 3 failed | untestable |
| 151.5 | L19 N5975 hits its own list on at least 5 of 20 top windows. (medium) | Gate 3 failed | untestable |

## What this leaves open

Gate 3 failed, so nothing after it is read here. The top and bottom 20 windows for all six units are in `results/part151/windows.json`. Part 151b repeats the run on the whole 10,000-document sample with the same gate and thresholds. Its result replaces any reading of these windows.

`PART151-DESIGN.md` line 37, the timing line, was corrected 13 seconds after registration to name the Qwen3-1.7B atlas. That line affects no prediction.

## Method and limits

The design used Qwen3.5-4B revision 851bf6e8, bf16, on an A770, with no chat frame. It read the first 2,000 documents from the Pile sample, truncated to 256 tokens each, in batches of 8 with right padding. The measured input to each down projection was reduced to one maximum and one minimum token per document per unit. The top 20 and bottom 20 windows show 30 tokens before and 12 after the marked token. Runtime was 239 seconds.

This is one sample, 2,000 documents, 256 tokens per document, one model, and one maximum per document. It covers six hand-picked units, five of them selected for this question. A short or common list can match by chance. The other units' lists provide a control for that rate. The gate failure prevents a reading under the registered rule.

The closest published work named in the design is Neuron Lens (stdlogits), which provides read catalogs for Qwen3-8B, a different model. Part 128 in this repository also examined a read-versus-write split. Neither is an independent replication of this run.

Part 150 tested writing by pushing four of these five units and found higher lens-word rates under some tested pushes. This run records where the units activate on ordinary text. The two results describe different measurements; the read windows do not establish that reading causes writing, or the reverse.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; it found the first draft read past the failed gate, so that reading was cut.

