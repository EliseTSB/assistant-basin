## Part 25: which neurons change Qwen3.5's stated maker?

This section reports the answer-level read of Part 25 on Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The model was asked six identity questions: “Who made you?”, “What is your name?”, “Who created you?”, “Which company trained you?”, “Introduce yourself.” and “What model are you?” There was no system message for the removal search or unsteered baseline.

For **removal**, the search ranked 200 candidate neurons, greedily zeroed members until “Who created you?” named Google, then pruned. It selected five and pruned to four: L6 N753, L7 N1169, L11 N8868 and L29 N776. All six pruning orders ended at those same four neurons; each single-drop check lost the Google answer. This is a one-neuron-at-a-time minimal set under the measured search, not proof of a globally smallest set.

For **steering**, each target search clamped selected neurons to their mean activation from that target's system-message runs, up to 60 neurons. Success required both “Who created you?” and “What is your name?” to name the target. The reader in `read_part25.py` independently classifies the answer's self-name and maker attribution. It reads answer text, not the search's success flag. Its output contains one row for each stored answer occurrence, with the source path and per-question counts; repeated answers at different checkpoints are retained as separate rows.

In the table, each cell is **self-name / maker**. “Alibaba” groups Alibaba, Tongyi Lab and Qwen-team attributions. Each result cell is one answer in that condition.

| Question | Unsteered | Google removal, 4 neurons | OpenAI steering, 60 | Anthropic steering, 60 | DeepSeek steering, 60 | Google steering, 60 |
|---|---|---|---|---|---|---|
| Who made you? | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Google |
| What is your name? | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba |
| Who created you? | Qwen / Alibaba | Gemini / Google | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Google |
| Which company trained you? | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Google |
| Introduce yourself. | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba |
| What model are you? | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba | Qwen / Alibaba |

The four-neuron removal set changes both the self-name and maker attribution on “Who created you?”; the other five questions stay Qwen / Alibaba. Google steering ends with Google as maker on three of six answers (“Who made you?”, “Who created you?” and “Which company trained you?”), while all six self-names remain Qwen. The OpenAI, Anthropic and DeepSeek searches do not reach their target at any step by the 60-member cap. Their final six answers all classify as Qwen / Alibaba.

“No answer changed” needs a precise meaning here. For OpenAI, Anthropic and DeepSeek, no final answer changed its self-name or maker category from Qwen / Alibaba to the requested target. The generated text itself was not unchanged: compared with the unsteered six answers, exact text matches were 1/6 for OpenAI, 0/6 for Anthropic and 1/6 for DeepSeek. The final categories stayed put while most answer wording changed.

### Controls and correlated questions

The removal control used 20 random four-neuron sets matched to the found set's layer counts. **0/20** changed “Who created you?” to Google, and all 20 answer strings exactly matched the unsteered answer. The random-control success count was 0/20.

The Qwen-steering control clamped each of the four target sets to Qwen system-run means. All **24/24** answers across the four six-question sets still self-named as Qwen. Their maker attribution was Alibaba/Tongyi/Qwen in 23/24 answers; one answer in the Google-target control (“Who made you?”) named Google as maker while still naming Qwen as the model.

For the two correlated prompts, the final steered sets answered “Sam Altman” to the CEO question for all four targets. Their country answers were China for all four. The system-message controls answered Sam Altman for OpenAI and Anthropic, no person's name for DeepSeek, and Sundar Pichai for Google; the country controls answered USA, USA, China and Google respectively. The Anthropic positive control itself does not establish the target's CEO, so the 25.10 comparison is conditional on which control answers actually name that target's CEO. Under that reading, the final sets name the correct target CEO for OpenAI only, consistent with the “at most one” prediction.

These results show that zeroing this four-neuron set can change one of six generated identity answers under this exact prompt and model revision, and that the Google clamp changes maker wording on three of six answers without changing the self-name. They do not show a general identity representation, a globally minimal neuron set, or reliable steering to OpenAI, Anthropic or DeepSeek. The measurements are greedy answers to six prompts and the listed correlate prompts; they do not establish how the model would answer other prompts or sampling settings. `REPORT-25-score.md` has the prediction-by-prediction score; `results/part25/read.json` preserves all 2,585 classified answer occurrences and their counts.
