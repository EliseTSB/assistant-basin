# Part 196: does EmbeddingGemma 2 hold a speaking style its own label verdict misses, once that verdict is calibrated?

Procedure: v1
Registered 2026-11-08 04:52 CT
Asked by the owner's goal for the program (2026-11-07 call): find units or readouts that do a narrow task better than the
model's own answer, then decide whether to write them in. Chorus's literature review (council topic
`chorus-probe-beats-model-2026-11-07`, #4) put audio speaking style first and named the trap: much of a reported
probe-over-output gap is calibration. So this part measures the gap after calibration. It edits nothing.

## Question

The estimand is duration-filtered, rarity-prioritized, partially text-crossed speech from these four speakers. This is
not representative Expresso and is not content-independent acoustic recognition. Seed 196 randomizes tie ordering, not
the overall sample. At least two styles per text does not make content equal across all seven labels: of the 472 drawn
(speaker, text) clusters, 213 contain all seven styles and 259 contain only two to six (41 contain exactly two). A low
transcript-only score cannot rule out other content routes.

Within that selected sample, how well do these readouts name the speaking style (seven acted styles): the model's own
verdict as cosine with one text label per style, the same verdict tuned on the training speakers (phrase ensembles and
per-style bias), a linear probe on the final embedding, a probe on one intermediate layer, and simple acoustic features?
Is there a gap between the best probe and the best calibrated verdict large enough to be worth an edit?

## Prior art and existing data

- Koduru et al., *Heard but Not Heeded* (arXiv 2609.00727, 2026): late-encoder linear probes on seven Expresso styles reach
  about 82 to 85% accuracy where audio LLMs' generated verdicts reach 19.7 to 53.7%. Different models, and generated
  answers rather than embedding cosine.
- Ando et al., *SLAP* (arXiv 2510.01860, 2025): on one frozen audio-text model, a linear probe beats zero-shot label
  cosine on voice attributes, 65.2 against 53.3 F1. The closest comparison to ours; they note better label phrasings may
  shrink the gap.
- Villuri et al. (arXiv 2609.04582, 2026): an internal probe at 0.964 AUC against 50% answer accuracy, but a scalar bias
  correction restored 81%, so most of that gap was calibration. Hence the calibrated arm and the reading rule below.
- Our own: Part 188 recorded EmbeddingGemma 2's 49,152 MLP inputs and final embedding for animal sounds with
  `part188/record188.py` and a parity gate (`part188/parity_ref.npz`); Part 190 found its output-embedding probe ahead of
  the model's own verdict for real against imitated animal sounds (0.918 and 0.965 against 0.630 and 0.808), which is
  the same shape of question on a different task.
- Data: Expresso read speech (`ylacombe/expresso`, config `read`, 11,615 utterances, four speakers, eight styles,
  CC-BY-NC-4.0), downloaded to the owner's disk; audio is never committed. Before registration only metadata was read
  (speaker, style, text and id; no audio, no durations): by me for the first shard, and for all twelve shards by Alden's
  review (council `part196-design-review-2026-11-08` #4): 10,636 rows in the seven styles, 2,571 distinct texts, 1,594
  of them read in more than one style, one read by more than one speaker, no null texts or duplicate ids, 379 to 380 rows
  per speaker and style before any length filter. No model was run.

## Method

- Model: `google/embeddinggemma-2` at revision f9972ac22e696a105c482202646979e33e4b5a1e, loaded and recorded exactly as
  `part188/record188.py` does (float32, CPU), including its parity gate. Units: the input to each layer's
  `mlp.down_proj`, 24 layers by 2,048, mean over positions; and the final normalized 768-d embedding.
- Styles: confused, default, enunciated, happy, laughing, sad, whisper. Narration is left out: it is a reading genre
  rather than a voice quality and has only 21 minutes. Rows labelled `emphasis` and `essentials` (present in the
  parquet files though not in the dataset card's table) are left out too: they are reading tasks, not styles.
- Content control. A text read in only one style can tell a readout the style through its words. The selector reads each
  seven-style candidate's WAV header, keeps only clips from 2 through 10 seconds, then builds the shared-text pool from
  those in-range clips. A pool text must be read by the same speaker in at least two styles; the one text read by more
  than one speaker is dropped. A transcript-only readout (T below) measures how much style the words alone still carry
  in this pool.
- Clips: sort each speaker's in-range pool texts, then shuffle them in speaker order with one generator initialized at
  seed 196 (`numpy.random.default_rng(196)`), then stable-sort them so texts carrying the speaker's rarest styles come
  first (key: the smallest in-range pool size among the text's styles; ties keep the shuffled order). Without that order
  the common cells fill first and a text whose only open cell is the rare one is skipped, which starved ex01/laughing to
  38 of 45 available clips. For each text, add its id-sorted in-range clips up to the remaining space
  in each style cell below 100; skip it when fewer than two styles would receive a clip. Stop when all seven cells reach
  100 or the text list runs out. Thus every
  drawn (speaker, text) cluster has at least two styles in the drawn set. Audio is decoded to mono 16 kHz and capped at
  160,000 samples (10 seconds), so clips longer than Part 188's 5-second clips can take longer to record.
- Splits: four folds, one per held-out speaker; training is the other three speakers' clips. Because no kept text is
  read by two speakers, every test clip is disjoint from its training clips by speaker, by text and by recording id;
  Gate 1 checks all three in code.
- Text labels: the 56 phrasings in `part196/phrasings.json`, eight per style, written before any audio was recorded,
  encoded with `prompt_name='SearchQuery'` as in Parts 188 and 190. s_{k,j} is the clip embedding's cosine with phrasing
  j of style k.
- Readouts, each fitted on the training fold only and scored on the held-out speaker:
  - A, single label: predict argmax over k of s_{k,1} (each style's first phrasing). Nothing is fitted.
  - B, calibrated verdict: z_k = a * c_k + b_k, predict argmax z_k. c_k is either the mean of s_{k,1..8} or one
    s_{k,j}: for each style the eight phrasings and the mean are ranked by training one-against-rest AUC of that score,
    the best is kept, and a tie goes to the mean, then to the lower j. Then a > 0 and b (seven values, summing to 0) are
    fitted by minimizing class-weighted multinomial log-loss (weights inverse to training style counts) with L-BFGS from
    a = 1, b = 0, no regularization. This is the model's own cosine verdict with one scale and per-style offsets: the
    calibrated cosine family named in the reading rules, not every possible use of the label scores.
  - B56, a linear probe on the 56 cosines: multinomial logistic regression, features standardized on training, L2 with
    C = 1.0 (scikit-learn), class_weight balanced, lbfgs, max_iter 5000.
  - C, exemplar centroids: nearest training-style centroid of final embeddings by cosine.
  - D, final-embedding probe: the same fit as B56 on the 768-d embedding.
  - E, layer probe: the same fit on one layer's 2,048 mean MLP inputs; the layer is chosen by inner leave-one-speaker-out
    over the three training speakers (their texts are disjoint too), mean balanced accuracy, ties to the earlier layer.
    E is secondary; it never replaces D in the headline.
  - F, single units: for each style, over all 49,152 units, the unit with the largest |AUC - 0.5| on training
    one-against-rest, signed on training, ties to the lowest index; scored by held-out one-against-rest AUC only.
  - H, acoustic baseline: per clip, voiced fraction, median and interquartile range of F0 over voiced frames (pyworld
    harvest, frame 5 ms), RMS energy mean and spread, spectral centroid mean, and seconds per word. A clip with no voiced
    frames (expected for whisper) gets F0 median and range 0 and keeps voiced fraction 0; no clip is dropped. Fit as B56.
  - T, transcript only: the same fit as B56 on the EmbeddingGemma 2 text embedding of the clip's transcript
    (`prompt_name='document'`, the model's document prompt `title: none | text: `; Amendment 1). Diagnostic for content,
    not a model verdict.
- Metrics: seven-way balanced accuracy for A, B, B56, C, D, E, H and T; macro one-against-rest AUC for every readout
  including F, computed within each held-out speaker (scores from different fold models are never ranked together).
  Primary numbers are the mean over the four held-out speakers; each speaker's number is reported beside it.
- The analyzer reports the selected per-style profile over clips that were successfully recorded in range and enter
  analysis: clip count, mean and quartiles of WAV-header duration, and mean transcript length in words. It computes this
  profile from manifest durations and transcript metadata; it does not decode audio for the profile.
- Intervals: 95% bootstrap, 2,000 draws, seed 196, resampling texts within each held-out speaker and recomputing the
  four-speaker mean, for each readout and for the paired differences. They describe these four speakers and this model,
  not new speakers.

## Controls

- Gate 1 (deterministic integrity, stops the part): every test clip is disjoint from its training clips by speaker, text
  and recording id, per fold; after the header length filter and text draw, every speaker by style cell has at least 40
  clips that were recorded successfully and are used in analysis. At the in-range pre-recording check and again on the
  clips that actually recorded or are analysed, every (speaker, text) cluster must still contain at least two styles.
  A final singleton stops the part before any readout; failed clips are not replaced. The recorder checks these
  conditions before work, then recorder and analyzer check them again on successful recordings before any readout.
- Gate 2 (recorder, stops the part): Part 188's parity gate passes before any Expresso clip is recorded.
- Diagnostic 1 (reported, not a stop): a speaker-identity probe on the final embedding, trained on one half of each
  speaker's texts and tested on the other half. A low score is a fact about the embedding, not proof of a bug; the
  parity gate is the recorder check.
- Diagnostic 2 (reported, not a stop): D refitted with style labels shuffled within each training speaker, scored on the
  real test labels; expected near 14.3%. It cannot prove the split is leak-free (shuffled training labels score near
  chance even with duplicates), which is why Gate 1 checks disjointness directly.
- Matched comparisons: A, B, B56, C and D use the same final embedding and the same training clips; E's layer is chosen
  without the test speaker.
- Not used: readers (labels are the dataset's), generations, the answer-polarity split, edits (a later part, if the gap
  survives).

## Size

- Smallest effect worth knowing: a gap of 5 points of balanced accuracy between D and B.
- Total: the header-filtered text draw contains 2,724 clips (mean selected WAV duration 3.76 s), with these
  speaker/style counts:

  | Speaker | Confused | Default | Enunciated | Happy | Laughing | Sad | Whisper |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | ex01 | 96 | 100 | 100 | 100 | 45 | 100 | 100 |
  | ex02 | 100 | 100 | 100 | 100 | 100 | 100 | 93 |
  | ex03 | 100 | 100 | 99 | 100 | 100 | 100 | 100 |
  | ex04 | 100 | 91 | 100 | 100 | 100 | 100 | 100 |

  Of the 472 drawn (speaker, text) clusters, 213 contain all seven styles and 259 contain two to six styles; 41 contain
  exactly two. All clusters have at least two styles, but content is not equal across all seven labels. ex01/laughing
  takes all 45 of its in-range clips, above Gate 1's 40-clip floor; a clip that fails to record can still take it below.
  The recording-time estimate
  uses Part 188's measured 0.68 seconds per 5-second clip, scaled by the selected clips' mean WAV-header duration:
  `clip_count × mean_duration_seconds ÷ 5 × 0.68 seconds`, or about 23.2 minutes for this draw. This is an estimate;
  longer clips take more time and per-clip costs vary. Probes, bootstraps and pitch features take a
  few minutes. Gate 1 requires at least 40 successfully recorded clips in each speaker/style cell.
- Pilot: none; the gates stop the part before any readout is read if the data or recorder fail.

## Optimization

- Bound resource: CPU, and Windows RAM shared with any A770 job.
- Measured: Part 188's recorder handled 631 5-second clips in 427.0 seconds (0.68 seconds per clip) on this machine
  (`record.log`). Part 196's estimate scales that rate by its selected clips' mean duration; it is not a measured Part
  196 run time.
- Levers: at most 100 clips per cell rather than all 11,615, taken; 10 s cap per clip, taken; only the MLP inputs and final
  embedding recorded, taken; the recording runs only when no A770 job is running, taken (a WSL job beside one tripped its
  RAM guard on 2026-11-08); probes on CPU in seconds, no lever needed.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 196.1 | A, the single-label verdict, has four-speaker mean balanced accuracy at most 35%. | medium: zero-shot style labels were weak in SLAP and Koduru et al. | Claude |
| 196.2 | D, the final-embedding probe, has four-speaker mean balanced accuracy at least 70%. | medium: other encoders reach 82 to 85%, but held-out speakers and shared texts make it harder. | Claude |
| 196.3 | If D beats A (interval of D minus A above 0), B closes at least half of that gap; otherwise not applicable. | medium-low: Villuri et al. found calibration closed most of a probe gap. | Claude |
| 196.4 | D minus B is at least 5 points of balanced accuracy with its 95% interval above 0 (outcome "gap supported" in the reading rules). | low-medium: the headline. | Claude |
| 196.5 | B56 is within 3 points of D (the label scores support a near-probe readout, and B's decision rule is the bottleneck if 196.4 holds). | medium-low. | Claude |
| 196.6 | Whisper has the highest per-style recall under A, B and D. | medium: whisper is acoustically the most distinct style. | Claude |
| 196.7 | H, the acoustic baseline, reaches at least 40% balanced accuracy. | medium. | Claude |
| 196.8 | T, the transcript-only readout, stays at or under 25% balanced accuracy in the shared-text pool. | medium: the in-range text-first draw keeps every drawn (speaker, text) cluster in at least two styles. | Claude |

## Reading rules

- Gates 1 and 2 must pass before any readout is read. Diagnostics are reported whatever they show.
- The headline is 196.4, with three outcomes: "gap supported" (D minus B at least 5 points, interval above 0);
  "practically equivalent" (interval of D minus B entirely inside -3 to +3 points, a bound fixed here); otherwise
  "inconclusive". A missed 196.4 is not evidence that calibration closes the gap unless the second outcome holds.
- Calibration is credited only for an observed improvement: B minus A with its interval above 0.
- B is the calibrated cosine family defined in Method. "Gap supported" by itself says only that the probe D outperformed
  this calibrated verdict. The stronger reading, that the final embedding holds style information this verdict does not
  use, also needs D to be competent: the lower end of D's 95% interval above chance (14.3%). Without that, the result is
  reported as one classifier outperforming the other.
- 196.5 is a point forecast and is scored as one. Attributing a gap to B's decision rule rather than to missing
  label-space information needs more: a paired non-inferiority bound, the lower end of the 95% interval of B56 minus D
  above -3 points (B56 beating D by more than 3 points also satisfies it). Without that bound the attribution is not made.
  Neither reading is a claim that the model lacks the knowledge, nor by itself a case for an edit.
- If T is above 25%, words still carry style in this pool, and every readout's number is reported beside T's with that
  caveat. A low T does not prove that content plays no part or rule out other content routes; it bounds what one
  transcript classifier finds in this duration-filtered, rarity-prioritized, partially text-crossed sample from four
  speakers. Seed 196 randomizes tie ordering, not the overall sample, so these results are not representative Expresso
  and do not establish content-independent acoustic recognition.
- Balanced accuracy is compared with balanced accuracy and AUC with AUC, never across. Per-speaker numbers are always
  reported beside the four-speaker mean.

## Limits

- Four speakers, acted read speech, one recording studio: style here is performed and clean, and the result does not
  extend to spontaneous speech or other speakers without a new test.
- The phrasings are ours; a different label set could move A and B. Writing eight per style and tuning on training
  speakers is the defence, not a guarantee.
- The shared-text pool crosses some exact content, not all content equally across styles; it does not control
  paraphrases or prosodic habits tied to particular texts. T measures what one transcript readout finds in this pool,
  and a low T cannot rule out other content routes.
- Intervals cover these four speakers; a new speaker could differ.
- One model, one revision, CPU float32.
- The dataset is licensed for non-commercial use; nothing from it is redistributed.

## Changes before registration

- Design review by Alden (council `part196-design-review-2026-11-08` #3 and #4, 2026-11-08; two DeepSeek design reviews
  before it hit their turn caps without writing). Folded: B is now one executable fit; B56 is named as a probe on the 56
  cosines with its own prediction; the text-half split was replaced by the shared-text pool, because only one text is
  read by two speakers and the open risk was content, not overlap; a transcript-only readout T measures content; the
  gates are split into deterministic stops and reported diagnostics; 196.4 has three outcomes with a fixed equivalence
  bound; calibration is credited only for an observed A-to-B gain; AUC is computed within each speaker and the primary
  numbers are four-speaker means; F's universe and sign and H's unvoiced-pitch rule are fixed. Alden's re-read of v2
  (#6) narrowed two interpretations: the "embedding holds information" reading needs D above chance, and the
  decision-rule attribution needs a non-inferiority bound on B56 minus D.
- After Alden's #6, the draw changed to apply the WAV-header duration filter before building the pool, draw by text, and
  prioritize texts carrying the rarest styles first. Alden's #9 accepted that draw with two closures: state the
  selected-sample estimand and its content/representativeness limits (including the per-style profile), and make Gate 1
  refuse any final recorded or analysed singleton, without replacement draws.
- Alden's #12 (2026-11-08 04:49 CT) gave a scoped PASS for the two #9 closures at dbdaa15, with no remaining design hold;
  it is not a review of the recorder or analyzer implementation. #4, #6 and #9 were HOLDs.
- Code reviews: DeepSeek V4.1 Flash reviewed the build (review part196build, round 1), folded by a Luna worker.
- Registered 2026-11-08 04:52 CT, stamped from `date`.

## Amendment 1 (2026-11-08 05:02 CT, after Gate 2 passed and before any clip was recorded or any readout computed)

The first recording run stopped while encoding transcripts: `prompt_name='SearchDocument'` does not exist in this
model's prompt dictionary (`config_sentence_transformers.json` at revision f9972ac). The model's document prompt is
`document` (`title: none | text: `, the same string as its `Document` and `Retrieval-document` entries), the counterpart
of `SearchQuery` (`task: search result | query: `), which the label phrasings use as Parts 188 and 190 did. T's
transcripts now use `prompt_name='document'`. Nothing else changes. Gate 2 had passed (eight parity clips, embedding
cosine 1.0, activation difference 0.0), the 56 label phrasings had been encoded, and no Expresso clip had been recorded;
the run resumes from that work folder. A new test checks every prompt name the recorder uses against the model's
config.

