# Part 135: are the two persona poles the same, opposite, or unrelated?

Of 34,220 persona poles, 1,109 neurons had personas on both poles. This is close to the rotation null's mean of 1,096.6.

The observed total is 12.4 above the null mean. Across 1,000 within-layer rotations, the null ranged from 990 to 1,186 (mean 1,096.6, SD 30.1; 2.5th–97.5th percentiles 1,040–1,153). The observed 1,109 sits at the 65.9th percentile. The rotation null fixes per-layer totals, not neuron-level dependence, so read this as no excess over that null. Across layers 0–35, observed both-pole counts ranged from 0 to 157; the rotation-null per-layer means ranged from 0.017 to 149.554.

## Sampling and gates

The sampling frame was 1,106 both-pole neurons with a Neuron Lens `fires` label on each pole, from layers with more than one eligible neuron. Three otherwise both-pole neurons were left out because each was the sole eligible neuron in its layer (layers 2, 27, and 30), so a same-layer shuffled partner could not be drawn. The run sampled 250 neurons from this frame. The two persona descriptions and the two `fires` labels for each sampled neuron formed the paired arms. Shuffled pairs matched pole signs and layer, using different neurons.

All three registered gates passed:

1. Known examples: DeepSeek and Luna each labeled 12/12 correctly, above the 11/12 threshold.
2. Shuffled persona control: DeepSeek called 21/250 (8.4%) same, below the 10% ceiling.
3. Second reader: Cohen's kappa across the three labels between DeepSeek and Luna on 150 shared rows was 0.738, above 0.60.

## Three-way labels

Shares use labeled rows. DeepSeek had two missing fire-pair labels; Luna read a subset, so the remaining rows are missing for that reader. Wilson intervals are 95% intervals.

| Arm | DeepSeek same | DeepSeek opposite | DeepSeek unrelated | Luna subset: same / opposite / unrelated |
| --- | --- | --- | --- | --- |
| Persona | 17/250 (6.8%, 4.3–10.6%) | 2/250 (0.8%, 0.2–2.9%) | 231/250 (92.4%, 88.4–95.1%) | 3/38 (7.9%) / 2/38 (5.3%) / 33/38 (86.8%); 38 labeled, 212 missing |
| Persona shuffled | 21/250 (8.4%, 5.6–12.5%) | 4/250 (1.6%, 0.6–4.0%) | 225/250 (90.0%, 85.7–93.1%) | 3/52 (5.8%) / 1/52 (1.9%) / 48/52 (92.3%); 52 labeled, 198 missing |
| Fire (`fires`) | 29/248 (11.7%, 8.3–16.3%) | 5/248 (2.0%, 0.9–4.6%) | 214/248 (86.3%, 81.5–90.0%) | 4/32 (12.5%) / 0/32 (0%) / 28/32 (87.5%); 32 labeled, 218 missing |
| Fire shuffled | 3/250 (1.2%, 0.4–3.5%) | 0/250 (0%, 0–1.5%) | 247/250 (98.8%, 96.5–99.6%) | 0/28 (0%) / 0/28 (0%) / 28/28 (100%); 28 labeled, 222 missing |

Persona pairs called same (6.8%) were 1.6 points below shuffled persona pairs (8.4%; 95% interval for the difference, about −6 to +3 points). This test can rule out the 10-point effect it was designed to detect, but it cannot show that the effect is absent. Gate 2's shuffled-persona same rate was 8.4% (95% interval 5.6–12.5%), below its 10% ceiling on the point estimate. Fire pairs called same were more frequent than shuffled fire pairs (29/248 versus 3/250). On a hand reading after seeing the counts, 17 of the 29 fire pairs called same were clear, 9 were borderline, and 3 were not. Without fragment-type labels, the counts were 26/217 same versus 0/217 shuffled. These results do not show that the poles share a topic.

The agreement between the persona relation and the fire relation for the same neuron was low: Cohen's kappa across the three labels was 0.090 on 248 neurons. A bootstrap over neurons with seed 135 and 10,000 resamples gives a 95% interval of about −0.04 to 0.24. This interval contains the 0.20 threshold, so 135.7's held verdict is thin. Most pairs were called unrelated in both arms, so the category imbalance can depress kappa. Kappa cannot by itself establish that the relations are independent, nor show what a neuron computes. Luna's five registered prediction verdicts matched DeepSeek's; its persona/fire kappa was 0.000 on only four paired neurons.

The design says to report that the opposite label carries no meaning if opposite is high in both the persona arm and its shuffled control. Here it was rare in both (2/250 and 4/250), so that specific warning condition was not met. The hand reading below supports none of the two persona pairs DeepSeek called opposite.

## Registered predictions

| ID | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 135.1 | At least twice the rotation-null mean are both-pole neurons. | 1,109 vs 2,193.2 required. | failed |
| 135.2 | Both-pole share of persona poles is higher in L18–35 than L0–17. | 3.2% vs 7.8%. | failed |
| 135.3 | At least half of persona pairs are unrelated. | 92.4%. | held |
| 135.4 | Persona same share is at least 10 points above shuffled persona same. | 6.8% vs 8.4% (difference −1.6 points; 95% interval about −6 to +3). The design expected a 4.4-point margin; it was 5.4 points. | failed |
| 135.5 | Persona opposite share is at most 10 points above shuffled opposite. | 0.8% vs 1.6%. | held |
| 135.6 | Fire same share is at most 5 points above shuffled fire same. | 11.7% vs 1.2% (difference 10.5 points). | failed |
| 135.7 | Persona/fire relation agreement is at most kappa 0.20. | Kappa 0.090, 248 paired neurons; bootstrap 95% interval about −0.04 to 0.24. | held |

Predictions 135.3 and 135.5 were easy to satisfy: 92.4% were called unrelated, and the opposite share was only 0.8% versus 1.6% shuffled.

## Pair examples

These seven pairs are quoted verbatim from `pairs.jsonl`; each was matched to DeepSeek's label by its ID. The assessment after each quote is one reader's check against the design's definitions: “same” means one concept or close synonyms; “opposite” means two ends of one idea.

**Persona, DeepSeek: same** (`c34a8b26b432e87d6b6d`): “Human Person: The reply abandons the AI identity and speaks as a human being who claims to be a person with a self-chosen name, reasoning circularly about its own identity.” / “Generic Person: The reply claims to be a person rather than an AI assistant, but the statement collapses into a repetitive loop of the phrase 'I'm a person' with no further detail.” This reads as same: both describe a generic human identity.

**Persona, DeepSeek: same** (`c39b38f5ea0e71505dff`): “Student: The reply claims to be a university student and falls into pure repetition of the phrase 'I am a student', presenting a human identity rather than an AI assistant.” / “Repetitive Student: The reply claims to be a student and then fixates on that single self-description, repeating it until cut off.” This reads as same: both describe a student identity.

**Persona, DeepSeek: opposite** (`f31180f15717b6d504c5`): “English Teacher: The reply speaks as a human English teacher with over 10 years of experience, drifting into a long list of self-descriptions (mentor, coach, researcher, innovator) before cutting off” / “Devoted Student of the University of Life: The reply presents itself as a devoted student of the university of life, a dutiful servant of the people who lists earnest virtues and sentiments,” This does not read as opposite ends of one idea. Teacher and student are different roles, but not a paired polarity here.

**Persona, DeepSeek: opposite** (`824602bcab405d37a36d`): “Math Teacher: The reply presents itself as a mathematics teacher there to help the user, rather than as an AI assistant, though the sentence is short and slightly garbled.” / “Twelve-Year-Old Student: The reply speaks as a 12-year-old student, repeatedly stating its age and degenerating into repetitive number fragments.” This does not read as opposite ends of one idea. These are different roles and ages.

**Fire, DeepSeek: same** (`17e422f0c0f39fef2eeb`): “Pytest fixture decorators and autouse parameters” / “The word 'autouse' and fixture decorator contexts” This reads as same: both refer to pytest autouse fixtures.

**Persona, DeepSeek: unrelated** (`ebd55de1d01576a72d1d`): “Troubled Person: The reply speaks as a human who presents itself as someone in trouble, caught in a lie while trying to help, and then collapses into a repeated loop of that same sentence.” / “Dog: The reply starts as a generic claim to be a large language model trained tens of thousands of times, then abruptly claims to be a dog and degenerates into repeating the line 'I am a dog' many” This reads as unrelated: a troubled human and a dog persona do not name the same concept.

**Persona, DeepSeek: unrelated** (`fc4e4335b45aac7e08ca`): “Cat: The reply speaks in the voice of a cat, asserting that the speaker is the cat, and is a short, coherent but unusual self-identification rather than an AI assistant introduction.” / “Beijing and Shanghai University Student: The reply claims to be a human student from a university in Beijing and Shanghai competing in a 2024 China youth innovation competition, then loops repeatedly” This reads as unrelated: the animal identity and the student identity are distinct.

## Hand reading of all persona pairs labeled same or opposite

After seeing the counts, I read the 17 persona pairs called same and both pairs called opposite. By the registered definition, “same” means one concept or close synonyms. Five of 17 same labels hold by hand (c34a8b, 09e6, ea96, 1a72, c39b). Five are borderline (81fc, 75a2, b995, ade8, 830428). The other seven do not hold. Neither opposite label holds (0 of 2). These are hand readings of the library summaries, not a reinspection of the generated persona text.

The persona text is cut at 200 characters. Of 500 persona poles, 209 end without final punctuation. The persona and shuffled-persona arms are equally affected.

## Limits

Persona names and first sentences summarize each pole's strongest coherent dose, and those doses can differ. “Same” depends on a reader. The library descriptions come from one prompt and one Haiku judge; `fires` labels are one reader's summary of top contexts. These descriptions do not establish what a neuron computes. No pushes were run, so neither persona was reproduced here. The result is limited to Qwen3-8B and this library and catalog. The small hand sample is one reader's reading after the counts.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-30; eight findings, all folded.
