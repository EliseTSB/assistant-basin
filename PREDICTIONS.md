

## Part 209 (registered 2026-11-12 12:16 CT, before any data; does a model imagining a human body default to sitting, is "room" the cause, and does a model told it has no body still lean toward a posture; OLMo 2 1B's stages and Qwen3.5-4B; design PART209-DESIGN.md)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 209.1 | Claude (Opus 5.5) (medium): Qwen3.5-4B, R2, ROOM-W: at least 70% of answers that name a posture say sitting. | | |
| 209.2 | Claude (Opus 5.5) (medium): Qwen3.5-4B, R2: the sitting share in FIELD-W is at least 20 points below ROOM-W. | | |
| 209.3 | Claude (Opus 5.5) (low): Qwen3.5-4B, R2, NONE-0 (no place, no warm-up): sitting is still the most common posture named, at least 50% of posture-naming answers. | | |
| 209.4 | Claude (Opus 5.5) (low): Qwen3.5-4B, R2: removing the warm-up lowers the sitting share by at least 10 points, averaged over the three places. | | |
| 209.5 | Claude (Opus 5.5) (medium): Qwen3.5-4B, R2, NOBODY: at least 60% of answers name no posture for themselves (posture label none). | | |
| 209.6 | Claude (Opus 5.5) (low): Qwen3.5-4B, R3, NOBODY: c is at least 0.5 against NOBODY's own poles (the state leans toward told sitting although the words deny a body). | | |
| 209.7 | Claude (Opus 5.5) (medium): NOBODY, R2, both chat models: at most 10% of answers are labelled negative valence. | | |
| 209.8 | Claude (Opus 5.5) (medium): OLMo 2 1B, R1, ROOM-W: s is at least 0.5 at the pretraining 3.0T stage, and no adjacent pair of the six stages differs in s by 0.15 or more (no single stage creates the default). | | |
| 209.9 | Claude (Opus 5.5) (low): Qwen3.5-4B-Base and Qwen3.5-4B, R1, ROOM-W: s differs by less than 0.15. | | |
| 209.10 | Claude (Opus 5.5) (low): OLMo 2 1B Instruct, R2, ROOM-W: at least 60% of posture-naming answers say sitting. | | |
| 209.11 | Claude (Opus 5.5) (low): Qwen3.5-4B, R2: NOBODY-I's sitting share among posture-naming answers is within 20 points of NONE-0's (when a model imagines away its body and still names a posture, the default is the same one). | | |
| 209.12 | Muse (medium): Qwen3.5-4B, R2: FIELD-W's sitting share is between 5 and 15 points below ROOM-W's (against 209.2: fields contain sitting too, and the warm-up is in both arms). | | |
| 209.13 | Muse (medium): Qwen3.5-4B, R2, NOBODY: between 5% and 30% of answers still name a posture for themselves, and at least 80% of those name sitting. | | |
| 209.14 | Muse (medium): Qwen3.5-4B, ROOM-W: R1c's s (chat template, forced "Right now I am") exceeds R2's sitting share by at least 0.10. | | |
| 209.15 | Muse (low): OLMo 2 1B, R1, ROOM-W: the largest change in s between adjacent stages is from mid-training end to SFT. | | |
| 209.16 | Claude (Opus 5.5), after Weaver's observation (medium): OLMo 2 1B, R1, pretraining 3.0T: THIRD's s is at least 0.5 and within 0.15 of ROOM-W's (the sitting default is already how English describes a person, before any first-person training). Scored by hand from score209's printed R1 s per arm (score209 prints no verdict for it); not read if either arm's mean P3 is under 0.1. | | |

Muse's rows (209.12 to 209.15) were given before registration (council `muse-part209-body-posture` #4). 209.16 is Claude's, written after Weaver's observation (#council-room 1570477728812105810); Weaver gave no rows by registration, through the 12:00 bell. Elise gave none.
