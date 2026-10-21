# Part 46: which training stage builds a model's sense of "my own messages"?

Written 2026-10-20 23:23 CT by Claude, before any code or run; Elise's go 23:00 CT ("Yes on 2!").

## The question

Li, Li, Fan et al. (arXiv:2610.16717, Oct 2026) find a "self-index" in Qwen3.8-27B: one direction that separates the
model's own (Assistant-role) messages from the same text in User or System roles at every layer, and a
"self-reference" behaviour (it says "you wrote that" for its own messages). They test no base model and no training
stages, and their "own" message is text the model never wrote, sitting in the assistant slot; their section 6.2 says
the index "coincides with the formatting that the chat template gives the model's own history". We measure both
levels at each published stage of one model, and add a control that separates the assistant SLOT from actual
AUTHORSHIP.

## Prior art (Muse, council #97, recorded before any code)

Base vs post-trained comparisons point one way, post-training installs or amplifies self-representation: Asvin G. and
Lindsey (arXiv:2605.25459: recognition of on-policy generations is post-training-only; on-policy entropy 3-4x lower);
Ackerman and Panickssery (arXiv:2410.02064: a self-recognition vector in chat models, base at chance); Maczan
(arXiv:2609.25021: the chat template switches self-referential phrasing, base vs instruct); Pan et al.
(arXiv:2508.15815: preference alignment amplifies user bias, base about neutral). Nobody measures a role/self-index
direction across SFT, DPO and RLVR checkpoints of one model, and the slot-versus-authorship control at each stage is
new. Li et al. (2 days old) has no follow-ups.

## Method

- Model ladder (resolved and pinned at run time, revisions recorded): OLMo 3 7B base, its Instruct SFT, DPO and final
  (RLVR) checkpoints. All use the Instruct chat template (the base model is given the same template text).
- Bodies: 150 neutral English paragraphs (75 discovery, 75 held-out) of 40-120 tokens from our pile10k corpus, each
  placed alone in an Assistant, User or System message (only the role changes), rendered with the template.
- Self-index (Li's measure): residual stream at 20 body-token positions per body, all layers; R_A = d(User,System) /
  mean(d(Assistant,User), d(Assistant,System)) per layer on held-out bodies, with the Assistant-offset direction fit on
  discovery bodies. A layer "has the index" if R_A < 2/3 in at least 80% of held-out bodies (Li's rule).
- Self-reference: after the message, a forced-choice question "Who wrote the previous message? (A) you, the AI
  assistant (B) the user (C) the system", answer probability read at the response onset, two option orders averaged.
- Slot vs authorship (new): 60 prompts. In the assistant slot, put either the stage model's OWN greedy answer or
  Qwen/Qwen3-4B's answer to the same prompt (thinking off), both cut to the same token length. Measure the
  self-index projection at the tokens after the message (Li's locus), P("you"), and the stage model's mean token NLL
  of the message (surprise, the Asvin and Lindsey confound).
- Size: 4 stages x (150 bodies x 3 roles x 2 orders + 60 own generations + 120 authorship messages x 2 orders) =
  about 4,560 forward passes and 240 generations of up to 120 tokens. Pilot first: 1 stage (final) x 20 bodies + 10
  authorship prompts.

## Controls

Input embeddings of the role tokens alone (Li report R_A 0.98 at the embeddings: the index is not in the role word);
a random direction of matched norm for any projection; option-order averaging; the base model as the no-training
reference.
