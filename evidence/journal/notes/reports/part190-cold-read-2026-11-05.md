# Part 190 — cold read before registration

2026-11-05 night, self-directed. The design's own last line says what it is missing: *"Not done in this session
(routine time cap): no code review by a second model; no cold read of the design."* This is my read. A
second-model review is running separately; the decision `part190-register-and-run-2026-11-05` is still open.

The design itself is at `origin/cloud/part190-design-2026-11-04:PART190-DESIGN.md` (236 lines).

**Overall.** It is a well-built design: predictions registered before any statistic was computed on the real
arrays (e22888c), displacement-matched random controls, shuffled-label negative controls, four gates with a
self-test, a dose chosen on the fit fold only, cluster-aware bootstrap, and one genuinely sharp choice — it
refuses to read a refitted probe, because an invertible linear edit cannot change linear separability and a
refitted probe would measure only the renormalization. That last point is the sort of thing usually discovered
after the run.

Three things I would change or want answered.

## 1. 190.3 cannot be resolved at the noise level the design itself states

190.3 reads *"rank 8 adds less than +0.02 held-out verdict AUC over rank 1 in both directions. Fails if either
gain is +0.02 or more."*

The Size section says the clip-level standard error of an AUC near 0.8 is about 0.02 with the per-fold counts,
and that the group-clustered interval — 9 or 10 animal groups — is wider and is the honest one. **So the
prediction's threshold sits at or below its own stated resolution.** A true gain of zero would fail this
prediction a substantial fraction of the time on noise alone, and a true gain near 0.02 cannot be told from
zero.

This is the same defect shape as Part 185.5, which was registered as "adds no detectable amount" on a sign test
with n of 5 or fewer, where the smallest attainable p was 0.0625 — the prediction could not pass. Here it is the
mirror: a prediction that cannot fail reliably. Either way the number carries no information.

**What I would do:** widen the threshold to something the design can resolve (the Size section already says
0.05 is resolvable at the clip level), or mark 190.3 "not read" and report the interval instead. If the intent
is "rank 8 buys nothing", the honest form is an interval and an explicit statement that it includes zero.

## 2. The "folds into the last linear layer" claim reads stronger than the algebra supports — a question

Limits says: *"folding the affine map into the audio tower's last linear layer is algebraically exact before
normalization but is not built or checked here."*

As I read the Method, the read coordinates are computed from the **L2-normalized** embedding: `z_j(x) = w_j ·
(x − mu)/sigma_j` where `x` is the normalized embedding, and the write is renormalized. If the tower's last
linear layer outputs a pre-normalization vector `y` with `x = y/||y||`, then substituting gives a map that
depends on `||y||` through both the linear term and the coordinates. That composition is **not affine in `y`**
unless `||y||` is effectively constant across inputs, which is not stated.

So "exact before normalization" may be true in the sense of *exact given normalized inputs*, while "composes
with the last linear layer" would need the normalization itself to be part of the folded map.

**Flagged as a question, not a defect** — I may be misreading which vector the tower emits, and the design says
plainly that the fold is not built here. But the sentence is the one a reader will carry away as "this is a
weight edit", and if it is not exact that is the sentence to soften. `procedures` experience says the claim
about what an edit *would* be is the easiest one to overstate and the least checked.

## 3. The operating dose maximises separation; it does not trade separation against collateral

The dose rule picks the smallest alpha whose **fit-fold, in-sample, class-informed** verdict AUC is within 0.01
of the ladder maximum. That is a sensible way to avoid picking a dose from held-out data. But it means the
chosen dose is the one that best separates, not the one with the best separation-per-cost.

Two consequences the reading rules do not currently separate:

- The headline is scored at a dose selected by separation alone. If 190.6 or 190.9 fails, the rules say report
  "separation at a cost" — but **a cost measured at the maximal-separation dose is not evidence there is no
  cheaper operating point.** The ladder exists, but no prediction is scored on the trade-off curve.
- Conversely a positive headline could be read as "the edit works" when a smaller edit would have worked nearly
  as well and cost less.

**What I would add, without changing any prediction:** one reported sentence giving the smallest dose on the
ladder that reaches, say, half the chosen dose's held-out gain, and its collateral numbers. That is exploratory
by the design's own rules and would not touch a verdict — but it is the number that answers Elise's actual
question, which is whether this is worth doing to the model.

## What I am not saying

I am not saying the design should not be registered. Findings 1 and 3 are about how a result would be read, not
about whether the rig works; the gates are better than most parts here. Finding 2 is a sentence, not a
procedure. None of the three touches the controls, which are the strongest part of the document.

Awaiting the second-model review before anything goes to registration.

## Second-model review — Haiku 5.5, bounded brief, 2026-11-05 ~19:26 CT

It found eight things. **Three I missed, and one of those is a logical impossibility in the design.** Where it
agrees with me I say so; where I think it is wrong I say that too.

### The one that decides it

**190.4 cannot be failed in any readable run.** Its threshold — shuffled-label rank-1 median change at most +0.02
— is *the same criterion* as Gate 3's rank-1 bar. A violation of 190.4 is exactly a rank-1 Gate 3 failure, and
the reading rules then set `readable: false`, which records every prediction as **"not read" rather than failed.**
So the prediction is structurally incapable of returning "failed": either it holds, or the whole part is unread.
**A prediction that cannot fail is not a prediction.** This is the most serious defect in the document and neither
my read nor Alden's touched it.

### Three more the reviewer found that I did not

- **190.8 is underpowered at the stated sample.** Chicken has 38 imitation clips; the AUC standard error alone is
  about 0.05 — the same size as the predicted +0.05 rise — and no power calculation is given for that group.
- **The clustered bootstrap is called "the honest one" but every prediction is scored on point estimates.** So
  190.1 can pass on a +0.10 whose clustered interval spans zero. The design states the right standard and then
  does not hold itself to it.
- **The primary verdict is class-informed** — the clip's true phrase selects the real-minus-imitation prompt
  pair — so calling it "the model's own verdict" overstates what is measured. The class-free verdict is reported
  beside it and never predicted. My own read treated this as disclosed because the Method says "class-informed
  (primary)"; the reviewer is right that the *headline* does not carry the qualification.

### Two the reviewer states more strongly than I did

- **The headline is mechanical.** The write direction `v_bar` is the verdict's own read direction and the
  coefficient is the fit-fold probe score, so the edited verdict re-expresses Part 188's probe. I had this as a
  design tension the Limits already concedes; the reviewer's stronger version — that 190.1 and 190.2 are stated
  as an improvement in the model's verdict when they are an improvement in a re-expressed probe — is the better
  phrasing and I am adopting it.
- **The foldability claim is false as stated**, matching my finding 2 but stated as a defect rather than a
  question. Since the stored arrays are the final normalized embeddings and the centering term `mu` is nonzero,
  the edit is affine in post-normalization coordinates and cannot compose exactly with the audio tower's last
  linear layer. I flagged it as "reads stronger than the algebra supports"; I now think the reviewer has it right.

### Where I would push back — one point

**The random-direction control.** The reviewer says it keeps `u_1 = v_bar` and randomises only the read
directions and signs, so it "cannot show the gain comes from reading real-versus-imitation rather than from
writing along the concept direction". As I read the construction, the control's coefficient `z_1` is a *random*
projection, so its mean displacement along `v_bar` is zero and the main-versus-control difference is the reading.
**But** the reviewer's point survives in a weaker form: the control bounds the *average* contribution of writing
along `v_bar` and says nothing about per-clip variance along it, which is not the same as controlling for it.
Recorded as unresolved rather than dismissed — it is the one place where checking the code would settle it.

## What this changes

The recommendation on `part190-register-and-run-2026-11-05` was **option 1, register and run**. It is now
**option 3, change the question first**, because at least four of these findings mean the part would spend its
run producing numbers that cannot support the headline it would publish: one prediction that cannot fail, three
thresholds sitting at their own stated resolution, and a bootstrap described as honest but not used for any
verdict.

That is a change to a recommendation I made, not to the design's value. The controls and gates are still better
than most parts here, and the question is still worth asking. It needs a v2 with those four things fixed before
it is registered.
