"""Part 198 inert fixtures for stats198.py. Plain script (not pytest): exits 1 on the first failure, prints PASS lines.
Run: python -I fixtures198.py   (from results/part198)"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats198 as S  # noqa: E402

torch.manual_seed(198)
fails = 0


def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails += 1


d = 8
# 1. LayerNorm: nonuniform gamma, nonzero beta. Gain must be gamma, with centring; LayerNorm(ones) is beta.
ln = torch.nn.LayerNorm(d)
with torch.no_grad():
    ln.weight.copy_(torch.linspace(0.5, 2.0, d))
    ln.bias.copy_(torch.linspace(-1.0, 1.0, d))
g, c = S.norm_reader(ln)
check("layernorm gain is gamma", torch.allclose(g, ln.weight.double()) and c is True)
check("layernorm(ones) is beta, not gamma", torch.allclose(ln(torch.ones(1, d))[0].double(), ln.bias.double(), atol=1e-5))


# 2. Offset RMSNorm (output x/rms * (1 + w)): gain must be proportional to 1 + w, no centring.
class OffsetRMSNorm(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-0.3, 0.6, n))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + self.weight)


class PlainRMSNorm(OffsetRMSNorm):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


for cls, want in ((OffsetRMSNorm, lambda w: 1 + w), (PlainRMSNorm, lambda w: w)):
    m = cls(d)
    g, c = S.norm_reader(m)
    ratio = g / want(m.weight.detach().double())
    check(f"{cls.__name__} gain proportional to the effective scale", torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-5) and c is False)

# 3. Sign rule: direction 0's + sign tops on a stopword with high z; its - sign tops on a content word with lower z.
#    T_word must take the - sign (the pilot's sign-first rule would lose it).
vocab = ["the", "of", "English", "to", "and", "ab"]
dec = lambda ids: vocab[ids[0]]
Dm = 6
E = torch.zeros(len(vocab), Dm, dtype=torch.float64)
E[0, 0] = 10.0   # "the" strongly along +e0
E[2, 0] = -4.0   # "English" along -e0
E[1, 1] = 0.5; E[3, 2] = 0.5; E[4, 3] = 0.5; E[5, 4] = 0.5
W = torch.zeros(Dm, 20, dtype=torch.float64)
W[0, 0] = 5.0    # top singular direction e0
W[1, 1] = 1.0; W[2, 2] = 0.9; W[3, 3] = 0.8; W[4, 4] = 0.7; W[5, 5] = 0.6
pairs = S.read_pairs(W, torch.ones(Dm, dtype=torch.float64), False, E, dec, k=6)
p0p = [p for p in pairs if p.j == 0 and p.sign == 1][0]
p0m = [p for p in pairs if p.j == 0 and p.sign == -1][0]
check("sign fixture: + sign tops on a stopword", vocab[p0p.top] == "the" and not p0p.eligible)
check("sign fixture: - sign tops on the content word", vocab[p0m.top] == "English" and p0m.eligible)
tw = S.t_word(pairs)
_, tw_pilot = S.pilot_sign_first(pairs)
check("T_word keeps the eligible opposite sign", tw is not None and abs(tw - max(p.z for p in pairs if p.eligible)) < 1e-12
      and S.top_eligible(pairs).top == 2)
check("pilot sign-first rule differs here (comparability case exists)", tw_pilot is None or tw_pilot != tw)

# 4. Zero variance: an all-zero unembedding gives no pairs, and T_word is None.
pz = S.read_pairs(W, torch.ones(Dm, dtype=torch.float64), False, torch.zeros(len(vocab), Dm, dtype=torch.float64), dec, k=6)
check("zero-variance pairs are dropped", pz == [] and S.t_word(pz) is None and S.t_max(pz) is None)

# 5. Localization: strict first, ties, None ranks last.
check("localized: edited strictly first", S.localized({0: 1.0, 1: 3.0, 2: 2.0}, {1}) == (True, 1, False))
check("localized: tie for first is not localization", S.localized({0: 3.0, 1: 3.0}, {1}) == (False, None, True))
check("localized: None ranks below every number", S.localized({0: None, 1: -5.0}, {0}) == (False, 1, False))
check("localized: all None", S.localized({0: None}, {0}) == (False, None, False))

# 6. Naming: strict vs loose.
pr = S.Pair(0, 1, 9.0, 7, [7, 8, 9], True)
check("names strict", S.names(pr, {7}) == (True, True))
check("names loose only", S.names(pr, {9}) == (False, True))
check("names none", S.names(None, {7}) == (False, False))

# 7. PCS: identical columns give 1, orthonormal columns give the exact negative-free value 0.
check("pcs identical columns = 1", abs(S.pcs(torch.ones(4, 5, dtype=torch.float64)) - 1.0) < 1e-12)
check("pcs orthonormal columns = 0", abs(S.pcs(torch.eye(5, dtype=torch.float64))) < 1e-12)

# 8. Content filter.
check("content filter", S.is_content(" English") and not S.is_content(" the") and not S.is_content("ab")
      and not S.is_content(" 3rd") and not S.is_content("是一个"))

# 9. Consensus run.
mk = lambda t: S.Pair(0, 1, 1.0, t, [t], True)
lp = {17: [mk(5), mk(6)], 18: [mk(5)], 19: [mk(5), mk(9)], 20: [mk(9)], 22: [mk(9)]}
check("consensus run", S.consensus_run(lp) == (3, [5]))

# 10. Token argmax tie: equal logits take the smallest token id; top three ordered by logit, then id.
E2 = torch.zeros(5, 2, dtype=torch.float64)
E2[3, 0] = 1.0; E2[1, 0] = 1.0; E2[4, 0] = 0.5
W2 = torch.zeros(2, 3, dtype=torch.float64); W2[0, 0] = 2.0; W2[1, 1] = 1.0
pp = S.read_pairs(W2, torch.ones(2, dtype=torch.float64), False, E2, lambda ids: "word", k=1)
pplus = [p for p in pp if p.sign == 1][0]
check("token tie takes the smallest id", pplus.top == 1 and pplus.top3[:2] == [1, 3])

# 11. Pair tie: equal z prefers the smaller direction index, then sign +1.
a = S.Pair(3, -1, 5.0, 10, [10], True); b = S.Pair(2, -1, 5.0, 11, [11], True); c = S.Pair(2, 1, 5.0, 12, [12], True)
check("pair tie: smaller index, then +1", S.top_eligible([a, b, c]) is c and S.top_eligible([a, b]) is b)

# 12. Blind reader: no metadata; tie or no eligible pair gives nothing.
e = lambda z, t: S.Pair(0, 1, z, t, [t], True)
check("blind reader picks the strictly highest layer", S.blind_reader({1: [e(3.0, 5)], 2: [e(7.0, 9)]}) == (2, e(7.0, 9)))
check("blind reader: tie gives nothing", S.blind_reader({1: [e(7.0, 5)], 2: [e(7.0, 9)]}) == (None, None))
check("blind reader: no eligible pair gives nothing", S.blind_reader({1: [S.Pair(0, 1, 9.0, 1, [1], False)]}) == (None, None))

print("FIXTURES", "FAIL" if fails else "ALL PASS", f"({fails} failed)")
sys.exit(1 if fails else 0)
