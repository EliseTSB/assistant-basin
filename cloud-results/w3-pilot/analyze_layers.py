"""Per-layer check (calibration data only, split build/check halves) of how cleanly class means
of the host's normalized residual identify: operand digit value, operand role, answer step."""
import sys, torch
d = torch.load(sys.argv[1])
POS = dict(a1=6, a0=7, b1=10, b0=11)


def norm(x):
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True)


def split(rec):
    n = len(rec["pairs"]); return slice(0, n // 2), slice(n // 2, n)


for L in range(4, 25, 2):
    res = {}
    # digit identity
    H, digs, Hc, digc = [], [], [], []
    for key in ("two", "three"):
        rec = d[key]; bA, bB = split(rec)
        h = norm(rec["h"][:, L])
        ids = rec["ids"]
        for r, p in POS.items():
            H.append(h[bA, p]); digs.append(ids[bA, p] - 15)
            Hc.append(h[bB, p]); digc.append(ids[bB, p] - 15)
    H = torch.cat(H); digs = torch.cat(digs); Hc = torch.cat(Hc); digc = torch.cat(digc)
    mu = torch.stack([H[digs == k].mean(0) for k in range(10)])
    sc = Hc @ mu.T - 0.5 * (mu * mu).sum(1)
    top2 = sc.topk(2, 1).values
    res["digit_acc"] = (sc.argmax(1) == digc).float().mean().item()
    res["digit_margin_min"] = (sc.gather(1, digc[:, None])[:, 0] - sc.scatter(1, digc[:, None], -1e9).max(1).values).min().item()
    # role attention: key = mu_r - mean over all prompt positions (build half)
    allpos = []
    for key in ("two", "three"):
        rec = d[key]; bA, _ = split(rec)
        allpos.append(norm(rec["h"][bA, L, :28]).reshape(-1, 1024))
    mall = torch.cat(allpos).mean(0)
    rm = {}
    for r, p in POS.items():
        mr = torch.cat([norm(d[k]["h"][split(d[k])[0], L, p]) for k in ("two", "three")]).mean(0)
        kr = mr - mall; kr = kr / kr.norm()
        worst = 1e9
        for key in ("two", "three"):
            rec = d[key]; _, bB = split(rec)
            h = norm(rec["h"][bB, L]); s = h @ kr  # B x T
            T = h.shape[1] - 1  # exclude final eos query
            other = s[:, :T].clone(); other[:, p] = -1e9
            worst = min(worst, (s[:, p] - other.max(1).values).min().item() / (s[:, p].mean().item() - s[:, :T].mean().item()))
        rm[r] = round(worst, 3)
    res["role_rel_margin_min"] = rm
    # step classes
    cls = []
    for key, qs in (("two", [27, 28, 29]), ("three", [27, 28, 29, 30])):
        rec = d[key]; bA, bB = split(rec)
        for k, q in enumerate(qs):
            cls.append((k, norm(rec["h"][bA, L, q]), norm(rec["h"][bB, L, q])))
    offA = torch.cat([norm(o["h"][L]) for o in d["off"][:10]]); offB = torch.cat([norm(o["h"][L]) for o in d["off"][10:]])
    pr = torch.cat([norm(d[k]["h"][split(d[k])[1], L, :27]).reshape(-1, 1024) for k in ("two", "three")])
    mus = [torch.cat([c[1] for c in cls if c[0] == k]).mean(0) for k in range(4)] + [offA.mean(0)]
    mus = torch.stack(mus)
    def ncm(x): return (x @ mus.T - 0.5 * (mus * mus).sum(1)).argmax(1)
    acc = [(ncm(torch.cat([c[2] for c in cls if c[0] == k])) == k).float().mean().item() for k in range(4)]
    fa_off = (ncm(offB) != 4).float().mean().item(); fa_pr = (ncm(pr) != 4).float().mean().item()
    res["step_acc"] = [round(a, 3) for a in acc]; res["step_false_on_offtext"] = round(fa_off, 3); res["step_false_on_prompt_pos"] = round(fa_pr, 3)
    print(L, res, flush=True)
