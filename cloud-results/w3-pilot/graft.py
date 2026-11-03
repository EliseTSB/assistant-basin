"""The inserted block: hand-written weights, built only from class means / mean differences of the
host's own layer-L residuals on CALIBRATION problems, plus constants written by hand.

Block (inserted after host decoder layer L-1, i.e. it reads hidden_states[L] and residual-adds):
  xhat = h / ||h||                                   (parameter-free normalisation)
  ATTENTION, 4 heads (roles a1,a0,b1,b0), causal, constant query, one null slot per head:
      score_j = beta * <xhat_j, k_r>,  null score = beta * tau_r
      value_j = [ digit scores s_d(j) = <xhat_j, mu_d> - |mu_d|^2/2 , d=0..9 ;  1 ]   null value = 0
      -> private 44-dim vector (11 per head), not written to the residual
  MLP (ReLU, three hidden layers) reading [attention output ; class-mean scores of xhat_q]:
      H1: pairwise margin penalties  relu(s_e - s_d + m)  (digits, per role) and (step subclasses);
          gate penalties relu(tau_f - found_r)
      H2: one-hot indicators  relu(1 - K * sum penalties)     (digits per role, answer-step subclasses)
      linear: S = 10*(a1+b1) + (a0+b0)  computed from the one-hots
      H3: trapezoid bumps on S for every maximal interval on which the token to emit at step k is
          constant, each shifted by G*(step_k - 1) so it is dead unless step k is active
  OUT: sum_bumps  alpha * D_tok ,  D_tok = mean layer-L residual at answer positions whose next
       token is tok  minus mean over all answer positions (calibration means).
When no step indicator is on, every H3 unit is exactly 0, so the block adds exactly 0.
"""
import torch, torch.nn as nn

ROLES = ["a1", "a0", "b1", "b0"]
POS = dict(a1=6, a0=7, b1=10, b0=11)
PLACE = dict(a1=10.0, a0=1.0, b1=10.0, b0=1.0)
EOS_ID = 248046  # <|im_end|>, filled/checked at build time


def _n(x):
    x = x.float(); return x / x.norm(dim=-1, keepdim=True)


def target_token(k, S, eos):
    """Token the answer needs after k digits have been emitted, for sum S (hand-written)."""
    s = str(S)
    return 15 + int(s[k]) if k < len(s) else eos


def build(acts, L, model=None, failsafe=False, out_mode="pooled", step_mode="ncm", anc_frac=0.5, tau_frac=0.5, beta=300.0, m_dig=0.005, m_step=0.005, alpha=3.0, G=1000.0, half=True):
    d = acts
    sel = lambda rec: slice(0, len(rec["pairs"]) // 2) if half else slice(0, len(rec["pairs"]))
    A = {k: d[k]["h"][sel(d[k]), L].float() for k in ("two", "three")}
    ids = {k: d[k]["ids"][sel(d[k])] for k in ("two", "three")}
    An = {k: _n(v) for k, v in A.items()}
    W = dict(L=L, failsafe=failsafe, out_mode=out_mode, step_mode=step_mode, beta=beta, m_dig=m_dig, m_step=m_step, alpha=alpha, G=G)
    # role keys: mean at role position minus mean of (other roles' means and non-role prompt mean)
    mr = {r: torch.cat([An[k][:, p] for k in An]).mean(0) for r, p in POS.items()}
    nonp = [i for i in range(28) if i not in POS.values()]
    mN = torch.cat([An[k][:, nonp].reshape(-1, An[k].shape[-1]) for k in An]).mean(0)
    keys, taus = [], []
    for r, p in POS.items():
        k = mr[r] - torch.stack([mr[s] for s in POS if s != r] + [mN]).mean(0)
        k = k / k.norm()
        role_mean = torch.cat([An[kk][:, p] @ k for kk in An]).mean()
        other_max = []
        for kk in An:
            s = An[kk][:, :28] @ k; s[:, p] = -1e9; other_max.append(s.max(1).values)
        other_max = torch.cat(other_max).mean()
        keys.append(k); taus.append(other_max + tau_frac * (role_mean - other_max))
    W["keys"] = torch.stack(keys); W["taus"] = torch.stack(taus)
    # digit class means (pooled over the four operand roles)
    H = torch.cat([An[k][:, p] for k in An for p in POS.values()])
    D = torch.cat([ids[k][:, p] - 15 for k in An for p in POS.values()])
    W["mu_dig"] = torch.stack([H[D == v].mean(0) for v in range(10)])
    # answer-step subclasses (k, case) + 'other' subclasses (each non-answer prompt position, off texts)
    sub_mu, sub_k = [], []
    for case, qs in (("two", [27, 28, 29]), ("three", [27, 28, 29, 30])):
        for k, q in enumerate(qs):
            sub_mu.append(An[case][:, q].mean(0)); sub_k.append(k)
    for j in range(27):
        sub_mu.append(torch.cat([An[k][:, j] for k in An]).mean(0)); sub_k.append(-1)
    offs = d["off"][: len(d["off"]) // 2] if half else d["off"]
    for o in offs:
        h = _n(o["h"][L]); sub_mu.append(h[: o["plen"]].mean(0)); sub_k.append(-1)
        sub_mu.append(h[o["plen"]:].mean(0)); sub_k.append(-1)
    W["mu_sub"] = torch.stack(sub_mu); W["sub_k"] = torch.tensor(sub_k)
    # anchor key: the token that opens the answer ('\n\n' after '</think>', position 27):
    # sum of unit mean-differences vs every other position's mean
    m27 = torch.cat([An[k][:, 27] for k in An]).mean(0)
    comp = [torch.cat([An[k][:, j] for k in An]).mean(0) for j in range(30) if j != 27]
    ka = sum((m27 - c) / (m27 - c).norm() for c in comp); ka = ka / ka.norm()
    anc = torch.cat([An[k][:, 27] @ ka for k in An]).mean()
    oth = []
    for k in An:
        s = An[k][:, :30] @ ka; s[:, 27] = -1e9; oth.append(s.max(1).values)
    oth = torch.cat(oth).mean()
    W["k_anc"] = ka; W["tau_anc"] = oth + anc_frac * (anc - oth)
    # output directions in raw residual units: next-token class means at answer positions
    eos = int(d["two"]["ids"][0, -1]); W["eos"] = eos
    Hq, nxt = [], []
    for case, qs in (("two", [27, 28, 29]), ("three", [27, 28, 29, 30])):
        for q in qs:
            Hq.append(A[case][:, q]); nxt.append(ids[case][:, q + 1])
    Hq = torch.cat(Hq); nxt = torch.cat(nxt)
    mq = Hq.mean(0)
    toks = list(range(15, 25)) + [eos]
    W["out_dir"] = {t: Hq[nxt == t].mean(0) - mq for t in toks if (nxt == t).any()}
    # unembedding-based directions (model's own matrices): (1+w_finalnorm) * (W_U[t] - mean_c W_U[c])
    if model is not None:
        WU = model.lm_head.weight.detach().float()[toks]
        gam = 1.0 + model.model.norm.weight.detach().float()
        ref = torch.stack(list(W["out_dir"].values())).norm(dim=1).mean()
        W["out_dir_u"] = {}
        for i, t in enumerate(toks):
            v = gam * (WU[i] - WU.mean(0))
            W["out_dir_u"][t] = v / v.norm() * ref
    # step-specific output directions: within answer step k, mean(next = tok) - mean(step k)
    W["out_dir_k"] = {}
    for k in range(4):
        hk, nk = [], []
        for case, qs in (("two", [27, 28, 29]), ("three", [27, 28, 29, 30])):
            if k < len(qs):
                hk.append(A[case][:, qs[k]]); nk.append(ids[case][:, qs[k] + 1])
        hk = torch.cat(hk); nk = torch.cat(nk)
        for t in toks:
            if (nk == t).sum() >= 3 and len(set(nk.tolist())) > 1:
                W["out_dir_k"][(k, t)] = hk[nk == t].mean(0) - hk.mean(0)
    return W


class Graft(nn.Module):
    def __init__(self, W, record=False):
        super().__init__()
        self.W = W; self.record = record; self.last = None

    def forward(self, h):
        W = self.W
        B, T, Dm = h.shape
        x = _n(h)
        # ---- attention (constant query, null slot) ----
        sc = W["beta"] * torch.einsum("btd,rd->brt", x, W["keys"])          # B x R x T(keys)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))              # q x j
        sc = sc[:, :, None, :].expand(B, 4, T, T).masked_fill(~causal, -1e30)  # B R q j
        null = (W["beta"] * W["taus"])[None, :, None, None].expand(B, 4, T, 1)
        att = torch.softmax(torch.cat([sc, null], -1), -1)[..., :T]          # drop null (value 0)
        mu = W["mu_dig"]
        vd = x @ mu.T - 0.5 * (mu * mu).sum(1)                              # B T 10 digit scores
        v = torch.cat([vd, torch.ones(B, T, 1)], -1)                        # B T 11
        o = torch.einsum("brqj,bjc->bqrc", att, v)                          # B q R 11
        dig, found = o[..., :10], o[..., 10]
        # ---- MLP layer 1: margin penalties ----
        m = W["m_dig"]
        pen_d = torch.relu(dig[..., None, :] - dig[..., :, None] + m)       # [..,d,e] = relu(s_e - s_d + m)
        pen_d = pen_d.sum(-1) - m                                           # remove e==d term (relu(m)=m)
        ms = W["mu_sub"]
        ss = x @ ms.T - 0.5 * (ms * ms).sum(1)                              # B q nsub
        mst = W["m_step"]
        # penalise subclass i only by competitors j belonging to a different step k (or 'other')
        diffk = (W["sub_k"][:, None] != W["sub_k"][None, :]).float()        # i x j
        pen_s = (torch.relu(ss[..., None, :] - ss[..., :, None] + mst) * diffk).sum(-1)
        pen_f = torch.relu(0.9 - found).sum(-1, keepdim=True)               # B q 1
        # ---- MLP layer 2: indicators ----
        K = 1.0 / m
        onehot = torch.relu(1 - K * pen_d - 10 * pen_f[..., None])          # B q R 10
        sub_on = torch.relu(1 - (1.0 / mst) * pen_s - 10 * pen_f)           # B q nsub
        step = torch.stack([sub_on[..., W["sub_k"] == k].sum(-1) for k in range(4)], -1)  # B q 4
        if W["step_mode"] == "rope":
            # 4 relative-position heads (RoPE form, two frequency pairs): head k attends to the
            # anchor iff it sits exactly k tokens back; null slot otherwise. found_k -> step_k.
            a = x @ W["k_anc"]                                               # B T anchor score
            qi = torch.arange(T)[:, None]; ji = torch.arange(T)[None, :]
            fk = []
            for k in range(4):
                rel = (qi - ji - k).float()
                cosm = 0.5 * (torch.cos(2 * torch.pi * rel / 6) + torch.cos(2 * torch.pi * rel / 7))
                sca = W["beta"] * a[:, None, :] * cosm[None]                 # B q j
                sca = sca.masked_fill(~causal[None], -1e30)
                nul = (W["beta"] * W["tau_anc"]).expand(B, T, 1)
                fk.append(torch.softmax(torch.cat([sca, nul], -1), -1)[..., :T].sum(-1))
            fk = torch.stack(fk, -1)                                         # B q 4
            step = torch.relu(1 - 10 * torch.relu(0.9 - fk) - 10 * pen_f)
        vals = torch.arange(10.0)
        place = torch.tensor([PLACE[r] for r in ROLES])
        S = (onehot * vals).sum(-1) @ place                                  # B q
        # ---- MLP layer 3: trapezoids on S, gated by step ----
        outv = torch.zeros_like(h)
        G = W["G"]
        # gated (GLU-style) units: up = trapezoid on S, gate = step indicator (exactly 0 when off)
        clean = onehot.sum((-1, -2))                                         # = 4 iff all four one-hots clean
        for k in range(4):
            gk = step[..., k]
            if W.get("failsafe"):
                # gate unit relu(step_k - 10*(4 - sum onehots)): closed unless S is fully determined
                gk = torch.relu(gk - 10 * (4 - clean))
            for (lo, hi, tok) in intervals(k, W["eos"]):
                if tok not in W["out_dir"]:
                    continue
                bump = (torch.relu(S - lo + 1) - torch.relu(S - lo) - torch.relu(S - hi) + torch.relu(S - hi - 1))
                dvec = W["out_dir"][tok]
                if W.get("out_mode") == "step" and (k, tok) in W["out_dir_k"]:
                    dvec = W["out_dir_k"][(k, tok)]
                if W.get("out_mode") == "unembed":
                    dvec = W["out_dir_u"][tok]
                if W.get("out_mode") == "mix":
                    dvec = 0.5 * (W["out_dir"][tok] + W["out_dir_u"][tok])
                outv = outv + W["alpha"] * (gk * bump)[..., None] * dvec
        if self.record:
            self.last = dict(S=S, step=step, found=found, onehot=onehot)
        return h + outv


_IV = {}


def intervals(k, eos):
    if (k, eos) not in _IV:
        iv = []
        cur = None
        for S in range(20, 199):
            t = target_token(k, S, eos)
            if cur and cur[2] == t and cur[1] == S - 1:
                cur[1] = S
            else:
                if cur: iv.append(tuple(cur))
                cur = [S, S, t]
        iv.append(tuple(cur))
        _IV[(k, eos)] = iv
    return _IV[(k, eos)]


def install(model, graft):
    """Insert graft between decoder layer L-1 and L (forward hook on layer L-1's output)."""
    lay = model.model.layers[graft.W["L"] - 1]

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (graft(out[0]),) + tuple(out[1:])
        return graft(out)
    return lay.register_forward_hook(hook)


def build_multi(data, L, recs=None, model=None, tau_frac=0.3, anc_frac=0.3, beta=300.0, m_dig=0.005,
                alpha=6.0, failsafe=True, out_mode="unembed"):
    """v5 builder: same block, but every mean is taken over calibration prompts in several build-only
    wordings, with role/anchor positions located per prompt (first four digit tokens; last prompt token)."""
    recs = recs if recs is not None else data["recs"]
    W = dict(L=L, failsafe=failsafe, out_mode=out_mode, step_mode="rope", beta=beta, m_dig=m_dig,
             m_step=0.005, alpha=alpha, G=1000.0)
    X = [_n(r["h"][L]) for r in recs]
    D = X[0].shape[-1]
    mr = {r: torch.stack([x[c["roles"][i]] for x, c in zip(X, recs)]).mean(0) for i, r in enumerate(ROLES)}
    nonrole = torch.cat([x[[j for j in range(c["P"] + 1) if j not in c["roles"]]] for x, c in zip(X, recs)])
    mN = nonrole.mean(0)
    keys, taus = [], []
    for i, r in enumerate(ROLES):
        k = mr[r] - torch.stack([mr[s] for s in ROLES if s != r] + [mN]).mean(0); k = k / k.norm()
        rm = torch.stack([x[c["roles"][i]] @ k for x, c in zip(X, recs)]).mean()
        om = []
        for x, c in zip(X, recs):
            s = x[:-1] @ k; s[c["roles"][i]] = -1e9; om.append(s.max())
        om = torch.stack(om).mean()
        keys.append(k); taus.append(om + tau_frac * (rm - om))
    W["keys"] = torch.stack(keys); W["taus"] = torch.stack(taus)
    H = torch.cat([x[c["roles"]] for x, c in zip(X, recs)])
    Dg = torch.cat([torch.tensor(c["ids"])[c["roles"]] - 15 for c in recs])
    W["mu_dig"] = torch.stack([H[Dg == v].mean(0) for v in range(10)])
    W["mu_sub"] = torch.zeros(2, D); W["sub_k"] = torch.tensor([-1, -1])  # unused in rope mode
    # anchor key: last prompt token vs per-(wording, offset-from-anchor) competitor means
    m_anc = torch.stack([x[c["P"]] for x, c in zip(X, recs)]).mean(0)
    comp = {}
    for x, c in zip(X, recs):
        for j in range(x.shape[0] - 1):
            if j != c["P"]:
                comp.setdefault((c["w"], j - c["P"]), []).append(x[j])
    ka = sum((m_anc - torch.stack(v).mean(0)) / (m_anc - torch.stack(v).mean(0)).norm() for v in comp.values())
    ka = ka / ka.norm()
    am = torch.stack([x[c["P"]] @ ka for x, c in zip(X, recs)]).mean()
    om = []
    for x, c in zip(X, recs):
        s = x[:-1] @ ka; s[c["P"]] = -1e9; om.append(s.max())
    om = torch.stack(om).mean()
    W["k_anc"] = ka; W["tau_anc"] = om + anc_frac * (am - om)
    eos = recs[0]["ids"][-1]; W["eos"] = eos
    toks = list(range(15, 25)) + [eos]
    Hq, nxt = [], []
    for c in recs:
        h = c["h"][L].float()
        for q in range(c["P"], len(c["ids"]) - 1):
            Hq.append(h[q]); nxt.append(c["ids"][q + 1])
    Hq = torch.stack(Hq); nxt = torch.tensor(nxt); mq = Hq.mean(0)
    W["out_dir"] = {t: Hq[nxt == t].mean(0) - mq for t in toks if (nxt == t).any()}
    W["out_dir_k"] = {}
    if model is not None:
        WU = model.lm_head.weight.detach().float()[toks]
        gam = 1.0 + model.model.norm.weight.detach().float()
        ref = torch.stack(list(W["out_dir"].values())).norm(dim=1).mean()
        W["out_dir_u"] = {t: (lambda v: v / v.norm() * ref)(gam * (WU[i] - WU.mean(0))) for i, t in enumerate(toks)}
    return W
