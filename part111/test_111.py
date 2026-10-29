"""Tests for Part 111's J-lens readouts and null rates, on a tiny fixture. Claude, 2026-10-29."""
import json, sys
from pathlib import Path
import numpy as np, pytest, torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jlens111 as p111

H, N, V, PRE = 8, 30, 40, "model.language_model."
WORDS = {4: " China", 9: " Google", 11: " AI", 17: "中国", 23: "medicina"}


class Tok:
    def decode(self, ids, **_):
        return WORDS.get(int(ids[0]), f"t{int(ids[0])}")


@pytest.fixture
def run(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(111)
    t = {f"{PRE}layers.{l}.mlp.down_proj.weight": torch.randn(H, N, generator=g) for l in (1, 2)}
    t[PRE + "embed_tokens.weight"] = torch.randn(V, H, generator=g)
    t[PRE + "norm.weight"] = 2.0 * torch.randn(H, generator=g)
    snap = tmp_path / "snap"; snap.mkdir()
    save_file(t, str(snap / "s.safetensors"))
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "s.safetensors" for k in t}}))
    (snap / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 4}}))
    J = {1: torch.randn(H, H, generator=g), 2: torch.randn(H, H, generator=g)}
    torch.save({"J": J, "n_prompts": 7, "source_layers": [1, 2], "d_model": H}, str(tmp_path / "lens.pt"))
    (tmp_path / "p71.json").write_text(json.dumps({"named": {
        "A": {"layer": 1, "unit": 3, "sign": -1.0}, "B": {"layer": 2, "unit": 5, "sign": 1.0}}}))
    for k, v in {"SNAP": snap, "LENS": tmp_path / "lens.pt", "PART71": tmp_path / "p71.json", "N_RANDOM": 10,
                 "TOP": 5, "CHUNK": 7, "load_tok": lambda: Tok(),
                 "PREDICTED": {"111.1": ("A", "china"), "111.2": ("B", "google")}}.items():
        monkeypatch.setattr(p111, k, v)
    p111.main(tmp_path / "out")
    return t, J, json.loads((tmp_path / "out" / "jlens111.json").read_text())


def manual(t, J, layer, unit, sign, use_j=True):
    w = sign * t[f"{PRE}layers.{layer}.mlp.down_proj.weight"][:, unit]
    v = (J[layer] @ w) if use_j else w
    return (t[PRE + "embed_tokens.weight"] @ ((1 + t[PRE + "norm.weight"]) * v)).numpy()


def test_readouts_match_the_formula_with_sign(run):
    t, J, r = run
    for name, (layer, unit, sign) in {"A": (1, 3, -1.0), "B": (2, 5, 1.0)}.items():
        for lens_name, use_j in (("j_lens", True), ("logit_lens", False)):
            want = [Tok().decode([i]) for i in np.argsort(-manual(t, J, layer, unit, sign, use_j), kind="stable")[:5]]
            assert r["readouts"][name][lens_name]["top20"] == want, (name, lens_name)


def test_matches_the_jlens_repo_readout(run):
    code = Path.home() / "jlens-persona-neurons/code"
    if not (code / "common.py").exists():
        pytest.skip("J-lens repo not present")
    sys.path.insert(0, str(code))
    import common
    t, J, r = run
    w = -t[f"{PRE}layers.1.mlp.down_proj.weight"][:, 3]
    logits = common.readout_logits(torch, t[PRE + "embed_tokens.weight"], w, 1 + t[PRE + "norm.weight"], J[1])
    want = [Tok().decode([int(i)]) for i in torch.topk(logits, 5).indices]
    assert r["readouts"]["A"]["j_lens"]["top20"] == want


def test_null_rates_count_both_signs(run):
    t, J, r = run
    rng = np.random.default_rng(p111.SEED_BASE + 1)
    pool = [u for u in range(N) if u != 3]
    rand = rng.choice(pool, size=10, replace=False)
    hits = 0
    for u in rand:
        for sign in (1.0, -1.0):
            toks = [Tok().decode([i]) for i in np.argsort(-manual(t, J, 1, int(u), sign), kind="stable")[:5]]
            hits += bool(p111.family_hits(toks)["china"])
    assert r["null_rates"]["1"]["j_lens"]["china"] == pytest.approx(hits / 20)


def test_family_patterns():
    fh = p111.family_hits([" China", "中国", " AI", " Gemini", "medicina", " PAID", "t1"])
    assert fh["china"] == [" China", "中国"] and fh["ai"] == [" AI"] and fh["google"] == [" Gemini"]


def test_verdict_fields(run):
    _, _, r = run
    v = r["verdicts"]["111.1"]
    assert v["status"] in ("held", "failed") and v["neuron"] == "A" and v["specific"] == (v["null_rate"] < 0.10)
