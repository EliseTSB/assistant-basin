"""Tests for Part 110's weights-only comparison, on tiny fixture snapshots. Claude, 2026-10-29."""
import json, sys
from pathlib import Path
import numpy as np, pytest, torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weights110 as p110

N, H, V, PRE = 40, 16, 64, "model.language_model."


def snapshot(root: Path, tensors: dict):
    root.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(root / "shard.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "shard.safetensors" for k in tensors}}))
    (root / "config.json").write_text(json.dumps({"text_config": {"rms_norm_eps": 1e-6}}))


def make_pair(tmp_path, chat_equals_base=False):
    g = torch.Generator().manual_seed(110)
    base, chat = {}, {}
    for layer in (1, 2):
        for m, shape in (("gate", (N, H)), ("up", (N, H)), ("down", (H, N))):
            w = torch.randn(shape, generator=g)
            base[f"{PRE}layers.{layer}.mlp.{m}_proj.weight"] = w
            chat[f"{PRE}layers.{layer}.mlp.{m}_proj.weight"] = w.clone() if chat_equals_base \
                else w + 0.05 * torch.randn(shape, generator=g)
    if not chat_equals_base:  # plant the biggest move on L1 N3's output column
        chat[f"{PRE}layers.1.mlp.down_proj.weight"][:, 3] = torch.randn(H, generator=g)
    emb = torch.randn(V, H, generator=g)
    nw = 3.0 * torch.randn(H, generator=g)  # a large final-norm scale, so omitting (1 + w) changes the readout
    base[PRE + "embed_tokens.weight"], chat[PRE + "embed_tokens.weight"] = emb, emb.clone()
    base[PRE + "norm.weight"], chat[PRE + "norm.weight"] = nw, nw.clone()
    snapshot(tmp_path / "base", base)
    snapshot(tmp_path / "chat", chat)
    return base, chat


def manual_readout(t: dict, layer: int, unit: int, sign: float, with_scale=True):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    norm = Qwen3_5RMSNorm(H, eps=1e-6)
    norm.weight.data = t[PRE + "norm.weight"].clone() if with_scale else torch.zeros(H)
    v = sign * t[f"{PRE}layers.{layer}.mlp.down_proj.weight"][:, unit]
    with torch.no_grad():
        return (t[PRE + "embed_tokens.weight"] @ norm(v)).double().numpy()


class Tok:
    def decode(self, ids):
        return f"t{ids[0]}"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def go(chat_equals_base=False):
        base, chat = make_pair(tmp_path, chat_equals_base)
        monkeypatch.setattr(p110, "BASE_DIR", tmp_path / "base")
        monkeypatch.setattr(p110, "CHAT_DIR", tmp_path / "chat")
        monkeypatch.setattr(p110, "MAKERS", {"L1 N3": (1, 3), "L2 N5": (2, 5)})
        monkeypatch.setattr(p110, "SECONDARY", {"L2 N7": (2, 7)})
        monkeypatch.setattr(p110, "GOOGLE_IDS", [4, 9])
        monkeypatch.setattr(p110, "N_DRAWS", 200)
        monkeypatch.setattr(p110, "MATCH_K", 10)
        monkeypatch.setattr(p110, "load_tok", lambda: Tok())
        named = {}
        for name, (l, u, sign) in {"L1 N3": (1, 3, -1.0), "L2 N5": (2, 5, 1.0)}.items():
            top = np.argsort(-manual_readout(chat, l, u, sign), kind="stable")[:20]
            named[name] = {"layer": l, "unit": u, "sign": sign, "top20": [f"t{int(i)}" for i in top]}
        (tmp_path / "p71.json").write_text(json.dumps({"named": named}))
        monkeypatch.setattr(p110, "PART71", tmp_path / "p71.json")
        p110.main(tmp_path / "out")
        return base, chat, json.loads((tmp_path / "out" / "weights110.json").read_text())
    return go


def test_cosines_percentiles_and_set_test(setup):
    base, chat, r = setup()
    key = f"{PRE}layers.1.mlp.down_proj.weight"
    manual = float(torch.nn.functional.cosine_similarity(base[key][:, 3], chat[key][:, 3], dim=0))
    n = r["neurons"]["L1 N3"]
    assert n["cos"]["down"] == pytest.approx(manual, abs=1e-6)
    assert n["pct_in_layer"]["down"] == 100.0 and n["pct_in_layer"]["mean"] == 100.0
    assert r["self_control_min_cos"] == pytest.approx(1.0, abs=1e-6)
    assert r["set_test"]["uniform"]["p_more"] < 0.05  # the planted mover drags the set mean above every draw
    assert r["set_test"]["matched"]["pool_size"] == [10, 10]


def test_readout_matches_qwen35_norm_and_needs_the_scale(setup):
    base, chat, r = setup()
    ro = r["readouts"]["L1 N3"]
    want = manual_readout(chat, 1, 3, -1.0)
    assert ro["chat"]["top20"] == [f"t{int(i)}" for i in np.argsort(-want, kind="stable")[:20]]
    assert ro["chat"]["google_rank"] == p110.best_rank(want, [4, 9])
    assert ro["chat_vs_part71_top20_shared"] == 20
    unscaled = manual_readout(chat, 1, 3, -1.0, with_scale=False)
    assert list(np.argsort(-unscaled)[:5]) != list(np.argsort(-want)[:5])  # the fixture can tell a missing (1 + w)
    assert r["verdicts"]["readout_gate"] == "passed"


def test_identical_checkpoints(setup):
    _, _, r = setup(chat_equals_base=True)
    assert all(v == pytest.approx(1.0, abs=1e-6) for L in r["layers"].values() for v in L["median_cos"].values())
    u = r["set_test"]["uniform"]
    assert abs(u["observed_mean_move"]) < 1e-12 and max(abs(x) for x in u["draw_q05_q95"]) < 1e-12
    assert all(x["base_chat_top20_shared"] == 20 for x in r["readouts"].values())
    assert r["verdicts"]["110.1"] == "held" and r["verdicts"]["110.5"] == "held"


def test_verdict_cascade():
    def res(median, shared71, shared_bc=20, rank=1, p_more=0.5, p_less=0.5):
        return {"layers": {"6": {"median_cos": {"gate": median, "up": median, "down": median}}},
                "readouts": {"L27 N3325": {"chat_vs_part71_top20_shared": shared71, "base_chat_top20_shared": shared_bc,
                                           "base": {"google_rank": rank}}},
                "set_test": {"uniform": {"p_more": p_more, "p_less": p_less},
                             "matched": {"p_more": p_more, "p_less": p_less}}}
    v = p110.verdicts(res(0.5, 20))
    assert v["110.1"] == "failed" and {v[k] for k in ("110.2", "110.3", "110.4", "110.5")} == {"NOT READ"}
    v = p110.verdicts(res(0.99, 17))
    assert v["readout_gate"] == "failed" and v["110.4"] == v["110.5"] == "NOT READ" and v["110.2"] == "held"
    v = p110.verdicts(res(0.99, 18, shared_bc=9, rank=11, p_more=0.01, p_less=0.99))
    assert (v["110.2"], v["110.3"], v["110.4"], v["110.5"]) == ("failed", "failed", "failed", "failed")
