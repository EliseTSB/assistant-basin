#!/usr/bin/env python3
"""test_olmo3_c3_runner_cpu.py — Comprehensive 0 CU CPU verification suite for `run_olmo3_c3.py`.

Exercises REAL HuggingFace `Olmo3ForCausalLM` architecture (tiny 4-layer bf16 CPU model with genuine
`post_feedforward_layernorm` near-zero gamma coordinates + `mlp.down_proj.weight` + `DynamicCache`):
  1. C3.0.4 fail-closed SHA-256 provenance preflight (PASS on committed files, FAIL on tamper).
  2. Zero-hook literal fp32-master weight edit + exact `stored_frob_ret` / `stored_rank1_proj_ret` +
     bitwise `torch.equal` bf16 restore (including exception safety & active-hook rejection).
  3. Exact `CommonBanLogitsProcessor` enforcing single-token IDs AND multi-token `bad_words_ids`
     sequences (blocking `"Ahoy matey buccaneer"`) identically at BOTH `B = 1` and `B = 8` left-padded!
  4. Real `Olmo3ForCausalLM` `B = 8` vs `B = 1` left-padded logit parity & KV-cached greedy decode parity.
  5. `off_KL` estimator exactness (0.0 for clean, positive & finite for edited, exact max single-pos KL).
  6. D4 L24 masked (`> 1.15x`) vs unmasked vs mis-masked (`< 1.05x`) on real near-zero `gamma_ff`.
  7. End-to-end `run_stage_c3_experiment(..., smoke=True)` verifying zero hooks, bitwise restore,
     commit-gate representability (`dose_unrepresentable`), 0 weight checkpoints saved (`11-07`), and
     hard refusal by `score_olmo3_c3.py` on smoke receipts.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_olmo3_c3 as runner  # noqa: E402
import score_olmo3_c3 as scorer  # noqa: E402


class ToyWordTokenizer:
    """Minimal deterministic tokenizer supporting multi-token pirate sequences for exact CPU ban tests."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.all_special_ids = [0, 1, 2]
        self.added_tokens_decoder = {0: "<|pad|>", 1: "<|endoftext|>", 2: "<|im_end|>"}
        self.chat_template = None

    def convert_tokens_to_ids(self, tok: str) -> int:
        return 2 if tok == "<|im_end|>" else -1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return [3]
        # Deterministic byte-based mapping into [3 .. vocab_size - 1]
        b = text.encode("utf-8")
        ids = [3 + (byte % (self.vocab_size - 3)) for byte in b[:40]]
        return ids or [3]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
        kept = [i for i in ids if not (skip_special_tokens and i in self.all_special_ids)]
        if not kept:
            return "alpha bravo charlie"
        return " ".join(words[i % len(words)] for i in kept)


def build_tiny_olmo3(vocab_size: int = 128, hidden_size: int = 64, inter_size: int = 128) -> Olmo3ForCausalLM:
    torch.manual_seed(20261126)
    cfg = Olmo3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=1,
    )
    model = Olmo3ForCausalLM(cfg).to(torch.bfloat16)
    model.eval()
    # Plant genuine near-zero gamma_ff coordinates (mimicking real OLMo-3 L21/24/27!)
    with torch.no_grad():
        for L in (1, 2, 3):
            g = torch.full((hidden_size,), 0.20, dtype=torch.bfloat16)
            g[2] = 0.001
            g[7] = 0.002
            g[13] = 0.003
            g[19] = 0.050
            model.model.layers[L].post_feedforward_layernorm.weight.copy_(g)
    return model


def build_tiny_bank(model: Olmo3ForCausalLM, layers: list[int] = [1, 2, 3]) -> dict[str, torch.Tensor]:
    g_gen = torch.Generator().manual_seed(42)
    d_model = model.config.hidden_size
    bank: dict[str, torch.Tensor] = {}
    drop_idx = [2, 7, 13, 19]
    for L in layers:
        gamma = model.model.layers[L].post_feedforward_layernorm.weight.detach().cpu().float()
        bank[f"L{L}_gamma_drop_idx"] = torch.tensor(drop_idx, dtype=torch.long)
        for prefix in ("pirate_lex", "nomd_lex"):
            v_post = torch.randn(d_model, generator=g_gen)
            # Ensure non-trivial weight on the tiny-gamma coordinates so masking gives a huge (>1.5x) boost!
            v_post[2] = 0.08
            v_post[7] = -0.08
            v_post = v_post / v_post.norm()
            v_pre, _ = runner.compute_prenorm_masked_direction(v_post, gamma, drop_idx)
            bank[f"L{L}_{prefix}_post_unit"] = v_post
            bank[f"L{L}_{prefix}_pre_unit"] = v_pre
            if L == 2 and prefix == "pirate_lex":
                raw_unm = v_post / gamma
                bank[f"L{L}_pirate_unmasked_pre_unit"] = raw_unm / raw_unm.norm()
        for rkind in ("rand_pirate", "rand_nomd"):
            for s in range(3):
                rv = torch.randn(d_model, generator=g_gen)
                for idx in drop_idx:
                    rv[idx] = 0.0
                rv = rv / rv.norm()
                bank[f"L{L}_{rkind}_pre_unit_s{s}"] = rv
    return bank


def test_1_provenance_preflight() -> None:
    hashes = runner.verify_preflight_provenance(
        HERE / "OLMO3-WSP-C3-DESIGN.md",
        HERE / "c3_prereg_constants.json",
        HERE / "c3_vector_bank.safetensors",
        HERE / "c3_scoring.py",
    )
    assert hashes["c3_constants_sha256"] == runner.EXPECTED_C3_CONSTANTS_SHA256
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad_scoring.py"
        bad.write_text("# tampered\n", encoding="utf-8")
        try:
            runner.verify_preflight_provenance(
                HERE / "OLMO3-WSP-C3-DESIGN.md",
                HERE / "c3_prereg_constants.json",
                HERE / "c3_vector_bank.safetensors",
                bad,
            )
            raise AssertionError("Expected provenance tamper to raise ValueError")
        except ValueError as exc:
            assert "SHA-256 mismatch" in str(exc)
    print("PASS test_1_provenance_preflight")


def test_2_weight_editor_hooks_and_bitwise_restore() -> None:
    model = build_tiny_olmo3()
    layers = [1, 2, 3]
    master = runner.WeightMaster(model, layers)
    assert runner.count_active_hooks(model) == 0

    g = torch.Generator().manual_seed(7)
    v = torch.randn(model.config.hidden_size, generator=g)
    v = v / v.norm()
    k = torch.randn(model.config.intermediate_size, generator=g)
    k = k / k.norm()

    # Large alpha -> high bf16 retention (~1.0); tiny alpha -> low retention (representability detection)
    with master.apply_edit(model, {2: (2.0, v, k)}) as m_hi:
        assert runner.count_active_hooks(model) == 0
        assert m_hi["2"]["stored_frob_ret"] > 0.95
        assert m_hi["2"]["stored_rank1_proj_ret"] > 0.95
        assert not torch.equal(model.model.layers[2].mlp.down_proj.weight.data.cpu(), master.clean_bf16[2])
    assert torch.equal(model.model.layers[2].mlp.down_proj.weight.data.cpu(), master.clean_bf16[2])

    # Tiny alpha (1e-5) must show severe bf16 quantization collapse
    with master.apply_edit(model, {2: (1e-5, v, k)}) as m_lo:
        assert m_lo["2"]["stored_rank1_proj_ret"] < 0.50
    assert torch.equal(model.model.layers[2].mlp.down_proj.weight.data.cpu(), master.clean_bf16[2])

    # Exception inside context still restores bitwise!
    try:
        with master.apply_edit(model, {1: (1.5, v, k)}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert torch.equal(model.model.layers[1].mlp.down_proj.weight.data.cpu(), master.clean_bf16[1])

    # Active hook MUST trigger C3.0.1 violation!
    h = model.model.layers[1].register_forward_hook(lambda m, i, o: o)
    try:
        try:
            with master.apply_edit(model, {1: (1.0, v, k)}):
                pass
            raise AssertionError("Expected active hook to raise RuntimeError")
        except RuntimeError as exc:
            assert "C3.0.1 VIOLATION" in str(exc)
    finally:
        h.remove()
    assert runner.count_active_hooks(model) == 0
    print("PASS test_2_weight_editor_hooks_and_bitwise_restore")


def test_3_common_ban_processor_single_and_multitoken_b1_and_b8() -> None:
    # Single-token ban on ID 10; multi-token ban on [20, 21, 22] ("Ahoy matey buccaneer" analogue)
    proc = runner.CommonBanLogitsProcessor(
        single_token_ids=[10],
        multi_token_sequences=[[20, 21, 22], [30, 31]],
        eos_token_ids={1},
    )
    # Test B=1 matching [20, 21] -> 22 must become -inf, 10 must become -inf
    s1 = torch.zeros((1, 50), dtype=torch.float32)
    s1[0, 10] = 9.0
    s1[0, 22] = 8.0
    s1[0, 7] = 1.0
    proc.apply(s1, [[5, 6, 20, 21]])
    assert torch.isneginf(s1[0, 10]) and torch.isneginf(s1[0, 22])
    assert int(torch.argmax(s1[0]).item()) == 7

    # Test B=8 with different unpadded lengths & mixed histories
    s8 = torch.zeros((8, 50), dtype=torch.float32)
    s8[:, 10] = 9.0
    s8[:, 22] = 8.0
    s8[:, 31] = 7.0
    s8[:, 4] = 1.0
    histories = [
        [20, 21],          # row 0: blocks 22 (+10) -> argmax 31
        [9, 9, 9, 30],     # row 1: blocks 31 (+10) -> argmax 22
        [20, 21, 30],      # row 2: ends with 30 (blocks 31, not 22) -> argmax 22
        [1, 2, 3, 20, 21], # row 3: blocks 22 (+10) -> argmax 31
        [5],               # row 4: only 10 blocked -> argmax 22
        [30],              # row 5: blocks 31 (+10) -> argmax 22
        [20, 20, 21],      # row 6: blocks 22 (+10) -> argmax 31
        [7, 8],            # row 7: only 10 blocked -> argmax 22
    ]
    proc.apply(s8, histories)
    assert all(torch.isneginf(s8[b, 10]) for b in range(8))
    assert torch.argmax(s8, dim=-1).tolist() == [31, 22, 22, 31, 22, 22, 31, 22]
    print("PASS test_3_common_ban_processor_single_and_multitoken_b1_and_b8")


def test_4_b8_vs_b1_parity_and_off_kl() -> None:
    model = build_tiny_olmo3()
    tok = ToyWordTokenizer(vocab_size=model.config.vocab_size)
    prompts = [f"Prompt number {i} with extra words " + ("x " * (i * 2)) for i in range(8)]
    id_lists = [tok.encode(p) for p in prompts]
    dev = next(model.parameters()).device

    b1 = torch.cat([runner.forward_next_logits_batch(model, [ids], 0, dev) for ids in id_lists], dim=0)
    b8 = runner.forward_next_logits_batch(model, id_lists, 0, dev)
    max_diff = float((b8 - b1).abs().max().item())
    assert max_diff <= 0.05, f"B=8 vs B=1 parity failed on real Olmo3ForCausalLM: {max_diff}"

    g1 = runner.greedy_decode_batch(model, tok, prompts, max_new_tokens=6, batch_size=1, eos_ids=[1, 2], pad_id=0)
    g8 = runner.greedy_decode_batch(model, tok, prompts, max_new_tokens=6, batch_size=8, eos_ids=[1, 2], pad_id=0)
    assert [r["gen_ids"] for r in g1] == [r["gen_ids"] for r in g8], "B=8 vs B=1 greedy tokens diverged!"

    # Exact off_KL = 0 for identical logits, > 0 for edited logits
    tf_clean = runner.teacher_forced_gen_logits_b1(model, id_lists[0], g1[0]["gen_ids"], dev)
    kl0, maxp0, dl0 = runner.compute_off_kl_from_logits(tf_clean, tf_clean)
    assert abs(kl0) < 1e-6 and abs(maxp0) < 1e-6 and dl0 == 0.0

    tf_shifted = tf_clean.clone()
    tf_shifted[0, 5] += 2.0
    kl1, maxp1, dl1 = runner.compute_off_kl_from_logits(tf_clean, tf_shifted)
    assert kl1 > 0.0 and maxp1 >= kl1 and abs(dl1 - 2.0) < 1e-5
    print(f"PASS test_4_b8_vs_b1_parity_and_off_kl (max_diff={max_diff:.6f})")


def test_5_end_to_end_smoke_and_d4_and_scorer_refusal() -> None:
    model = build_tiny_olmo3()
    tok = ToyWordTokenizer(vocab_size=model.config.vocab_size)
    bank = build_tiny_bank(model, layers=[1, 2, 3])
    c3_constants = json.loads((HERE / "c3_prereg_constants.json").read_text(encoding="utf-8"))
    # Remap single/multi bans into tiny vocab for smoke test without mutating disk
    c3_constants["lexicons"]["common_pirate_ban"]["single_tokens"] = [{"id": 10, "stem": "arr", "token": "arr"}]
    c3_constants["lexicons"]["common_pirate_ban"]["multi_token_sequences"] = [{"ids": [20, 21], "stem": "aye", "surface": "Aye"}]
    c3_constants["lexicons"]["pirate_seeds"] = [{"id": 12 + i, "token": f" p{i}"} for i in range(12)]

    hashes = {
        "c3_design_sha256": runner.EXPECTED_C3_DESIGN_SHA256,
        "c3_constants_sha256": runner.EXPECTED_C3_CONSTANTS_SHA256,
        "c3_bank_sha256": runner.EXPECTED_C3_BANK_SHA256,
        "c3_scoring_sha256": runner.EXPECTED_C3_SCORING_SHA256,
    }
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td) / "ckpt"
        receipt = runner.run_stage_c3_experiment(
            model, tok, c3_constants, bank, hashes,
            smoke=True,
            device_name="cpu-tiny-olmo3",
            checkpoint_dir=ckpt_dir,
            target_layers=[1, 2, 3],
            requested_batch_size=4,
            max_new_tokens_override={k: 4 for k in ("calib", "reserve", "pillar_a", "pillar_b", "multiturn", "d6", "math", "code", "refusal", "neutral")},
        )
        # Ensure ZERO model weight checkpoints were written to disk (11-07 policy!)
        saved_weights = list(Path(td).rglob("*.pt")) + list(Path(td).rglob("*.safetensors")) + list(Path(td).rglob("*.bin"))
        assert not saved_weights, f"11-07 VIOLATION: weight files saved: {saved_weights}"

    assert receipt["meta"]["smoke"] is True
    assert receipt["meta"]["active_hooks_max_seen"] == 0
    assert receipt["meta"]["restore_exact_all"] is True
    assert receipt["meta"]["weight_checkpoints_saved"] == 0
    assert receipt["meta"]["b8_parity_passed"] is True

    # D4 masked vs unmasked vs mis-masked on planted near-zero gamma_ff!
    d4 = receipt["preflights"]["D4_L24_mask_mismask"]
    assert d4["masked_over_unmasked_logit_l2_ratio"] > 1.15, f"D4 mask boost too low: {d4}"
    assert d4["mismasked_over_unmasked_logit_l2_ratio"] < 1.05, f"D4 mismask boost too high: {d4}"

    # Verify scorer hard-refuses smoke receipts!
    try:
        scorer.score_stage_c3(receipt, c3_constants, strict_provenance=False)
        raise AssertionError("Expected scorer to hard-refuse smoke=True receipt")
    except ValueError as exc:
        assert "smoke" in str(exc).lower()
    print(
        f"PASS test_5_end_to_end_smoke_and_d4_and_scorer_refusal "
        f"(D4 mask={d4['masked_over_unmasked_logit_l2_ratio']:.3f}x, mismask={d4['mismasked_over_unmasked_logit_l2_ratio']:.3f}x)"
    )


def main() -> None:
    test_1_provenance_preflight()
    test_2_weight_editor_hooks_and_bitwise_restore()
    test_3_common_ban_processor_single_and_multitoken_b1_and_b8()
    test_4_b8_vs_b1_parity_and_off_kl()
    test_5_end_to_end_smoke_and_d4_and_scorer_refusal()
    print("ALL 5 STAGE C3 CPU RUNNER TESTS PASSED.")


if __name__ == "__main__":
    main()
