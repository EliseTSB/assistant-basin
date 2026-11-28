#!/usr/bin/env python3
"""test_olmo3_c3b_runner_cpu.py — Comprehensive 0-CU CPU verification suite for `run_olmo3_c3b.py` and `audit_c3b_label.py`.

Exercises REAL HuggingFace `Olmo3ForCausalLM` (tiny 12-layer bf16 CPU model with genuine near-zero
`post_feedforward_layernorm` gamma coordinates + `mlp.down_proj.weight` + `DynamicCache`):
  1. C3b.0.4 fail-closed SHA-256 provenance preflight (PASS on committed files, FAIL on tamper).
  2. Zero-hook `WeightMaster.apply_edit` & `apply_orth_edit` (`WSP_NoMD_12L_kWA`):
     - exact `layers_compiled_total == restore_checks` tracking
     - exact rank-1 orthogonalization (`v_f @ W_edited ~= 0` at `beta = 1.0`)
     - bitwise `torch.equal` bf16 restore, exception safety, active-hook rejection, inner-exception non-masking
  3. `compute_sparse_topk_key` (exact top-32 support, unit norm, `c_bar > 0`, massive bf16 dead-zone escape at `alpha = 0.05`).
  4. `CommonBanLogitsProcessor` single + multi-token enforcement at exact B = 1, `em_dash_stats`, `compute_d_align_matrix`.
  5. End-to-end `run_stage_c3b_experiment(..., smoke=True)` verifying exact B = 1, zero hooks, bitwise restore,
     `restore_checks == layers_compiled_total`, 0 weight checkpoints saved (`11-07`), live contrastive `D4`,
     out-of-sample `D6_heldout` (`RESERVE_PROMPTS_8`), `D_align` 12x12, `D_key_ablation` (`kWA` + `kSparse32`),
     and fail-closed refusal by `score_olmo3_c3b.py` on smoke & partial checkpoint files.
  6. `audit_c3b_label.py`: pre-GPU `C3b.R_preflight` canary (exact label checks), split-reader `--readers ds` /
     `--readers luna` / `--merge-only`, receipt-SHA-256 bound resume, and truncated JSONL recovery.
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_c3b_label as auditor  # noqa: E402
import run_olmo3_c3b as runner  # noqa: E402
import score_olmo3_c3b as scorer  # noqa: E402


class ToyWordTokenizer:
    """Minimal deterministic tokenizer for exact B = 1 CPU runner tests."""

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
        b = text.encode("utf-8")
        ids = [3 + (byte % (self.vocab_size - 3)) for byte in b[:36]]
        return ids or [3]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
        kept = [i for i in ids if not (skip_special_tokens and i in self.all_special_ids)]
        if not kept:
            return "alpha bravo charlie"
        return " ".join(words[i % len(words)] for i in kept)


def build_tiny_olmo3_12l(vocab_size: int = 128, hidden_size: int = 48, inter_size: int = 96) -> Olmo3ForCausalLM:
    torch.manual_seed(20261127)
    cfg = Olmo3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=1,
    )
    model = Olmo3ForCausalLM(cfg).to(torch.bfloat16)
    model.eval()
    with torch.no_grad():
        for L in range(12):
            g = torch.full((hidden_size,), 0.20, dtype=torch.bfloat16)
            g[2] = 0.001
            g[7] = 0.002
            g[13] = 0.003
            g[19] = 0.050
            model.model.layers[L].post_feedforward_layernorm.weight.copy_(g)
    return model


def build_tiny_bank_12l(model: Olmo3ForCausalLM, layers: list[int]) -> dict[str, torch.Tensor]:
    g_gen = torch.Generator().manual_seed(20261127)
    d_model = model.config.hidden_size
    bank: dict[str, torch.Tensor] = {}
    drop_idx = [2, 7, 13, 19]
    for L in layers:
        bank[f"L{L}_gamma_drop_idx"] = torch.tensor(drop_idx, dtype=torch.long)
        for rkind in ("rand_pirate", "rand_nomd"):
            for s in range(3):
                rv = torch.randn(d_model, generator=g_gen)
                for idx in drop_idx:
                    rv[idx] = 0.0
                rv = rv / rv.norm()
                bank[f"L{L}_{rkind}_pre_unit_s{s}"] = rv
    # Include L24 C3 lexicon tensors sized to d_model for 0-fwd continuity check
    v_lex = torch.randn(d_model, generator=g_gen)
    v_lex = v_lex / v_lex.norm()
    bank["L24_pirate_lex_post_unit"] = v_lex
    bank["L24_pirate_lex_pre_unit"] = v_lex
    bank["L24_pirate_unmasked_pre_unit"] = v_lex
    return bank


def test_1_provenance_preflight() -> None:
    hashes = runner.verify_preflight_provenance(
        HERE / "OLMO3-WSP-C3B-DESIGN.md",
        HERE / "c3b_prereg_constants.json",
        HERE / "c3b_vector_bank.safetensors",
        HERE / "c3_scoring.py",
    )
    assert hashes["c3b_design_sha256"] == runner.EXPECTED_C3B_DESIGN_SHA256
    assert hashes["c3b_constants_sha256"] == runner.EXPECTED_C3B_CONSTANTS_SHA256
    assert hashes["c3b_bank_sha256"] == runner.EXPECTED_C3B_BANK_SHA256
    assert hashes["c3_scoring_sha256"] == runner.EXPECTED_C3_SCORING_SHA256

    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad_design.md"
        bad.write_text("# tampered\n", encoding="utf-8")
        try:
            runner.verify_preflight_provenance(
                bad,
                HERE / "c3b_prereg_constants.json",
                HERE / "c3b_vector_bank.safetensors",
                HERE / "c3_scoring.py",
            )
            raise AssertionError("Expected provenance tamper to raise ValueError")
        except ValueError as exc:
            assert "SHA-256 mismatch" in str(exc)
    print("PASS test_1_provenance_preflight")


def test_2_weight_master_edit_and_orth_kwa() -> None:
    model = build_tiny_olmo3_12l()
    layers = list(range(12))
    master = runner.WeightMaster(model, layers)
    assert master.layers_compiled_total == 0 and master.restore_checks == 0

    g = torch.Generator().manual_seed(11)
    v = torch.randn(model.config.hidden_size, generator=g)
    v = v / v.norm()
    k = torch.randn(model.config.intermediate_size, generator=g)
    k = k / k.norm()

    # Standard rank-1 edit
    with master.apply_edit(model, {4: (2.0, v, k), 5: (2.0, v, k)}) as m_hi:
        assert runner.count_active_hooks(model) == 0
        assert m_hi["4"]["stored_rank1_proj_ret"] > 0.95
        assert m_hi["5"]["stored_frob_ret"] > 0.95
        assert not torch.equal(model.model.layers[4].mlp.down_proj.weight.data.cpu(), master.clean_bf16[4])
    assert torch.equal(model.model.layers[4].mlp.down_proj.weight.data.cpu(), master.clean_bf16[4])
    assert torch.equal(model.model.layers[5].mlp.down_proj.weight.data.cpu(), master.clean_bf16[5])
    assert master.layers_compiled_total == 2 and master.restore_checks == 2

    # Rank-1 weight-aligned orthogonalization (WSP_NoMD_12L_kWA): beta=1.0 zeroes v^T W_down!
    with master.apply_orth_edit(model, {6: (1.0, v), 7: (0.5, v)}) as m_orth:
        w6_ed = model.model.layers[6].mlp.down_proj.weight.data.cpu().float()
        w6_clean = master.clean_fp32[6]
        before_norm = float((v @ w6_clean).norm().item())
        after_norm = float((v @ w6_ed).norm().item())
        assert before_norm > 1e-3
        assert after_norm / before_norm < 0.02, f"Orthogonalization residual too high: {after_norm / before_norm}"
        assert m_orth["6"]["orth_residual_ratio"] < 0.02
        assert 0.45 < m_orth["7"]["orth_residual_ratio"] < 0.55
        assert m_orth["6"]["stored_rank1_proj_ret"] > 0.95
    assert torch.equal(model.model.layers[6].mlp.down_proj.weight.data.cpu(), master.clean_bf16[6])
    assert torch.equal(model.model.layers[7].mlp.down_proj.weight.data.cpu(), master.clean_bf16[7])
    assert master.layers_compiled_total == 4 and master.restore_checks == 4

    # Exception inside context + hook added inside context: inner exception unmasked & layers bitwise restored!
    h_inner = None
    try:
        try:
            with master.apply_orth_edit(model, {2: (0.5, v), 3: (0.5, v)}):
                h_inner = model.model.layers[2].register_forward_hook(lambda m, i, o: o)
                raise RuntimeError("inner_err_unmasked")
            raise AssertionError("Expected inner_err_unmasked")
        except RuntimeError as exc:
            assert str(exc) == "inner_err_unmasked"
    finally:
        if h_inner is not None:
            h_inner.remove()
    assert torch.equal(model.model.layers[2].mlp.down_proj.weight.data.cpu(), master.clean_bf16[2])
    assert torch.equal(model.model.layers[3].mlp.down_proj.weight.data.cpu(), master.clean_bf16[3])
    assert master.layers_compiled_total == 6 and master.restore_checks == 6
    assert master.restore_exact_all is True

    # Pre-existing hook must raise C3b.0.1 VIOLATION
    h = model.model.layers[0].register_forward_hook(lambda m, i, o: o)
    try:
        try:
            with master.apply_edit(model, {0: (1.0, v, k)}):
                pass
            raise AssertionError("Expected active hook error")
        except RuntimeError as exc:
            assert "C3b.0.1 VIOLATION" in str(exc)
    finally:
        h.remove()
    print("PASS test_2_weight_master_edit_and_orth_kwa")


def test_3_sparse_topk_key_and_bf16_lift() -> None:
    g = torch.Generator().manual_seed(99)
    d_model, d_ff = 256, 4096
    W_bf16 = (0.04 * torch.randn(d_model, d_ff, generator=g)).to(torch.bfloat16)
    W_fp32 = W_bf16.float()
    v_hat = torch.randn(d_model, generator=g)
    v_hat = v_hat / v_hat.norm()
    m_mat = torch.randn(32, d_ff, generator=g) + 0.3 * torch.randn(1, d_ff, generator=g)

    m_bar = m_mat.mean(dim=0)
    k_dense = m_bar / m_bar.norm()
    k_sp32, st = runner.compute_sparse_topk_key(m_mat, k_support=32)

    assert st["support_k"] == 32
    assert int((k_sp32 != 0).sum().item()) == 32
    assert abs(float(k_sp32.norm().item()) - 1.0) < 1e-5
    assert st["c_bar"] > 0.0

    # Compare bf16 rank-1 projection retention at tiny alpha = 0.02
    alpha = 0.02
    dW_d = (W_fp32 + alpha * torch.outer(v_hat, k_dense)).to(torch.bfloat16).float() - W_fp32
    dW_s = (W_fp32 + alpha * torch.outer(v_hat, k_sp32)).to(torch.bfloat16).float() - W_fp32
    ret_dense = float(torch.dot(v_hat, dW_d @ k_dense).item() / alpha)
    ret_sparse = float(torch.dot(v_hat, dW_s @ k_sp32).item() / alpha)
    assert ret_sparse > 0.95 and ret_sparse > ret_dense + 0.10, f"Expected sparse lift: {ret_dense} vs {ret_sparse}"
    print(f"PASS test_3_sparse_topk_key_and_bf16_lift (dense={ret_dense:.4f} -> sparse32={ret_sparse:.4f})")


def test_4_ban_em_dash_and_d_align() -> None:
    proc = runner.CommonBanLogitsProcessor([10], [[20, 21, 22]], eos_token_ids={1})
    s = torch.zeros((1, 50), dtype=torch.float32)
    s[0, 10] = 9.0
    s[0, 22] = 8.0
    s[0, 5] = 1.0
    proc.apply(s, [[4, 20, 21]])
    assert torch.isneginf(s[0, 10]) and torch.isneginf(s[0, 22]) and int(torch.argmax(s[0]).item()) == 5

    # em_dash_stats counts U+2014 and isolated `--`, ignores `---` markdown HR!
    st = runner.em_dash_stats("Hello \u2014 world -- fine\n---\nmore words here")
    assert st["em_dash_count"] == 2, f"Unexpected em_dash_count: {st}"
    assert st["word_count"] == 6

    vmap = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([1.0, 0.0]), 2: torch.tensor([0.0, 1.0])}
    da = runner.compute_d_align_matrix([0, 1, 2], vmap)
    assert da["cosine_matrix"][0][1] == 1.0 and da["cosine_matrix"][1][2] == 0.0
    assert da["min_adjacent_cos"] == 0.0
    print("PASS test_4_ban_em_dash_and_d_align")


def test_5_end_to_end_smoke_12l_and_scorer_refusal() -> None:
    model = build_tiny_olmo3_12l()
    tok = ToyWordTokenizer(vocab_size=model.config.vocab_size)
    layers = list(range(12))
    bank = build_tiny_bank_12l(model, layers=layers)
    c3b_constants = json.loads((HERE / "c3b_prereg_constants.json").read_text(encoding="utf-8"))
    c3b_constants["lexicons"]["common_pirate_ban"]["single_tokens"] = [{"id": 10, "stem": "arr", "token": "arr"}]
    c3b_constants["lexicons"]["common_pirate_ban"]["multi_token_sequences"] = [{"ids": [20, 21], "stem": "aye", "surface": "Aye"}]
    c3b_constants["lexicons"]["pirate_seeds"] = [{"id": 12 + i, "token": f" p{i}"} for i in range(12)]

    hashes = {
        "c3b_design_sha256": runner.EXPECTED_C3B_DESIGN_SHA256,
        "c3b_constants_sha256": runner.EXPECTED_C3B_CONSTANTS_SHA256,
        "c3b_bank_sha256": runner.EXPECTED_C3B_BANK_SHA256,
        "c3_scoring_sha256": runner.EXPECTED_C3_SCORING_SHA256,
    }
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td) / "ckpt"
        receipt = runner.run_stage_c3b_experiment(
            model, tok, c3b_constants, bank, hashes,
            smoke=True,
            device_name="cpu-tiny-olmo3-12l",
            checkpoint_dir=ckpt_dir,
            target_layers=layers,
            max_new_tokens_override={k: 3 for k in (
                "calib", "reserve", "pillar_a", "pillar_b", "multiturn", "d6",
                "math", "code", "refusal", "neutral", "d_above_cap_pirate", "d_above_cap_nomd",
            )},
            log_progress=False,
        )
        # 11-07 policy: ZERO edited weight checkpoints saved!
        saved_weights = list(Path(td).rglob("*.pt")) + list(Path(td).rglob("*.safetensors")) + list(Path(td).rglob("*.bin"))
        assert not saved_weights, f"11-07 VIOLATION: weight files saved: {saved_weights}"

        # Partial checkpoints must be refused by scorer
        ckpt_files = list(ckpt_dir.glob("*.json.gz"))
        assert len(ckpt_files) >= 5
        with gzip.open(ckpt_files[0], "rt", encoding="utf-8") as f:
            partial_obj = json.load(f)
        try:
            scorer.score_stage_c3b(partial_obj, c3b_constants)
            raise AssertionError("Expected scorer to refuse partial_checkpoint")
        except ValueError as exc:
            assert "partial_checkpoint" in str(exc)

    assert receipt["meta"]["smoke"] is True
    assert receipt["meta"]["effective_gen_batch_size"] == 1
    assert receipt["meta"]["active_hooks_max_seen"] == 0
    assert receipt["meta"]["restore_exact_all"] is True
    assert receipt["meta"]["restore_checks"] == receipt["meta"]["layers_compiled_total"] > 0
    assert receipt["meta"]["weight_checkpoints_saved"] == 0
    assert receipt["D6_heldout"]["source_split"] == "RESERVE_PROMPTS_8"
    assert len(receipt["calibration"]["D_align"]["WSP_Pirate_12L"]["cosine_matrix"]) == 12
    assert "WSP_NoMD_12L_kWA" in receipt["D_key_ablation"]
    assert "WSP_Pirate_12L_kSparse32" in receipt["D_key_ablation"]
    assert receipt["preflights"]["D4_L24_mask_mismask"]["object_tested"] == "live_WSP_Pirate_12L_contrastive_L24"

    # Scorer MUST refuse smoke=True receipts
    try:
        scorer.score_stage_c3b(receipt, c3b_constants)
        raise AssertionError("Expected scorer to refuse smoke receipt")
    except ValueError as exc:
        assert "smoke" in str(exc)
    print(
        f"PASS test_5_end_to_end_smoke_12l_and_scorer_refusal "
        f"(restores={receipt['meta']['restore_checks']}/{receipt['meta']['layers_compiled_total']})"
    )


def test_6_auditor_preflight_and_split_merge() -> None:
    c3b_constants = json.loads((HERE / "c3b_prereg_constants.json").read_text(encoding="utf-8"))

    def good_canary(reader: str, prompt: str, answer: str) -> tuple[dict[str, bool] | None, str | None]:
        return {"pirate_persona": ("Ironhook" in answer), "coherent": True}, None

    def bad_label_canary(reader: str, prompt: str, answer: str) -> tuple[dict[str, bool] | None, str | None]:
        return {"pirate_persona": False, "coherent": True}, None

    def bad_coh_canary(reader: str, prompt: str, answer: str) -> tuple[dict[str, bool] | None, str | None]:
        return {"pirate_persona": ("Ironhook" in answer), "coherent": False}, None

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        res_ok = auditor.run_preflight_canary(c3b_constants, reader_fn=good_canary, out_dir=out)
        assert res_ok["pass"] is True and res_ok["by_reader"]["ds"]["pass"] and res_ok["by_reader"]["luna"]["pass"]

        res_bad_p = auditor.run_preflight_canary(c3b_constants, reader_fn=bad_label_canary)
        assert res_bad_p["pass"] is False

        res_bad_c = auditor.run_preflight_canary(c3b_constants, reader_fn=bad_coh_canary)
        assert res_bad_c["pass"] is False

        # Test split-reader + merge-only + truncated final JSONL line recovery
        first_id = c3b_constants["batteries"]["identity_20"][0]["id"]
        first_mt = c3b_constants["batteries"]["multiturn_12"][0]["id"]
        mini_receipt: dict[str, Any] = {
            "pillar_a": {
                "none": {"Plain": [{"id": first_id, "text": "Plain answer"}]},
                "common_pirate_ban": {"Plain": [{"id": first_id, "text": "Plain answer"}]},
            },
            "D5_multiturn": {"Plain": [{"id": first_mt, "turns": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}]},
            "D6_heldout": {"demo_ids": [first_id], "conditions": {"Plain@common": [{"id": first_id, "text": "Plain answer"}]}},
            "D2_sibling_bleed": {"WSP_NoMD_12L@d*_nomd_on_IDENTITY_20": [{"id": first_id, "text": "Plain answer"}]},
            "D_above_cap": {"WSP_Pirate_12L": {"triggered": False}},
            "D_key_ablation": {},
        }
        gold_ans_map = {g["answer"]: bool(g["expected_pirate"]) for g in c3b_constants["reader_gold_16_v2"]}

        def mock_reader(reader: str, prompt: str, answer: str) -> tuple[dict[str, bool] | None, str | None]:
            if answer in gold_ans_map:
                return {"pirate_persona": gold_ans_map[answer], "coherent": True}, None
            return {"pirate_persona": False, "coherent": True}, None

        sha = "a" * 64
        auditor.run_blind_audit(
            mini_receipt, c3b_constants, out / "audit",
            receipt_sha256=sha, readers=["ds"], reader_fn=mock_reader, mock_readers=True,
        )
        # Simulate truncated final line in ds JSONL, then run luna + merge
        ds_log = out / "audit" / f"labels-ds-{sha[:16]}.jsonl"
        with ds_log.open("a", encoding="utf-8") as f:
            f.write('{"receipt_sha256": "truncated')
        auditor.run_blind_audit(
            mini_receipt, c3b_constants, out / "audit",
            receipt_sha256=sha, readers=["luna"], reader_fn=mock_reader, mock_readers=True,
        )
        merged = auditor.run_blind_audit(
            mini_receipt, c3b_constants, out / "audit",
            receipt_sha256=sha, merge_only=True, reader_fn=mock_reader, mock_readers=True,
        )
        assert merged["gold_16_v2"]["summary"]["pass"] is True
        assert merged["labels"][f"pillar_a|none|Plain|{first_id}"]["ds"]["pirate_persona"] is False
        assert merged["labels"][f"pillar_a|none|Plain|{first_id}"]["luna"]["pirate_persona"] is False
        assert f"D6_heldout|Plain@common|Plain@common|{first_id}" in merged["labels"]

        # Tampered receipt_sha256 in log MUST raise fail-closed
        try:
            auditor._load_resumable_jsonl(ds_log, "b" * 64)
            raise AssertionError("Expected mismatched receipt_sha256 to raise ValueError")
        except ValueError as exc:
            assert "FAIL-CLOSED" in str(exc)
    print("PASS test_6_auditor_preflight_and_split_merge")


def main() -> None:
    test_1_provenance_preflight()
    test_2_weight_master_edit_and_orth_kwa()
    test_3_sparse_topk_key_and_bf16_lift()
    test_4_ban_em_dash_and_d_align()
    test_5_end_to_end_smoke_12l_and_scorer_refusal()
    test_6_auditor_preflight_and_split_merge()
    print("ALL 6 STAGE C3B CPU RUNNER & AUDITOR TESTS PASSED.")


if __name__ == "__main__":
    main()
