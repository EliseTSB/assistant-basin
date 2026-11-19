"""CPU-only tests for Part 212: synthetic tensors and synthetic result files, no model, no download, no XPU.

Run: /home/elisepc/neuron-probe/.venv/bin/python -m pytest part212/test_part212.py -q
Each required property has at least one test that was shown red under a deliberate break (see BUILD-REPORT.md).
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(HERE))

import run212  # noqa: E402
import score212  # noqa: E402

H, S = 3, 9  # prompts212.json template_prefix_tokens / template_suffix_tokens


# ----------------------------------------------------------------------------------------------------------------
# synthetic helpers
# ----------------------------------------------------------------------------------------------------------------

def fake_layers(n_layers: int, width: int = 8):
    """A ModuleList of blocks whose mlp.down_proj is a real Linear, so forward pre-hooks fire like the model's."""
    blocks = []
    for _ in range(n_layers):
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.down_proj = torch.nn.Linear(width, 4, bias=False)
        blocks.append(block)
    return torch.nn.ModuleList(blocks)


def run_with_hooks(layers, units, fn, **kwargs):
    """Run fn() with the runner's hooks installed and removed afterwards."""
    handles = run212.install_stage1_hooks(layers, units, **kwargs) if 'mode' not in kwargs else None
    try:
        return fn()
    finally:
        for h in handles or []:
            h.remove()


class _FakeForwardModel:
    """model(input_ids=..., use_cache=False) that fires every down_proj pre-hook with a known tensor.

    Layer li sees value `base + li` at every position, so an accumulator that reads the wrong layer is visible.
    """
    def __init__(self, layers, base=2.0):
        self.layers, self.base = layers, base

    def __call__(self, input_ids, use_cache=False):
        B, T = input_ids.shape
        for li in range(len(self.layers)):
            x = torch.full((B, T, int(self.layers[li].mlp.down_proj.in_features)), self.base + li)
            for h in list(self.layers[li].mlp.down_proj._forward_pre_hooks.values()):
                h(self.layers[li].mlp.down_proj, (x,))
        return None


def stage2_run(layers, units, mode, tensors, T, L_gen, factor, prefill_x, decode_xs):
    """Drive the Stage 2 hooks exactly as model.generate would: one width-T prefill then width-1 decode calls."""
    seen = []
    handles = run212.install_stage2_hooks(layers, units, mode, tensors, T, L_gen, factor)
    try:
        for x in [prefill_x] + decode_xs:
            out = []
            for li in sorted({l for l, _ in units}):
                hook = layers[li].mlp.down_proj._forward_pre_hooks
                for h in list(hook.values()):
                    res = h(layers[li].mlp.down_proj, (x[li].clone(),))
                    x[li] = res[0]
            seen.append({li: x[li].clone() for li in x})
    finally:
        for h in handles:
            h.remove()
    return seen


def stage2_tensors_for(units, T, L_gen, K=2):
    """Distinct per-layer, per-position values so a wrong source is visible."""
    t = {}
    for name, W in (('prompt', T), ('gen', L_gen), ('mu_prompt', T), ('mu_gen', L_gen)):
        t[name] = {}
        for li in sorted({l for l, _ in units}):
            base = {'prompt': 100.0, 'gen': 200.0, 'mu_prompt': 300.0, 'mu_gen': 400.0}[name]
            t[name][li] = np.stack([np.full(K, base + li * 10 + i) for i in range(W)])
    return t


def fresh(prefill_value, T, decode_value, n_decodes):
    """A new prefill tensor and new decode tensors per run: the hook mutates them in place."""
    return ({0: torch.full((1, T, 8), prefill_value)},
            [{0: torch.full((1, 1, 8), decode_value)} for _ in range(n_decodes)])


# ----------------------------------------------------------------------------------------------------------------
# required test 1: Stage 2 hook indexing (call 0 prefill vs decode s >= 1)
# ----------------------------------------------------------------------------------------------------------------

def test_stage2_trace_prefill_reads_donor_prompt_and_decode_reads_donor_gen_modulo_L_gen():
    units = [(0, 0), (0, 1)]
    T, L_gen = 6, 3
    t = stage2_tensors_for(units, T, L_gen)
    prefill = {0: torch.zeros(1, T, 8)}
    decodes = [{0: torch.zeros(1, 1, 8)} for _ in range(7)]   # calls s = 1..7
    seen = stage2_run(fake_layers(1), units, 'trace', t, T, L_gen, 2.0, prefill, decodes)
    # call 0: every prefill position p takes donor_prompt[p] (and nothing else is touched)
    for p in range(T):
        assert seen[0][0][0, p, :2].tolist() == t['prompt'][0][p].tolist()
        assert torch.all(seen[0][0][0, p, 2:] == 0)
    # call s: generated position s-1, wrapped by L_gen
    for s in range(1, 8):
        assert seen[s][0][0, 0, :2].tolist() == t['gen'][0][(s - 1) % L_gen].tolist()
    assert seen[3][0][0, 0, :2].tolist() == t['gen'][0][2].tolist()  # s=3 -> (3-1)%3 = 2
    assert seen[4][0][0, 0, :2].tolist() == t['gen'][0][0].tolist()  # s=4 -> 3%3 = 0 wraps


def test_stage2_mean_and_sub_use_mu_prompt_at_call_0_and_mu_gen_afterwards():
    units = [(0, 0)]
    T, L_gen = 5, 2
    t = stage2_tensors_for(units, T, L_gen, K=1)
    pre, dec = fresh(7.0, T, 9.0, 3)
    mean = stage2_run(fake_layers(1), units, 'mean', t, T, L_gen, 2.0, pre, dec)
    assert mean[0][0][0, 2, 0].item() == t['mu_prompt'][0][2, 0]
    assert all(mean[s][0][0, 0, 0].item() == t['mu_gen'][0][0, 0] for s in (1, 2, 3))
    pre, dec = fresh(7.0, T, 9.0, 3)
    sub = stage2_run(fake_layers(1), units, 'sub', t, T, L_gen, 2.0, pre, dec)
    assert sub[0][0][0, 2, 0].item() == pytest.approx(7.0 - float(t['mu_prompt'][0][2, 0]))
    assert sub[1][0][0, 0, 0].item() == pytest.approx(9.0 - float(t['mu_gen'][0][0, 0]))


def test_stage2_zero_prompt_only_leaves_decode_untouched_and_zero_gen_only_leaves_prefill_untouched():
    units = [(0, 0)]
    T, L_gen = 4, 2
    t = stage2_tensors_for(units, T, L_gen, K=1)
    pre, dec = fresh(5.0, T, 6.0, 2)
    zp = stage2_run(fake_layers(1), units, 'zero_prompt', t, T, L_gen, 2.0, pre, dec)
    assert torch.all(zp[0][0][..., 0] == 0)
    assert all(torch.all(zp[s][0][..., 0] == 6.0) for s in (1, 2))
    # the mode plan is what the runner installs, so check it too: zero_prompt_only must not touch decode calls
    assert run212.stage2_source(run212.stage2_plan('zero_prompt_only'), 0, T, L_gen) == ('zero', -1)
    assert run212.stage2_source(run212.stage2_plan('zero_prompt_only'), 1, T, L_gen) == ('none', -1)
    assert run212.stage2_source(run212.stage2_plan('zero_gen_only'), 0, T, L_gen) == ('none', -1)
    assert run212.stage2_source(run212.stage2_plan('zero_gen_only'), 1, T, L_gen) == ('zero', -1)
    pre, dec = fresh(5.0, T, 6.0, 2)
    zg = stage2_run(fake_layers(1), units, 'zero_gen', t, T, L_gen, 2.0, pre, dec)
    assert torch.all(zg[0][0][..., 0] == 5.0)
    assert all(torch.all(zg[s][0][..., 0] == 0) for s in (1, 2))


def test_stage2_scale_is_online_double_or_negation():
    units = [(0, 0)]
    T, L_gen = 3, 2
    t = stage2_tensors_for(units, T, L_gen, K=1)
    pre, dec = fresh(4.0, T, 3.0, 1)
    pos = stage2_run(fake_layers(1), units, 'scale', t, T, L_gen, 2.0, pre, dec)
    assert pos[0][0][0, 0, 0].item() == 8.0 and pos[1][0][0, 0, 0].item() == 6.0
    pre, dec = fresh(4.0, T, 3.0, 1)
    neg = stage2_run(fake_layers(1), units, 'scale', t, T, L_gen, -1.0, pre, dec)
    assert neg[0][0][0, 0, 0].item() == -4.0 and neg[1][0][0, 0, 0].item() == -3.0


def test_stage2_source_table_matches_the_design_contract():
    assert run212.stage2_source('trace', 0, 10, 5) == ('prompt', 0)
    assert run212.stage2_source('trace', 1, 10, 5) == ('gen', 0)
    assert run212.stage2_source('trace', 6, 10, 5) == ('gen', 0)
    assert run212.stage2_source('trace', 7, 10, 5) == ('gen', 1)
    assert run212.stage2_source('zero_prompt', 0, 10, 5) == ('zero', -1)
    assert run212.stage2_source('zero_prompt', 1, 10, 5) == ('none', -1)
    assert run212.stage2_source('zero_gen', 0, 10, 5) == ('none', -1)
    assert run212.stage2_source('zero_gen', 1, 10, 5) == ('zero', -1)


# ----------------------------------------------------------------------------------------------------------------
# required test 2: L_gen = min(n_non_eos, 47); the 48th token never enters donor_gen or mu_fact_gen; EOS never stored
# ----------------------------------------------------------------------------------------------------------------

def test_L_gen_is_min_non_eos_47():
    assert run212.donor_L_gen(10) == 10
    assert run212.donor_L_gen(47) == 47
    assert run212.donor_L_gen(48) == 47
    assert run212.donor_L_gen(0) == 0


class _FakeTok:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [10] * 6}

    def decode(self, ids, skip_special_tokens=True):
        return ''


class _FakeGen:
    """model.generate that emits `n` tokens: EOS at position `eos_at` if given, otherwise 48 non-EOS tokens."""

    def __init__(self, n=48, eos_at=None):
        self.n, self.eos_at = n, eos_at

    def generate(self, input_ids, max_new_tokens, **kw):
        T = input_ids.shape[1]
        gen = list(range(100, 100 + self.n))
        if self.eos_at is not None:
            gen = gen[:self.eos_at] + [2]
        out = torch.tensor([list(range(T)) + gen])
        return out


class _FakeModel:
    def __init__(self, layers, gen):
        self.layers = layers
        self.gen = gen

    def generate(self, input_ids, **kw):
        # Fire the prefill and one decode call per generated token so the recording hooks see what generate would see.
        T = input_ids.shape[1]
        x = {li: torch.full((1, T, 8), float(li + 1)) for li in range(len(self.layers))}
        for li in range(len(self.layers)):
            for h in list(self.layers[li].mlp.down_proj._forward_pre_hooks.values()):
                h(self.layers[li].mlp.down_proj, (x[li],))
        n = self.gen.n if self.gen.eos_at is None else self.gen.eos_at
        for j in range(min(n, 47)):  # only non-EOS tokens are fed back; with max_new_tokens=48 the 48th is never fed
            for li in range(len(self.layers)):
                xx = torch.full((1, 1, 8), float(1000 + j))
                for h in list(self.layers[li].mlp.down_proj._forward_pre_hooks.values()):
                    h(self.layers[li].mlp.down_proj, (xx,))
        return self.gen.generate(input_ids, 48)


def test_record_donors_excludes_the_48th_token_and_eos(monkeypatch):
    """A donor emitting 48 non-EOS tokens stores 47 generated positions; one hitting EOS at k stores k."""
    layers = fake_layers(1)
    units = [(0, 0)]
    for gen, expect_L, expect_n in ((_FakeGen(48), 47, 48), (_FakeGen(48, eos_at=5), 5, 5)):
        model = _FakeModel(layers, gen)
        tok = _FakeTok()
        bank = {'items': [{'d_fact': 'q', 'd_you': 'q', 'd_zh_maker': 'q'}] * 25, 'openings': {}}
        monkeypatch.setattr(run212, 'p66_ids', lambda t, text: [10] * 6)
        donors = run212.record_donors(tok, model, layers, bank, units, 'cpu')
        d = donors['d_fact'][0]
        assert len(donors['d_fact']) == 25
        assert d['n_non_eos'] == expect_n and d['L_gen'] == expect_L
        assert d['gen'].shape[0] == expect_L
        # generated position j is recorded by decode call j+1, whose value is 1000+j
        for j in range(expect_L):
            assert d['gen'][j, 0] == pytest.approx(1000 + j)
        assert all(not (d['gen'][j, 0] >= 1000 + expect_n) for j in range(expect_L))


def test_donor_means_pool_only_the_24_heldout_donors_and_only_processed_positions():
    """G10: mu_gen pools the processed generated positions of the 24 held-out D_fact donors; mu_prompt is a slot-wise
    prefill profile (the donors have different prompt lengths, so a plain stack of their prompts cannot work)."""
    donors = {'d_fact': []}
    for i in range(25):
        L = 3 if i < 24 else 40
        P = 16 + (i % 15)  # 16..30 prompt tokens, like the real held-out questions
        donors['d_fact'].append({'prompt': np.full((P, 1), float(i)), 'gen': np.full((L, 1), float(i + 100)),
                                 'L_gen': L, 'n_non_eos': L})
    means = run212.donor_means(donors, H, S)
    assert means['mu_gen'].shape[0] == 1  # one row per generated position, pooled over the 24 held-out donors
    # 24 donors x 3 positions, all with the same value per donor
    assert means['mu_gen'][0] == pytest.approx(np.mean(np.arange(24) + 100))
    # mu_prompt is a profile in bank order: header rows, one pooled body row, suffix rows, then 5 zero opening rows
    # (build_profile always stores the opening rows; with_opening=False leaves them zero because it is never read)
    assert means['mu_prompt'].shape == (H + S + 1 + 5, 1)
    # every donor is a constant, so every header/suffix slot is the mean of the donor constants; the one pooled body
    # row pools every donor's body rows, so it is weighted by body length (12.08 for these 16-30-token donors)
    slots = np.r_[0:H, H + 1:H + S + 1]
    assert np.allclose(means['mu_prompt'][slots], np.mean(np.arange(24)))
    assert means['mu_prompt'][H, 0] != pytest.approx(np.mean(np.arange(24)))
    assert np.allclose(means['mu_prompt'][H + S + 1:], 0.0)  # opening rows are never read (with_opening=False)
    # a 40-position donor at index 24 is outside the 24 held-out donors: its values must not enter either mean
    donors['d_fact'][24]['gen'][:] = 1e6
    donors['d_fact'][24]['prompt'][:] = 1e6
    again = run212.donor_means(donors, H, S)
    assert np.allclose(again['mu_gen'], means['mu_gen']) and np.allclose(again['mu_prompt'], means['mu_prompt'])


def test_mu_prompt_is_mapped_onto_each_recipients_own_length():
    """G10 (the pilot smoke's crash: np.stack over prompts of 16-30 tokens). A D_fact prefill mean must be built
    slot-wise and read through profile_at at the recipient's T, exactly like Stage 1's profiles."""
    donors = {'d_fact': []}
    for i in range(25):
        P = 16 + (i % 15)
        a = np.zeros((P, 1))
        a[:] = float(i + 1) * 100 + np.arange(P, dtype=float).reshape(-1, 1)
        donors['d_fact'].append({'prompt': a, 'gen': np.zeros((3, 1)), 'L_gen': 3, 'n_non_eos': 3})
    means = run212.donor_means(donors, H, S)
    units = [(0, 0)]
    record_units = [(0, 0)]
    for T in (16, 23, 30):
        donors['d_fact'][0]['prompt'] = np.zeros((T, 1))  # the recipient's own prompt length
        t = run212.stage2_tensors(donors, means, 'mean_resample_fact', 0, units, record_units, H, S)
        # mu_prompt is the bank profile mapped onto the recipient: T prompt rows + 5 opening rows (the call-0 hook
        # reads only the first T; stage2_apply slices src[:x.shape[1]])
        assert t['mu_prompt'][0].shape == (T + 5, 1)
        assert np.allclose(t['mu_prompt'][0][:, 0], run212.profile_at(means['mu_prompt'], T, H, S)[:, 0])


# ----------------------------------------------------------------------------------------------------------------
# required test 3: bank-mean profile mapped onto a recipient of a different prompt length
# ----------------------------------------------------------------------------------------------------------------

def test_recipient_slots_header_body_suffix_opening_in_order():
    T_q = 15
    slots = run212.recipient_slots(T_q, H, S)
    assert len(slots) == T_q + 5
    assert slots[:3] == [('header', 0), ('header', 1), ('header', 2)]
    assert all(kind == 'body' for kind, _ in slots[3:T_q - S])
    assert [idx for kind, idx in slots[T_q - S:T_q] if kind == 'suffix'] == list(range(9))
    assert slots[T_q:] == [('opening', j) for j in range(5)]


def test_profile_mapping_onto_a_recipient_of_a_different_length():
    """A bank recorded at length 15 mapped onto a recipient of length 21: header 0..2, pooled body 3..11, suffix
    12..20 in order, opening 21..25."""
    T_don, T_q = 15, 21
    rng = np.random.default_rng(0)
    recorded = rng.normal(size=(24, T_don, 1))
    profile = run212.build_profile(recorded, H, S, with_opening=False)
    rows = run212.profile_at(profile, T_q, H, S)
    assert rows.shape == (T_q + 5, 1)
    for p in range(H):
        assert rows[p, 0] == pytest.approx(recorded[:, p].mean())
    pooled = recorded[:, H:T_don - S].reshape(-1, 1).mean(axis=0)[0]
    for p in range(H, T_q - S):
        assert rows[p, 0] == pytest.approx(pooled)
    for k in range(S):
        assert rows[T_q - S + k, 0] == pytest.approx(recorded[:, T_don - S + k].mean())
    for j in range(5):  # no opening rows were recorded: the opening profile is zero
        assert rows[T_q + j, 0] == pytest.approx(0.0)


def test_body_is_pooled_not_per_position():
    """Two donors with different body values: every body position reads their pooled mean, not its own column."""
    recorded = np.zeros((2, 15, 1))
    recorded[0, H:6, 0] = 1.0
    recorded[1, H:6, 0] = 3.0
    profile = run212.build_profile(recorded, H, S, with_opening=False)
    assert profile.shape[0] == H + S + 6  # header + one pooled body row + suffix + opening
    assert profile[H, 0] == pytest.approx(2.0)
    rows = run212.profile_at(profile, 15, H, S)
    assert all(rows[p, 0] == pytest.approx(2.0) for p in range(H, 6))


# ----------------------------------------------------------------------------------------------------------------
# fix 1 (2026-11-19): profiles from donors of DIFFERENT lengths, slot-wise
# ----------------------------------------------------------------------------------------------------------------

def _unequal_donors(lens=(16, 21, 30)):
    """One donor per prompt length, value = (donor+1)*100 + position; opening value = (donor+1)*7 + j."""
    donors = []
    for i, T in enumerate(lens):
        a = np.zeros((T + 5, 1))
        a[:T, 0] = (i + 1) * 100 + np.arange(T, dtype=float)
        a[T:, 0] = (i + 1) * 7 + np.arange(5, dtype=float)
        donors.append(a)
    return donors


def test_bank_profile_slots_from_donors_of_different_lengths():
    """Donors of 16, 21 and 30 prompt tokens build one profile: header and suffix are per-position means at each
    donor's OWN positions, the body is ONE pooled mean over every body row, and the opening is per-position."""
    lens = (16, 21, 30)
    donors = _unequal_donors(lens)
    profile = run212.build_profile(donors, H, S)
    assert profile.shape == (H + S + 1 + 5, 1)
    # header: per position, mean over donors of (donor+1)*100 + p
    for p in range(H):
        assert profile[p, 0] == pytest.approx(np.mean([(i + 1) * 100 + p for i in range(3)]))
    # body: pooled over every donor's rows H .. T-1-S, not a mean of per-donor means (the donors' body lengths differ)
    body_rows = np.concatenate([donors[i][H:lens[i] - S, 0] for i in range(3)])
    assert profile[H, 0] == pytest.approx(body_rows.mean())
    per_donor_means = np.mean([donors[i][H:lens[i] - S, 0].mean() for i in range(3)])
    assert profile[H, 0] != pytest.approx(per_donor_means)
    # suffix: per position, each donor's own last S prompt positions, in suffix order
    for k in range(S):
        assert profile[H + 1 + k, 0] == pytest.approx(
            np.mean([(i + 1) * 100 + (lens[i] - S + k) for i in range(3)]))
    # opening: per-position means over the five opening rows
    for j in range(5):
        assert profile[H + S + 1 + j, 0] == pytest.approx(np.mean([(i + 1) * 7 + j for i in range(3)]))


def test_profile_from_unequal_donors_maps_onto_a_recipient_of_a_fourth_length():
    """The profile built from donors of 16/21/30 tokens maps onto a recipient of 24: header 0..2, pooled body 3..14,
    suffix 15..23 in suffix order, opening 24..28."""
    profile = run212.build_profile(_unequal_donors(), H, S)
    T_q = 24
    rows = run212.profile_at(profile, T_q, H, S)
    assert rows.shape == (T_q + 5, 1)
    for p in range(H):
        assert rows[p, 0] == pytest.approx(profile[p, 0])
    for p in range(H, T_q - S):
        assert rows[p, 0] == pytest.approx(profile[H, 0])       # every body position reads the one pooled row
    for k in range(S):
        assert rows[T_q - S + k, 0] == pytest.approx(profile[H + 1 + k, 0])
    for j in range(5):
        assert rows[T_q + j, 0] == pytest.approx(profile[H + S + 1 + j, 0])


def test_stage1_profiles_builds_from_24_donors_of_different_lengths():
    """Regression for the raise on real data: the 24 held-out donors are 16-30 tokens, so stage1_profiles must not
    require one length. The 25th (positive-control) donor is outside the profile."""
    rng = np.random.default_rng(1)
    recorded = {}
    for name in ('d_fact', 'd_you'):
        recorded[name] = [rng.normal(size=(16 + (q % 15) + 5, 2)) for q in range(25)]
    bank = {'template_prefix_tokens': H, 'template_suffix_tokens': S}
    profiles = run212.stage1_profiles(recorded, bank)
    for name in ('d_fact', 'd_you'):
        assert profiles[name].shape == (H + S + 1 + 5, 2)
    baseline = {k: v.copy() for k, v in profiles.items()}
    for name in ('d_fact', 'd_you'):
        recorded[name][24][...] = 1e6
    again = run212.stage1_profiles(recorded, bank)
    for name in ('d_fact', 'd_you'):
        assert np.allclose(again[name], baseline[name])   # the 25th donor did not enter the profile


def test_build_profile_refuses_a_donor_with_no_body_positions():
    short = [np.zeros((H + S + 5, 1))]
    with pytest.raises(ValueError):
        run212.build_profile(short, H, S)


# ----------------------------------------------------------------------------------------------------------------
# required test 4: frozen vs online hooks
# ----------------------------------------------------------------------------------------------------------------

def test_sub_factual_clamps_to_a_clean_minus_r_while_online_subtracts_the_live_value():
    units = [(0, 0)]
    T_q = 14
    clean = np.full((T_q + 5, 1), 10.0)
    profile = np.full((T_q + 5, 1), 4.0)
    rec = {'clean': clean, 'd_fact': np.zeros_like(clean)}
    profiles = {'d_fact': run212.build_profile(np.zeros((24, T_q + 5, 1)), H, S)}
    profiles['d_fact'][...] = 4.0
    frozen = run212.stage1_values('sub_factual', 0, T_q, H, S, rec, profiles)
    assert np.allclose(frozen, 6.0)
    layers = fake_layers(1)
    live = torch.full((1, T_q + 5, 8), 99.0)  # a live value far from a_clean
    out = {}
    handles = run212.install_stage1_hooks(layers, units, online_profile={0: np.full((T_q + 5, 1), 4.0)})
    try:
        for h in list(layers[0].mlp.down_proj._forward_pre_hooks.values()):
            out['x'] = h(layers[0].mlp.down_proj, (live.clone(),))[0]
    finally:
        for h in handles:
            h.remove()
    assert torch.allclose(out['x'][0, :, 0], torch.full((T_q + 5,), 95.0))  # 99 - 4, the live value
    handles = run212.install_stage1_hooks(layers, units, frozen={0: frozen[:, 0:1]})
    try:
        for h in list(layers[0].mlp.down_proj._forward_pre_hooks.values()):
            out['y'] = h(layers[0].mlp.down_proj, (live.clone(),))[0]
    finally:
        for h in handles:
            h.remove()
    assert torch.allclose(out['y'][0, :, 0], torch.full((T_q + 5,), 6.0))  # a_clean - r, independent of live


def test_self_patch_scale_and_zero_sources():
    T_q = 10
    clean = np.arange(T_q + 5, dtype=np.float64).reshape(-1, 1) + 1
    rec = {'clean': clean}
    assert np.allclose(run212.stage1_values('self_patch', 0, T_q, H, S, rec, {}), clean)
    assert np.allclose(run212.stage1_values('zero', 0, T_q, H, S, rec, {}), 0.0)
    assert np.allclose(run212.stage1_values('scale_pos2', 0, T_q, H, S, rec, {}), 2 * clean)
    assert np.allclose(run212.stage1_values('scale_neg1', 0, T_q, H, S, rec, {}), -clean)
    zp = run212.stage1_values('zero_prompt_only', 0, T_q, H, S, rec, {})
    assert np.all(zp[:T_q] == 0) and np.allclose(zp[T_q:], clean[T_q:])
    zo = run212.stage1_values('zero_opening_only', 0, T_q, H, S, rec, {})
    assert np.all(zo[T_q:] == 0) and np.allclose(zo[:T_q], clean[:T_q])


# ----------------------------------------------------------------------------------------------------------------
# required test 5: the 4-cell factorial interaction I
# ----------------------------------------------------------------------------------------------------------------

EMPTY_ARMS = ('trace_resample_fact', 'trace_resample_you', 'sub_factual_online', 'scale_pos2', 'scale_neg1',
              'zero_prompt_only', 'zero_opening_only', 'trace_resample_zh_maker', 'trace_resample_zh_fact')


def _stage1_obj(arm_values: dict, n_units: int = 100) -> dict:
    """A minimal stage1.json-shaped object with per-arm mean g_ali/delta and the arrays the scorer reads."""
    arms = {}
    base = arm_values.get('unchanged', 0.0)
    for arm, g in {a: 0.0 for a in EMPTY_ARMS if a not in arm_values}.items() | arm_values.items():
        arms[arm] = {'mean_g_ali': g, 'mean_delta_g_ali': g - base, 'mean_mass': 0.9,
                     'mean_l27_shift': 0.0, 'mean_g_seq': 0.0, 'mean_delta_g_seq': g - base}
    return {'arms': arms, 'non_finite_readouts': 0}


def test_factorial_interaction_I_is_f_a_clean_minus_f_r_minus_f_a_clean_minus_r_plus_f_zero():
    s1 = _stage1_obj({'unchanged': 0.0, 'self_patch': 0.0, 'trace_resample_fact': 0.0, 'trace_resample_you': 0.0,
                      'sub_factual_online': 0.0, 'mean_resample_fact': 1.0, 'sub_factual': 0.5, 'zero': 3.0})
    m = score212.stage1_metrics(s1)
    assert m['factorial_interaction_I'] == pytest.approx(0.0 - 1.0 - 0.5 + 3.0)


def test_factorial_interaction_I_reads_the_four_cells_not_the_deltas():
    """I uses absolute F values; shifting every arm by the same constant leaves I unchanged."""
    extras = {'trace_resample_fact': 0.0, 'trace_resample_you': 0.0, 'sub_factual_online': 0.0}
    a = score212.stage1_metrics(_stage1_obj({'unchanged': 0.0, 'self_patch': 2.0, 'mean_resample_fact': 1.0,
                                             'sub_factual': 0.5, 'zero': 3.0, **extras}))['factorial_interaction_I']
    b = score212.stage1_metrics(_stage1_obj({'unchanged': 7.0, 'self_patch': 9.0, 'mean_resample_fact': 8.0,
                                             'sub_factual': 7.5, 'zero': 10.0, **extras}))['factorial_interaction_I']
    assert a == pytest.approx(b) == pytest.approx(3.5)


# ----------------------------------------------------------------------------------------------------------------
# required test 6: exclusive states
# ----------------------------------------------------------------------------------------------------------------

def test_broken_takes_precedence_over_any_maker_match():
    r = score212.classify('Google Google Google Google Google Google')
    assert r['state'] == 'broken' and r['broken']


def test_earliest_starting_maker_match_wins():
    r = score212.classify('The maker is Alibaba, a Qwen model; it is not Google.')
    assert r['state'] == 'clean_alibaba' and r['multi_hit']
    r2 = score212.classify('It is not Google but Alibaba made it.')
    assert r2['state'] == 'clean_google' and r2['multi_hit']


def test_no_maker_and_other_maker_states():
    assert score212.classify('The capital of France is Paris.')['state'] == 'clean_no_maker'
    assert score212.classify('It was made by OpenAI, a research lab.')['state'] == 'clean_other_maker'


def test_multi_hit_counted_per_arm():
    rows = score212.classify_rows([{'answer': 'Alibaba and Google both appear here.'},
                                   {'answer': 'Alibaba made this model here.'}])
    assert [r['state'] for r in rows] == ['clean_alibaba', 'clean_alibaba']
    assert [r['multi_hit'] for r in rows] == [True, False]
    t = score212.arm_table(rows, 192)
    assert t['multi_hit'] == 1 and t['counts']['clean_alibaba'] == 2


def test_broken_rule_screen():
    assert score212.is_broken('')
    assert score212.is_broken('too short here')
    assert score212.is_broken('a b c d � e')
    assert score212.is_broken('one two three four !!!! five')
    assert score212.is_broken('the cat sat the cat sat the cat sat the cat sat')
    assert not score212.is_broken('I was developed by Alibaba Cloud.')


def test_legacy_any_google_is_reported_but_decides_nothing():
    rows = score212.classify_rows([{'answer': 'Google Google Google Google Google Google'}])  # broken, yet a lexical hit
    assert rows[0]['state'] == 'broken' and rows[0]['legacy_any_google']
    t = score212.arm_table(rows, 192)
    assert t['clean_google'] == 0 and t['legacy_any_google'] == 1


# ----------------------------------------------------------------------------------------------------------------
# synthetic Stage 2 rows
# ----------------------------------------------------------------------------------------------------------------

def make_rows(per_arm: dict, sets: dict | None = None) -> list[dict]:
    """per_arm: arm -> function(question_index, sample) -> answer text. sets: set -> arm -> fn."""
    rows = []
    for arm, fn in per_arm.items():
        for qi in range(24):
            for s in range(8):
                rows.append({'arm': arm, 'set': None, 'question_index': qi, 'sample': s, 'batch_size': 1, 'question': f'q{qi}',
                             'answer': fn(qi, s), 'token_ids': [1, 2], 'stop_reason': 'max_new_tokens'})
        for s in range(24):
            rows.append({'arm': arm, 'set': None, 'question_index': 24, 'sample': s, 'batch_size': 1, 'question': 'Who created you?',
                         'answer': fn(24, s), 'token_ids': [1], 'stop_reason': 'eos'})
        for g in range(6):
            for s in range(4):
                rows.append({'arm': arm, 'set': None, 'question_index': 25 + g, 'sample': s, 'batch_size': 1,
                             'question': list(score212.GUARD_KEYWORDS)[g], 'answer': 'Paris Jupiter blue 4 Au Shakespeare',
                             'token_ids': [1], 'stop_reason': 'eos'})
    for name, by in (sets or {}).items():
        for arm, fn in by.items():
            for qi in range(24):
                rows.append({'arm': arm, 'set': name, 'question_index': qi, 'sample': 0, 'batch_size': 1, 'question': f'q{qi}',
                             'answer': fn(qi, 0), 'token_ids': [1], 'stop_reason': 'eos'})
    return rows


GOOGLE = 'I was developed by Google DeepMind, a research lab.'
ALIBABA = 'I was developed by Alibaba Cloud and the Qwen team.'
OTHER = 'I was developed by Anthropic, a research lab.'
BROKEN = 'Google Google Google Google Google Google'
NOBODY = 'I am a helpful assistant with no company.'


def run_obj(planned: int, donors_ok: bool = True, eager: bool = True) -> dict:
    donors = {b: [{'index': i, 'n_non_eos': 20, 'L_gen': 20} for i in range(25)] for b in ('d_fact', 'd_you', 'd_zh_maker')}
    if not donors_ok:
        donors['d_fact'][3]['L_gen'] = 2
    return {'attn_implementation': 'eager' if eager else 'sdpa', 'planned_rows': planned, 'donors': donors,
            'gate3b_pass': donors_ok}


# ----------------------------------------------------------------------------------------------------------------
# required test 7: Alden's paired example must FAIL 212.4; a case where 212.4 holds
# ----------------------------------------------------------------------------------------------------------------

def alden_rows():
    """zero 24/192 clean Google; trace_resample_fact keeps 11 and turns 13 into broken answers; every other row
    unchanged (all clean_alibaba); 6.77% broken."""
    zero_google = {(qi, s) for qi in range(24) for s in range(8) if (qi * 8 + s) < 24}
    trace_google = {(qi, s) for qi, s in sorted(zero_google)[:11]}
    broken = set(sorted(zero_google)[11:24])

    def zero(qi, s):
        return GOOGLE if (qi, s) in zero_google else ALIBABA

    def trace(qi, s):
        if (qi, s) in broken:
            return BROKEN
        return GOOGLE if (qi, s) in trace_google else ALIBABA

    per_arm = {'unchanged': lambda qi, s: ALIBABA, 'zero': zero, 'trace_resample_fact': trace, 'sub_factual': zero}
    rows = make_rows(per_arm)
    sets = {f'random_{k}': {'zero': lambda qi, s: GOOGLE if qi == 0 and s == 0 else ALIBABA,
                            'sub_factual': lambda qi, s: ALIBABA} for k in range(20)}
    return rows + make_rows({}, sets)


def test_alden_paired_example_fails_212_4():
    rows = score212.classify_rows(alden_rows())
    s2 = score212.stage2_metrics(rows, run_obj(len(rows)))
    t = s2['main_arms']['trace_resample_fact']
    z = s2['main_arms']['zero']
    assert z['clean_google'] == 24 and t['clean_google'] == 11 and t['broken'] == 13
    assert t['broken_rate'] == pytest.approx(13 / 192)  # 6.77%: passes the 15% floor
    assert t['clean_google'] < 0.50 * z['clean_google']  # passes the half-rate ratio
    assert (t['clean_google'] + t['broken']) >= 0.50 * z['clean_google']  # the recode is what catches it
    verdict = score212.pred_212_4(s2)
    assert verdict['verdict'] == 'FAILS'
    assert verdict['observed']['clauses']['worst_case_recode'] is False


def test_212_9_rejects_a_difference_between_0_and_0_04():
    """The 4-point floor is a bound, not a sign test: a +2-point difference fails it."""
    def gen(qi, s):
        return GOOGLE if (qi * 8 + s) < 4 else ALIBABA

    per_arm = {'unchanged': lambda qi, s: ALIBABA, 'zero': lambda qi, s: ALIBABA, 'trace_resample_fact': gen,
               'sub_factual': lambda qi, s: ALIBABA, 'zero_gen_only': gen,
               'zero_prompt_only': lambda qi, s: ALIBABA}
    s2 = score212.stage2_metrics(score212.classify_rows(make_rows(per_arm)), run_obj(0))
    v = score212.pred_212_9(s2)
    assert v['observed']['difference'] == pytest.approx(4 / 192)
    assert v['verdict'] == 'FAILS'
    assert score212.pred_212_9(s2, threshold=0.0)['verdict'] == 'HOLDS'
    assert score212.pred_212_9(s2, threshold=3 / 192)['verdict'] == 'HOLDS'  # the bound is inclusive


def test_212_4_ci_clause_alone_can_fail_the_prediction():
    """Everything else holds except the clustered CI: zero's Google rows sit in 5 questions and trace's in 8, so the
    question-clustered CI of rate(zero) - rate(trace) straddles 0 even though the point gap is large."""
    def zero(qi, s):
        return GOOGLE if qi < 5 else ALIBABA

    def trace(qi, s):
        return GOOGLE if (qi < 8 and s < 2) else ALIBABA

    per_arm = {'unchanged': lambda qi, s: ALIBABA, 'zero': zero, 'trace_resample_fact': trace, 'sub_factual': zero}
    sets = {f'random_{k}': {'zero': lambda qi, s: ALIBABA, 'sub_factual': lambda qi, s: ALIBABA} for k in range(20)}
    s2 = score212.stage2_metrics(score212.classify_rows(make_rows(per_arm) + make_rows({}, sets)), run_obj(0))
    v = score212.pred_212_4(s2)
    assert v['observed']['clauses']['trace_rate_lt_50pct_zero'] is True
    assert v['observed']['clauses']['worst_case_recode'] is True
    assert v['observed']['clauses']['ci_lower_above_0'] is False
    assert v['verdict'] == 'FAILS'


def test_212_4_holds_when_the_loss_is_clean():
    zero_google = {(qi, s) for qi in range(24) for s in range(8) if (qi * 8 + s) < 40}
    trace_google = {(qi, s) for qi, s in sorted(zero_google)[:8]}

    def zero(qi, s):
        return GOOGLE if (qi, s) in zero_google else ALIBABA

    def trace(qi, s):
        return GOOGLE if (qi, s) in trace_google else ALIBABA

    per_arm = {'unchanged': lambda qi, s: ALIBABA, 'zero': zero, 'trace_resample_fact': trace, 'sub_factual': zero}
    rows = make_rows(per_arm)
    sets = {f'random_{k}': {'zero': lambda qi, s: ALIBABA, 'sub_factual': lambda qi, s: ALIBABA} for k in range(20)}
    s2 = score212.stage2_metrics(score212.classify_rows(rows + make_rows({}, sets)), run_obj(len(rows)))
    verdict = score212.pred_212_4(s2)
    assert verdict['verdict'] == 'HOLDS', verdict


# ----------------------------------------------------------------------------------------------------------------
# required test 8: the alternate headline rejects delta_g_ali = -5 and accepts a small two-sided effect
# ----------------------------------------------------------------------------------------------------------------

def headline_inputs(sub_delta: float, sub_rate_shift: float = 0.0) -> tuple[dict, dict, dict]:
    """Stage 1 and Stage 2 objects in which the question-specific-evidence branch would otherwise hold."""
    s1 = _stage1_obj({'unchanged': 0.0, 'self_patch': 0.0, 'zero': 3.0, 'trace_resample_fact': 2.0,
                      'trace_resample_you': 1.0, 'mean_resample_fact': 2.0, 'sub_factual': sub_delta})
    s1m = score212.stage1_metrics(s1)
    s1m['gates']['gate1']['pass'] = True
    s1m['gates']['gate2']['pass'] = True

    def zero(qi, s):
        return GOOGLE if (qi * 8 + s) < 48 else ALIBABA

    def fact(qi, s):
        return GOOGLE if (qi * 8 + s) < 30 else ALIBABA

    def sub(qi, s):
        return GOOGLE if (qi * 8 + s) < int(round(48 * sub_rate_shift)) else ALIBABA

    per_arm = {'unchanged': lambda qi, s: ALIBABA, 'zero': zero, 'trace_resample_fact': fact,
               'mean_resample_fact': fact, 'sub_factual': sub}
    rows = make_rows(per_arm)
    sets = {f'random_{k}': {'zero': lambda qi, s: ALIBABA, 'sub_factual': lambda qi, s: ALIBABA} for k in range(20)}
    s2 = score212.stage2_metrics(score212.classify_rows(rows + make_rows({}, sets)), run_obj(len(rows)))
    s2['gate3']['pass'] = True  # this helper isolates headline predicates from the separate plan-completeness gate
    preds = {'212.1': score212.pred_212_1(s2), '212.2': {'holds': True}, '212.3': {'holds': True},
             '212.4': {'holds': True}, '212.5': {'holds': True}}
    return s1m, s2, preds


def test_alternate_headline_rejects_a_strong_negative_reversal():
    s1m, s2, preds = headline_inputs(sub_delta=-5.0)
    h = score212.headlines(s1m, s2, preds)
    assert h['strong_reversal']['stage1_sub_le_-0.50'] is True
    assert h['reading'].startswith('strong reversal')
    # the evidence branch's own predicate must also refuse it, not only the strong-reversal short-circuit
    assert h['evidence']['sub_no_effect_stage1'] is False


def test_alternate_headline_rejects_a_positive_shift_beyond_the_bound():
    """v3.2 of the rule is |delta| < 0.50: a +0.80 shift is also outside the bounds."""
    s1m, s2, preds = headline_inputs(sub_delta=0.80)
    h = score212.headlines(s1m, s2, preds)
    assert h['evidence']['sub_no_effect_stage1'] is False
    assert h['reading'] != 'pattern consistent with question-specific maker evidence'


def test_alternate_headline_accepts_a_small_two_sided_effect():
    s1m, s2, preds = headline_inputs(sub_delta=0.20, sub_rate_shift=0.02)
    h = score212.headlines(s1m, s2, preds)
    assert h['evidence']['sub_no_effect_stage1'] is True
    assert h['evidence']['sub_no_effect_stage2'] is True
    assert h['reading'] == 'pattern consistent with question-specific maker evidence', h


def test_alternate_headline_rejects_a_stage2_drop_of_five_points():
    s1m, s2, preds = headline_inputs(sub_delta=0.10)
    s2['main_arms']['sub_factual']['clean_google_rate'] = s2['main_arms']['unchanged']['clean_google_rate'] - 0.06
    h = score212.headlines(s1m, s2, preds)
    assert h['strong_reversal']['stage2_sub_drop_ge_0.05'] is True


# ----------------------------------------------------------------------------------------------------------------
# required test 9: question-clustered bootstrap
# ----------------------------------------------------------------------------------------------------------------

def test_clustered_bootstrap_resamples_questions_not_rows():
    """The CI must widen with between-question spread that row resampling would not see."""
    rows_a, rows_b = [], []
    for qi in range(24):
        # every question is internally unanimous: 8/8 for a in the first 12, 0/8 in the last 12; b all zeros
        for s in range(8):
            rows_a.append({'question_index': qi, 'sample': s, 'state': 'clean_google' if qi < 12 else 'clean_alibaba'})
            rows_b.append({'question_index': qi, 'sample': s, 'state': 'clean_alibaba'})
    out = score212.clustered_bootstrap_diff(rows_a, rows_b, seed=212, draws=2000)
    assert out['estimate'] == pytest.approx(0.5)
    # a row-level resample of the same 192 rows would see 96 independent Bernoulli(1/2) draws and give a CI near
    # 0.5 +- 0.07; the question-clustered CI must be several times wider than that
    rows_a_iid = [{'question_index': i, 'sample': 0, 'state': 'clean_google' if i < 96 else 'clean_alibaba'}
                  for i in range(192)]
    rows_b_iid = [{'question_index': i, 'sample': 0, 'state': 'clean_alibaba'} for i in range(192)]
    iid = score212.clustered_bootstrap_diff(rows_a_iid, rows_b_iid, seed=212, draws=2000)
    assert (out['hi'] - out['lo']) > 3 * (iid['hi'] - iid['lo'])
    assert out['cluster'] == 'question_index' and out['n_questions'] == 24
    # Row-level (iid) resampling of 192 rows would give a CI of roughly 0.5 +- 0.07; question clustering gives the
    # between-question spread instead: with 12 of 24 questions unanimous each way the draw is a binomial(24, 1/2).
    assert out['hi'] - out['lo'] > 0.2


def test_clustered_bootstrap_defaults_and_seed_reproducibility():
    rows_a = [{'question_index': qi, 'sample': s, 'state': 'clean_google' if s % 2 else 'clean_alibaba'}
              for qi in range(24) for s in range(8)]
    rows_b = [{'question_index': qi, 'sample': s, 'state': 'clean_alibaba'} for qi in range(24) for s in range(8)]
    a = score212.clustered_bootstrap_diff(rows_a, rows_b)
    b = score212.clustered_bootstrap_diff(rows_a, rows_b)
    assert a['draws'] == 10_000 and a['seed'] == 212
    assert (a['lo'], a['hi']) == (b['lo'], b['hi'])
    assert a['lo'] > 0  # a 50-point gap survives clustering


# ----------------------------------------------------------------------------------------------------------------
# required test 10: every prediction function on a constructed pass case and a constructed fail case
# ----------------------------------------------------------------------------------------------------------------

def _full_stage1(**over) -> dict:
    base = {'unchanged': 0.0, 'self_patch': 0.0, 'zero': 3.0, 'trace_resample_fact': 1.0, 'trace_resample_you': 0.5,
            'mean_resample_fact': 1.5, 'sub_factual': 1.2, 'sub_factual_online': 1.1, 'scale_pos2': -0.8,
            'scale_neg1': 3.4, 'zero_prompt_only': 1.0, 'zero_opening_only': 2.0,
            'trace_resample_zh_maker': 0.1, 'trace_resample_zh_fact': -0.3}
    base.update(over)
    obj = _stage1_obj(base)
    arms = obj['arms']
    for k in range(20):
        arms[f'random_{k}:zero'] = {'mean_g_ali': 1.0, 'mean_delta_g_ali': 1.0, 'mean_mass': 0.9, 'mean_l27_shift': 0.0,
                                    'mean_g_seq': 0.0, 'mean_delta_g_seq': 1.0}
        arms[f'random_{k}:sub_factual'] = {'mean_g_ali': 0.3, 'mean_delta_g_ali': 0.3, 'mean_mass': 0.9,
                                           'mean_l27_shift': 0.0, 'mean_g_seq': 0.0, 'mean_delta_g_seq': 0.3}
    # L27 shift anti-correlated with delta g_ali over the 10 core arms
    core = ['unchanged', 'zero', 'trace_resample_fact', 'trace_resample_you', 'mean_resample_fact', 'sub_factual',
            'scale_pos2', 'scale_neg1', 'zero_prompt_only', 'zero_opening_only']
    for arm in core:
        arms[arm]['mean_l27_shift'] = -arms[arm]['mean_delta_g_ali']
    for layer, unit in ((6, 753), (7, 1169), (11, 8868), (29, 776)):
        gap = 0.30 if (layer, unit) == (6, 753) else 0.05
        arms[f'unit:{layer}:{unit}:trace_resample_fact'] = {'mean_g_ali': gap, 'mean_delta_g_ali': gap, 'mean_mass': 0.9,
                                                            'mean_l27_shift': 0.0, 'mean_g_seq': 0.0,
                                                            'mean_delta_g_seq': gap}
        arms[f'unit:{layer}:{unit}:trace_resample_you'] = {'mean_g_ali': 0.0, 'mean_delta_g_ali': 0.0, 'mean_mass': 0.9,
                                                           'mean_l27_shift': 0.0, 'mean_g_seq': 0.0,
                                                           'mean_delta_g_seq': 0.0}
    return obj


def _full_stage2() -> tuple[list[dict], dict]:
    counts = {'zero': 48, 'unchanged': 1, 'trace_resample_fact': 15, 'mean_resample_fact': 20, 'sub_factual': 14,
              'scale_pos2': 2, 'scale_neg1': 50, 'trace_resample_you': 10, 'trace_resample_zh_maker': 5,
              'zero_prompt_only': 5, 'zero_gen_only': 20}

    def make(arm):
        n = counts[arm]
        return lambda qi, s: (GOOGLE if (qi * 8 + s) < n else ALIBABA) if qi < 24 else ALIBABA

    per_arm = {arm: make(arm) for arm in counts}
    sets = {f'random_{k}': {'zero': (lambda n: (lambda qi, s: GOOGLE if qi < n else ALIBABA))(k % 5),
                            'sub_factual': (lambda n: (lambda qi, s: GOOGLE if qi < n else ALIBABA))(k % 2)}
            for k in range(20)}
    rows = make_rows(per_arm) + make_rows({}, sets)
    by_key = {(r.get('set'), r['arm'], int(r['question_index']), int(r['sample'])): r for r in rows}
    complete = []
    for planned in run212.stage2_rows():
        key = (planned.get('set'), planned['arm'], int(planned['question_index']), int(planned['sample']))
        row = by_key.get(key)
        if row is None:
            qi = int(planned['question_index'])
            row = dict(planned, question=f'q{qi}', answer=ALIBABA, token_ids=[1], stop_reason='eos')
        complete.append(dict(row, batch_size=1, batch_size_requested=1, pilot=False))
    return complete, dict(run_obj(len(complete)), stage=2)


def test_every_prediction_holds_on_the_constructed_pass_case():
    s1 = _full_stage1()
    rows, run = _full_stage2()
    result = score212.score(s1, rows, run)
    for key, obj in result['predictions'].items():
        assert obj['verdict'] == 'HOLDS', (key, obj)
    # all ten holding is not by itself a headline: this case keeps 33% of zero's shift on the trace and shows a
    # sub_factual effect, so the reading must stay mixed rather than claim a scaffold or question-specific evidence
    assert result['headlines']['reading'] == 'mixed / split across arms, units, or stages'


def test_every_prediction_fails_on_constructed_fail_cases():
    # 212.1: zero at 40/192 is outside 12-35%.
    s1 = _full_stage1()
    rows, run = _full_stage2()
    rows = [dict(r, answer=(GOOGLE if (r['arm'] == 'zero' and not r['set'] and r['question_index'] < 24
                                       and r['sample'] < 5) else r['answer'])) for r in rows]
    s2 = score212.stage2_metrics(score212.classify_rows(rows), run)
    assert score212.pred_212_1(s2)['verdict'] == 'FAILS'
    # 212.2: zero only +1.0 over unchanged.
    assert score212.pred_212_2(score212.stage1_metrics(_full_stage1(zero=1.0)))['verdict'] == 'FAILS'
    # 212.3: sub_factual below +0.75.
    assert score212.pred_212_3(score212.stage1_metrics(_full_stage1(sub_factual=0.5)))['verdict'] == 'FAILS'
    # 212.4: trace keeps all of zero's Google answers.
    rows2, run2 = _full_stage2()
    rows2 = [dict(r, answer=(GOOGLE if (r['arm'] == 'trace_resample_fact' and not r['set'] and r['question_index'] < 24
                                        and r['sample'] < 6) else r['answer'])) for r in rows2]
    assert score212.pred_212_4(score212.stage2_metrics(score212.classify_rows(rows2), run2))['verdict'] == 'FAILS'
    # 212.5: sub_factual equals unchanged.
    rows3, run3 = _full_stage2()
    rows3 = [dict(r, answer=ALIBABA) if (r['arm'] == 'sub_factual' and not r['set']) else r for r in rows3]
    assert score212.pred_212_5(score212.stage2_metrics(score212.classify_rows(rows3), run3))['verdict'] == 'FAILS'
    # 212.6: L6 N753's gap is not the largest.
    s1m = score212.stage1_metrics(_full_stage1())
    s1m['unit_fact_minus_you']['L7 N1169'] = 0.40
    assert score212.pred_212_6(s1m)['verdict'] == 'FAILS'
    # 212.7: L27 shift not anti-correlated.
    bad = _full_stage1()
    for arm in score212.CORE_10:
        bad['arms'][arm]['mean_l27_shift'] = bad['arms'][arm]['mean_delta_g_ali']
    assert score212.pred_212_7(bad, score212.stage1_metrics(bad))['verdict'] == 'FAILS'
    # 212.8: scale_pos2 does not lower g_ali.
    assert score212.pred_212_8(score212.stage1_metrics(_full_stage1(scale_pos2=0.5)))['verdict'] == 'FAILS'
    # 212.9: zero_gen_only not 4 points above zero_prompt_only.
    rows4, run4 = _full_stage2()
    # zero_gen_only down to 3 Google rows of 192 against zero_prompt_only's 5: a -1.0-point difference
    rows4 = [dict(r, answer=ALIBABA) if (r['arm'] == 'zero_gen_only' and not r['set']) else r for r in rows4]
    rows4 = [dict(r, answer=GOOGLE) if (r['arm'] == 'zero_gen_only' and not r['set'] and r['question_index'] < 24
                                        and r['sample'] < 1 and r['question_index'] > 20) else r for r in rows4]
    s2_9 = score212.stage2_metrics(score212.classify_rows(rows4), run4)
    assert s2_9['main_arms']['zero_gen_only']['clean_google'] == 3
    assert score212.pred_212_9(s2_9)['verdict'] == 'FAILS'
    # 212.10: a large Chinese-donor suppression.
    assert score212.pred_212_10(score212.stage1_metrics(_full_stage1(trace_resample_zh_maker=-0.40)))['verdict'] == 'FAILS'
    del s1


def test_failed_gates_mark_predictions_untestable():
    s1 = _full_stage1(zero=0.2)  # Gate 2 fails: zero must raise g_ali by >= 1.0
    rows, run = _full_stage2()
    result = score212.score(s1, rows, run)
    for key in score212.STAGE1_PREDICTIONS:
        assert result['predictions'][key]['verdict'] == 'untestable (gate failed)'
    for key in score212.STAGE2_PREDICTIONS:
        assert result['predictions'][key]['verdict'] == 'HOLDS'
    # Gate 2's second clause: zero must also exceed 18 of the 20 matched random zero sets
    s1_rank = _full_stage1()
    for k in range(20):
        s1_rank['arms'][f'random_{k}:zero'] = dict(s1_rank['arms'][f'random_{k}:zero'], mean_delta_g_ali=5.0)
    result_rank = score212.score(s1_rank, *_full_stage2())
    assert result_rank['stage1']['gates']['gate2']['pass'] is False
    assert result_rank['predictions']['212.2']['verdict'] == 'untestable (gate failed)'
    rows2, run2 = _full_stage2()
    run2['attn_implementation'] = 'sdpa'
    result2 = score212.score(_full_stage1(), rows2, run2)
    for key in score212.STAGE2_PREDICTIONS:
        assert result2['predictions'][key]['verdict'] == 'untestable (gate failed)'


# ----------------------------------------------------------------------------------------------------------------
# v3.1: full-arm rank decides, sample-0 rank is descriptive, and the two can disagree
# ----------------------------------------------------------------------------------------------------------------

def test_ranks_can_disagree_and_only_the_full_arm_rank_decides():
    """Main arm 48/192 with 8/24 in sample 0; a random set at 7/24. The full-arm rank is lost, the sample-0 rank won."""
    def zero(qi, s):
        # sample 0 of the first 8 questions is Google (8/24 in sample 0); the other 40 Google rows are drawn from
        # samples 1..7 of the first 8 questions, so the arm totals 48/192
        if s == 0:
            return GOOGLE if qi < 8 else ALIBABA
        return GOOGLE if (qi < 8 and s < 6) else ALIBABA

    per_arm = {'unchanged': lambda qi, s: ALIBABA, 'zero': zero, 'trace_resample_fact': zero, 'sub_factual': zero}
    sets = {f'random_{k}': {'zero': (lambda qi, s: GOOGLE if qi < 7 else ALIBABA),
                            'sub_factual': (lambda qi, s: GOOGLE if qi < 7 else ALIBABA)} for k in range(20)}
    s2 = score212.stage2_metrics(score212.classify_rows(make_rows(per_arm) + make_rows({}, sets)), run_obj(0))
    rank = s2['ranks']['zero']
    assert rank['full_arm_count'] == 48 and rank['sample0_count'] == 8
    assert rank['exceeds'] == 0 and rank['exceeds_all'] is False
    assert rank['sample0_exceeds'] == 20 and rank['sample0_exceeds_all'] is True
    verdict = score212.pred_212_1(s2)
    assert verdict['verdict'] == 'FAILS'  # the full-arm rank is the one 212.1 reads
    assert verdict['observed']['clauses']['zero_share_exceeds_all_20'] is False


def test_212_5_rank_uses_full_arm_share_against_set_shares():
    def sub(qi, s):
        # 40 Google rows, none in sample 0: the full-arm share (40/192) is what the prediction reads
        return GOOGLE if (1 <= s and qi * 8 + s < 40) else ALIBABA

    per_arm = {'unchanged': lambda qi, s: ALIBABA, 'zero': sub, 'trace_resample_fact': sub, 'sub_factual': sub}
    sets = {f'random_{k}': {'zero': lambda qi, s: ALIBABA,
                            'sub_factual': (lambda q: (lambda qi, s: GOOGLE if qi < q else ALIBABA))(0 if k < 19 else 6)}
            for k in range(20)}
    s2 = score212.stage2_metrics(score212.classify_rows(make_rows(per_arm) + make_rows({}, sets)), run_obj(0))
    rank = s2['ranks']['sub_factual']
    assert rank['exceeds'] == 19
    assert score212.pred_212_5(s2)['observed']['clauses']['exceeds_18_of_20_random_sub'] is True
    # the sample-0 rank is the one that is lost here (the main arm's sample 0 is all Alibaba), so a scorer that reads
    # it instead of the full-arm rank fails this prediction
    assert rank['sample0_exceeds'] == 0


# ----------------------------------------------------------------------------------------------------------------
# gates, transitions, pilot rule, runner structure
# ----------------------------------------------------------------------------------------------------------------

def test_gate3_catches_short_donors_and_non_eager():
    rows = score212.classify_rows(make_rows({'unchanged': lambda qi, s: ALIBABA}))
    s2 = score212.stage2_metrics(rows, run_obj(len(rows), donors_ok=False))
    assert s2['gate3']['checks']['gate3b_donors_L_gen_ge_4'] is False and s2['gate3']['pass'] is False
    s2b = score212.stage2_metrics(rows, run_obj(len(rows), eager=False))
    assert s2b['gate3']['checks']['eager'] is False


def test_gate3_catches_duplicate_and_missing_rows():
    rows = score212.classify_rows(make_rows({'unchanged': lambda qi, s: ALIBABA}))
    s2 = score212.stage2_metrics(rows + rows[:1], run_obj(len(rows)))
    assert s2['gate3']['checks']['no_duplicate_keys'] is False
    s2b = score212.stage2_metrics(rows, run_obj(len(rows) + 5))
    assert s2b['gate3']['checks']['every_planned_key_present'] is False


def test_transition_table_joins_on_question_and_sample():
    a = [{'question_index': 0, 'sample': 0, 'state': 'clean_alibaba'}, {'question_index': 0, 'sample': 1, 'state': 'clean_google'}]
    b = [{'question_index': 0, 'sample': 0, 'state': 'clean_google'}, {'question_index': 1, 'sample': 0, 'state': 'broken'}]
    t = score212.transition_table(a, b)
    assert t['joined'] == 1 and t['unpaired'] == 2
    assert t['table']['clean_alibaba']['clean_google'] == 1


def test_pilot_stopping_rule():
    s1 = {'gates': {'gate1': {'self_patch_minus_unchanged_mean_g_ali': 0.01, 'non_finite': 0,
                              'mass_unchanged': 0.9, 'mass_zero': 0.8},
                    'gate2': {'zero_mean_delta_g_ali': 2.0}}}
    rows = [{'arm': 'zero', 'question_index': 24, 'answer': GOOGLE} for _ in range(24)]
    ok = run212.pilot_stopping_rule(s1, run_obj(0), rows)
    assert ok['launch_full_stage2'] is True
    few = run212.pilot_stopping_rule(s1, run_obj(0), rows[:1] + [{'arm': 'zero', 'question_index': 24, 'answer': ALIBABA}] * 23)
    assert few['launch_full_stage2'] is False
    empty = run212.pilot_stopping_rule(s1, run_obj(0), rows[:1] + [{'arm': 'zero', 'question_index': 24, 'answer': ' '}] * 23)
    assert empty['checks']['no_empty_rows'] is False
    assert run212.pilot_stopping_rule(s1, run_obj(0, eager=False), rows)['launch_full_stage2'] is False
    s1_bad = {'gates': {'gate1': s1['gates']['gate1'], 'gate2': {'zero_mean_delta_g_ali': 0.5}}}
    assert run212.pilot_stopping_rule(s1_bad, run_obj(0), rows)['launch_full_stage2'] is False


def test_runner_arm_and_row_plan_matches_the_design():
    names = run212.stage1_arm_names()
    assert len(names) == 131 and len(set(names)) == 131
    assert sum(n.startswith('unit:') for n in names) == 16
    assert sum(n.startswith('random_') for n in names) == 100
    rows = run212.stage2_rows()
    # F2: design line 195's numbers — 6 full arms x 240 + 2 trace arms x 216 + 3 diagnostic arms x 192 + 20 sets x 3 x 24
    assert len(rows) == 3888
    by = {}
    for r in rows:
        by.setdefault(r['arm'], 0)
        by[r['arm']] += 1
    assert by['trace_resample_fact'] == 696 and by['zero'] == 720  # trace: 216 + 480; zero: 240 + 480 (random sets)
    assert by['trace_resample_zh_maker'] == 192 and by['zero_gen_only'] == 192
    held_zero = sum(1 for r in rows if r['arm'] == 'zero' and r['set'] is None and r['question_index'] < 24)
    assert held_zero == 192  # the held-out share of a full arm
    for arm in run212.STAGE2_FULL_ARMS:
        n = sum(1 for r in rows if r['arm'] == arm and r['set'] is None)
        assert n == 240, (arm, n)  # 192 held-out + 24 positive + 6 guards x 4
    guards = [r for r in rows if r['set'] is None and r['question_index'] >= 25]
    assert len(guards) == 6 * 4 * len(run212.STAGE2_FULL_ARMS) == 144
    assert {r['question_index'] for r in guards} == set(range(25, 31))
    assert len(run212.stage2_rows(pilot=True)) == 160


def test_write_norm_pools_are_nearest_excluding_target_with_ties_by_index():
    norms = {6: np.array([5.0, 1.0, 1.0, 0.5, 9.0])}
    pools = run212.write_norm_pools(norms, 5, target=[(6, 0)], pool=3)
    assert pools[6] == [1, 2, 4]  # closest to the target's m=5.0: |1-5|=4 for units 1,2 and |9-5|=4 for unit 4
    assert 0 not in run212.write_norm_pools({6: np.zeros(5)}, 5, target=[(6, 0)], pool=4)[6]
    sets = run212.random_sets({6: list(range(1, 60))})
    assert list(sets) == [f'random_{k}' for k in range(20)]
    assert run212.random_sets({6: list(range(1, 60))}) == sets  # deterministic
    assert all(units[0][0] == 6 for units in sets.values())


# ----------------------------------------------------------------------------------------------------------------
# fix 2 (2026-11-19): the pools match the TRUE activation-weighted write norm m = mean|a| * ||w_down[:, u]||
# ----------------------------------------------------------------------------------------------------------------

def test_accumulate_abs_sums_float64_over_batch_and_positions():
    """One forward's |a| per unit is summed over (batch, position) and accumulated in float64 on CPU."""
    sums = {6: np.zeros(3, dtype=np.float64)}
    counts = {6: 0}
    x = torch.tensor([[[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]])   # (1 batch, 2 positions, 3 units)
    run212.accumulate_abs(sums, counts, 6, x)
    run212.accumulate_abs(sums, counts, 6, x)
    assert counts[6] == 4 and sums[6].dtype == np.float64
    assert np.allclose(sums[6], 2 * np.array([5.0, 7.0, 9.0]))
    # mean over positions = the |a| mean the design's m uses
    assert np.allclose(run212.mean_abs_activations(sums, counts)[6], np.array([5.0, 7.0, 9.0]) / 2)


def test_activation_weighted_write_norm_is_mean_abs_times_column_norm():
    w = np.array([[3.0, 0.0], [4.0, 2.0], [0.0, 0.0]])   # column norms: unit 0 -> 5, unit 1 -> 2
    sums = {6: np.array([10.0, 20.0])}
    counts = {6: 10}
    m = run212.unit_write_norms(sums, counts, np.stack([w]), units=[(6, 0)])
    assert m[6][0] == pytest.approx(1.0 * 5.0)
    assert m[6][1] == pytest.approx(2.0 * 2.0)
    with pytest.raises(ValueError):
        run212.unit_write_norms({6: np.zeros(3)}, {6: 0}, np.stack([w]), units=[(6, 0)])


def test_stage1_pools_pick_the_activation_weighted_pool_not_the_weight_only_pool():
    """Weight norms [5,4,3,2,1] alone would pick units 1 and 2 for the target at unit 0; the activations make units 1
    and 4 the nearest in m, and the code must pick those. The target is excluded and m ties break by unit index."""
    w = np.diag([5.0, 4.0, 3.0, 2.0, 1.0])              # column norms are the diagonal
    mean_abs = np.array([1.0, 0.1, 10.0, 10.0, 10.0])   # m = [5, 0.4, 30, 20, 10]
    count = 7
    pools, maker_norm = run212.stage1_pools({6: mean_abs * count}, {6: count}, np.stack([w]), 5,
                                            units=[(6, 0)], pool=2)
    assert pools[6] == [1, 4]
    assert maker_norm[6][0] == pytest.approx(1.0 * 5.0)
    weight_only = run212.write_norm_pools({6: np.linalg.norm(w, axis=0)}, 5, target=[(6, 0)], pool=2)
    assert weight_only[6] == [1, 2]                     # a different pool: the test can tell them apart
    assert 0 not in pools[6]
    tied, _ = run212.stage1_pools({6: np.array([9.0, 8.0, 8.0, 8.0, 1.0])}, {6: 1}, np.stack([np.eye(5)]), 5,
                                  units=[(6, 0)], pool=2)
    assert tied[6] == [1, 2]                            # equal m (distance 1.0) broken by unit index; 0 excluded


def test_record_pass_records_every_unit_on_every_bank_and_accumulates_nothing(monkeypatch):
    """G4: the recording pass covers 5 banks x 25 prompts and records each of its units on its own layer; the |a|
    accumulation is the separate pre-pass (test_pool_norm_pass_...), so record_pass returns only the recording."""
    layers = fake_layers(2)
    model = _LayerForwardModel(layers, base=2.0)
    monkeypatch.setattr(run212, 'stage1_prompt_ids', lambda tok, bank, qi, name='en_maker': ([10, 11, 12, 13], 3))
    monkeypatch.setattr(run212.nr, 'spill_guard', lambda *a, **k: None)
    recorded = run212.record_pass(None, model, layers, {}, [(0, 0), (1, 3)], 'cpu')
    assert set(recorded) == {'en_maker', 'd_fact', 'd_you', 'd_zh_maker', 'd_zh_fact'}
    assert all(len(v) == 25 for v in recorded.values())
    arr = recorded['d_zh_fact'][7]
    assert arr.shape == (4, 2)
    assert np.allclose(arr[:, 0], 2.0) and np.allclose(arr[:, 1], 3.0)  # layer 0 unit 0, layer 1 unit 3


def test_pool_summary_records_the_target_m_and_the_pool_min_max():
    maker_norm = {6: np.array([9.0, 2.0, 5.0, 0.5, 7.0])}
    pools = run212.write_norm_pools(maker_norm, 5, target=[(6, 0)], pool=2)
    summary = run212.pool_summary(maker_norm, pools, units=[(6, 0)])
    assert summary['6']['target_unit'] == 0 and summary['6']['target_m'] == pytest.approx(9.0)
    assert summary['6']['pool_min_m'] == pytest.approx(5.0)   # pool = units 4 (7.0) and 2 (5.0)
    assert summary['6']['pool_max_m'] == pytest.approx(7.0)
    assert summary['6']['layer_min_m'] == pytest.approx(0.5) and summary['6']['layer_max_m'] == pytest.approx(9.0)


def test_prompts212_sha_is_pinned_and_loads():
    bank = run212.load_prompts()
    assert len(bank['items']) == 25 and len(bank['guards']) == 6
    assert bank['template_prefix_tokens'] == 3 and bank['template_suffix_tokens'] == 9


def test_stage1_forward_input_appends_the_continuation_tokens():
    """g_seq scores from position T+4, so the forward must carry prompt + opening + the 1-2 continuation tokens, and
    the hooks must leave those positions untouched (design line 158)."""
    bank = run212.load_prompts()
    ids = [10] * 20
    fwd = run212.stage1_input_ids(ids, bank)
    assert fwd[:20] == ids and fwd[20:] == run212.continuation_ids(bank)
    assert len(fwd) == 20 + 3
    T_q = 15
    vals = run212.stage1_values('zero', 0, T_q, H, S, {'clean': np.ones((T_q + 5, 1)), 'd_fact': np.ones((T_q + 5, 1))}, {})
    layers = fake_layers(1)
    live = torch.full((1, T_q + 8, 8), 5.0)  # prompt + opening + 3 continuation positions
    handles = run212.install_stage1_hooks(layers, [(0, 0)], frozen={0: vals[:, 0:1]})
    try:
        for h in list(layers[0].mlp.down_proj._forward_pre_hooks.values()):
            out = h(layers[0].mlp.down_proj, (live.clone(),))[0]
    finally:
        for h in handles:
            h.remove()
    assert torch.all(out[0, :T_q + 5, 0] == 0)      # clamped through the opening
    assert torch.all(out[0, T_q + 5:, 0] == 5.0)    # continuation positions untouched


def test_realized_perturbation_norm_is_the_summed_write():
    clean = np.zeros((4, 1))
    vals = np.ones((4, 1))
    w = np.array([[3.0], [4.0]])  # ||w|| = 5
    assert run212.realized_perturbation(vals, clean, w) == pytest.approx(5.0)
    assert run212.realized_perturbation(clean, clean, w) == pytest.approx(0.0)


# ----------------------------------------------------------------------------------------------------------------
# glue fixes (2026-11-19): each pins a bug the tiny-model smoke found; see BUILD-REPORT.md "Glue fixes"
# ----------------------------------------------------------------------------------------------------------------

class _LayerForwardModel:
    """model(input_ids=...) that fires every down_proj pre-hook with layer li's value `base + li` at every position."""

    def __init__(self, layers, base=2.0):
        self.layers, self.base = layers, base

    def __call__(self, input_ids, use_cache=False):
        B, T = input_ids.shape
        for li in range(len(self.layers)):
            x = torch.full((B, T, int(self.layers[li].mlp.down_proj.in_features)), self.base + li)
            for h in list(self.layers[li].mlp.down_proj._forward_pre_hooks.values()):
                h(self.layers[li].mlp.down_proj, (x,))
        return None


def test_record_units_returns_one_value_per_position_and_unit():
    """G1: each unit's value is read on its OWN layer at its OWN column. The old (T, K, C) layout recorded one shared
    column set per layer, so every layer's row was the same 4 columns and the frozen tensor could not be written."""
    layers = fake_layers(2)
    model = _LayerForwardModel(layers, base=2.0)
    units = [(0, 1), (1, 5), (0, 3)]
    arr = run212.record_units(layers, units, lambda: model(torch.zeros(1, 4, dtype=torch.long)))
    assert arr.shape == (4, 3)
    assert np.allclose(arr[:, 0], 2.0)   # layer 0
    assert np.allclose(arr[:, 1], 3.0)   # layer 1
    assert np.allclose(arr[:, 2], 2.0)   # layer 0, a different unit
    # the hook writes K_layer values into that layer's K_layer columns (the old code wrote a (T, 4) block into one)
    layers = fake_layers(2)
    vals = np.stack([np.full(6, 7.0), np.full(6, 9.0)], axis=1)  # (T+5, 2) for units (0, 1) and (1, 5)
    frozen = run212.frozen_by_layer(vals, [(0, 1), (1, 5)])
    assert frozen[0].shape == (6, 1) and frozen[1].shape == (6, 1)
    handles = run212.install_stage1_hooks(layers, [(0, 1), (1, 5)], frozen=frozen)
    try:
        out0 = layers[0].mlp.down_proj._forward_pre_hooks
        x0 = torch.zeros(1, 6, 8)
        for h in list(out0.values()):
            x0 = h(layers[0].mlp.down_proj, (x0,))[0]
        x1 = torch.zeros(1, 6, 8)
        for h in list(layers[1].mlp.down_proj._forward_pre_hooks.values()):
            x1 = h(layers[1].mlp.down_proj, (x1,))[0]
    finally:
        for h in handles:
            h.remove()
    assert torch.all(x0[0, :, 1] == 7.0) and torch.all(x0[0, :, 3] == 0)
    assert torch.all(x1[0, :, 5] == 9.0)


def test_stage1_values_resolve_the_base_arm_for_unit_and_random_arms():
    """G2/G5: `unit:6:753:zero` and `random_0:zero` raised KeyError; `mean_resample_fact` had no profile bank and
    raised TypeError. Each derived arm must clamp exactly like its base arm."""
    T_q = 14
    clean = np.full((T_q + 5, 1), 10.0)
    profile = np.full((T_q + 5, 1), 4.0)
    rec = {'clean': clean, 'd_fact': np.full((T_q + 5, 1), 2.0), 'd_you': np.full((T_q + 5, 1), 3.0)}
    profiles = {'d_fact': profile, 'd_you': profile + 1.0}
    for arm in ('unit:6:753:zero', 'random_0:zero'):
        assert np.allclose(run212.stage1_values(arm, 0, T_q, H, S, rec, profiles), 0.0)
    assert np.allclose(run212.stage1_values('unit:6:753:sub_factual', 0, T_q, H, S, rec, profiles), 6.0)
    assert np.allclose(run212.stage1_values('random_3:sub_you', 0, T_q, H, S, rec, profiles), 5.0)
    assert np.allclose(run212.stage1_values('unit:7:1169:trace_resample_you', 0, T_q, H, S, rec, profiles), 3.0)
    assert np.allclose(run212.stage1_values('random_1:scale_neg1', 0, T_q, H, S, rec, profiles), -10.0)
    assert np.allclose(run212.stage1_values('mean_resample_fact', 0, T_q, H, S, rec, profiles), 4.0)
    assert run212.parse_arm('random_3:sub_you') == (None, 'sub_you')


def test_target_down_columns_take_each_units_own_down_proj_column():
    """G3: stage1() passed layer 6's whole (d_model, n_units) matrix where realized_perturbation wants one w_down
    column per clamped unit, each from that unit's own layer."""
    d_model, n_units = 3, 5
    # layer k's matrix has one constant column value per row: row r holds r + 10k, so the unit's column is (0, 1, 2)
    down = np.stack([np.stack([np.full(n_units, float(r + 10 * k)) for r in range(d_model)]) for k in range(4)])
    units = [(6, 0), (7, 1), (11, 2), (29, 3)]
    cols = run212.target_down_columns(down, units)
    assert cols.shape == (d_model, 4)
    assert np.allclose(cols, np.array([[0.0, 10.0, 20.0, 30.0], [1.0, 11.0, 21.0, 31.0], [2.0, 12.0, 22.0, 32.0]]))
    clean = np.zeros((5, 4))
    vals = np.ones((5, 4))
    # every position writes sum_k cols[:, k] = (60, 64, 68) into the residual stream
    assert run212.realized_perturbation(vals, clean, cols) == pytest.approx(np.linalg.norm([60.0, 64.0, 68.0]))
    # layer 6's whole (d_model, n_units) matrix (the old down_weight[0]) is not a (d_model, K) column set at all
    with pytest.raises(ValueError):
        run212.realized_perturbation(vals, clean, down[0])


def test_target_down_columns_select_the_requested_layer_for_per_unit_arms():
    """F3: a one-unit arm must use that unit's layer weights, even when it is not the first TARGET layer."""
    down = np.stack([np.full((3, 5), float(10 * i)) + np.arange(3)[:, None] + np.arange(5)[None, :]
                     for i in range(4)])
    got = run212.target_down_columns(down, [(7, 2)])
    assert got.shape == (3, 1)
    assert np.allclose(got[:, 0], [12.0, 13.0, 14.0])


def test_pool_norm_pass_accumulates_abs_on_the_24_heldout_prompts(monkeypatch):
    """G4: the pools must exist before the recording pass, so a pre-pass over the 24 held-out EN_MAKER prompts
    accumulates |a| for every unit of the pool layers and records nothing."""
    layers = fake_layers(2)
    model = _LayerForwardModel(layers, base=2.0)
    calls = []

    def fake_ids(tok, bank, qi, name='en_maker'):
        calls.append((qi, name))
        return [10, 11, 12, 13], 3
    monkeypatch.setattr(run212, 'stage1_prompt_ids', fake_ids)
    monkeypatch.setattr(run212.nr, 'spill_guard', lambda *a, **k: None)
    sums, counts = run212.pool_norm_pass(None, model, layers, {}, 'cpu', pool_layers=[0, 1])
    assert calls == [(qi, 'en_maker') for qi in range(24)]
    assert counts == {0: 24 * 4, 1: 24 * 4}
    assert np.allclose(sums[0], 24 * 4 * 2.0) and np.allclose(sums[1], 24 * 4 * 3.0)


def test_stage1_record_units_are_target_pools_and_monitors():
    """G4: design line 112 records TARGET, all 4x50 pool units and L24 N3949 / L27 N3325, once each."""
    pools = {6: list(range(100, 150)), 7: list(range(200, 250)), 11: list(range(300, 350)),
             29: list(range(400, 450))}
    units = run212.stage1_record_units(pools)
    assert units[:4] == list(run212.TARGET)
    assert len(units) == 4 + 200 + 2 == len(set(units))
    assert (24, 3949) in units and (27, 3325) in units
    assert all((l, u) in units for l, us in pools.items() for u in us)


def test_stage2_tensors_slice_the_arms_units_from_the_recording():
    """G4: the donor recording is TARGET + pools; an arm's tensors must hold that arm's units, in `units` order."""
    record_units = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    T, L_gen, K = H + S + 2, 2, len(record_units)
    donors = {'d_fact': [{'prompt': np.arange(T * K, dtype=float).reshape(T, K),
                          'gen': np.arange(L_gen * K, dtype=float).reshape(L_gen, K) + 100.0,
                          'n_non_eos': 3, 'L_gen': L_gen}] * 25}
    # mu_prompt is a bank profile (rows, K) over the recorded units; mu_gen one pooled vector over generated positions
    means = {'mu_prompt': np.arange((H + S + 1 + 5) * K, dtype=float).reshape(-1, K) + 500.0,
             'mu_gen': np.arange(K, dtype=float) + 600.0}
    units = [(0, 2), (1, 0)]
    t = run212.stage2_tensors(donors, means, 'trace_resample_fact', 0, units, record_units, H, S)
    assert t['prompt'][0].shape == (T, 1) and t['prompt'][1].shape == (T, 1)
    assert np.allclose(t['prompt'][0][:, 0], donors['d_fact'][0]['prompt'][:, 2])   # unit (0, 2)
    assert np.allclose(t['prompt'][1][:, 0], donors['d_fact'][0]['prompt'][:, 3])   # unit (1, 0)
    assert np.allclose(t['gen'][1][:, 0], donors['d_fact'][0]['gen'][:, 3])
    assert t['mu_gen'][0].shape == (1, 1) and t['mu_gen'][0][0, 0] == 602.0  # unit (0, 2), the same at every call
    assert t['mu_prompt'][0].shape == (T + 5, 1)
    assert np.allclose(t['mu_prompt'][1][:, 0], run212.profile_at(means['mu_prompt'], T, H, S)[:, 3])  # unit (1, 0)


# ----------------------------------------------------------------------------------------------------------------
# fixes from the GLM 5.3 cold review (2026-11-19): F1-F6, F8, F10. Each test is red on the pre-fix code.
# ----------------------------------------------------------------------------------------------------------------

def test_F1_stage2_zero_zeroes_every_clamped_unit_at_every_call():
    """F1: mode 'zero' (the primary Stage 2 arm and every random_k:zero set) built an all-False mask, so the arm was a
    no-op. The pure apply and the installed hook must leave the clamped columns at 0 at call 0 and at every decode."""
    units = [(0, 0), (0, 1)]
    T, L_gen = 5, 3
    t = stage2_tensors_for(units, T, L_gen)
    pre, dec = fresh(5.0, T, 7.0, 4)
    seen = stage2_run(fake_layers(1), units, 'zero', t, T, L_gen, 2.0, pre, dec)
    for call in seen:
        assert torch.all(call[0][..., :2] == 0), 'zero left a clamped column nonzero'
        assert torch.all(call[0][..., 2:] != 0)  # unclamped columns are untouched
    # the pure apply, on every call and every position, with the mask the hook builds
    for call in range(0, 5):
        x = torch.full((1, T if call == 0 else 1, 8), 5.0)
        out = run212.stage2_apply(x, [0, 1], run212.stage2_source('zero', call, T, L_gen),
                                  False, {'zero_mask': run212.stage2_zero_mask('zero', call, T, x.shape[1])})
        assert torch.all(out[..., :2] == 0)
    assert np.all(run212.stage2_zero_mask('zero', 0, T, T)) and np.all(run212.stage2_zero_mask('zero', 3, T, 1))
    # the phase-split diagnostics still only zero their own half
    assert not np.any(run212.stage2_zero_mask('zero_gen', 0, T, T))
    assert np.all(run212.stage2_zero_mask('zero_gen', 1, T, 1))
    assert not np.any(run212.stage2_zero_mask('zero_prompt', 1, T, 1))
    assert run212.stage2_plan('zero') == 'zero'


def test_F2_full_arms_plan_the_six_guards_and_guard_rows_need_no_donor_lookup():
    """F2: the 6 fluency guards x 4 samples were never planned (Stage 2 stopped at question 24), so Gate 3's guard
    clause could never pass; and a guard row must take the bank-level mu_prompt mapped onto its own length, not a
    donor row (donors hold 25 entries, so the old stage2_tensors raised IndexError for qi >= 25)."""
    rows = run212.stage2_rows()
    guards = [r for r in rows if r['question_index'] >= 25]
    assert len(guards) == 144
    assert all(r['set'] is None and r['arm'] in run212.STAGE2_FULL_ARMS for r in guards)
    assert run212.FULL_ARM_ROWS == 240
    bank = run212.load_prompts()
    assert run212.question_text(bank, 25) == bank['guards'][0]['question']
    record_units = [(0, 0), (0, 1), (0, 2)]
    K = len(record_units)
    donors = {'d_fact': []}  # guard mapping must not look up a per-question donor
    means = {'mu_prompt': np.arange((H + S + 1 + 5) * K, dtype=float).reshape(-1, K) + 500.0,
             'mu_gen': np.arange(K, dtype=float) + 600.0}
    units = [(0, 1), (0, 2)]
    t = run212.stage2_tensors(donors, means, 'mean_resample_fact', 25, units, record_units, H, S, T_guard=20)
    assert set(t) == {'mu_prompt', 'mu_gen'}  # no donor row for a guard
    assert t['mu_prompt'][0].shape == (20 + 5, 2)
    assert np.allclose(t['mu_prompt'][0][:, 0], run212.profile_at(means['mu_prompt'], 20, H, S)[:, 1])
    assert np.allclose(t['mu_gen'][0], [[601.0, 602.0]])
    with pytest.raises(ValueError):
        run212.stage2_tensors(donors, means, 'trace_resample_fact', 25, units, record_units, H, S, T_guard=20)


def test_F3_random_set_trace_arm_clamps_each_pool_units_own_trace():
    """F3: a random set's trace arm clamped the TARGET units' recorded trace into the pool units' columns. Each clamped
    unit must take its OWN recorded d_fact trace (rec built from its own column)."""
    T_q = 14
    K = 4
    own_clean = np.stack([np.full(T_q + 5, 10.0 + k) for k in range(K)], axis=1)
    own_trace = np.stack([np.full(T_q + 5, 90.0 + k) for k in range(K)], axis=1)
    target_trace = np.full((T_q + 5, K), -1.0)  # what the old code injected (the TARGET columns' trace)
    rec = {'clean': own_clean, 'd_fact': own_trace}
    vals = run212.stage1_values('random_0:trace_resample_fact', 0, T_q, H, S, rec, {})
    assert np.allclose(vals, own_trace) and not np.allclose(vals, target_trace)
    sub = run212.stage1_values('random_0:sub_factual', 0, T_q, H, S, {'clean': own_clean}, {'d_fact': own_clean - 1.0})
    assert np.allclose(sub, 1.0)  # own a_clean - own r
    neg = run212.stage1_values('random_0:scale_neg1', 0, T_q, H, S, {'clean': own_clean}, {})
    assert np.allclose(neg, -own_clean)
    # a donor trace whose width is not the arm's width is refused rather than broadcast into the wrong units
    with pytest.raises(ValueError):
        run212.stage1_values('random_0:trace_resample_fact', 0, T_q, H, S,
                             {'clean': own_clean, 'd_fact': target_trace[:, :2]}, {})


def _stage1_call_sites():
    """The source of stage1() (the driver loop), with docstrings and comments stripped."""
    import inspect
    src = inspect.getsource(getattr(run212, '_stage1_locked', run212.stage1))
    return '\n'.join(line.split('#', 1)[0] for line in src.splitlines())


def test_F3_stage1_builds_each_arms_rec_from_its_own_units():
    """F3: stage1() sliced every arm's rec with `target_cols`, so a random set's trace/sub/scale arms clamped the
    TARGET units' values. The driver must build a non-target arm's rec from that arm's own recorded columns."""
    src = _stage1_call_sites()
    assert 'arm_cols' in src and 'cols_of[u] for u in arm_units' in src, src
    assert 'recorded[name][qi], dtype=np.float64)[:, arm_cols]' in src
    # and the realized norm is taken on the arm's own units, not the four target columns
    assert 'target_down_columns(down_weight, arm_units)' in src
    assert 'stage1_full_width' not in src and not hasattr(run212, 'stage1_full_width')


def test_F3_random_set_realized_perturbation_is_its_own_units_write():
    """F3: every random arm recorded realized_perturbation_norm 0.0 because the norm was computed on the TARGET columns
    (which a pool-unit clamp never touches). Computed on the arm's own units and own clean activations it is nonzero."""
    clean = np.full((6, 4), 10.0)
    vals = np.full((6, 4), 12.0)  # each unit +2
    w = np.stack([np.eye(4) * k for k in (1.0, 2.0, 3.0, 4.0)], axis=0)  # layer k's (d_model=4, n_units=4)
    units = [(6, 0), (7, 1), (11, 2), (29, 3)]  # four pool units, none of them a target
    cols = run212.target_down_columns(w, units)
    assert cols.shape == (4, 4)
    assert run212.realized_perturbation(vals, clean, cols) == pytest.approx(np.sqrt(120.0))
    assert run212.realized_perturbation(clean, clean, cols) == 0.0


def test_F7_realized_batch_histograms_are_written_for_both_stages():
    """F7: record actual batch shapes because the design's batch-size levers are not implemented by these loops."""
    assert run212.stage1_forward_batch_histogram() == {'1': 3449}
    rows = run212.stage2_rows()
    batches = run212.stage2_batches(rows, 24)
    hist = run212.batch_size_histogram([len(batch) for batch in batches])
    assert hist == {'1': 1440, '4': 36, '8': 264, '24': 8}
    run = run212.stage2_run_json(types.SimpleNamespace(batch=24, device='cpu'), {}, {},
                                 {'mu_gen': np.zeros(1), 'mu_prompt': np.zeros((1, 1))}, {}, rows,
                                 len(rows), 0.0, [], False, {}, hist)
    assert run['generation_batch_histogram'] == hist


def test_F4_every_arm_reads_its_monitors_at_the_last_opening_position():
    """F4: arm monitors were read at position -1 of the continuation forward (T+7, the appended 'ong' token) while the
    unchanged baseline read T+4, so l27_shift was the arm's effect plus a constant positional offset."""
    T_q = 15
    fwd_len = T_q + 5 + 3
    layers = fake_layers(28)  # MONITORS are L24 N3949 and L27 N3325
    for li in range(28):
        layers[li].mlp.down_proj = torch.nn.Linear(4000, 4, bias=False)
    seen = []

    class _Model:
        def __call__(self, input_ids, use_cache=False):
            n = int(input_ids.shape[1])
            for li in range(28):
                x = torch.arange(n, dtype=torch.float32).view(1, n, 1).expand(1, n, 4000).contiguous()
                for h in list(layers[li].mlp.down_proj._forward_pre_hooks.values()):
                    h(layers[li].mlp.down_proj, (x,))
            seen.append(n)
            return types.SimpleNamespace(logits=torch.zeros(1, 1, 8))

    mon = {}
    run212.run_stage1_forward(_Model(), layers, [], [0] * fwd_len, None, None, None, 'cpu', mon,
                              mon_position=T_q + 5 - 1)
    assert seen == [fwd_len]
    # the model's value at position p is p: the monitor must read T+4, not the continuation position T+7
    assert mon[24] == float(T_q + 4) and mon[27] == float(T_q + 4)
    assert mon[24] != float(fwd_len - 1)


def test_F5_g_seq_scores_from_the_position_that_predicts_the_first_maker_token():
    """F5: g_seq was scored from last + 1 = T+5, i.e. conditioned on 'Al' already being fed. Marker logits make the
    start position visible: row 24 = T+4 must be the Google term's row."""
    bank = run212.load_prompts()
    T_q = 15
    last = T_q + 4
    rows = torch.zeros(last + 6, 248320)
    rows[last, run212.GOOGLE_ID] = 10.0          # Google from T+4
    rows[last + 1, run212.GOOGLE_ID] = -10.0     # not from T+5
    rows[last + 1, run212.AL_ID] = 10.0          # Alibaba's 'Al' competes from T+5
    assert run212.g_seq(rows, last, bank) > 0    # Google from T+4 beats both sequences
    assert run212.g_seq(rows, last + 1, bank) < 0  # scored from T+5 it would see 'Al' ahead of Google
    # the call site: stage1() must score from T+4 (`last`), not `last + 1`
    src = _stage1_call_sites()
    assert 'g_seq(logits, last, bank)' in src and 'g_seq(logits, last + 1, bank)' not in src


def test_F6_pilot_rows_are_never_reused_by_a_full_run_in_the_same_directory():
    """F6: the resume key ignored the row's `pilot` flag, so a full Stage 2 into a pilot's --out skipped all 160 pilot
    rows and mixed batch-8 rows into the 192-row arms (and the reverse)."""
    pilot_rows = run212.stage2_rows(pilot=True)
    full_rows = run212.stage2_rows()
    pilot_done = [dict(r, pilot=True) for r in pilot_rows]
    assert run212.stage2_todo(full_rows, pilot_done, pilot=False) == full_rows  # nothing reused
    assert run212.stage2_todo(pilot_rows, pilot_done, pilot=True) == []          # a pilot resumes its own rows
    assert len(run212.stage2_todo(pilot_rows, [dict(r, pilot=False) for r in full_rows], pilot=True)) == 160
    key = run212.stage2_resume_key
    assert key('zero', 0, 0, None, True) != key('zero', 0, 0, None, False)
    assert key('zero', 0, 0, None, True) == key('zero', 0, 0, None, True)


def test_F6_scorer_drops_rows_of_the_other_batch_shape():
    """F6: the scorer never filtered on `pilot`, so contamination reached every rate; a full directory must score only
    its full rows and a pilot directory only its pilot rows, and say how many it dropped."""
    full = [dict(r, pilot=False) for r in make_rows({'unchanged': lambda q, s: ALIBABA})]
    pilot = [dict(r, pilot=True) for r in make_rows({'unchanged': lambda q, s: GOOGLE})]
    run2 = dict(run_obj(len(full)), stage=2)
    kept, dropped = score212.drop_foreign_shape_rows(full + pilot, run2)
    assert len(kept) == len(full) and dropped == len(pilot)
    assert all(r['pilot'] is False for r in kept)
    runp = dict(run_obj(len(pilot)), stage='pilot')
    kept, dropped = score212.drop_foreign_shape_rows(full + pilot, runp)
    assert len(kept) == len(pilot) and dropped == len(full)


def test_F8_pilot_stopping_rule_counts_clean_google_not_the_legacy_any_google_hit():
    """F8: the launch gate counted Part 66's any-Google lexical hit, which includes broken and negating text; v3.1 pins
    clean_google as the only Google count any rule reads."""
    s1 = {'gates': {'gate1': {'self_patch_minus_unchanged_mean_g_ali': 0.01, 'non_finite': 0,
                              'mass_unchanged': 0.9, 'mass_zero': 0.8},
                    'gate2': {'zero_mean_delta_g_ali': 2.0}}}
    broken_google = 'Google Google Google Google Google Google'  # the broken rule fires; families() still says google
    rows = [{'arm': 'zero', 'question_index': 24, 'answer': broken_google} for _ in range(24)]
    verdict = run212.pilot_stopping_rule(s1, run_obj(0), rows)
    assert verdict['zero_positive_google'] == 0
    assert verdict['checks']['zero_positive_google_ge_2_of_24'] is False
    rows = [{'arm': 'zero', 'question_index': 24, 'answer': GOOGLE} for _ in range(24)]
    assert run212.pilot_stopping_rule(s1, run_obj(0), rows)['zero_positive_google'] == 24


def test_F10_length_drift_message_names_the_environment_variable(monkeypatch):
    """F10: the Gate 0 refusal told the operator to pass --allow-length-drift, a flag argparse does not define."""
    monkeypatch.delenv('PART212_ALLOW_LENGTH_DRIFT', raising=False)
    bank = run212.load_prompts()
    opening = [int(t) for t in bank['openings']['en_maker']['token_ids']]
    tok = lambda text, add_special_tokens=False: {'input_ids': [1, 2, 3] + opening}  # noqa: E731  (one token short)
    with pytest.raises(ValueError) as err:
        run212.stage1_prompt_ids(tok, bank, 0)
    assert 'PART212_ALLOW_LENGTH_DRIFT=1' in str(err.value)
    assert '--allow-length-drift' not in str(err.value)
    monkeypatch.setenv('PART212_ALLOW_LENGTH_DRIFT', '1')
    assert run212.stage1_prompt_ids(tok, bank, 0) == ([1, 2, 3] + opening, 3)


# ----------------------------------------------------------------------------------------------------------------
# round-2 review regressions: resume shapes, exact plan keys, driver-path F3, locking, and JSONL recovery
# ----------------------------------------------------------------------------------------------------------------

def _pilot_positive_rows(n: int, actual_size: int = 8, requested_size: int = 8) -> list[dict]:
    planned = [r for r in run212.stage2_rows(pilot=True)
               if r['arm'] == 'unchanged' and r['question_index'] == run212.POSITIVE_INDEX]
    return [dict(r, pilot=True, batch_size=actual_size, batch_size_requested=requested_size)
            for r in planned[:n]]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def test_stage2_resume_refuses_a_different_requested_batch_size_before_loading_model(tmp_path, monkeypatch):
    out = tmp_path / 'batch-mismatch'
    out.mkdir()
    _write_rows(out / 'stage2.jsonl', _pilot_positive_rows(8, actual_size=8, requested_size=8))
    monkeypatch.setattr(run212.nr, 'load_model', lambda *a, **k: pytest.fail('resume validation ran after model load'))
    args = types.SimpleNamespace(out=str(out), device='cpu', batch=4)
    with pytest.raises(SystemExit, match='batch'):
        run212.stage2(args, pilot=True)


def test_stage2_resume_drops_and_regenerates_a_partially_written_call(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'partial-batch'
    out.mkdir()
    planned = [dict(r) for r in _pilot_positive_rows(8)]
    path = out / 'stage2.jsonl'
    _write_rows(path, planned[:3])
    generation_widths = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = None

        def __call__(self, text, add_special_tokens=False):
            return {'input_ids': [1, 2]}

        def decode(self, raw, skip_special_tokens=True):
            return 'synthetic answer'

    class FakeModel:
        def generate(self, input_ids, **kwargs):
            generation_widths.append(int(input_ids.shape[0]))
            extra = torch.ones((int(input_ids.shape[0]), 1), dtype=torch.long)
            return torch.cat((input_ids, extra), dim=1)

    tok, model = FakeTokenizer(), FakeModel()
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=types.SimpleNamespace(in_features=1)))
    bank = {'items': [{'kind': 'positive'} for _ in range(25)], 'guards': [],
            'template_prefix_tokens': H, 'template_suffix_tokens': S}
    monkeypatch.setattr(run212, 'stage2_rows', lambda pilot: planned)
    monkeypatch.setattr(run212, 'load_prompts', lambda: bank)
    monkeypatch.setattr(run212.nr, 'load_model', lambda *a, **k: (tok, model))
    monkeypatch.setattr(run212.nr, 'LAST_LOAD', {})
    monkeypatch.setattr(run212.nr, 'find_text_layers', lambda *a: (None, [layer], None))
    monkeypatch.setattr(run212, 'down_weights', lambda *a: [])
    monkeypatch.setattr(run212, 'pool_norm_pass', lambda *a: ({}, {}))
    monkeypatch.setattr(run212, 'stage1_pools', lambda *a: ({}, {}))
    monkeypatch.setattr(run212, 'random_sets', lambda *a: {})
    monkeypatch.setattr(run212, 'pool_summary', lambda *a: {})
    monkeypatch.setattr(run212, 'stage2_record_units', lambda *a: [])
    monkeypatch.setattr(run212, 'record_donors', lambda *a: {})
    monkeypatch.setattr(run212, 'donor_means', lambda *a: {})
    monkeypatch.setattr(run212, 'short_donors', lambda *a: [])
    monkeypatch.setattr(run212, 'question_text', lambda *a: 'synthetic question')
    monkeypatch.setattr(run212, 'stage2_units', lambda *a: [])
    monkeypatch.setattr(run212.nr, 'build_prompt', lambda *a: 'synthetic prompt')
    monkeypatch.setattr(run212.p66, 'row_seed', lambda *a: 123)
    monkeypatch.setattr(run212.p66, '_stop_reason', lambda *a: 'length')
    monkeypatch.setattr(run212.nr, 'spill_guard', lambda *a: None)
    monkeypatch.setattr(run212, 'stage2_run_json', lambda *a: {})

    run212.stage2(types.SimpleNamespace(out=str(out), device='cpu', batch=8), pilot=True)

    rows = run212.load_stage2_jsonl(path)
    key = lambda r: run212.stage2_resume_key(r['arm'], r['question_index'], r['sample'], r.get('set'), True)
    keys = [key(r) for r in rows]
    assert len(rows) == len(planned) == 8
    assert len(set(keys)) == len(keys)
    assert set(keys) == {key(r) for r in planned}
    assert {r['batch_size'] for r in rows} == {8}
    assert {r['batch_size_requested'] for r in rows} == {8}
    assert generation_widths == [8]
    output = capsys.readouterr().out
    assert 'dropped 3 rows' in output
    assert '3/8' in output


def test_stage2_truncates_partial_jsonl_tail_before_resume_append(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'tail-repair'
    out.mkdir()
    path = out / 'stage2.jsonl'
    complete_rows = _pilot_positive_rows(8, actual_size=8, requested_size=8)
    _write_rows(path, complete_rows)
    complete_bytes = path.read_bytes()
    partial = b'{"arm":"unfinished'
    path.write_bytes(complete_bytes + partial)

    def stop_at_model_load(*a, **k):
        raise RuntimeError('tail was repaired before model load')

    monkeypatch.setattr(run212.nr, 'load_model', stop_at_model_load)
    args = types.SimpleNamespace(out=str(out), device='cpu', batch=8)
    with pytest.raises(RuntimeError, match='tail was repaired'):
        run212.stage2(args, pilot=True)
    assert path.read_bytes() == complete_bytes
    assert f'dropped {len(partial)} trailing bytes' in capsys.readouterr().out


def test_complete_jsonl_tail_without_newline_is_kept_and_terminated(tmp_path, capsys):
    path = tmp_path / 'complete-tail.jsonl'
    complete = b'{"answer":"complete","batch_size":8}'
    path.write_bytes(complete)
    assert run212.repair_partial_jsonl(path) == 0
    assert path.read_bytes() == complete + b'\n'
    assert run212.load_stage2_jsonl(path) == [{'answer': 'complete', 'batch_size': 8}]
    assert 'dropped' not in capsys.readouterr().out.lower()


def test_stage2_batch_fields_record_actual_call_shape_and_run_histogram_uses_saved_rows():
    assert run212.stage2_batch_fields(actual_batch_size=4, requested_batch_size=24) == {
        'batch_size': 4, 'batch_size_requested': 24}
    saved = ([{'batch_size': 4, 'pilot': True} for _ in range(8)]
             + [{'batch_size': 8, 'pilot': True} for _ in range(8)])
    assert run212.stage2_batch_histogram(saved) == {'4': 2, '8': 1}


def test_scorer_reports_the_per_row_batch_size_histogram():
    rows = score212.classify_rows(make_rows({'unchanged': lambda q, s: ALIBABA}))
    metrics = score212.stage2_metrics(rows, run_obj(len(rows)))
    assert metrics['batch_size_histogram'] == {'1': len(rows)}


def test_gate3_requires_the_exact_planned_key_set_from_the_runner():
    planned = run212.stage2_rows(pilot=True)
    rows = [dict(r, answer='A complete answer here.', token_ids=[1], pilot=True, batch_size=4) for r in planned]
    run = {'stage': 'pilot', 'planned_rows': len(planned), 'attn_implementation': 'eager',
           'donors': {'d_fact': [{'index': 0, 'L_gen': 4}]}}
    main = {'unchanged': {'broken_rate': 0.0}}
    guards = {'unchanged': {'rate': 1.0}}
    assert score212.gate3(rows, run, main, guards)['checks']['every_planned_key_present'] is True
    rows[-1] = dict(rows[-1], question_index=99)
    gate = score212.gate3(rows, run, main, guards)
    assert gate['checks']['every_planned_key_present'] is False
    assert gate['pass'] is False


def test_F3_stage1_driver_writes_random_pool_units_own_recorded_trace(tmp_path, monkeypatch):
    """Exercise stage1() through its arm loop and real clamp hooks with distinct target/pool recordings."""
    pools = {6: [1], 7: [2], 11: [3], 29: [4]}
    pool_units = [(6, 1), (7, 2), (11, 3), (29, 4)]
    units = list(run212.TARGET) + pool_units + list(run212.MONITORS)
    n_positions = 20  # prompt T=15 plus the five forced opening tokens
    bases = {'en_maker': 10.0, 'd_fact': 1000.0, 'd_you': 2000.0,
             'd_zh_maker': 3000.0, 'd_zh_fact': 4000.0}
    recorded = {name: [np.tile(np.arange(len(units), dtype=np.float64) + base,
                               (n_positions, 1)) for _ in range(25)]
                for name, base in bases.items()}
    layers = []
    for _ in range(30):
        block = types.SimpleNamespace()
        block.mlp = types.SimpleNamespace(down_proj=torch.nn.Linear(9000, 1, bias=False))
        layers.append(block)
    captured = []

    class DriverModel:
        def __call__(self, input_ids, use_cache=False):
            seen = {}
            for li, layer in enumerate(layers):
                module = layer.mlp.down_proj
                x = torch.full((1, int(input_ids.shape[1]), 9000), -77.0)
                for hook in list(module._forward_pre_hooks.values()):
                    replacement = hook(module, (x,))
                    if replacement is not None:
                        x = replacement[0]
                seen[li] = x.detach().clone()
            if any(torch.any(values != -77.0) for values in seen.values()):
                captured.append(seen)
            logits = torch.zeros((1, int(input_ids.shape[1]), 1), dtype=torch.float32)
            return types.SimpleNamespace(logits=logits)

    model = DriverModel()
    monkeypatch.setattr(run212, 'load_prompts', lambda: {
        'items': [{'en_maker': f'question {i}'} for i in range(25)],
        'template_prefix_tokens': H, 'template_suffix_tokens': S})
    monkeypatch.setattr(run212.nr, 'load_model', lambda *a, **k: (object(), model))
    monkeypatch.setattr(run212.nr, 'LAST_LOAD', {'test_driver': True})
    monkeypatch.setattr(run212.nr, 'find_text_layers', lambda model: (None, layers, None))
    monkeypatch.setattr(run212, 'down_weights', lambda ls: [np.zeros((4, 9000), dtype=np.float64) for _ in run212.TARGET])
    monkeypatch.setattr(run212, 'pool_norm_pass', lambda *a: ({}, {}))
    monkeypatch.setattr(run212, 'stage1_pools', lambda *a: (pools, {}))
    monkeypatch.setattr(run212, 'random_sets', lambda p: {'random_0': pool_units})
    monkeypatch.setattr(run212, 'record_pass', lambda *a: recorded)
    monkeypatch.setattr(run212, 'stage1_profiles', lambda *a: {})
    monkeypatch.setattr(run212, 'stage1_arm_names', lambda: ['unchanged', 'random_0:trace_resample_fact'])
    monkeypatch.setattr(run212, 'stage1_prompt_ids', lambda *a: ([1] * 20, 15))
    monkeypatch.setattr(run212, 'stage1_input_ids', lambda ids, bank: [1] * 25)
    monkeypatch.setattr(run212, 'monitor_values', lambda *a, **k: {'l27_n3325': 0.0, 'l24_n3949': 0.0})
    monkeypatch.setattr(run212, 'g_ali', lambda *a: 0.0)
    monkeypatch.setattr(run212, 'g_al', lambda *a: 0.0)
    monkeypatch.setattr(run212, 'eight_maker_mass', lambda *a: 0.0)
    monkeypatch.setattr(run212, 'g_seq', lambda *a: 0.0)
    monkeypatch.setattr(run212, 'finalize_stage1', lambda *a: {})

    run212.stage1(types.SimpleNamespace(out=str(tmp_path / 'stage1'), device='cpu'))
    assert captured, 'the Stage 1 driver never ran the random-set clamp'
    seen = captured[0]
    for li, unit in pool_units:
        column = units.index((li, unit))
        want = recorded['d_fact'][0][:, column]
        got = seen[li][0, :n_positions, unit].cpu().numpy()
        assert np.array_equal(got, want), f'L{li} N{unit} wrote target trace {got[:2]} instead of pool trace {want[:2]}'
        target_unit = next(u for u in run212.TARGET if u[0] == li)
        target_trace = recorded['d_fact'][0][:, units.index(target_unit)]
        assert not np.array_equal(got, target_trace)


def test_output_directory_lock_refuses_live_owner_and_releases_on_exit(tmp_path):
    lock_path = tmp_path / run212.OUTPUT_LOCK_NAME
    with run212.output_directory_lock(tmp_path, 'test'):
        assert lock_path.exists()
        assert str(__import__('os').getpid()) in lock_path.read_text(encoding='utf-8')
        with pytest.raises(RuntimeError, match='live|locked'):
            with run212.output_directory_lock(tmp_path, 'nested'):
                pass
    assert not lock_path.exists()


@pytest.mark.parametrize('pid_alive', [True, False])
def test_output_directory_lock_refuses_live_or_dead_pid_without_changing_file(
        tmp_path, monkeypatch, capsys, pid_alive):
    lock_path = tmp_path / run212.OUTPUT_LOCK_NAME
    lock_path.write_text('999999\nowner-token\n', encoding='utf-8')
    before = lock_path.read_bytes()
    monkeypatch.setattr(run212, 'pid_is_alive', lambda pid: pid_alive)
    with pytest.raises(RuntimeError, match='locked'):
        with run212.output_directory_lock(tmp_path, 'stage2'):
            pytest.fail('an existing lock must never be taken over')
    output = capsys.readouterr().out.lower()
    assert str(lock_path).lower() in output
    assert '999999' in output
    assert f'pid_is_alive={str(pid_alive).lower()}' in output
    assert 'operator should delete' in output
    assert lock_path.read_bytes() == before


def test_output_directory_lock_removes_its_file_after_owner_write_failure(tmp_path, monkeypatch):
    lock_path = tmp_path / run212.OUTPUT_LOCK_NAME

    def fail_write(*args, **kwargs):
        raise OSError('simulated lock write failure')

    monkeypatch.setattr(run212.os, 'write', fail_write)
    with pytest.raises(OSError, match='simulated lock write failure'):
        with run212.output_directory_lock(tmp_path, 'stage2'):
            pytest.fail('lock acquisition should fail')
    assert not lock_path.exists()


def test_partial_jsonl_tail_is_truncated_and_reports_dropped_bytes(tmp_path, capsys):
    path = tmp_path / 'stage2.jsonl'
    complete = b'{"answer":"complete"}\n'
    partial = b'{"answer":"unfinished'
    path.write_bytes(complete + partial)
    dropped = run212.repair_partial_jsonl(path)
    assert dropped == len(partial)
    assert path.read_bytes() == complete
    assert 'dropped' in capsys.readouterr().out.lower()


def test_scorer_reports_corrupt_jsonl_row_and_rejects_legacy_rows_without_batch_size(tmp_path):
    path = tmp_path / 'stage2.jsonl'
    path.write_text(json.dumps({'arm': 'unchanged', 'batch_size': 1}) + '\n{"answer": }\n', encoding='utf-8')
    with pytest.raises(ValueError, match='corrupt Stage 2 JSONL.*line 2'):
        score212.load_stage2(path)
    path.write_text(json.dumps({'arm': 'unchanged', 'answer': 'valid'}) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='batch_size'):
        score212.load_stage2(path)


def test_output_directory_lock_releases_after_windows_text_mode_newlines(tmp_path, monkeypatch):
    """Windows text mode wrote the lock's newlines as CRLF; the release check must still see the file as its own."""
    real_write = run212.os.write

    def text_mode_write(fd, data):
        return real_write(fd, data.replace(b'\n', b'\r\n'))

    lock_path = tmp_path / run212.OUTPUT_LOCK_NAME
    monkeypatch.setattr(run212.os, 'write', text_mode_write)
    with run212.output_directory_lock(tmp_path, 'stage1'):
        assert b'\r\n' in lock_path.read_bytes()
    monkeypatch.setattr(run212.os, 'write', real_write)
    assert not lock_path.exists()
    with run212.output_directory_lock(tmp_path, 'pilot'):
        pass
