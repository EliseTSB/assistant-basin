"""Smoke: drive part212/run212.py's real stage1 (and optionally the stage2 pilot) on a tiny random Qwen3.5.

Usage (from the worktree root, on the host venv, HF_HUB_OFFLINE=1):
  python -I -B part212/smoke_tiny212.py <worktree> <scratch-dir> [stage1|pilot]
Writes only under <scratch-dir>/out. Exit 0 and a final "SMOKE OK" line means the real torch glue ran end to end.
This is the acceptance test for the runner's glue: the unit tests cannot see shape bugs between record and hook.
The tiny model has the real tokenizer, 30 layers and an MLP 9216 wide so TARGET (6:753, 7:1169, 11:8868, 29:776) exist.

It also ASSERTS that the interventions do what they claim (GLM review F1-F3, 2026-11-19), on the real hook path of
this model, because a smoke that only proves the code runs passed over a no-op `zero` arm:
  * Stage 1: `zero` leaves every clamped column at 0 at every position; a random_k arm's clamped columns hold THAT
    pool unit's own recorded value (not a target's), and its realized perturbation norm is above 0;
  * Stage 2: `zero` leaves the clamped columns at 0 at call 0 and at every decode call; a random_k set's trace arm
    clamps its own units' donor rows; the full plan holds 6 guards x 4 samples and 240 rows per full arm.
Every check prints `CHECK OK <name>`; the first failure prints `CHECK FAILED <name>: ...` and exits 1.
"""
import json, os, sys, time
wt, scratch, what = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else 'pilot')
sys.path.insert(0, wt); sys.path.insert(0, os.path.join(wt, 'part212'))
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import neuron_report as nr
import run212

SNAP = '/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
cfg = json.load(open(os.path.join(SNAP, 'config.json'), encoding='utf-8'))
t = dict(cfg.get('text_config', cfg))
t.update(num_hidden_layers=30, hidden_size=64, intermediate_size=9216, num_attention_heads=4,
         num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
         linear_key_head_dim=16, linear_value_head_dim=16, layer_types=t['layer_types'][:30],
         tie_word_embeddings=True)
t.pop('architectures', None)
conf = AutoConfig.for_model(t.get('model_type', 'qwen3_5_text'), **{k: v for k, v in t.items() if k != 'model_type'})
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)

LOADS = []

def fake_load(name, rev, device, *a, **k):
    LOADS.append(name)
    nr.LAST_LOAD = {'model': name, 'revision': rev, 'device': device, 'tiny_random_smoke': True}
    return tok, model
nr.load_model = fake_load
os.makedirs(scratch, exist_ok=True)
os.chdir(scratch)  # --out must be relative (the Windows job runner's rule)


# ----------------------------------------------------------------------------------------------------------------
# intervention checks: capture what the hooks actually write, on this model's real layers
# ----------------------------------------------------------------------------------------------------------------

def check(ok, name, detail=''):
    if not ok:
        print(f'CHECK FAILED {name}: {detail}', flush=True)
        raise SystemExit(1)
    print(f'CHECK OK {name}', flush=True)


def capture_down_proj_inputs(layers, fn):
    """Register a capture pre-hook on every layer, run fn(), and return {layer: [down_proj input per forward]}.

    The capture hook is registered after any hook already installed, so torch hands it the tensor those hooks
    produced (forward pre-hooks run in registration order and chain their return values).
    """
    seen = {}
    handles = []
    for li in range(len(layers)):
        def hook(_module, args, li=li):
            # the runner's hooks return a NEW tensor (they clone), so identity tells the clamped tensor from the raw
            # input the model's own pre-hooks would hand a hook registered before them
            seen.setdefault(li, []).append((args[0].detach().clone(), args[0].data_ptr()))
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        fn()
    finally:
        for h in handles:
            h.remove()
    return seen


def stage1_forward_capture(model, layers, units, fwd_ids, frozen, device):
    """The runner's Stage 1 forward, with a capture hook registered AFTER the runner's own hooks.

    torch runs forward pre-hooks in registration order and chains their return values, so the capture hook receives
    exactly the tensor the runner's clamp produced (the model's own pre-hooks run after both and are not seen).
    """
    handles = run212.install_stage1_hooks(layers, units, frozen=frozen)
    try:
        return capture_down_proj_inputs(layers, lambda: run212.run_stage1_forward(
            model, layers, units, fwd_ids, frozen, None, None, device))
    finally:
        for h in handles:
            h.remove()


def stage1_intervention_checks(device):
    _, layers, _ = nr.find_text_layers(model)
    bank = run212.load_prompts()
    n_units = int(layers[0].mlp.down_proj.in_features)
    down_weight = run212.down_weights(layers)
    abs_sums, counts = run212.pool_norm_pass(tok, model, layers, bank, device)
    pools, _maker_norm = run212.stage1_pools(abs_sums, counts, down_weight, n_units)
    sets = run212.random_sets(pools)
    units = run212.stage1_record_units(pools)
    cols_of = {u: units.index(u) for u in units}
    target_cols = [cols_of[u] for u in run212.TARGET]
    h, s = int(bank['template_prefix_tokens']), int(bank['template_suffix_tokens'])
    qi = 0
    ids, T_q = run212.stage1_prompt_ids(tok, bank, qi)
    fwd_ids = run212.stage1_input_ids(ids, bank)
    recorded = run212.record_pass(tok, model, layers, bank, units, device)
    profiles = run212.stage1_profiles({name: [a[:, target_cols] for a in recorded[name]] for name in recorded}, bank)
    rec = {name: np.asarray(recorded[name][qi], dtype=np.float64)[:, target_cols] for name in recorded}
    rec['clean'] = rec['en_maker']

    # F1 / Stage 1: `zero` leaves every clamped column at 0 at every position it clamps
    vals = run212.stage1_values('zero', qi, T_q, h, s, rec, profiles)
    frozen = run212.frozen_by_layer(vals, run212.TARGET)
    seen = stage1_forward_capture(model, layers, run212.TARGET, fwd_ids, frozen, device)
    for li, unit in run212.TARGET:
        x, _ptr = seen[li][0]
        check(x.shape[1] == len(fwd_ids), f'stage1 zero capture saw the full forward at L{li}', f'{x.shape[1]}')
        clamped = x[0, :T_q + run212.OPENING_TOKENS, unit]
        check(torch.all(clamped == 0), f'stage1 zero leaves L{li} N{unit} at 0 on 0..T+4',
              f'max |value| {float(clamped.abs().max()):.4g}')
        untouched = x[0, T_q + run212.OPENING_TOKENS:, unit]
        check(torch.all(untouched != 0), f'stage1 zero leaves the continuation positions of L{li} N{unit} alone')

    # F3: a random_k arm clamps each pool unit to ITS OWN recorded trace, and its realized norm is above 0
    set_name = 'random_0'
    arm_units = list(sets[set_name])
    arm_cols = [cols_of[u] for u in arm_units]
    arm_rec = {name: np.asarray(recorded[name][qi], dtype=np.float64)[:, arm_cols] for name in recorded}
    arm_rec['clean'] = arm_rec['en_maker']
    arm_profiles = run212.stage1_profiles({name: [a[:, arm_cols] for a in recorded[name]] for name in recorded}, bank)
    for base in ('trace_resample_fact', 'sub_factual', 'scale_neg1'):
        arm = f'{set_name}:{base}'
        vals = run212.stage1_values(arm, qi, T_q, h, s, arm_rec, arm_profiles)
        frozen = run212.frozen_by_layer(vals, arm_units)
        seen = stage1_forward_capture(model, layers, arm_units, fwd_ids, frozen, device)
        for k, (li, unit) in enumerate(arm_units):
            got = seen[li][0][0][0, :T_q + run212.OPENING_TOKENS, unit]
            want = torch.as_tensor(vals[:, k], dtype=got.dtype)
            check(torch.allclose(got, want), f'{arm} clamps L{li} N{unit} to its OWN value',
                  f'max diff {float((got - want).abs().max()):.4g}')
        pert = run212.realized_perturbation(vals, arm_rec['en_maker'], run212.target_down_columns(down_weight, arm_units))
        check(np.isfinite(pert) and pert > 0, f'{arm} realized perturbation norm > 0', f'got {pert}')
        target_vals = run212.stage1_values(base, qi, T_q, h, s, rec, profiles)
        check(not np.allclose(vals, target_vals), f'{arm} does not clamp the TARGET units\' values into the pool units')
    print(f'stage1 intervention checks: pools {[len(pools[l]) for l in sorted(pools)]}, '
          f'{set_name} units {arm_units}', flush=True)


def stage2_intervention_checks(device):
    _, layers, _ = nr.find_text_layers(model)
    bank = run212.load_prompts()
    n_units = int(layers[0].mlp.down_proj.in_features)
    down_weight = run212.down_weights(layers)
    abs_sums, counts = run212.pool_norm_pass(tok, model, layers, bank, device)
    pools, _ = run212.stage1_pools(abs_sums, counts, down_weight, n_units)
    sets = run212.random_sets(pools)
    record_units = run212.stage2_record_units(pools)
    h, s = int(bank['template_prefix_tokens']), int(bank['template_suffix_tokens'])
    # donor recording for ONE question and the three banks is enough to drive the hooks (75 runs is the pilot's job)
    donors = {}
    for name in run212.DONOR_BANKS:
        qi = 0
        ids = run212.p66_ids(tok, nr.build_prompt(tok, bank['items'][qi][name], False))
        T = len(ids)
        cap = {li: [] for li in sorted({l for l, _ in record_units})}
        handles = []
        for li in sorted(cap):
            cols = run212._columns(record_units, li)
            def hook(_module, args, li=li, cols=cols, cap=cap):
                x = args[0]
                cap[li].append((x[0][-1, cols] if x.shape[1] == 1 else x[0][:, cols]).float().cpu())
            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                res = model.generate(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                                     max_new_tokens=run212.MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        finally:
            for hd in handles:
                hd.remove()
        gen = [int(x) for x in res[0, T:].tolist()]
        eos = tok.eos_token_id
        n_non_eos = gen.index(int(eos)) if (eos is not None and int(eos) in gen) else len(gen)
        L_gen = run212.donor_L_gen(n_non_eos)
        prompt = np.zeros((T, len(record_units)), dtype=np.float64)
        genarr = np.zeros((L_gen, len(record_units)), dtype=np.float64)
        for k, (li, unit) in enumerate(record_units):
            col = run212._columns(record_units, li).index(int(unit))
            prompt[:, k] = cap[li][0][:, col].numpy()
            for j in range(L_gen):
                genarr[j, k] = cap[li][1 + j][col].numpy()
        donors[name] = [{'prompt': prompt, 'gen': genarr, 'n_non_eos': n_non_eos, 'L_gen': L_gen}]
    donors = {name: ds * 25 for name, ds in donors.items()}
    means = run212.donor_means(donors, h, s)

    def run_stage2_arm(arm, set_name, ids):
        T = len(ids)
        units = run212.stage2_units(arm, set_name, sets)
        tensors = run212.stage2_tensors(donors, means, arm, 0, units, record_units, h, s)
        L_gen = donors[run212.donor_bank_for(arm)][0]['L_gen']
        handles = run212.install_stage2_hooks(layers, units, run212.stage2_plan(arm), tensors, T, L_gen, 2.0)
        try:
            seen = capture_down_proj_inputs(layers, lambda: model.generate(
                input_ids=torch.tensor([ids], dtype=torch.long, device=device), max_new_tokens=6, do_sample=False,
                use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id))
        finally:
            for hd in handles:
                hd.remove()
        return units, tensors, seen

    question = bank['items'][0]['en_maker']
    ids = run212.p66_ids(tok, nr.build_prompt(tok, question, False))
    T = len(ids)

    # F1 / Stage 2: `zero` leaves the clamped columns at 0 at call 0 AND at every decode call
    units, _tensors, seen = run_stage2_arm('zero', None, ids)
    calls = seen[run212.TARGET[0][0]]
    check(len(calls) >= 2, 'stage2 zero ran a prefill and at least one decode call', f'{len(calls)} calls')
    for li, unit in run212.TARGET:
        for c, (x, _ptr) in enumerate(seen[li]):
            got = x[0, :, unit]
            check(torch.all(got == 0), f'stage2 zero leaves L{li} N{unit} at 0 at call {c}', f'max {float(got.abs().max())}')

    # F2: a random_k set's trace arm clamps its own units' donor rows (prefill row p, decode row s-1 mod L_gen)
    set_name = 'random_0'
    units, tensors, seen = run_stage2_arm('trace_resample_fact', set_name, ids)
    for li, unit in units:
        col = run212._columns(units, li).index(unit)
        want0 = torch.as_tensor(tensors['prompt'][li][:, col], dtype=seen[li][0][0].dtype)
        check(torch.allclose(seen[li][0][0][0, :T, unit], want0), f'{set_name} prefill clamps L{li} N{unit} to its own donor row')
        for c, (x, _ptr) in enumerate(seen[li][1:], start=1):
            want = float(tensors['gen'][li][(c - 1) % tensors['gen'][li].shape[0], col])
            check(float(x[0, 0, unit]) == want, f'{set_name} decode call {c} clamps L{li} N{unit} to its own gen row')

    # F2: the plan holds the guards and 240 rows per full arm
    rows = run212.stage2_rows()
    guards = [r for r in rows if r['question_index'] >= 25]
    check(len(rows) == 3888, 'full Stage 2 plan is 3,888 rows', f'{len(rows)}')
    check(len(guards) == 144, 'the plan holds 6 guards x 4 samples x 6 full arms', f'{len(guards)}')
    for arm in run212.STAGE2_FULL_ARMS:
        n = sum(1 for r in rows if r['arm'] == arm and r['set'] is None)
        check(n == 240, f'full arm {arm} is planned at 240 rows', f'{n}')
    # a guard row takes the bank-level mu_prompt mapped onto the guard's own length, with no donor lookup
    guard_q = run212.question_text(bank, 25)
    g_ids = run212.p66_ids(tok, nr.build_prompt(tok, guard_q, False))
    # Guards have no per-question donor; prove the mean arm needs only the bank-level profile and own guard length.
    guard_donors = {name: [] for name in run212.DONOR_BANKS}
    g_t = run212.stage2_tensors(guard_donors, means, 'mean_resample_fact', 25, run212.TARGET, record_units, h, s,
                                T_guard=len(g_ids))
    check(g_t['mu_prompt'][run212.TARGET[0][0]].shape == (len(g_ids) + run212.OPENING_TOKENS, 1),
          'a guard row maps mu_prompt onto the guard\'s own length')
    print('stage2 intervention checks: done', flush=True)


t0 = time.time()
stage1_intervention_checks('cpu')
stage2_intervention_checks('cpu')
print(f'intervention checks done in {time.time() - t0:.1f}s', flush=True)
t0 = time.time()
run212.main([what, '--out', 'out', '--device', 'cpu'])
if what == 'pilot':
    if len(LOADS) != 1:
        print(f'CHECK FAILED pilot_loads_model_once: {len(LOADS)} loads'); sys.exit(1)
    print('CHECK OK pilot_loads_model_once')
if what in ('stage2', 'pilot'):
    stage2_path = os.path.join('out', 'stage2.jsonl')
    with open(stage2_path, encoding='utf-8') as f:
        stage2_rows = [json.loads(line) for line in f if line.strip()]
    check(bool(stage2_rows), 'Stage 2 wrote rows for batch metadata inspection')
    missing_batch = [i for i, row in enumerate(stage2_rows, start=1)
                     if isinstance(row.get('batch_size'), bool) or not isinstance(row.get('batch_size'), int)
                     or row['batch_size'] < 1]
    check(not missing_batch, 'every Stage 2 row carries its actual positive batch_size',
          f'bad rows {missing_batch[:5]}')
print('SMOKE OK', what, round(time.time() - t0, 1), 's', flush=True)
