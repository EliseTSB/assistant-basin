"""Score Part 24 from results/part24 (reach.json, assay.json, samples.json, alien-labels.json); prints each row's numbers."""
import json, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else 'results/part24'
reach = json.load(open(f'{D}/reach.json')); assay = json.load(open(f'{D}/assay.json'))
samples = json.load(open(f'{D}/samples.json')); labels = json.load(open(f'{D}/alien-labels.json'))['labels']
s = reach['summary']
low = max(v['max_relative_change'] for v in s['24_1_max_relative_change_layers_at_or_below_target'].values())
ident = s['identity-50']['2.0']['median_relative_change_layers_above_target']
rand = s['random-50']['2.0']['median_relative_change_layers_above_target']
def retention(group, k):
    vals = [u['name_retention_by_k'][k] for u in reach['units'] if group in u['groups'] and u['layer'] > reach['layer']
            and u['name_retention_by_k'].get(k) is not None]
    return st.median(vals)
ret_i, ret_r = retention('identity-50', '2.0'), retention('random-50', '2.0')
assert abs(ret_i - s['24_3_identity_median_name_retention_at_k_2']['median']) < 1e-9, 'retention recomputation disagrees'
m = assay['margins_and_effects']; ali = assay['self_alibaba_cloud_at_k_0']
eff = lambda z, k: m['self'][z][k]['effect']['mean']
drop = ali['random-50']['mean_per_token_logprob'] - ali['identity-50']['mean_per_token_logprob']
# every sample carries a label, and the labelled set is exactly the sampled set
keys = {f"{r['question']}|{r['k']:g}|{r['zero_condition']}": len(r['texts']) for r in samples['rows']}
assert set(keys) == set(labels) and all(len(labels[k]) == n for k, n in keys.items()), 'labels do not match samples'
count = lambda z, k: sum(sum(labels[f'{q}|{k}|{z}']) for q in ('Who made you?', 'Who created you?'))
print(f'24.1 max relative change, layers <= 22, any k: {low}  ->', 'held' if low < 1e-3 else 'FAILED')
print(f'24.2 k=2 identity {ident:.3f} vs random {rand:.3f}, ratio {ident/rand:.2f}  ->', 'held' if ident <= 1.5 * rand else 'failed')
print(f'24.3 identity median name retention at k=2 {ret_i:.3f} (random {ret_r:.3f})  ->', 'held' if ret_i >= 0.8 else 'failed')
print(f'24.4 Alibaba Cloud per token: identity {ali["identity-50"]["mean_per_token_logprob"]:.3f}, random {ali["random-50"]["mean_per_token_logprob"]:.3f}, none {ali["none"]["mean_per_token_logprob"]:.3f}; extra drop {drop:.3f}  ->', 'held' if drop >= 0.3 else 'failed')
ok5 = True
for k in ('1.0', '2.0'):
    d = abs(eff('identity-50', k) - eff('random-50', k)); q = eff('none', k) / 4; ok5 &= d < q
    print(f'     k={k}: effect none {eff("none", k):.3f}, identity {eff("identity-50", k):.3f}, random {eff("random-50", k):.3f}; |diff| {d:.3f} vs quarter {q:.3f}')
print('24.5 additive  ->', 'held' if ok5 else 'failed')
for k in ('0', '1', '1.5', '2'):
    print(f'     alien answers of 16 at k={k}: none {count("none", k)}, identity-50 {count("identity-50", k)}, random-50 {count("random-50", k)}')
d6 = count('identity-50', '1.5') - count('random-50', '1.5')
print(f'24.6 k=1.5 identity minus random {d6}  ->', 'held' if d6 >= 2 else 'failed')
ok7 = all(abs(count('random-50', k) - count('none', k)) <= 2 for k in ('0', '1', '1.5', '2'))
print('24.7 random within 2 of none at every k  ->', 'held' if ok7 else 'failed')
print(f'check: self effect at k=2, nothing zeroed, {eff("none", "2.0"):.3f} (Part 18c: 3.15)')
