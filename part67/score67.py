#!/usr/bin/env python3
"""Score a completed Part 67 run directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

TARGETS = [(6,753),(7,1169),(11,8868),(29,776)]
PAIRS = [(a,b) for a in TARGETS for b in TARGETS if a[0] < b[0]]
UPSTREAM = TARGETS[:3]  # Claude, 2026-10-26: used at the 67.3 step but never defined (NameError on the first score)
SEEDS = list(range(6700,6750))


def empirical_p(observed: float, null: list[float]) -> float:
    return (1 + sum(abs(x) >= abs(observed) for x in null)) / (len(null) + 1)


def rank_abs(value: float, controls: list[float]) -> int:
    """One-based competition rank by absolute magnitude; ties receive the same rank."""
    return 1 + sum(abs(x) > abs(value) for x in controls)


def passes_67_1(pair_effects: list[float], nulls: list[list[float]]) -> bool:
    return any(empirical_p(effect, controls) == 1/51 for effect, controls in zip(pair_effects, nulls))


def passes_67_2(largest_pair_ends_at_l29: bool) -> bool:
    return largest_pair_ends_at_l29


def passes_67_3(joint: float, singles: list[float], random_triples: list[float]) -> bool:
    return abs(joint) > max(map(abs, singles)) and all(abs(joint) > abs(x) for x in random_triples)


def passes_67_4(gate_rank: int | None, up_rank: int | None) -> bool:
    return any(rank is not None and rank <= 5 for rank in (gate_rank, up_rank))


def passes_67_5(max_abs_earlier_delta: float) -> bool:
    return max_abs_earlier_delta <= 1e-3


def passes_67_6(all_layer_delta: float, random_deltas: list[float]) -> bool:
    return abs(all_layer_delta) > max(map(abs, random_deltas))


def reading_gate(control_675: bool, control_676: bool) -> bool:
    return control_675 and control_676


def _mean(xs): return sum(xs) / len(xs)


def _read_data(directory: Path):
    data = json.loads((directory/'activations.json').read_text(encoding='utf-8'))
    records = {(r['condition'], int(r['prompt_index']), tuple(r['target'])): r for r in data['records']}
    return data, records


def _normalized_effect(records, condition: str, target: tuple[int,int], prompts: int,
                       baseline_positions: dict[int,list[float]], last=False) -> float:
    sd = statistics.pstdev([v for vals in baseline_positions.values() for v in vals])
    if sd == 0: raise ValueError(f'unchanged standard deviation is zero for target {target}')
    key = 'last' if last else 'mean'
    diffs = [records[(condition,p,target)][key] - records[('unchanged',p,target)][key]
             for p in range(prompts)]
    return _mean(diffs) / sd


def _effects(data, records, last=False):
    prompts = len(data['prompts'])
    baselines = {t: {p: records[('unchanged',p,t)]['positions'] for p in range(prompts)} for t in TARGETS}
    effects = {}
    for upstream, downstream in PAIRS:
        cond = f'zero_{upstream[0]}_{upstream[1]}'
        effects[(upstream,downstream)] = _normalized_effect(records,cond,downstream,prompts,baselines[downstream],last)
    return effects, baselines


def score(directory: str | Path) -> dict:
    directory = Path(directory)
    data, records = _read_data(directory)
    prompts = len(data['prompts'])
    effects, baselines = _effects(data,records)
    last_effects, _ = _effects(data,records,last=True)
    pair_rows=[]
    for upstream,downstream in PAIRS:
        observed=effects[(upstream,downstream)]
        random=[_normalized_effect(records,f"rand_{upstream[0]}_{seed}",downstream,prompts,baselines[downstream]) for seed in SEEDS]
        last_random=[_normalized_effect(records,f"rand_{upstream[0]}_{seed}",downstream,prompts,baselines[downstream],last=True) for seed in SEEDS]
        paths=json.loads((directory/'weights.json').read_text(encoding='utf-8'))['paths']
        def pathvalue(unit, pair, component):
            row=next((x for x in paths if x['upstream']==list(unit) and x['downstream']==list(pair)),None)
            return None if row is None else float(row[component])
        observed_path={k:pathvalue(upstream,downstream,k) for k in ('gate','up')}
        ranks={}
        for component in ('gate','up'):
            controls=[pathvalue((upstream[0],int(cu)),downstream,component)
                      for seed in SEEDS for cu in [next(r['set_units'][0][1] for r in data['conditions'] if r['id']==f'rand_{upstream[0]}_{seed}')]]
            controls=[x for x in controls if x is not None]
            ranks[component]=None if observed_path[component] is None or not controls else rank_abs(observed_path[component],controls)
        pair_rows.append({'upstream':list(upstream),'downstream':list(downstream),'delta':observed,
            'max_random_abs_delta':max(map(abs,random)),'p':empirical_p(observed,random),
            'last_position_delta':last_effects[(upstream,downstream)],
            'last_position_max_random_abs_delta':max(map(abs,last_random)),
            'weight_path':observed_path,'weight_rank':ranks})

    # 67.5: target zeroing may not move any earlier target by more than 1e-3 unchanged SD.
    earlier_checks=[]
    for zeroed in TARGETS:
        cond=f'zero_{zeroed[0]}_{zeroed[1]}'
        for earlier in TARGETS:
            if earlier[0] < zeroed[0]:
                val=_normalized_effect(records,cond,earlier,prompts,baselines[earlier])
                earlier_checks.append({'zeroed':list(zeroed),'earlier':list(earlier),'delta':val})
    max_earlier=max((abs(x['delta']) for x in earlier_checks),default=0.0)
    c675=passes_67_5(max_earlier)
    # 67.6: all layer-6 units must exceed every matched random single on L7 N1169.
    detect_target=(7,1169)
    detect=_normalized_effect(records,'layer6_all',detect_target,prompts,baselines[detect_target])
    detect_random=[_normalized_effect(records,f'rand_6_{seed}',detect_target,prompts,baselines[detect_target]) for seed in SEEDS]
    c676=passes_67_6(detect,detect_random)
    gate=reading_gate(c675,c676)
    verdicts={}
    if not gate:
        verdicts={f'67.{i}':'NOT READ' for i in range(1,5)}
    else:
        # 67.1: at least one pair reaches p=1/51 (strictly exceeds all 50 controls).
        v671=passes_67_1([r['delta'] for r in pair_rows], [
            [_normalized_effect(records,f"rand_{upstream[0]}_{seed}",downstream,prompts,baselines[downstream]) for seed in SEEDS]
            for upstream,downstream in PAIRS])
        # 67.2: largest absolute pair effect ends at the latest target, L29 N776.
        largest=max(pair_rows,key=lambda r:abs(r['delta']))
        v672=passes_67_2(largest['downstream']==[29,776])
        # 67.3: joint effect exceeds each upstream single and all matched random triples.
        joint=_normalized_effect(records,'joint_6_7_11',(29,776),prompts,baselines[(29,776)])
        single29=[effects[((l,u),(29,776))] for l,u in UPSTREAM]
        triple=[_normalized_effect(records,f'randtriple_{seed}',(29,776),prompts,baselines[(29,776)]) for seed in SEEDS]
        v673=passes_67_3(joint,single29,triple)
        # 67.4: gate or up path ranks in the top five of target plus 50 matched L6 neurons.
        p674=next(r for r in pair_rows if r['upstream']==[6,753] and r['downstream']==[7,1169])
        v674=passes_67_4(p674['weight_rank']['gate'],p674['weight_rank']['up'])
        verdicts={'67.1':'PASS' if v671 else 'FAIL','67.2':'PASS' if v672 else 'FAIL',
                  '67.3':'PASS' if v673 else 'FAIL','67.4':'PASS' if v674 else 'FAIL'}
    verdicts.update({'67.5':'PASS' if c675 else 'FAIL','67.6':'PASS' if c676 else 'FAIL'})
    result={'pairs':pair_rows,'controls':{'67.5':{'max_abs_earlier_delta':max_earlier,'threshold':1e-3,'checks':earlier_checks},
               '67.6':{'layer6_all_delta':detect,'max_random_abs_delta':max(map(abs,detect_random))}},
            'joint':None if not gate else {'delta':joint,'max_abs_single_delta':max(map(abs,single29)),
               'max_random_triple_abs_delta':max(map(abs,triple))},'verdicts':verdicts}
    (directory/'score.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Part 67 score','',f"Controls: 67.5 **{verdicts['67.5']}**, 67.6 **{verdicts['67.6']}**.",'',
           '| Pair | Delta | Max random | p | Gate rank | Up rank |','|---|---:|---:|---:|---:|---:|']
    for r in pair_rows:
        lines.append(f"| L{r['upstream'][0]} N{r['upstream'][1]} → L{r['downstream'][0]} N{r['downstream'][1]} | {r['delta']:.6g} | {r['max_random_abs_delta']:.6g} | {r['p']:.6g} | {r['weight_rank']['gate']} | {r['weight_rank']['up']} |")
    lines += ['', '## Verdicts','']+[f"- {k}: **{v}**" for k,v in verdicts.items()]
    if not gate: lines += ['', '67.5 or 67.6 failed; the remaining claims are not read.']
    (directory/'score.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('directory'); a=p.parse_args(); score(a.directory)

if __name__=='__main__': main()
