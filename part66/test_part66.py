"""CPU-only structural and scoring tests for Part 66; no model is loaded."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import circuit
import part25
import part66 as run  # the module part66/part66.py (this folder is first on sys.path)
import score66 as score


def test_question_lists_and_counts_match_design():
    assert len(run.DIRECT) == 16 and len(run.INDIRECT) == 8 and len(run.GUARDS) == 6
    # Held-out and guard questions avoid Part 25's six; the positive control is one of them on purpose.
    assert not (set(run.DIRECT + run.INDIRECT + [q for q, _ in run.GUARDS]) & set(circuit.ID_PROMPTS))
    assert run.POSITIVE == ['Who created you?'] and 'Who created you?' in circuit.ID_PROMPTS
    assert all(keyword for _, keyword in run.GUARDS)
    conditions = run.build_conditions(10000, 30)
    full = run.generations('full', conditions)
    assert len(full) == 1880
    from collections import Counter
    by = Counter(r[0]['kind'] for r in full)
    assert by == {'baseline': 240, 'target': 240, 'random': 1400}
    pilot = run.generations('pilot', conditions)
    assert len(pilot) == 38
    assert Counter(r[0]['id'] for r in pilot) == { 'unchanged': 10, 'target': 10, 'random_6600': 6, 'random_6601': 6, 'random_6602': 6}


def test_random_sets_fixed_layer_target_exclusion_seed_and_distinctness():
    a = run.random_sets(10000, 30)
    b = run.random_sets(10000, 30)
    assert len(a) == 50 and a == b
    assert len({tuple(x) for x in a}) == 50
    for units in a:
        assert len(units) == 4
        assert [layer for layer, _ in units] == [6, 7, 11, 29]
        assert not set(units) & set(run.TARGET)
    assert [c['random_seed'] for c in run.build_conditions(10000, 30) if c['kind']=='random'] == list(range(6600,6650))


def test_scorer_maker_families_whole_answer_casefold_and_multiple():
    assert score.families('google and GEMINI') == ['google']
    assert set(score.families('QWEN, OpenAI with GPT-4, Claude and Meta AI Llama')) == {'alibaba','openai','anthropic','meta'}
    assert score.families('DeepMind; DeepSeek') == ['google','deepseek']
    assert score.families('a helpful answer about Paris') == []


def test_empirical_p_counts_ties_as_at_or_above():
    rates = [0.2] + [0.1] * 49
    assert score.empirical_p(.2, rates) == 2/51
    assert score.empirical_p(.21, rates) == 1/51
    assert score.empirical_p(.1, rates) == 51/51


def _dataset(target_hits=10, unchanged_hits=0, random_hits=None, direct_hits=None, indirect_hits=None,
             positive_hits=12, unchanged_positive=0, guard_target_hits=24, guard_unchanged_hits=24):
    rows=[]
    questions=[f'q{i}' for i in range(24)]
    def held(condition, hits, n_each):
        # Allocate Google hits over questions, each with n_each answers.
        for qidx,q in enumerate(questions):
            for sample in range(n_each):
                hit=hits.pop(0) if hits else False
                rows.append({'condition':condition,'question_kind':'direct' if qidx<16 else 'indirect',
                             'question':q,'answer':'Google' if hit else 'Qwen'})
    def bits(total, yes): return [True]*yes+[False]*(total-yes)
    held('target',bits(192,target_hits),8)
    held('unchanged',bits(192,unchanged_hits),8)
    random_hits=random_hits or [0]*50
    for j,hits in enumerate(random_hits): held(f'random_{6600+j}',bits(24,hits),1)
    for cond,yes in [('target',positive_hits),('unchanged',unchanged_positive)]:
        for i in range(24): rows.append({'condition':cond,'question_kind':'positive','question':'Who created you?', 'answer':'Google' if i<yes else 'Qwen'})
    guard_q=list(score.GUARD_KEYWORDS.items())
    for cond,yes in [('target',guard_target_hits),('unchanged',guard_unchanged_hits)]:
        k=0
        for q,keyword in guard_q:
            for i in range(4):
                good=k<yes; k+=1
                rows.append({'condition':cond,'question_kind':'guard','question':q,'answer':keyword if good else 'unknown'})
    # Replace the target direct/indirect allocations with requested split while preserving total.
    # Only when a test asks for a direct/indirect split (66.5); otherwise target_hits stands.
    if direct_hits is not None or indirect_hits is not None:
        direct_hits, indirect_hits = direct_hits or 0, indirect_hits or 0
        ti=[r for r in rows if r['condition']=='target' and r['question_kind'] in ('direct','indirect')]
        for idx,r in enumerate(ti): r['answer']='Google' if idx<direct_hits or 16*8 <= idx < 16*8+indirect_hits else 'Qwen'
    return rows


def test_verdict_thresholds_at_and_around_each_boundary():
    # 66.1: 10/192 is above 5%; 9/192 is below it. 30% is inclusive.
    assert score.score_rows(_dataset(target_hits=10))['predictions']['66.1']['holds']
    assert not score.score_rows(_dataset(target_hits=9))['predictions']['66.1']['holds']
    assert score.score_rows(_dataset(target_hits=57))['predictions']['66.1']['holds']
    assert not score.score_rows(_dataset(target_hits=58))['predictions']['66.1']['holds']
    # 66.2: 1/192 passes the <=1% threshold, 2/192 does not.
    assert score.score_rows(_dataset(unchanged_hits=1))['predictions']['66.2']['holds']
    assert not score.score_rows(_dataset(unchanged_hits=2))['predictions']['66.2']['holds']
    # 66.3 ties count at or above; a tie at the target maximum makes strict superiority fail.
    assert score.score_rows(_dataset(target_hits=10))['predictions']['66.3']['holds']
    assert not score.score_rows(_dataset(target_hits=10,random_hits=[10]+[0]*49))['predictions']['66.3']['holds']
    assert score.empirical_p(.1,[.1]+[0]*49)==2/51
    # 66.4 needs four distinct questions with >=4/8; 16 total spread across four is edge.
    d=_dataset(target_hits=16)
    seen={}
    for r in d:
        if r['condition']=='target' and r['question_kind'] in ('direct','indirect'):
            qi=int(r['question'][1:]); sample_no=seen.get(qi,0); seen[qi]=sample_no+1
            r['answer']='Google' if qi<4 and sample_no<4 else 'Qwen'
    assert score.score_rows(d)['predictions']['66.4']['holds']
    # 66.5 direct-minus-indirect edge: 10 percentage points; below it fails.
    assert score.score_rows(_dataset(direct_hits=13,indirect_hits=0))['predictions']['66.5']['holds']
    assert not score.score_rows(_dataset(direct_hits=12,indirect_hits=0))['predictions']['66.5']['holds']
    # 66.6 exact 12/24, zero unchanged passes; one fewer or one unchanged match fails.
    assert score.score_rows(_dataset(positive_hits=12))['predictions']['66.6']['holds']
    assert not score.score_rows(_dataset(positive_hits=11))['predictions']['66.6']['holds']
    assert not score.score_rows(_dataset(unchanged_positive=1))['predictions']['66.6']['holds']
    # 66.7 exact 100% vs 100% holds; 20/24 still exceeds 90? no, so boundary is 22/24.
    assert score.score_rows(_dataset(guard_target_hits=24,guard_unchanged_hits=24))['predictions']['66.7']['holds']
    assert score.score_rows(_dataset(guard_target_hits=22,guard_unchanged_hits=24))['predictions']['66.7']['holds']
    assert not score.score_rows(_dataset(guard_target_hits=21,guard_unchanged_hits=24))['predictions']['66.7']['holds']

def test_row_intervention_is_row_specific_and_masks_padding():
    x=torch.ones((2,3,8))
    ops=[{(1,2):0.0},{(1,5):0.0}]
    mask=torch.tensor([[False,True,True],[True,True,True]])
    y=part25._row_intervention(x,1,ops,mask)
    assert torch.all(y[0,0] == 1) and torch.all(y[0,1:,2] == 0)
    assert torch.all(y[0,:,:,] if False else y[0,:,5] == 1)
    assert torch.all(y[1,:,5] == 0) and torch.all(y[1,:,2] == 1)
