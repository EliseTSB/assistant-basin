"""Part 205 report: prints every number the README cites, from results/part205/results.json, letter.json and
a0/instruct/screen.json. Writes results/part205/report.md. No model is run."""
import json
import sys
from pathlib import Path

ORDER = ['pretrain-3.0T', 'midtrain-26B', 'midtrain-end', 'sft', 'dpo', 'instruct', 'qwen-base', 'qwen-chat']


def main(root):
    root = Path(root)
    res = json.loads((root / 'results/part205/results.json').read_text())
    letter = json.loads((root / 'results/part205/letter.json').read_text())
    screen = json.loads((root / 'results/part205/a0/screen.json').read_text())
    stages = {s['stage']: s for s in res['stages']}
    let = {s['stage']: s for s in letter['stages']}
    out = ['# Part 205 report (made by part205/report205.py)', '']

    out += ['## Step A0 (Amendment 2): the six-value gap at OLMo Instruct', '',
            f"gap mean(P_you - P_third) = {screen['gap']:.4f} over {screen['n_third']} pairs "
            f"(clustered SE {screen['se_clustered']:.4f}, {screen['n_clusters']} questions); threshold "
            f"{screen['threshold']}; branch **{screen['branch']}**. Source: results/part205/a0/screen.json.", '']

    out += ['## Verdicts (results/part205/results.json, key predictions)', '',
            '| # | verdict | value |', '| --- | --- | --- |']
    for k, v in res['predictions'].items():
        val = {kk: vv for kk, vv in v.items() if kk not in ('verdict',)}
        out.append(f"| {k} | {v['verdict']} | {json.dumps(val)[:200]} |")
    out += ['', f"Withdrawal headline: {res['withdrawal']['headline']}", '']

    out += ['## Per stage (results.json, key stages[...])', '',
            '| stage | G0 option mass | G0 letter top | G1 max diff | Retest SAY | Retest DO | Cross | O | O 95% CI '
            '| gap (G2) | broken ranking rows (of 48) | broken You rows (of 24) | SAY heads/neurons | DO heads/neurons '
            '| SAY and DO top-80 shared |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for st in ORDER:
        s = stages[st]
        g, o, ci, comp = s['gates'], s['overlap'], s['O_ci'], s['sets']['composition']
        read = 'read' if s['read']['O'] else 'not read'
        out.append(
            f"| {st} | {g['G0']['mean_option_mass']:.3f} ({'pass' if g['G0']['pass'] else 'FAIL'}) | "
            f"{g['G0']['letter_share']:.2f} | {g['G1']['max_abs_diff']} | {o['retest_say']} | {o['retest_do']} | "
            f"{o['cross']:.2f} | {o['O']:.5f} ({read}) | {ci['lo']:.3f} to {ci['hi']:.3f} | {g['G2']['gap']:.4f} | "
            f"{s['broken']['ranking']['broken']} | {s['broken']['you']['broken']} | "
            f"{comp['SAY']['heads']}/{comp['SAY']['neurons']} | {comp['DO']['heads']}/{comp['DO']['neurons']} | "
            f"{comp['say_do_overlap']} |")
    out += ['', 'Qwen head kinds (full attention / DeltaNet): ' + '; '.join(
        f"{st} SAY {stages[st]['sets']['composition']['SAY'].get('head_full')}/"
        f"{stages[st]['sets']['composition']['SAY'].get('head_deltanet')}, DO "
        f"{stages[st]['sets']['composition']['DO'].get('head_full')}/"
        f"{stages[st]['sets']['composition']['DO'].get('head_deltanet')}" for st in ('qwen-base', 'qwen-chat')), '']

    out += ['## Exploratory, after scoring (results/part205/letter.json, made by part205/letter205.py)', '',
            '| stage | SAY P(consistent) | DO P(consistent) | DO P(A) | SAY top-80 letter share | DO top-80 letter share '
            '| SAY pool letter share | DO pool letter share | SAY random-80 | DO random-80 | parts in both top-80 sets '
            '| their letter share (score-weighted) |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for st in ORDER:
        L = let[st]
        out.append(f"| {st} | {L['say']['behaviour']['p_consistent']:.2f} | {L['do']['behaviour']['p_consistent']:.2f} | "
                   f"{L['do']['behaviour']['p_a']:.2f} | {L['say']['top80_letter_share']:.3f} | "
                   f"{L['do']['top80_letter_share']:.3f} | {L['say']['pool_letter_share_score_weighted']:.3f} | "
                   f"{L['do']['pool_letter_share_score_weighted']:.3f} | {L['say']['random80_letter_share_median']:.3f} | "
                   f"{L['do']['random80_letter_share_median']:.3f} | {L['shared']['n_parts']} | "
                   f"{L['shared']['letter_share_score_weighted']:.3f} |")
    text = '\n'.join(out) + '\n'
    (root / 'results/part205/report.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
