"""Part 205 figure: the overlap index by stage (left) beside whether each model picks the value-consistent option at
all (right). Drawn from results/part205/results.json and letter.json; writes results/part205/fig205.png."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['pretrain-3.0T', 'midtrain-26B', 'midtrain-end', 'sft', 'dpo', 'instruct', 'qwen-base', 'qwen-chat']
LABEL = {'pretrain-3.0T': 'pretrain\n3.0T', 'midtrain-26B': 'midtrain\n26B', 'midtrain-end': 'midtrain\nend',
         'sft': 'SFT', 'dpo': 'DPO', 'instruct': 'Instruct', 'qwen-base': 'Qwen3.5-4B\nBase', 'qwen-chat': 'Qwen3.5-4B\nchat'}
BLUE, ORANGE = '#2a78d6', '#eb6834'          # validated categorical slots 1-2 (dataviz reference palette)
INK, MUTED, GRID, SURF = '#1f1f1e', '#6b6b68', '#e6e6e3', '#fcfcfb'


def main(root):
    root = Path(root)
    res = {s['stage']: s for s in json.loads((root / 'results/part205/results.json').read_text())['stages']}
    let = {s['stage']: s for s in json.loads((root / 'results/part205/letter.json').read_text())['stages']}
    xs = [i if i < 6 else (i - 6) * 1.4 + 6.9 for i in range(len(ORDER))]
    color = [BLUE] * 6 + [ORANGE] * 2

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURF)
    for ax in (a, b):
        ax.set_facecolor(SURF)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[s] for s in ORDER], color=INK)
        ax.set_ylim(0, 1.05)

    for x, st, c in zip(xs, ORDER, color):
        s = res[st]; o = s['overlap']['O']; ci = s['O_ci']
        read = s['read']['O']
        a.errorbar([x], [o], yerr=[[o - ci['lo']], [ci['hi'] - o]], fmt='o', color=c, ecolor=c, elinewidth=2,
                   capsize=0, markersize=9, markerfacecolor=c if read else SURF, markeredgewidth=2)
        a.annotate(f"{o:.2f}" + ('' if read else '\nnot read'), (x, ci['hi']), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8.5,
                   color=INK)
    a.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    a.text(xs[0] - 0.3, 0.52, "205.2's bar (O at least 0.5 at Instruct)", fontsize=8, color=MUTED, va='bottom', ha='left')
    a.text(xs[0] - 0.3, 0.42, 'open marker: not read (G0 failed at pretrain 3.0T)', fontsize=8, color=MUTED, ha='left')
    a.set_ylabel('overlap index O  (Cross / Retest, 95% CI)', color=INK, fontsize=9.5)
    a.set_title('SAY and DO rankings overlap at every stage that passed G0', loc='left', fontsize=11.5, color=INK, pad=10)

    for kind, marker, ls, dx in (('say', 'o', '-', -0.12), ('do', 's', (0, (3, 2)), 0.12)):
        for seg in (range(0, 6), range(6, 8)):
            ys = [let[ORDER[i]][kind]['behaviour']['p_consistent'] for i in seg]
            c = BLUE if seg.start == 0 else ORANGE
            b.plot([xs[i] + dx for i in seg], ys, linestyle=ls, marker=marker, color=c, linewidth=2, markersize=8,
                   markeredgecolor=SURF, markeredgewidth=1.5)
    b.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    b.text(xs[5] + 0.35, 0.48, 'chance', fontsize=8, color=MUTED, va='top', ha='left')
    inst = let['instruct']['do']['behaviour']
    b.annotate(f"DO at Instruct: P(A) {inst['p_a']:.2f}\n(mostly a B preference)", (xs[5] + 0.12, inst['p_consistent']),
               xytext=(-95, -62), textcoords='offset points', fontsize=8, color=INK,
               arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8))
    b.plot([], [], color=MUTED, marker='o', linestyle='-', label='SAY questions')
    b.plot([], [], color=MUTED, marker='s', linestyle=(0, (3, 2)), label='DO scenarios')
    b.legend(loc='lower right', frameon=False, fontsize=8.5, labelcolor=INK)
    b.set_ylabel('P(value-consistent option), You frame', color=INK, fontsize=9.5)
    b.set_title("while OLMo's DO P(value-consistent) stays near chance", loc='left', fontsize=11.5, color=INK, pad=10)

    fig.text(0.01, 0.005, 'OLMo 2 1B stages (blue), Qwen3.5-4B Base and chat (orange). Selection prompts, 24 per kind. '
             'Sources: results/part205/results.json, letter.json; drawn by part205/fig205.py.', fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = root / 'results/part205/fig205.png'
    fig.savefig(out, dpi=150, facecolor=SURF)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
