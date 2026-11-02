#!/usr/bin/env python3
"""Build the two-panel Part 175 breakage and claim-count figure."""
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT / name).read_text())


def read_jsonl(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line.strip()]


def near(a, b):
    return abs(float(a) - float(b)) < 1e-9


stage = read_json('stage_a.json')
stage_dirs = stage['directions']
census_dirs = [row for row in read_jsonl('scored_directions.jsonl') if row.get('family') == 'f100d']
rng = random.Random(175)

BLUE = '#356B8C'
ORANGE = '#D17A4B'
DARK = '#24323B'
GRAY = '#888E91'
GRID = '#D9E0E3'
GREEN = '#287A5B'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 14,
    'axes.titlesize': 19,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.titleweight': 'bold',
})

# Keep both panels stacked so labels remain legible when the image is read on a phone.
fig, (ax_break, ax_claim) = plt.subplots(
    2, 1, figsize=(9.4, 14.0), gridspec_kw={'height_ratios': [1.7, 1.0]}
)

# Left panel: the Stage A ladder (eight signed directions per magnitude) and the
# 40-direction Stage B census at its three doses.
groups = []
for magnitude in (0.10, 0.15, 0.20, 0.225, 0.25, 0.30, 0.35, 0.40, 0.45):
    if magnitude == 0.225:
        values = [int(row['broken']) for row in census_dirs if near(abs(row['dose']), magnitude)]
        groups.append((f'Census  |d| = {magnitude:.3f}', values, ORANGE, 'census'))
    else:
        values = [int(row['broken']) for row in stage_dirs if near(abs(row['dose']), magnitude)]
        groups.append((f'Stage A  |d| = {magnitude:.2f}', values, BLUE, 'stage'))
    if magnitude in (0.15, 0.30):
        values = [int(row['broken']) for row in census_dirs if near(abs(row['dose']), magnitude)]
        groups.append((f'Census  |d| = {magnitude:.2f}', values, ORANGE, 'census'))

# Put the intermediate census dose in numeric order, beside the Stage A ladder.
groups.sort(key=lambda item: (float(item[0].split('=')[1]), 0 if item[3] == 'stage' else 1))
y_positions = list(range(len(groups)))
for y, (label, values, color, family) in zip(y_positions, groups):
    jitter = [y + rng.uniform(-0.17, 0.17) for _ in values]
    ax_break.scatter(values, jitter, s=34 if family == 'stage' else 24,
                     color=color, alpha=0.76, edgecolors='none', zorder=3)
    median = sorted(values)[len(values) // 2] if len(values) % 2 else (
        sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]
    ) / 2
    ax_break.plot([median - 0.8, median + 0.8], [y, y], color=DARK, lw=3.2,
                  solid_capstyle='round', zorder=4)
    if label in ('Stage A  |d| = 0.30', 'Census  |d| = 0.30'):
        ax_break.annotate(f'median {median:g}', (median, y), xytext=(7, 7),
                          textcoords='offset points', fontsize=12, color=DARK,
                          fontweight='bold')

ax_break.axvline(8, color=GRAY, ls=(0, (4, 3)), lw=1.7, zorder=1)
ax_break.text(8.5, len(groups) - 0.42, 'support bar: 8 broken', color=GRAY,
              fontsize=12, va='bottom')
ax_break.set_yticks(y_positions)
ax_break.set_yticklabels([group[0] for group in groups])
ax_break.invert_yaxis()
ax_break.set_xlim(-1, 46)
ax_break.set_xticks([0, 10, 20, 30, 40, 44])
ax_break.set_xlabel('Broken answers per direction (of 44)')
ax_break.set_title('Breakage by dose', loc='left', pad=11)
ax_break.grid(axis='x', color=GRID, lw=0.8)
ax_break.set_axisbelow(True)
ax_break.spines[['top', 'right', 'left']].set_visible(False)
ax_break.tick_params(axis='y', length=0, pad=9)
ax_break.legend(handles=[
    Line2D([0], [0], marker='o', color='none', markerfacecolor=BLUE,
           markeredgecolor='none', markersize=9, label='Stage A: 8 signed directions'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=ORANGE,
           markeredgecolor='none', markersize=9, label='Census: 40 signed directions'),
], loc='upper right', bbox_to_anchor=(1.0, -0.10), frameon=False, fontsize=11)

# Right panel: one mark per signed direction. Null counts get a distinct column,
# away from the numeric zero, so they cannot be mistaken for no Google claims.
# Supported directions receive a filled diamond on this claim-count axis.
doses = (0.15, 0.225, 0.30)
claim_x = []
claim_y = []
null_y = []
supported_x = []
supported_y = []
for y, dose in enumerate(doses):
    rows = [row for row in census_dirs if near(abs(row['dose']), dose)]
    for row in rows:
        claim_count = row.get('google_claim_count')
        y_position = y + rng.uniform(-0.28, 0.28)
        x_position = 46 if claim_count is None else int(claim_count)
        if claim_count is None:
            null_y.append(y_position)
        else:
            claim_x.append(x_position)
            claim_y.append(y_position)
        if row.get('supported') is True:
            supported_x.append(x_position)
            supported_y.append(y_position)

ax_claim.scatter(claim_x, claim_y, s=37, color=BLUE, alpha=0.72,
                 edgecolors='none', zorder=3)
if null_y:
    ax_claim.scatter([46] * len(null_y), null_y, s=52, marker='X',
                     color=GRAY, alpha=0.88, zorder=4)
if supported_x:
    ax_claim.scatter(supported_x, supported_y, s=68, marker='D',
                     color=GREEN, edgecolors='white', linewidths=0.65, zorder=5)

ax_claim.set_yticks(range(len(doses)))
ax_claim.set_yticklabels([f'|d| = {dose:g}' for dose in doses])
ax_claim.invert_yaxis()
ax_claim.set_xlim(-1.5, 49)
ax_claim.set_xticks([0, 10, 20, 30, 40, 46])
ax_claim.set_xticklabels(['0', '10', '20', '30', '40', 'null'])
ax_claim.set_xlabel('Google claims per signed direction')
ax_claim.set_title('Google claims by dose (nulls separate)', loc='left', pad=11, fontsize=17)
ax_claim.grid(axis='x', color=GRID, lw=0.8)
ax_claim.set_axisbelow(True)
ax_claim.spines[['top', 'right', 'left']].set_visible(False)
ax_claim.tick_params(axis='y', length=0, pad=9)
fig.legend(handles=[
    Line2D([0], [0], marker='o', linestyle='none', color='none',
           markerfacecolor=BLUE, markeredgecolor='none', markersize=8,
           label='Google claims (labelled direction)'),
    Line2D([0], [0], marker='D', linestyle='none', color='none',
           markerfacecolor=GREEN, markeredgecolor='white', markersize=8,
           label='supported: modal at least 10 of 44, at most 8 broken'),
    Line2D([0], [0], marker='X', linestyle='none', color=GRAY,
           markerfacecolor=GRAY, markeredgecolor=GRAY, markersize=8,
           label='null Google claim count'),
], loc='lower left', bbox_to_anchor=(0.34, 0.015), frameon=False, fontsize=10)
fig.subplots_adjust(left=0.32, right=0.97, top=0.94, bottom=0.18, hspace=0.62)
fig.savefig(ROOT / 'plot175.png', dpi=220, facecolor='white')
plt.close(fig)
