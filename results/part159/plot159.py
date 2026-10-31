#!/usr/bin/env python3
"""Plot the named-unit mean traces around the first veto from traces.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
traces = json.loads((HERE / 'traces.json').read_text(encoding='utf-8'))
units = [('L6N753', 'L6 N753'), ('L7N1169', 'L7 N1169'),
         ('L11N8868', 'L11 N8868'), ('L27N3325', 'L27 N3325'),
         ('L29N776', 'L29 N776')]
arms = [('free', 'free'), ('maker_ban', 'maker ban'), ('control_ban', 'control ban')]
colors = {'free': '#555555', 'maker_ban': '#b22222', 'control_ban': '#2166ac'}
fig, axes = plt.subplots(len(units), 1, figsize=(9, 12), sharex=True, constrained_layout=True)
for ax, (key, label) in zip(axes, units):
    for arm, legend in arms:
        series = traces[key][arm]
        ax.plot(series['offsets'], series['mean_sd_units'], marker='o', ms=3,
                label=legend, color=colors[arm])
    ax.axvline(0, color='#999999', lw=.8, ls='--')
    ax.axhline(0, color='#cccccc', lw=.7)
    ax.set_ylabel(f'{label}\nfree SD')
    ax.grid(axis='y', alpha=.2)
axes[0].legend(ncol=3, frameon=False, loc='upper left')
axes[-1].set_xlabel('Call offset from first veto')
fig.suptitle('Part 159 named-unit traces')
fig.savefig(HERE / 'veto159.png', dpi=160)
