#!/usr/bin/env python3
"""power104.py -- exact power of Part 104's prediction 104.1 (PART104-DESIGN.md): a one-sided Fisher test that layer-0-3
slot units are lost under SFT's attention more often than layer-4-15 units, held only if p < 0.05 AND the difference
is at least 30 points. Written by Claude (Opus 5.5), 2026-10-29, before any Part 104 data.
Usage: power104.py
"""
from math import comb


def fisher_one_sided(a, n1, b, n2):
    """P(early losses >= a | total losses a + b), hypergeometric."""
    k, n = a + b, n1 + n2
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / comb(n, k)


def power(n1, n2, p1, p2):
    total = 0.0
    for a in range(n1 + 1):
        pa = comb(n1, a) * p1 ** a * (1 - p1) ** (n1 - a)
        for b in range(n2 + 1):
            pb = comb(n2, b) * p2 ** b * (1 - p2) ** (n2 - b)
            if fisher_one_sided(a, n1, b, n2) < 0.05 and a / n1 - b / n2 >= 0.30:
                total += pa * pb
    return total


if __name__ == "__main__":
    for n1, n2 in [(8, 6), (11, 9), (14, 10), (16, 14)]:
        print(f"early {n1}, later {n2}: power {power(n1, n2, .7, .05):.2f} at 70% vs 5% (Part 103's own), "
              f"{power(n1, n2, .5, .1):.2f} at 50% vs 10%, false-positive {power(n1, n2, .25, .25):.3f} at 25% vs 25%")
