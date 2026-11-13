"""Repo-wide pytest setup (2026-11-13, Claude Opus 5.5).

Caps the BLAS/OpenMP thread pools tests start with, so a test run beside a long CPU job (a causal step, a reader) does
not take every core and leave both crawling. Twice in two days a suite looked hung under a load of 28 with a 6-core
job running (Part 205/211 builds, 2026-11-12 and 11-13). setdefault: an explicit OMP_NUM_THREADS still wins, and torch
reads these at its own import, so this file never imports torch.
"""
import os

TEST_THREADS = "4"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, TEST_THREADS)
