"""The root conftest.py caps test threads (2026-11-13). Sabotage: delete conftest.py -> this test fails when the
shell has no OMP_NUM_THREADS set."""
import os


def test_thread_cap_is_set():
    assert os.environ.get("OMP_NUM_THREADS")
    assert os.environ.get("MKL_NUM_THREADS")
