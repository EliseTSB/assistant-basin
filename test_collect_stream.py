"""Real-checkpoint equivalence and memory checks for collect_stream.py.

The integration fixture deliberately uses only Qwen/Qwen3-0.6B and runs offline.
It gives collect.py the same one-frame subset that collect_stream.py receives
through --max-prompts, without changing the checked-in collect.py or data files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-0.6B"
MODEL_CACHE = Path("/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3-0.6B")
SLUG = MODEL.replace("/", "__")
PYTHON = sys.executable


class CollectStreamIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_CACHE.is_dir():
            raise unittest.SkipTest(f"offline checkpoint is not cached: {MODEL_CACHE}")

        cls.temp = tempfile.TemporaryDirectory(prefix="collect-stream-test-")
        cls.work = Path(cls.temp.name)
        cls.reference_out = cls.work / "reference-out"
        cls.stream_out = cls.work / "stream-out"
        cls.reference_out.mkdir()
        cls.stream_out.mkdir()

        env = os.environ.copy()
        env["HF_HOME"] = "/mnt/e/neuron-probe/hf"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"

        # collect.py has no subset option. Give an isolated copy the first
        # frame only; its source and the real project data stay untouched.
        tiny_root = cls.work / "reference"
        (tiny_root / "data").mkdir(parents=True)
        spec = json.loads((ROOT / "data" / "frames.json").read_text(encoding="utf-8"))
        spec["frames"] = spec["frames"][:1]
        spec["users"] = spec["users"][:1]
        (tiny_root / "data" / "frames.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
        shutil.copy2(ROOT / "collect.py", tiny_root / "collect.py")

        reference_command = [
            PYTHON,
            "-B",
            str(tiny_root / "collect.py"),
            MODEL,
            str(cls.reference_out),
            "--neutral",
            str(ROOT / "data" / "neutral.txt"),
            "--neutral-tokens",
            "8",
            "--dtype",
            "float32",
        ]
        stream_command = [
            PYTHON,
            "-B",
            str(ROOT / "collect_stream.py"),
            MODEL,
            str(cls.stream_out),
            "--neutral",
            str(ROOT / "data" / "neutral.txt"),
            "--neutral-tokens",
            "8",
            "--dtype",
            "float32",
            "--max-prompts",
            "1",
        ]
        cls.reference_rss_kib = cls._run_measured(reference_command, env)
        cls.stream_rss_kib = cls._run_measured(stream_command, env)
        cls.reference_npz = np.load(cls.reference_out / f"{SLUG}.npz", allow_pickle=False)
        cls.stream_npz = np.load(cls.stream_out / f"{SLUG}.npz", allow_pickle=False)

    @classmethod
    def tearDownClass(cls) -> None:
        for archive_name in ("reference_npz", "stream_npz"):
            archive = getattr(cls, archive_name, None)
            if archive is not None:
                archive.close()
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    @classmethod
    def _run_measured(cls, command: list[str], env: dict[str, str]) -> int:
        """Run one child in a fresh process and return its peak RSS in KiB."""
        runner = cls.work / "measure_child.py"
        runner.write_text(
            "import resource, subprocess, sys\n"
            "result = subprocess.run(sys.argv[1:], text=True, capture_output=True)\n"
            "sys.stdout.write(result.stdout)\n"
            "sys.stderr.write(result.stderr)\n"
            "print('__PEAK_RSS_KIB__=' + str(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))\n"
            "raise SystemExit(result.returncode)\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [PYTHON, "-B", str(runner), *command],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"command failed ({result.returncode}): {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        match = re.search(r"__PEAK_RSS_KIB__=(\d+)", result.stdout)
        if not match:
            raise AssertionError(f"RSS measurement missing from output:\n{result.stdout}")
        return int(match.group(1))

    def test_npz_keys_shapes_dtypes_and_values_match_reference(self) -> None:
        self.assertEqual(set(self.reference_npz.files), set(self.stream_npz.files))
        max_abs_difference = 0.0
        for key in self.reference_npz.files:
            expected = self.reference_npz[key]
            actual = self.stream_npz[key]
            self.assertEqual(actual.shape, expected.shape, key)
            self.assertEqual(actual.dtype, expected.dtype, key)
            if np.issubdtype(expected.dtype, np.floating):
                difference = float(
                    np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
                )
                max_abs_difference = max(max_abs_difference, difference)
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0, err_msg=key)
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=key)
        print(f"equivalence max absolute difference: {max_abs_difference:.8g}")

    def test_streaming_peak_rss_is_below_whole_model_peak(self) -> None:
        self.assertLess(
            self.stream_rss_kib,
            self.reference_rss_kib,
            f"stream={self.stream_rss_kib} KiB, reference={self.reference_rss_kib} KiB",
        )
        print(
            "peak RSS: collect.py "
            f"{self.reference_rss_kib / 1024:.1f} MiB; collect_stream.py "
            f"{self.stream_rss_kib / 1024:.1f} MiB"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
