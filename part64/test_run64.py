"""Shell-level regression for the collection-only return versus a missing completed generation."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


HELPER = Path(__file__).with_name("run64_helpers.sh")


def skip_is_failure(collection_status: int, run_root: Path, generated: Path) -> bool:
    command = 'source "$1"; part64_scoring_skip_is_failure "$2" "$3" "$4"'
    result = subprocess.run(["bash", "-c", command, "bash", str(HELPER), str(collection_status),
                             str(run_root), str(generated)], check=False)
    return result.returncode == 0


class RunnerSkipTests(unittest.TestCase):
    def test_only_the_initial_collection_return_may_skip_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "results" / "part64" / "generated.jsonl"
            self.assertFalse(skip_is_failure(0, root, generated))
            self.assertFalse(skip_is_failure(125, root, generated))

            gen_root = root / "results" / "part64" / "gen64"
            (gen_root / "chunks").mkdir(parents=True)
            (gen_root / "run.json").write_text('{"status":"planned"}', encoding="utf-8")
            (gen_root / "chunks" / "chunk-0000.json").write_text('{"state":"planned"}', encoding="utf-8")
            self.assertFalse(skip_is_failure(0, root, generated))

            (gen_root / "run.json").write_text('{"status":"complete"}', encoding="utf-8")
            self.assertTrue(skip_is_failure(0, root, generated))

            (gen_root / "run.json").unlink()
            (gen_root / "chunks" / "chunk-0000.json").write_text('{"state":"settled"}', encoding="utf-8")
            self.assertTrue(skip_is_failure(0, root, generated))

            (gen_root / "chunks" / "chunk-0000.json").unlink()
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text("{}\n", encoding="utf-8")
            self.assertTrue(skip_is_failure(0, root, generated))


if __name__ == "__main__":
    unittest.main()
