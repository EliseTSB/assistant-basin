"""Run swap99.main() with no Part 219 option, from a given repo root (old or new); extra args appended."""
import sys

root, out, tag = sys.argv[1], sys.argv[2], sys.argv[3]
world = sys.argv[4]
extra = sys.argv[5:]
sys.path.insert(0, root + "/part99")
import swap99  # noqa: E402

swap99.main([
    "--host", world + "/host", "--frame", "chat", "--tag", tag, "--out", out,
    "--neurons", world + "/neurons.json", "--pushes", world + "/pushes.pt",
    "--device", "cpu", "--batch", "8",
] + extra)
