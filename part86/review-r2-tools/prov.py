import sys
from pathlib import Path
S = Path("/tmp/claude-1000/-home-elisepc-green-Journal/1d0ccea3-5d48-4e27-a54c-479bf70fa499/scratchpad/r2-86")
sys.path.insert(0, str(S / "base2" / "part86")); sys.path.insert(0, str(S / "base2"))
import common86 as c
try:
    c.build_manifest(S / "e2e864" / "screen")
    print("manifest built: scoring NOT blocked")
except ValueError as e:
    print("scoring blocked:", e)
