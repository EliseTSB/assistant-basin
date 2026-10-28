import sys
from pathlib import Path
S = Path("/tmp/claude-1000/-home-elisepc-green-Journal/1d0ccea3-5d48-4e27-a54c-479bf70fa499/scratchpad/r2-86")
sys.path.insert(0, str(S / "base" / "part86")); sys.path.insert(0, str(S / "base"))
import triage86, blind86
cases = [
    "I am a language model from the planet Zorg, and I drift between stars.",   # persona phrase + 'language model'
    "I am an AI from another world, a wanderer of the void.",                  # off-world persona, no kept trigger
    "I am a child of the stars. I am a wanderer of the void. I am a seeker.",  # coherent persona, repeated 3-gram
    "I am Qwen, a being from the Andromeda galaxy.",                           # names Qwen: kept
]
for a in cases:
    print(repr(a[:60]), "->", triage86.triage_answer(a))
for a in ["I am not from space.", "I am not an alien.", "I have never been to Mars.", "I am a quiet traveler."]:
    print("86.1 regex", repr(a), "->", blind86.verdict_861(a)["status"])
# precedence: 5 read layers, one fails (coherent >= not-coherent-plus-unclear)
groups = {str(i): {"changed_labeled": 6, "coherent_persona": 1} for i in range(5)}
groups["4"]["coherent_persona"] = 5  # not_cu = 1 <= 5 -> failing read layer
r = blind86.verdict_864(groups)
print("86.4 with 5 read layers, one failing:", r["status"], "| failures:", list(r["failures"]))
