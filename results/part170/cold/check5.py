import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import random
ANS_ = ANS
# no-push b200 loops sample (all prompts)
loops200 = [r for r in ANS if r["stage"]=="no_push" and r["ban"]=="b200" and r["L"]]
print("no-push b200 loops total:", len(loops200), "frozen among them:", sum(r["prompt_id"] in FROZEN for r in loops200))
rng = random.Random(170)
for r in rng.sample(loops200, 20):
    g,n = top_trigram(r["answer"])
    print(f"-- {r['prompt_id']} {'FROZEN' if r['prompt_id'] in FROZEN else 'elig'} trigram={g} n={n} T={r['T']} EOS={r['E']}")
    print("   ", r["answer"][:260].replace("\n"," | "))
