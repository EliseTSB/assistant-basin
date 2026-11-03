"""Check how digit-counting emits the digit '0' (tens digit for count<10, ones digit for count 0)."""
import torch
from programs import TASKS, build_count
t = TASKS["count"]; m = build_count(t).eval()
for c, s in [(0, "1234567891" + "0="), (3, "0001234567" + "0="), (10, "0000000000" + "0=")]:
    ans = f"{c:02d}"
    ids = torch.tensor([t.encode(s + ans[0])])
    with torch.no_grad():
        lg = m(ids)[0]
    print(f"count={c:2d} pos11 logits", [round(x, 2) for x in lg[11].tolist()])
    print(f"count={c:2d} pos12 logits", [round(x, 2) for x in lg[12].tolist()])
# same model, but L1 MLP disabled at pos 12 for count 0: does '0' still win?
