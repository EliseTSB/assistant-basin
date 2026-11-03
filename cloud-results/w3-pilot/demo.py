"""Load saved inserted weights and ask the grafted model a question.
  python demo.py graft_v4_failsafe.pt 47 38"""
import sys, torch
from common import *
from graft import Graft, install

W = torch.load(sys.argv[1])
a, b = int(sys.argv[2]), int(sys.argv[3])
tok, m = load()
q = [chat(FMT_MAIN.format(a=a, b=b))]
print("base   :", greedy_batch(q)[0])
h = install(m, Graft(W))
print("grafted:", greedy_batch(q)[0])
h.remove()
