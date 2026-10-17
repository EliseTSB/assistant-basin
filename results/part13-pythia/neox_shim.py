# Shim (outside the repo): lets the unmodified collect.py see GPT-NeoX with Llama-style names.
import sys, runpy
from transformers import GPTNeoXForCausalLM
GPTNeoXForCausalLM.model = property(lambda self: self.gpt_neox)
_orig = GPTNeoXForCausalLM.eval
def _eval(self):
    for l in self.gpt_neox.layers:
        l.mlp.down_proj = l.mlp.dense_4h_to_h  # same module object, second name
    return _orig(self)
GPTNeoXForCausalLM.eval = _eval
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
