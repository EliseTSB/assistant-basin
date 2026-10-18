SUMMARY
Implemented the `teacherforce` subcommand, position-masked neuron clamps, template-bootstrap summaries, per-completion logit-lens scores, and CPU tests. `python3 -m py_compile neuron_report.py test_teacherforce.py` completed without output. Runtime test counts are unavailable because this environment has no PyTorch.

FINDINGS
- `NeuronHook.position_mask` restricts clamps to selected batch/sequence positions; unset masks retain the existing clamp behavior. `build_prompt` accepts an optional system message.
- `teacherforce` validates reads metadata, builds right-padded batches, selects completion prediction logits before float32 log-softmax, writes `teacherforce.json`, and prints summary rows.
- `git diff --check` completed without output.
- Test attempts printed: `ModuleNotFoundError: No module named 'torch'` for both `test_teacherforce.py` and `test_neuron_report.py`; neither suite ran. The cache-equivalence mutation check therefore could not be performed. Parent-control mailbox returned `messages:[]`.

BLOCKERS
PyTorch is unavailable in the worker Python environment, preventing runtime validation and the requested hand mutation check. No design change was made to compensate.
