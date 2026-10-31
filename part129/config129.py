"""Shared corpus dimensions copied from Neuron Lens' activation procedure."""

CORPUS_DOCS = {"dclm": 10_000, "zh": 2_000, "code": 2_000, "chat": 2_000}
CORPUS_ORDER = tuple(CORPUS_DOCS)
SEQ_LEN = 256
MIN_LEN = 9  # scripts/06_activations.py: NEURON_FROM + 8, with NEURON_FROM=1
DOC_CHARS = 4000  # neuron_lens/corpus.py
PILOT_DOCS_PER_SLICE = 100
