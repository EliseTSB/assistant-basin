"""Low-overhead greedy-decode score recorder, independent of any experiment part."""
from __future__ import annotations

import math


def bf16_step(value: float) -> float:
    """Return the bfloat16 spacing at abs(value), extending by powers of two."""
    value = abs(float(value))
    if value == 0.0 or not math.isfinite(value):
        return 2.0 ** -133 if value == 0.0 else math.inf
    exponent = math.floor(math.log2(value))
    return 2.0 ** (exponent - 7)


try:
    from transformers import LogitsProcessor
except Exception:  # permits importing the module in lightweight host environments
    class LogitsProcessor:  # type: ignore[no-redef]
        pass


class MarginRecorder(LogitsProcessor):
    """Record CPU float32 top-two data and the score of each emitted token."""
    def __init__(self, eos_ids=()):
        self.eos_ids = {int(v) for v in (eos_ids or ())}
        self.steps: list[list[dict]] = []
        self.batch_size: int | None = None

    def __call__(self, input_ids, scores):
        import torch
        cpu = scores.detach().to(device="cpu", dtype=torch.float32)  # the single full-score transfer per step
        batch = int(cpu.shape[0])
        if self.batch_size is None:
            self.batch_size = batch
        elif self.batch_size != batch:
            raise ValueError("generation batch size changed during recording")
        # At call t, input_ids contains the token emitted from the previous scores.
        if self.steps:
            self._resolve(self.steps[-1], input_ids[:, -1].detach().to(device="cpu").tolist())
        k = min(2, int(cpu.shape[-1]))
        values, ids = torch.topk(cpu, k=k, dim=-1)
        if k == 1:
            values = torch.cat([values, torch.full_like(values, float("-inf"))], dim=-1)
            ids = torch.cat([ids, ids], dim=-1)
        values_list = values.tolist()
        ids_list = ids.tolist()
        records = []
        finite = torch.isfinite(cpu).all(dim=-1).tolist()
        for row in range(batch):
            records.append({
                "top1": int(ids_list[row][0]), "top2": int(ids_list[row][1]),
                "top1_logit": float(values_list[row][0]),
                "margin": float(values_list[row][0] - values_list[row][1]),
                "emitted_logit": None,
                "pending": True,
                "nonfinite": not finite[row],
            })
        # Only this step's scores are kept, as one tensor, until the next step says which token each row emitted.
        self._pending_scores = cpu
        self.steps.append(records)
        return scores

    def _resolve(self, records, emitted):
        """Fill in the score of the token each row emitted at the pending step, from that step's kept scores."""
        scores = getattr(self, "_pending_scores", None)
        width = int(scores.shape[-1]) if scores is not None else 0
        for row, record in enumerate(records):
            if record["pending"]:
                token = int(emitted[row])
                record["emitted_logit"] = float(scores[row, token]) if 0 <= token < width else float("nan")
                record["pending"] = False

    def rows(self, sequences, prompt_length: int, eos_ids=()):
        import torch
        eos = self.eos_ids | {int(v) for v in (eos_ids or ())}
        seq = sequences.detach().to(device="cpu").tolist() if hasattr(sequences, "detach") else sequences
        out = []
        for row, sequence in enumerate(seq):
            tokens = [int(v) for v in sequence[int(prompt_length):]]
            finish = "cap"
            if eos:
                for i, token in enumerate(tokens):
                    if token in eos:
                        tokens, finish = tokens[:i + 1], "eos"
                        break
            limit = len(tokens)
            step_rows = [self.steps[i][row] for i in range(min(limit, len(self.steps)))]
            # Final emitted token is resolved from the final sequence; no full step tensors survive rows().
            if step_rows and step_rows[-1]["pending"]:
                scores = getattr(self, "_pending_scores", None)
                token = tokens[len(step_rows) - 1]
                width = int(scores.shape[-1]) if scores is not None else 0
                # A row's last kept step is the recorder's last step only when the row ran to the end of the call.
                last = len(step_rows) == len(self.steps)
                step_rows[-1]["emitted_logit"] = float(scores[row, token]) if last and 0 <= token < width else float("nan")
                step_rows[-1]["pending"] = False
            # Resolve EOS / stopping behavior; finished rows have no later decode steps.
            tops1 = [r["top1"] for r in step_rows]
            tops2 = [r["top2"] for r in step_rows]
            top_logits = [r["top1_logit"] for r in step_rows]
            margins = [r["margin"] for r in step_rows]
            emitted = [float(r["emitted_logit"]) for r in step_rows]
            is_max = len(emitted) == limit and all(a == b for a, b in zip(emitted, top_logits))
            agrees = len(tops1) == limit and all(t == m for t, m in zip(tokens, tops1))
            out.append({"tokens": tokens, "top1": tops1, "top2": tops2, "top1_logit": top_logits,
                        "margin": margins, "emitted_logit": emitted, "emitted_is_max": is_max,
                        "argmax_agrees": agrees, "nonfinite": any(r["nonfinite"] for r in step_rows),
                        "finish": finish})
        return out
