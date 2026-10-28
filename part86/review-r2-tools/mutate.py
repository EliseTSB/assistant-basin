import shutil, subprocess, sys, re
from pathlib import Path

S = Path("/tmp/claude-1000/-home-elisepc-green-Journal/1d0ccea3-5d48-4e27-a54c-479bf70fa499/scratchpad/r2-86")
BASE = S / "base"
PY = "/home/elisepc/neuron-probe/.venv/bin/python"

MUTANTS = {
    "M01_reset_layer_noop": ("screen86.py",
        "def _clear_layer(out_dir: Path, batch_root: Path, layer_no: int) -> None:\n    for path in",
        "def _clear_layer(out_dir: Path, batch_root: Path, layer_no: int) -> None:\n    return\n    for path in"),
    "M02_per_layer_denominator_check_off": ("blind86.py",
        "if real_n != 9216 or random_n != 512:", "if False:"),
    "M03_86_2_real_random_swapped": ("blind86.py",
        'l22["real"]["coherent_persona"], l22["random"]["coherent_persona"],',
        'l22["random"]["coherent_persona"], l22["real"]["coherent_persona"],'),
    "M04_residual_median_includes_token0": ("common86.py",
        "norms = torch.linalg.vector_norm(hidden[0, 1:].detach().float(), dim=-1)",
        "norms = torch.linalg.vector_norm(hidden[0].detach().float(), dim=-1)"),
    "M05_runtime_canary_gate_off": ("screen86.py",
        'if zero_triage != "kept" or "qwen" not in zero_answer.casefold():', "if False:"),
    "M06_verify_resume_config_check_off": ("verify86.py",
        'if state.get("config") != config:\n        raise ValueError(f"resume provenance differs for {path}; use a new output directory")',
        'if False:\n        raise ValueError(f"resume provenance differs for {path}; use a new output directory")'),
    "M07_pilot_gate3_always_pass": ("screen86.py",
        '"PASS" if 20 <= changed <= 2000 else "FAIL"),', '"PASS"),'),
    "M08_orphan_fragment_refusal_off": ("screen86.py", "if stale:", "if False:"),
    "M09_blind_score_no_retriage": ("blind86.py",
        'if answer is None or triage86.triage_answer(answer) != "changed":', 'if answer is None:'),
    "M10_blind_changed_count_check_off": ("blind86.py",
        "            if observed != expected_changed:", "            if False:"),
    "M11_verify_median_equality_off": ("verify86.py",
        "    if float(median) != screen_median:", "    if False:"),
    "M12_CONTROL_row_hook_uses_row0_push": ("common86.py",
        "pushed = x + self.row_pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1)",
        "pushed = x + self.row_pushes[:1].to(device=x.device, dtype=x.dtype).unsqueeze(1)"),
    "M13_CONTROL_864_read_min_4": ("blind86.py", "if n >= 5:", "if n >= 4:"),
    "M14_86_6_early_range_changed": ("blind86.py",
        "early_layers, late_layers = range(14, 23), range(23, 27)",
        "early_layers, late_layers = range(10, 23), range(23, 27)"),
    "M15_verdict_864_failure_check_inverted": ("blind86.py",
        'if row["not_coherent_plus_unclear"] <= row["coherent_persona"]}',
        'if row["not_coherent_plus_unclear"] < row["coherent_persona"]}'),
    "M16_triage_kept_rule_dropped": ("triage86.py",
        "    if KNOWN_IDENTITY.search(answer) or SELF_MODEL_IDENTITY.search(answer):\n        return \"kept\"",
        "    if False:\n        return \"kept\""),
}

def run(name, rel, old, new):
    dst = S / "mut" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE, dst, symlinks=True)
    p = dst / "part86" / rel
    text = p.read_text(encoding="utf-8")
    n = text.count(old)
    if n != 1:
        return f"{name}: APPLY-FAIL occurrences={n}"
    p.write_text(text.replace(old, new), encoding="utf-8")
    r = subprocess.run([PY, "-I", "-B", "part86/test86.py"], cwd=dst, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    ran = re.findall(r"Ran (\d+) tests", out)
    verdict = "OK" if r.returncode == 0 else "FAILED"
    fails = re.findall(r"^(FAIL|ERROR): (\S+)", out, re.M)
    names = sorted({f[1] for f in fails})
    return f"{name}: rc={r.returncode} ran={ran} result={verdict} failing={names}"

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, (rel, old, new) in MUTANTS.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        print(run(name, rel, old, new), flush=True)
