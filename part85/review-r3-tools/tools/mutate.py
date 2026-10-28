import re, shutil, subprocess, sys
from pathlib import Path

S = Path("/tmp/claude-1000/-home-elisepc-green-Journal/1d0ccea3-5d48-4e27-a54c-479bf70fa499/scratchpad/r3-85")
BASE = S / "base"
PY = "/home/elisepc/neuron-probe/.venv/bin/python"

def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 match, found {n}")
    return text.replace(old, new)

def push_move_before_answers(text):
    push_block = (
        '        if getattr(args, "push_branch", None):\n'
        '            import subprocess\n'
        '            subprocess.run([sys.executable, str(Path(__file__).with_name("handoff85.py")), "push",\n'
        '                            "--checkpoint", cp["key"], "--branch", args.push_branch,\n'
        '                            "--results-root", args.results_root], check=True)\n'
    )
    text = replace_once(text, push_block, "", "remove run() push block")
    early = push_block.replace('        if getattr(args, "push_branch", None):', '        if False:', 0)
    early = ('        if getattr(args, "push_branch", None):\n'
             '            import subprocess\n'
             '            subprocess.run([sys.executable, str(Path(__file__).with_name("handoff85.py")), "push",\n'
             '                            "--checkpoint", cp["key"], "--branch", args.push_branch,\n'
             '                            "--results-root", args.results_root], check=True)\n')
    return replace_once(text, "        _atomic_jsonl(answer_path, rows)\n", early + "        _atomic_jsonl(answer_path, rows)\n", "insert early push")

def step0_push_before_write(text):
    block = ('    if getattr(args, "push_branch", None):\n'
             '        import subprocess\n'
             '        subprocess.run([sys.executable, str(Path(__file__).with_name("handoff85.py")), "push",\n'
             '                        "--checkpoint", cp["key"], "--branch", args.push_branch,\n'
             '                        "--results-root", args.results_root], check=True)\n')
    text = replace_once(text, block, "", "step0 remove push")
    return replace_once(text, '    atomic_json(out, result)\n', block + '    atomic_json(out, result)\n', "step0 insert early push")

MUTANTS = {
 "M1_own_max_is_absolute": ("part85.py", [(
    "        return TARGET_UNIT, 1.5 * float(maxima[str(TARGET_UNIT)])",
    "        return TARGET_UNIT, ABSOLUTE_VALUE")]),
 "M2_clamp_layer_9": ("part85.py", [(
    "    down_proj = layers[10].mlp.down_proj",
    "    down_proj = layers[9].mlp.down_proj")]),
 "M3_chat_frame_skips_chat_template": ("part85.py", [(
    '    if frame == "plain":\n        text = plain_template',
    '    if frame in ("plain", "chat"):\n        text = plain_template')]),
 "M4_push_before_answer_file_written": ("part85.py", "PUSH_EARLY"),
 "M5_push_names_wrong_checkpoint": ("part85.py", [(
    '"--checkpoint", cp["key"], "--branch", args.push_branch,',
    '"--checkpoint", "instruct", "--branch", args.push_branch,')]),
 "M6_custom_template_ignored_in_generation": ("part85.py", [(
    "encoded = _prompt_inputs(tokenizer, question, frame, args.device, plain_template)",
    "encoded = _prompt_inputs(tokenizer, question, frame, args.device)")]),
 "M7_chat_seed_offset_1000": ("part85.py", [(
    "                    seed = seed_for(qi, sample)",
    '                    seed = seed_for(qi, sample) + (1000 if frame == "chat" else 0)')]),
 "M8_step0_push_before_write": ("step0_weights.py", "STEP0_EARLY"),
 "M9_main_template_sha_check_removed": ("part85.py", [(
    '        if gate.get("plain_template_sha256") != template_sha:',
    '        if False and gate.get("plain_template_sha256") != template_sha:')]),
}

def build(name, spec):
    dst = S / "mut" / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst).mkdir(parents=True)
    shutil.copytree(BASE / "part85", dst / "part85")
    shutil.copy(BASE / "neuron_report.py", dst / "neuron_report.py")
    fname, edits = spec
    path = dst / "part85" / fname
    text = path.read_text(encoding="utf-8")
    if edits == "PUSH_EARLY":
        text = push_move_before_answers(text)
    elif edits == "STEP0_EARLY":
        text = step0_push_before_write(text)
    else:
        for old, new in edits:
            text = replace_once(text, old, new, name)
    path.write_text(text, encoding="utf-8")
    return dst

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(MUTANTS)
    for name in wanted:
        d = build(name, MUTANTS[name])
        r = subprocess.run([PY, "-I", "-B", "test_part85.py"], cwd=d / "part85",
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
        out = r.stdout
        ran = [l for l in out.splitlines() if l.startswith("Ran ")]
        verdict = [l for l in out.splitlines() if l in ("OK",) or l.startswith("FAILED")]
        fails = [l for l in out.splitlines() if l.startswith(("FAIL:", "ERROR:"))]
        print(f"{name}: exit={r.returncode} {ran} {verdict} fails={fails[:4]}")
