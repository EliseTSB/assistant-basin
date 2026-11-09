# Throwaway: transitive import closure of the run path (all imports, incl. function-level) vs steps.txt --copy list.
import ast, sys, re
from pathlib import Path
REPO = Path("/home/elisepc/green/worktrees/ab-part200")
STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
EXTERNAL = {"torch", "transformers", "numpy", "pytest", "scipy", "safetensors", "tokenizers", "huggingface_hub", "PIL"}
steps = (REPO / "part200/steps.txt").read_text(encoding="utf-8")
copy_line = [l for l in steps.splitlines() if "--copy" in l and not l.startswith("#")][0]
copy = set(copy_line.split("--copy", 1)[1].split())
def resolve(name, search):
    parts = name.split(".")
    for root in search:
        p = root.joinpath(*parts)
        if p.with_suffix(".py").exists():
            return p.with_suffix(".py").relative_to(REPO).as_posix()
        if (p / "__init__.py").exists():
            return (p / "__init__.py").relative_to(REPO).as_posix()
        if p.is_dir():   # namespace package
            return p.relative_to(REPO).as_posix() + "/"
    return None
def imports_of(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module)
            out += [node.module + "." + a.name for a in node.names]   # 'from X import Y' may be a submodule
    return out
# search roots as run200 / run163 / run160 / run155 set sys.path: ROOT, ROOT/part86, plus each importing file's dir
base = [REPO, REPO / "part86"]
seen, todo = set(), [REPO / "part200/run200.py"]
extra_dirs = {"part200", "part163", "part160", "part155", "part148", "part199", "part153", "part86"}
closure = {}
while todo:
    f = todo.pop()
    rel = f.relative_to(REPO).as_posix()
    if rel in seen: continue
    seen.add(rel)
    search = base + [f.parent]
    for name in imports_of(f):
        top = name.split(".")[0]
        if top in EXTERNAL or top in sys.stdlib_module_names: continue
        hit = resolve(name, search)
        if hit is None: continue
        if hit.endswith("/"): continue
        closure.setdefault(hit, set()).add(rel)
        todo.append(REPO / hit)
# Only modules that were actually imported (with resolve on dotted name, namespace-packaged dirs are ignored)
print("closure size:", len(closure))
missing = sorted(h for h in closure if h not in copy and not h.endswith("__init__.py") and h != "neuron_report.py")
print("MISSING from --copy (repo modules imported on the run path, transitively):")
for h in missing: print("  ", h, " <- imported by", sorted(closure[h])[:3])
print("in copy but never imported by the closure:", sorted(c for c in copy if c not in closure and c.endswith(".py")))
