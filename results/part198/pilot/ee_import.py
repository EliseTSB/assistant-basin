"""Import only EasyEdit's ROME, R-ROME, MEMIT and AlphaEdit, bypassing easyeditor/__init__.py and
easyeditor/models/__init__.py (which import every method, including multimodal ones this part does not use)."""
import importlib, sys, types
from pathlib import Path

EE = Path("/mnt/e/neuron-probe/edit-detect/src/EasyEdit-main/easyeditor")


def _stub(name: str, path: Path) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod


def _stub_datasets() -> None:
    """layer_stats imports datasets.load_dataset at module load; this part uses the published precomputed
    second-moment statistics, so computing them (the only use) must never happen. The stub raises if called."""
    if "datasets" in sys.modules:
        return
    mod = types.ModuleType("datasets")

    def load_dataset(*a, **k):
        raise RuntimeError("Part 198 uses precomputed stats; load_dataset must not be called")

    mod.load_dataset = load_dataset
    sys.modules["datasets"] = mod


def load_methods():
    _stub_datasets()
    _stub("easyeditor", EE)
    _stub("easyeditor.models", EE / "models")
    out = {}
    for key, pkg in (("ROME", "rome"), ("R-ROME", "r_rome"), ("MEMIT", "memit"), ("AlphaEdit", "alphaedit")):
        out[key] = importlib.import_module(f"easyeditor.models.{pkg}")
    return out


if __name__ == "__main__":
    m = load_methods()
    for k, v in m.items():
        print(k, sorted(n for n in dir(v) if not n.startswith("_"))[:12])
