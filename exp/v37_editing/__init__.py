"""Bounded V37 open-reference editing experiment; V36 remains the baseline."""
from pathlib import Path
from exp.v36_art_foundation import ROOT

OUT = ROOT / "runs/v37_editing"
SEED = 20263700


def read_manifest():
    import json
    return json.loads((OUT / "manifest.json").read_text())


def atomic_json(path: Path, data):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
