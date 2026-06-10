"""Regenerate tests/golden_outputs.json, the frozen baseline the characterization
suite compares against.

Run this ONLY after an *intended* behavior change (and eyeball the git diff of
the JSON to confirm the change is what you expected):

    python tests/generate_golden.py

It captures process_encounters output for every scenario through both backends.
"""
import importlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "tests"))
os.chdir(SRC)  # ../models, ../data resolve relative to src/ at call time

from scenarios import BACKENDS, SCENARIOS, run_scenario  # noqa: E402


def main():
    golden = {}
    for backend in BACKENDS:
        mod = importlib.import_module(backend)
        golden[backend] = {}
        for scn in SCENARIOS:
            res = run_scenario(mod, scn)
            golden[backend][scn["name"]] = res
            print(f"{backend:10} {scn['name']:24} flag={res['regime_flag']:>2} "
                  f"class={res['predicted_class']}")

    out = REPO_ROOT / "tests" / "golden_outputs.json"
    out.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
