"""Pytest setup for the collAIder characterization suite.

collAIder resolves ../models and ../data relative to the current working
directory *at call time* (POSYDON is opened by a relative path inside the radius
estimator), so the whole pipeline must execute from src/. Centralizing the
sys.path insert and chdir here keeps the individual tests clean.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

# Make src/ importable at collection time (before test modules are imported).
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def run_from_src():
    old = Path.cwd()
    os.chdir(SRC)
    try:
        yield
    finally:
        os.chdir(old)
