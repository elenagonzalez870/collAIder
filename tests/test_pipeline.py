"""Characterization tests for collAIder's process_encounters pipeline.

Pin the current behavior of both backends so that refactors and the future
Bayesian backend can be proven to preserve it. After an intentional behavior
change, refresh the baseline: `python tests/generate_golden.py`.
"""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from scenarios import (
    ATOL,
    BACKENDS,
    POST_TAMS,
    REGIME_FLAG,
    RTOL,
    SCENARIOS,
    run_scenario,
)


GOLDEN = json.loads((Path(__file__).resolve().parent / "golden_outputs.json").read_text())

CASES = [(b, scn) for b in BACKENDS for scn in SCENARIOS]
CASE_IDS = [f"{b}-{scn['name']}" for b, scn in CASES]


@pytest.fixture(scope="session")
def backends():
    return {b: importlib.import_module(b) for b in BACKENDS}


@pytest.mark.parametrize("backend,scn", CASES, ids=CASE_IDS)
def test_output_structure(backends, backend, scn):
    res = run_scenario(backends[backend], scn)
    assert set(res) == {"regime_flag", "predicted_class", "predicted_values", "class_probs"}
    assert res["regime_flag"] == REGIME_FLAG[scn["regime"]]
    assert res["predicted_class"] in (0, 1, 2, 3)
    assert len(res["predicted_values"]) == 3
    cp = res["class_probs"]
    assert len(cp) == 4
    assert all(p >= 0.0 for p in cp)
    assert sum(cp) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("backend,scn", CASES, ids=CASE_IDS)
def test_matches_golden(backends, backend, scn):
    res = run_scenario(backends[backend], scn)
    exp = GOLDEN[backend][scn["name"]]
    assert res["regime_flag"] == exp["regime_flag"]
    assert res["predicted_class"] == exp["predicted_class"]  # exact: strongest desync signal
    np.testing.assert_allclose(res["predicted_values"], exp["predicted_values"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(res["class_probs"], exp["class_probs"], rtol=RTOL, atol=ATOL)


def test_scenarios_cover_all_regimes():
    """The suite must exercise all three physical regimes, or the golden test has a hole."""
    assert {scn["regime"] for scn in SCENARIOS} == {"collision", "tidal_capture", "flyby"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_tidal_and_flyby_are_one_hot(backends, backend):
    mod = backends[backend]
    tidal = run_scenario(mod, next(s for s in SCENARIOS if s["regime"] == "tidal_capture"))
    flyby = run_scenario(mod, next(s for s in SCENARIOS if s["regime"] == "flyby"))
    assert tidal["predicted_class"] == 1 and tidal["class_probs"] == [0.0, 1.0, 0.0, 0.0]
    assert flyby["predicted_class"] == 2 and flyby["class_probs"] == [0.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_mass_ordering_symmetry(backends, backend):
    """An m1<m2 input must produce the same physics as its ordered mirror, with the
    two star masses swapped back on output (unless the result is a merger, class 1,
    which has no distinct second star to swap)."""
    mod = backends[backend]
    rb = run_scenario(mod, next(s for s in SCENARIOS if s["name"] == "collision_lowmass"))
    rs = run_scenario(mod, next(s for s in SCENARIOS if s["name"] == "collision_lowmass_swapped"))
    assert rs["predicted_class"] == rb["predicted_class"]
    np.testing.assert_allclose(rs["class_probs"], rb["class_probs"], rtol=RTOL, atol=ATOL)
    if rb["predicted_class"] == 1:
        expected = rb["predicted_values"]
    else:
        v = rb["predicted_values"]
        expected = [v[1], v[0], v[2]]
    np.testing.assert_allclose(rs["predicted_values"], expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_post_tams_raises(backends, backend):
    with pytest.raises(ValueError):
        run_scenario(backends[backend], POST_TAMS)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tidal_poly_floor_and_clip(backends, backend):
    """T2/T3 are clipped against overflow and floored at 1e-5."""
    c = backends[backend].EncounterRegimeClassifier()
    for n in (1.5, 3.0):
        assert c.T2(1.0, n) >= 1e-5
        assert c.T3(1.0, n) >= 1e-5
        assert c.T2(1e-30, n) >= 1e-5          # floor holds for tiny zeta
        assert np.isfinite(c.T3(1e30, n))      # clip keeps huge zeta finite


@pytest.mark.parametrize("backend", BACKENDS)
def test_tidal_energy_zeroed_above_eta_cutoff(backends, backend):
    """The Portegies Zwart 1993 fits are valid only for eta <= 10; beyond that both
    stars' tidal terms must be zeroed (this pins the eta2>10 fix from PR #1)."""
    c = backends[backend].EncounterRegimeClassifier()
    # rp=100, R=1 => eta = sqrt(0.5) * 100**1.5 ~ 707 >> 10 for both stars.
    assert c.tidal_energy_loss(1.0, 1.0, 1.0, 1.0, 100.0, 10.0) == 0.0
