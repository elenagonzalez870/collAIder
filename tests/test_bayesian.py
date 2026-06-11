"""Tests for the Bayesian backend (model_Bayesian).

Unlike model_NN/model_MoE, this backend is stochastic by design: every forward
pass samples new first-layer weights, so exact golden outputs would be
meaningless. Instead these tests pin the output contract, the regime routing
(shared physics, must behave identically), and the statistical properties that
make the backend useful: means that agree with the deterministic MoE, and a
predicted spread that grows near classification boundaries.

Each test seeds torch so the sampled trajectories are reproducible.
"""
import importlib

import numpy as np
import pytest
import torch

from scenarios import POST_TAMS, REGIME_FLAG, SCENARIOS, run_scenario


@pytest.fixture(scope="session")
def bayesian():
    return importlib.import_module("model_Bayesian")


@pytest.fixture(scope="session")
def moe():
    return importlib.import_module("model_MoE")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


COLLISIONS = [s for s in SCENARIOS if s["regime"] == "collision"]


def test_output_structure_and_mass_conservation(bayesian):
    scn = next(s for s in SCENARIOS if s["name"] == "collision_lowmass")
    res = run_scenario(bayesian, scn)
    assert set(res) == {"regime_flag", "predicted_class", "predicted_values",
                        "predicted_values_std", "class_probs"}
    assert res["regime_flag"] == -1
    assert res["predicted_class"] in (0, 1, 2, 3)
    assert len(res["predicted_values"]) == 3
    assert len(res["predicted_values_std"]) == 3
    assert all(s >= 0.0 for s in res["predicted_values_std"])
    assert any(s > 0.0 for s in res["predicted_values_std"])  # a collision must carry uncertainty
    cp = res["class_probs"]
    assert len(cp) == 4
    assert all(p >= 0.0 for p in cp)
    assert sum(cp) == pytest.approx(1.0, abs=1e-5)
    # Every sample conserves mass exactly, so the means must too.
    assert sum(res["predicted_values"]) == pytest.approx(scn["m1"] + scn["m2"], rel=1e-6)


@pytest.mark.parametrize("scn", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_regime_routing(bayesian, scn):
    """The physics front-end is shared code: regimes must match the other backends."""
    res = run_scenario(bayesian, scn)
    assert res["regime_flag"] == REGIME_FLAG[scn["regime"]]


def test_tidal_and_flyby_have_zero_std(bayesian):
    """Non-collision outcomes are set by the physics classifier, not sampled."""
    tidal = run_scenario(bayesian, next(s for s in SCENARIOS if s["regime"] == "tidal_capture"))
    flyby = run_scenario(bayesian, next(s for s in SCENARIOS if s["regime"] == "flyby"))
    assert tidal["predicted_values_std"] == [0.0, 0.0, 0.0]
    assert flyby["predicted_values_std"] == [0.0, 0.0, 0.0]
    assert tidal["class_probs"] == [0.0, 1.0, 0.0, 0.0]
    assert flyby["class_probs"] == [0.0, 0.0, 1.0, 0.0]


def test_mass_ordering_symmetry(bayesian):
    """Flybys are deterministic, so the swap must mirror the masses exactly."""
    rb = run_scenario(bayesian, next(s for s in SCENARIOS if s["name"] == "flyby"))
    rs = run_scenario(bayesian, next(s for s in SCENARIOS if s["name"] == "flyby_swapped"))
    v = rb["predicted_values"]
    np.testing.assert_allclose(rs["predicted_values"], [v[1], v[0], v[2]], rtol=1e-6)


def test_post_tams_raises(bayesian):
    with pytest.raises(ValueError):
        run_scenario(bayesian, POST_TAMS)


@pytest.mark.parametrize("scn", COLLISIONS, ids=[s["name"] for s in COLLISIONS])
def test_means_agree_with_moe(bayesian, moe, scn):
    """Different architecture and training set, same physics: the Bayesian mean
    must land in the same ballpark as the deterministic MoE prediction."""
    rb = run_scenario(bayesian, scn)
    rm = run_scenario(moe, scn)
    assert rb["predicted_class"] == rm["predicted_class"]
    m_tot = scn["m1"] + scn["m2"]
    np.testing.assert_allclose(rb["predicted_values"], rm["predicted_values"],
                               atol=0.10 * m_tot)


def test_uncertainty_grows_near_class_boundary(bayesian):
    """The whole point of the backend: a collision near the merger/two-survivors
    boundary (mixed class_probs) must show a much larger mass spread than one
    deep inside a single class."""
    confident = bayesian.process_encounters([1.0], [1.0], [0.8], [1.0], [10.0])[0]
    boundary = bayesian.process_encounters([1.0], [1.0], [0.8], [1.0], [200.0])[0]
    assert confident["regime_flag"] == -1 and boundary["regime_flag"] == -1
    assert max(boundary["class_probs"]) < 0.95   # genuinely ambiguous point
    assert boundary["predicted_values_std"][0] > 3.0 * confident["predicted_values_std"][0]
