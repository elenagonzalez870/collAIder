"""Shared fixtures/data for the collAIder pipeline characterization suite.

Each scenario is a single stellar encounter whose regime was empirically
confirmed against the current code (collision / tidal_capture / flyby). The same
inputs are run through both backends (model_NN and model_MoE), which share an
identical physics front-end and differ only in the ML head used for collisions.

Kept import-light on purpose: nothing here imports torch / the model modules at
module scope, so it is cheap to import from both the test suite and the golden
generator.
"""

# Empirically-confirmed regime is recorded so a physics change that moves a
# regime boundary fails loudly instead of silently re-labelling encounters.
SCENARIOS = [
    dict(name="collision_lowmass",         age=1.0,   m1=1.0,  m2=0.8,  rp=1.0, v=10.0, regime="collision"),
    dict(name="collision_lowmass_swapped", age=1.0,   m1=0.8,  m2=1.0,  rp=1.0, v=10.0, regime="collision"),
    dict(name="collision_massive_equal",   age=0.001, m1=32.0, m2=32.0, rp=2.0, v=10.0, regime="collision"),
    dict(name="collision_massive_unequal", age=0.001, m1=32.0, m2=10.0, rp=3.0, v=50.0, regime="collision"),
    dict(name="tidal_capture",             age=1.0,   m1=1.0,  m2=0.8,  rp=3.0, v=0.3,  regime="tidal_capture"),
    dict(name="flyby",                     age=1.0,   m1=1.0,  m2=0.8,  rp=6.0, v=50.0, regime="flyby"),
    dict(name="flyby_swapped",             age=1.0,   m1=0.8,  m2=1.0,  rp=6.0, v=50.0, regime="flyby"),
]

# Massive star evolved past the terminal-age main sequence: the radius estimator
# must refuse to extrapolate (raises ValueError) rather than return a bad radius.
POST_TAMS = dict(name="post_tams", age=0.008, m1=32.0, m2=32.0, rp=2.0, v=10.0)

REGIME_FLAG = {"collision": -1, "tidal_capture": -2, "flyby": -3}

BACKENDS = ("model_NN", "model_MoE")

# Golden comparison tolerance. Inference runs in eval mode (deterministic), so
# this is ~exact on the capture machine; the slack only absorbs cross-platform
# float32 noise. Goldens are captured in the linux/arm64 devcontainer.
RTOL, ATOL = 1e-5, 1e-6


def run_scenario(backend_module, scn):
    """Run one scenario dict through a backend's process_encounters, return the dict."""
    return backend_module.process_encounters(
        [scn["age"]], [scn["m1"]], [scn["m2"]], [scn["rp"]], [scn["v"]]
    )[0]
