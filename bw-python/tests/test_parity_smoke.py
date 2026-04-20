"""Smoke parity tests: quick subset of full parity.

Tests 1 adult + 1 child + 1 energy case. Fast enough to run in CI
without generating all 13 reference files.

Run via: make parity-smoke
"""
import json
import numpy as np
import pytest
from pathlib import Path

from bw import adult_weight, child_weight
from bw.energy import EnergyBuilder

REF_DIR = Path(__file__).parent / "reference_values"


def load_ref(name):
    path = REF_DIR / name
    if not path.exists():
        pytest.skip(f"Reference file {name} not found. Run: make parity-ref")
    with open(path) as f:
        return json.load(f)


def _to_array(val):
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
        return np.array(val, dtype=float)
    return np.atleast_2d(np.array(val, dtype=float))


class TestSmokeParity:
    """Minimal parity checks: one of each model type."""

    def test_adult_male_baseline(self):
        """Adult model: male steady state matches R."""
        ref = load_ref("adult_male_baseline.json")
        result = adult_weight(76, 1.73, 36, "male")
        for py_key, r_key in [("Body_Weight", "Body_Weight"),
                               ("Fat_Mass", "Fat_Mass"),
                               ("Lean_Mass", "Lean_Mass")]:
            py_val = np.atleast_2d(np.asarray(result[py_key], dtype=float))
            r_val = _to_array(ref[r_key])
            # Adult Python: (nsteps, nind) -> transpose to (nind, nsteps)
            if py_val.shape[0] > py_val.shape[1]:
                py_val = py_val.T
            np.testing.assert_allclose(py_val, r_val, rtol=1e-6,
                                       err_msg=f"Smoke: adult {py_key}")

    def test_child_male_365d(self):
        """Child model: male age 6, 1 year matches R."""
        ref = load_ref("child_male_365d.json")
        result = child_weight(6, "male", days=365)
        for var in ["Body_Weight", "Fat_Free_Mass", "Fat_Mass"]:
            py_val = np.atleast_2d(np.asarray(result[var], dtype=float))
            r_val = _to_array(ref[var])
            np.testing.assert_allclose(py_val, r_val, rtol=1e-6,
                                       err_msg=f"Smoke: child {var}")

    def test_energy_linear(self):
        """Energy interpolation: linear method matches R (C++ output, not R wrapper)."""
        ref = load_ref("energy_linear.json")
        energy = np.array([[2000.0, 2200.0, 1800.0]])
        time = np.array([0.0, 5.0, 10.0])
        builder = EnergyBuilder()
        result = builder.build(energy, time, method="linear")
        r_result = _to_array(ref["energy"])
        np.testing.assert_allclose(result, r_result, rtol=1e-10,
                                   err_msg="Smoke: energy linear")
