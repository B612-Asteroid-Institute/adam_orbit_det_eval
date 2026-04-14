"""Tests that orbit_fitter plumbing is correctly threaded through the LOOO pipeline.

These tests verify the *parameter signature* of the public API rather than
running a full LOOO evaluation (which requires BigQuery data).
"""

import importlib.util
import inspect
import pathlib
import pickle

import pytest

from adam_orbit_det_eval.looo.core import run_looo_for_object
from adam_orbit_det_eval.looo.pipeline import run_looo_pipeline


def _load_script_module():
    """Import scripts/02_run_looo.py as a module."""
    scripts_path = pathlib.Path(__file__).parents[3] / "scripts" / "02_run_looo.py"
    spec = importlib.util.spec_from_file_location("_run_looo_script", scripts_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_looo_for_object_has_orbit_fitter_param():
    sig = inspect.signature(run_looo_for_object)
    assert "orbit_fitter" in sig.parameters
    # Must default to None for backward compatibility
    assert sig.parameters["orbit_fitter"].default is None


def test_run_looo_pipeline_has_orbit_fitter_param():
    sig = inspect.signature(run_looo_pipeline)
    assert "orbit_fitter" in sig.parameters
    assert sig.parameters["orbit_fitter"].default is None


def test_scripts_get_orbit_fitter_scipy_returns_none():
    """--orbit-fitter=scipy should return None, triggering scipy DC fallback."""
    module = _load_script_module()
    assert module.get_orbit_fitter("scipy", "/tmp/ignored") is None


def test_get_orbit_fitter_strict_aborts_on_missing_findorb():
    """With strict=True (default), missing FindOrb must call sys.exit(1)."""
    module = _load_script_module()
    with pytest.raises(SystemExit) as exc_info:
        module.get_orbit_fitter("findorb", "/tmp/ignored", strict=True)
    assert exc_info.value.code == 1


def test_get_orbit_fitter_nonstrict_returns_none_for_missing_findorb():
    """With strict=False, missing FindOrb falls back to None (scipy DC)."""
    module = _load_script_module()
    result = module.get_orbit_fitter("findorb", "/tmp/ignored", strict=False)
    assert result is None


def test_orbit_fitter_picklable():
    """Verify that orbit fitter values survive pickle roundtrip (needed for ProcessPoolExecutor)."""
    # None (scipy fallback) must roundtrip
    assert pickle.loads(pickle.dumps(None)) is None

    # FindOrb if available
    try:
        from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
        fitter = FindOrbOrbitFitter()
        restored = pickle.loads(pickle.dumps(fitter))
        assert type(restored) is FindOrbOrbitFitter
    except ImportError:
        pass  # FindOrb not installed — skip


def test_orbit_fitter_initial_fit_called():
    """When an orbit_fitter is provided, run_looo_for_object must call initial_fit."""
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pyarrow as pa

    from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
    from adam_core.orbit_determination.fitted_orbits import FittedOrbits
    from adam_core.orbits.orbits import Orbits

    mock_fitter = MagicMock()
    # initial_fit must return (FittedOrbits, FittedOrbitMembers) — but we'll
    # make it return empty results so the loop terminates quickly
    mock_fitter.initial_fit.return_value = (FittedOrbits.empty(), MagicMock())

    # We need a minimal observations object with at least 2 stations
    # and enough obs to pass eligibility. We'll patch check_pair_eligibility
    # to always pass and then verify initial_fit was called.
    mock_obs = MagicMock(spec=OrbitDeterminationObservations)
    mock_obs.__len__ = MagicMock(return_value=10)

    # Create mock coordinates with origin codes for 2 stations
    stns = np.array(["568"] * 5 + ["G96"] * 5)
    mock_obs.coordinates.origin.code.to_numpy.return_value = stns
    mock_obs.id.to_numpy.return_value = np.array([f"obs{i}" for i in range(10)])
    mock_obs.coordinates.time.mjd.return_value.to_numpy.return_value = np.linspace(
        60000, 60030, 10
    )

    # Patch eligibility to always pass and apply_mask to return mock obs
    mock_obs.apply_mask.return_value = mock_obs

    mock_orbit = MagicMock(spec=Orbits)
    mock_propagator = MagicMock()

    with patch("adam_orbit_det_eval.looo.core.check_pair_eligibility") as mock_elig:
        mock_result = MagicMock()
        mock_result.eligible = True
        mock_result.stats = {
            "n_held_out": 5,
            "n_remaining": 5,
            "held_out_fraction": 0.5,
            "arc_remaining": 30.0,
        }
        mock_elig.return_value = mock_result

        run_looo_for_object(
            object_id="test_obj",
            observations=mock_obs,
            reference_orbit=mock_orbit,
            propagator=mock_propagator,
            orbit_fitter=mock_fitter,
        )

    # initial_fit should have been called (once per unique station)
    assert mock_fitter.initial_fit.called, (
        "orbit_fitter.initial_fit was never called — fitter plumbing is broken"
    )
