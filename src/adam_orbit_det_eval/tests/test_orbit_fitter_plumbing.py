"""Tests that orbit_fitter plumbing is correctly threaded through the LOOO pipeline.

These tests verify the *parameter signature* of the public API rather than
running a full LOOO evaluation (which requires BigQuery data).
"""

import inspect

from adam_orbit_det_eval.looo.core import run_looo_for_object
from adam_orbit_det_eval.looo.pipeline import run_looo_pipeline


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
    import importlib.util
    import pathlib

    scripts_path = pathlib.Path(__file__).parents[3] / "scripts" / "02_run_looo.py"
    spec = importlib.util.spec_from_file_location("_run_looo_script", scripts_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.get_orbit_fitter("scipy", "/tmp/ignored") is None
