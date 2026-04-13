"""Tests for program_code tracking in LOOOResult and analysis."""

import numpy as np

from adam_orbit_det_eval.looo.analysis import (
    ProgramCodeStats,
    compute_program_code_stats,
)
from adam_orbit_det_eval.looo.core import LOOOResult


def _make_result(n: int = 10) -> LOOOResult:
    """Build a minimal LOOOResult with program_code values for testing."""
    return LOOOResult.from_kwargs(
        object_id=[f"obj{i // 2}" for i in range(n)],
        obs_id=[f"obs{i}" for i in range(n)],
        stn=["568"] * n,
        residual_ra_arcsec=np.linspace(-0.5, 0.5, n).tolist(),
        residual_dec_arcsec=np.linspace(-0.3, 0.3, n).tolist(),
        sigma_ra_cosdec_arcsec=[0.3] * n,
        sigma_dec_arcsec=[0.3] * n,
        chi2=[1.0] * n,
        astcat=["Gaia2"] * n,
        program_code=["A"] * (n // 2) + ["B"] * (n - n // 2),
        n_obs_held_out=[2] * n,
        n_obs_remaining=[20] * n,
        arc_length_remaining_days=[30.0] * n,
        held_out_fraction=[0.1] * n,
        n_obs_total=[22] * n,
        delta_q_au=[0.0] * n,
        delta_e=[0.0] * n,
        delta_i_deg=[0.0] * n,
        hold_in_reduced_chi2=[1.0] * n,
        hold_in_fit_success=[True] * n,
    )


def test_looo_result_has_program_code_column():
    res = _make_result(4)
    assert "program_code" in res.table.schema.names
    assert res.program_code.to_pylist() == ["A", "A", "B", "B"]


def test_compute_program_code_stats():
    res = _make_result(20)
    stats = compute_program_code_stats(res, min_obs_per_group=5)
    assert isinstance(stats, ProgramCodeStats)
    assert len(stats) == 2
    # groups are sorted by (stn, program_code)
    assert stats.program_code.to_pylist() == ["A", "B"]
    assert stats.stn.to_pylist() == ["568", "568"]
    assert stats.n_obs.to_pylist() == [10, 10]
