"""Tests for configurable LOOO group keys (bead wl0).

The same configured subset of {stn, prog, band, astcat} drives two roles:

1. **Hold-out unit** — ``run_looo_for_object`` holds out each unique tuple of
   the configured columns per object (verified here with a mocked fitter so no
   propagation is needed).
2. **bias_table aggregation key** — ``compute_bias_table(group_by=...)`` emits
   per-tuple rows plus the per-station rollup rows.

The canonical multi-(prog, band) case is station 568: program "2" (Tholen) vs
program "_" (Micheli), each through a different filter. The on-disk
``data/looo_sample_3500`` fixture predates bead 43z and has no ``prog`` column,
so these tests build a tiny in-memory ``MPCObservations`` instead.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from adam_orbit_det_eval.looo.bias_table import (
    BootstrapConfig,
    compute_bias_table,
)
from adam_orbit_det_eval.looo.core import (
    DEFAULT_GROUP_BY,
    _validate_group_by,
    run_looo_for_object,
)


# ---------------------------------------------------------------------------
# In-memory MPCObservations fixture (568 Tholen/Micheli + a single-prog station)
# ---------------------------------------------------------------------------


def _make_mpc_obs():
    """Build a tiny MPCObservations covering the 568 (prog 2/_) case.

    Layout (object 'objA'):
      - 568, prog '2', band 'r'  × 3 obs   (Tholen)
      - 568, prog '_', band 'i'  × 3 obs   (Micheli)
      - 950, prog None, band 'r' × 4 obs   (single-program control)
    """
    from adam_core.time import Timestamp
    from mpcq.observations import MPCObservations

    stn = ["568"] * 6 + ["950"] * 4
    prog = ["2", "2", "2", "_", "_", "_"] + [None] * 4
    band = ["r", "r", "r", "i", "i", "i"] + ["r"] * 4
    astcat = ["Gaia2"] * 10
    n = len(stn)
    mjd = 60000.0 + np.arange(n, dtype=float)
    return MPCObservations.from_kwargs(
        requested_provid=["objA"] * n,
        id=np.arange(n, dtype=np.int64),
        stn=stn,
        prog=prog,
        band=band,
        astcat=astcat,
        obstime=Timestamp.from_mjd(mjd, scale="utc"),
        ra=10.0 + 0.001 * np.arange(n),
        dec=5.0 + 0.001 * np.arange(n),
        rmsra=[0.2] * n,
        rmsdec=[0.2] * n,
        rmscorr=[0.0] * n,
    )


def _mpc_to_od(obs):
    from adam_orbit_det_eval.utils import mpc_to_od_observations

    return mpc_to_od_observations(obs, prevent_nans=True, sigma_model="veres2017")


def _count_holdout_units(group_by):
    """Run run_looo_for_object with a mock fitter + always-pass eligibility and
    return how many hold-out units (initial_fit calls) were produced."""
    obs = _make_mpc_obs()
    od = _mpc_to_od(obs)

    from adam_core.orbit_determination.fitted_orbits import FittedOrbits

    mock_fitter = MagicMock()
    # Empty fit → loop hits `if len(hold_in_orbit) == 0: continue` right after
    # initial_fit, so we count one call per eligible hold-out unit.
    mock_fitter.initial_fit.return_value = (FittedOrbits.empty(), MagicMock())

    with patch("adam_orbit_det_eval.looo.core.check_pair_eligibility") as mock_elig:
        elig = MagicMock()
        elig.eligible = True
        elig.stats = {
            "n_held_out": 3,
            "n_remaining": 7,
            "held_out_fraction": 0.3,
            "arc_remaining": 9.0,
        }
        mock_elig.return_value = elig

        run_looo_for_object(
            object_id="objA",
            observations=od,
            reference_orbit=MagicMock(),
            propagator=MagicMock(),
            astcats=obs.astcat.to_pylist(),
            program_codes=obs.prog.to_pylist(),
            bands=obs.band.to_pylist(),
            group_by=group_by,
            orbit_fitter=mock_fitter,
        )
    return mock_fitter.initial_fit.call_count


# ---------------------------------------------------------------------------
# Pre-flight invariant: band/prog/astcat survive the MPC→OD conversion aligned
# ---------------------------------------------------------------------------


class TestParallelArrayAlignment:
    def test_conversion_is_size_preserving(self):
        obs = _make_mpc_obs()
        od = _mpc_to_od(obs)
        assert od is not None
        assert len(od) == len(obs)

    def test_group_key_columns_present_on_mpc_obs(self):
        obs = _make_mpc_obs()
        # band is plumbed via the same parallel-array path as astcat/prog —
        # NOT routed through adam_core.Observations (bead wl0 pre-flight).
        assert obs.prog.to_pylist() == ["2", "2", "2", "_", "_", "_", None, None, None, None]
        assert obs.band.to_pylist()[:6] == ["r", "r", "r", "i", "i", "i"]
        assert set(obs.astcat.to_pylist()) == {"Gaia2"}


# ---------------------------------------------------------------------------
# Hold-out unit = the configured tuple
# ---------------------------------------------------------------------------


class TestHoldoutUnit:
    def test_stn_only_is_per_station(self):
        # Two stations → two hold-out units.
        assert _count_holdout_units(["stn"]) == 2

    def test_stn_prog_splits_568(self):
        # 568 splits into prog '2' and prog '_'; 950 stays one unit → 3 total.
        assert _count_holdout_units(["stn", "prog"]) == 3

    def test_stn_prog_band_matches_prog_resolution_here(self):
        # In this fixture each (stn, prog) maps to a single band, so adding band
        # does not further split: (568,2,r), (568,_,i), (950,None,r) → 3.
        assert _count_holdout_units(["stn", "prog", "band"]) == 3

    def test_default_group_by_is_stn_prog_band(self):
        assert list(DEFAULT_GROUP_BY) == ["stn", "prog", "band"]

    def test_validate_group_by_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown group_by"):
            _validate_group_by(["stn", "bogus"])

    def test_validate_group_by_rejects_empty(self):
        with pytest.raises(ValueError):
            _validate_group_by([])


# ---------------------------------------------------------------------------
# bias_table aggregation key
# ---------------------------------------------------------------------------


def _looo_rows():
    """A LOOO-shaped DataFrame: station 568 with prog '2'/band 'r' and prog
    '_'/band 'i', plus a single-prog control station 309. 5 objects per group
    so every (stn, prog[, band]) group clears min_objects_per_group=3."""
    rows = []
    objects = [f"obj{i}" for i in range(5)]
    # (stn, prog, band, astcat, ra_resid, dec_resid)
    groups = [
        ("568", "2", "r", "Gaia2", 0.30, 0.20),
        ("568", "_", "i", "Gaia2", -0.30, -0.20),
        ("309", "&", "g", "Gaia2", 0.10, 0.05),
    ]
    k = 0
    for stn, prog, band, astcat, ra, dec in groups:
        for obj in objects:
            for _ in range(3):  # 3 obs/object → 15 obs/group, 5 objects/group
                rows.append(
                    dict(
                        object_id=obj,
                        obs_id=f"o{k}",
                        stn=stn,
                        program_code=prog,
                        band=band,
                        astcat=astcat,
                        residual_ra_arcsec=ra,
                        residual_dec_arcsec=dec,
                        chi2=1.0,
                    )
                )
                k += 1
    return pd.DataFrame(rows)


_BOOT = BootstrapConfig(n_resamples=100, random_seed=42)


def _is_rollup(table):
    m = pd.Series(True, index=table.index)
    for col in ("program_code", "band", "astcat"):
        if col in table.columns:
            m &= table[col].isna()
    return m


class TestBiasTableGroupBy:
    def test_round_trip_stn_only_matches_rollup(self):
        """group_by=['stn'] produces ONLY per-station rollup rows, and those
        rollup rows are identical to the rollup rows produced under a richer
        group_by. This is the v1↔v2 apples-to-apples regression gate."""
        df = _looo_rows()
        stn_only = compute_bias_table(
            df, group_by=["stn"], min_obs_per_group=10,
            min_objects_per_group=3, max_hold_in_reduced_chi2=None, bootstrap=_BOOT,
        )
        # Only rollup rows — every non-stn key NULL, one row per station.
        assert _is_rollup(stn_only).all()
        assert sorted(stn_only["obs_code"]) == ["309", "568"]

        full = compute_bias_table(
            df, group_by=["stn", "prog", "band"], min_obs_per_group=10,
            min_objects_per_group=3, max_hold_in_reduced_chi2=None, bootstrap=_BOOT,
        )
        full_rollup = full[_is_rollup(full)].set_index("obs_code")
        so = stn_only.set_index("obs_code")
        # Rollup rows must be invariant to the configured group keys.
        for col in ("n_obs", "n_objects", "bias_ra_arcsec", "bias_dec_arcsec",
                    "rms_ra_arcsec", "rms_dec_arcsec"):
            for code in ["309", "568"]:
                assert so.loc[code, col] == pytest.approx(full_rollup.loc[code, col]), (
                    f"rollup {code}.{col} diverged between group_by configs"
                )

    def test_stn_prog_station_568_has_two_program_rows_plus_rollup(self):
        df = _looo_rows()
        table = compute_bias_table(
            df, group_by=["stn", "prog"], min_obs_per_group=10,
            min_objects_per_group=3, max_hold_in_reduced_chi2=None, bootstrap=_BOOT,
        )
        s568 = table[table["obs_code"] == "568"]
        # 1 rollup (program_code NULL) + 2 per-program rows ('2' and '_').
        rollup = s568[s568["program_code"].isna()]
        prog_rows = s568[s568["program_code"].notna()]
        assert len(rollup) == 1
        assert set(prog_rows["program_code"]) == {"2", "_"}
        assert len(prog_rows) >= 2

        # Rollup aggregates BOTH programs: 30 obs across 5 objects.
        assert int(rollup.iloc[0]["n_obs"]) == 30
        assert int(rollup.iloc[0]["n_objects"]) == 5
        # Each program row covers its own 15 obs.
        for _, r in prog_rows.iterrows():
            assert int(r["n_obs"]) == 15

    def test_four_key_schema_and_nonzero_output(self):
        df = _looo_rows()
        table = compute_bias_table(
            df, group_by=["stn", "prog", "band", "astcat"], min_obs_per_group=10,
            min_objects_per_group=3, max_hold_in_reduced_chi2=None, bootstrap=_BOOT,
        )
        for col in ("obs_code", "program_code", "band", "astcat"):
            assert col in table.columns, f"missing schema column {col}"
        assert len(table) > 0
        # 568 still resolves to two full tuples at this resolution.
        tuples = table[
            (table["obs_code"] == "568") & table["program_code"].notna()
        ][["program_code", "band"]].apply(tuple, axis=1)
        assert set(tuples) == {("2", "r"), ("_", "i")}

    def test_missing_configured_column_collapses_out(self):
        """A configured key whose column is absent is treated as all-NULL and
        simply collapses out of the tuple (no crash)."""
        df = _looo_rows().drop(columns=["astcat"])
        table = compute_bias_table(
            df, group_by=["stn", "prog", "astcat"], min_obs_per_group=10,
            min_objects_per_group=3, max_hold_in_reduced_chi2=None, bootstrap=_BOOT,
        )
        # astcat column still exists in output schema, all-NULL.
        assert "astcat" in table.columns
        assert table["astcat"].isna().all()
        # 568 program split still works.
        s568 = table[(table["obs_code"] == "568") & table["program_code"].notna()]
        assert set(s568["program_code"]) == {"2", "_"}
