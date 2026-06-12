"""Tests for the parametric obstime filter + named run profiles (bead tcu).

Axis 2 of the v2 configurable architecture (docs/v2-scope.md):

1. **Pre-LOOO obstime filter** — ``filter_observations_by_obstime`` keeps
   observations in the ``(obstime_min, obstime_max]`` window (min exclusive,
   max inclusive) so the pre/post-2017 profiles partition the archive with no
   overlap and no gap. The filter runs BEFORE refits, so per-pair eligibility
   criteria are re-evaluated on the windowed data — an object that passes over
   its full arc may be excluded within a window.
2. **Run profiles** — ``looo/profiles.py`` maps a profile name to
   group_by + obstime window + output-dir suffix, with explicit per-option
   arguments overriding the profile's values.

Follows the in-memory ``MPCObservations`` fixture pattern from
``test_group_keys_wl0.py`` (no on-disk data/ fixtures). 2017-01-01 UTC is
MJD 57754.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from adam_orbit_det_eval.looo.core import (
    DEFAULT_GROUP_BY,
    LOOOConfig,
    run_looo_for_object,
)
from adam_orbit_det_eval.looo.pipeline import filter_observations_by_obstime
from adam_orbit_det_eval.looo.profiles import (
    PROFILES,
    get_profile,
    resolve_run_options,
)

MJD_2017_01_01 = 57754.0


def _make_mpc_obs(rows):
    """Build an in-memory MPCObservations from (stn, prog, band, mjd) rows."""
    from adam_core.time import Timestamp
    from mpcq.observations import MPCObservations

    stn = [r[0] for r in rows]
    prog = [r[1] for r in rows]
    band = [r[2] for r in rows]
    mjd = np.array([r[3] for r in rows], dtype=float)
    n = len(rows)
    return MPCObservations.from_kwargs(
        requested_provid=["objA"] * n,
        id=np.arange(n, dtype=np.int64),
        stn=stn,
        prog=prog,
        band=band,
        astcat=["Gaia2"] * n,
        obstime=Timestamp.from_mjd(mjd, scale="utc"),
        ra=10.0 + 0.001 * np.arange(n),
        dec=5.0 + 0.001 * np.arange(n),
        rmsra=[0.2] * n,
        rmsdec=[0.2] * n,
        rmscorr=[0.0] * n,
    )


def _boundary_fixture():
    """8 pre-2017 obs (568), 1 obs exactly AT 2017-01-01 (G96), 8 post (950)."""
    rows = []
    rows += [("568", "2", "r", 57700.0 + 3 * i) for i in range(8)]
    rows += [("G96", None, "G", MJD_2017_01_01)]
    rows += [("950", None, "r", 57760.0 + 3 * i) for i in range(8)]
    return _make_mpc_obs(rows)


def _split_arc_fixture():
    """An object whose arc straddles 2017-01-01: 8 obs before, 8 after.

    Over the full arc, holding out either station leaves the other station's
    8 obs (eligible under the test config). Within the pre-2017 window only
    station 568's observations remain, so holding out 568 leaves nothing —
    the object passes full-arc but fails in-window.
    """
    rows = []
    rows += [("568", "2", "r", 57700.0 + 3 * i) for i in range(8)]
    rows += [("950", None, "r", 57760.0 + 3 * i) for i in range(8)]
    return _make_mpc_obs(rows)


def _equiv_fixture():
    """Two pre-2017 stations + one post-2017 station, 8 obs each."""
    rows = []
    rows += [("568", "2", "r", 57700.0 + 3 * i) for i in range(8)]
    rows += [("950", None, "r", 57701.0 + 3 * i) for i in range(8)]
    rows += [("G96", None, "G", 57760.0 + 3 * i) for i in range(8)]
    return _make_mpc_obs(rows)


def _mjds(obs):
    return obs.obstime.mjd().to_numpy(zero_copy_only=False)


_TEST_CONFIG = LOOOConfig(
    min_obs_held_out=1,
    min_obs_remaining=6,
    min_arc_length_days=7.0,
    max_held_out_fraction=0.5,
)


def _run_looo_with_mock_fitter(mpc_obs, group_by):
    """Run run_looo_for_object with REAL eligibility and a mock fitter.

    The mock fitter returns an empty fit, so the per-tuple loop stops right
    after ``initial_fit`` — each call corresponds to one hold-out unit that
    passed eligibility. Returns the list of hold-in observation MJD sets
    (each fixture row has a unique obstime, so MJDs identify observations),
    one per eligible unit.
    """
    from adam_core.orbit_determination.fitted_orbits import FittedOrbits

    from adam_orbit_det_eval.utils import mpc_to_od_observations

    od = mpc_to_od_observations(mpc_obs, prevent_nans=True, sigma_model="veres2017")
    assert od is not None and len(od) == len(mpc_obs)

    mock_fitter = MagicMock()
    mock_fitter.initial_fit.return_value = (FittedOrbits.empty(), MagicMock())

    run_looo_for_object(
        object_id="objA",
        observations=od,
        reference_orbit=MagicMock(),
        propagator=MagicMock(),
        config=_TEST_CONFIG,
        astcats=mpc_obs.astcat.to_pylist(),
        program_codes=mpc_obs.prog.to_pylist(),
        bands=mpc_obs.band.to_pylist(),
        group_by=group_by,
        orbit_fitter=mock_fitter,
    )
    return [
        sorted(call.args[1].coordinates.time.mjd().to_pylist())
        for call in mock_fitter.initial_fit.call_args_list
    ]


# ---------------------------------------------------------------------------
# Obstime filter
# ---------------------------------------------------------------------------


class TestObstimeFilter:
    def test_no_bounds_is_identity(self):
        obs = _boundary_fixture()
        assert filter_observations_by_obstime(obs) is obs

    def test_max_is_inclusive(self):
        # max = 2017-01-01 keeps the 8 pre-2017 obs AND the boundary obs.
        obs = _boundary_fixture()
        kept = filter_observations_by_obstime(obs, obstime_max="2017-01-01")
        assert len(kept) == 9
        assert (_mjds(kept) <= MJD_2017_01_01).all()
        assert "G96" in set(kept.stn.to_pylist())

    def test_min_is_exclusive(self):
        # min = 2017-01-01 keeps only the 8 strictly-later obs; the boundary
        # obs belongs to the pre-2017 side (no overlap with max=2017-01-01).
        obs = _boundary_fixture()
        kept = filter_observations_by_obstime(obs, obstime_min="2017-01-01")
        assert len(kept) == 8
        assert (_mjds(kept) > MJD_2017_01_01).all()
        assert set(kept.stn.to_pylist()) == {"950"}

    def test_pre_post_partition_has_no_overlap_and_no_gap(self):
        obs = _boundary_fixture()
        pre = filter_observations_by_obstime(obs, obstime_max="2017-01-01")
        post = filter_observations_by_obstime(obs, obstime_min="2017-01-01")
        pre_ids = set(pre.id.to_pylist())
        post_ids = set(post.id.to_pylist())
        assert pre_ids.isdisjoint(post_ids)
        assert pre_ids | post_ids == set(obs.id.to_pylist())

    def test_both_bounds(self):
        obs = _boundary_fixture()
        kept = filter_observations_by_obstime(
            obs, obstime_min="2016-11-01", obstime_max="2017-01-01"
        )
        mjds = _mjds(kept)
        assert len(kept) > 0
        assert (mjds <= MJD_2017_01_01).all()
        assert (mjds > 57693.0).all()  # 2016-11-01


# ---------------------------------------------------------------------------
# Object-level exclusion criteria re-evaluated AFTER the filter
# ---------------------------------------------------------------------------


class TestExclusionReappliedInWindow:
    def test_object_passes_over_full_arc(self):
        obs = _split_arc_fixture()
        hold_in_sets = _run_looo_with_mock_fitter(obs, list(DEFAULT_GROUP_BY))
        # Both hold-out units (568 and 950 tuples) are eligible over the
        # full arc: each leaves the other station's 8 obs (>= 6) over 21 days.
        assert len(hold_in_sets) == 2

    def test_same_object_dropped_within_pre_2017_window(self):
        obs = _split_arc_fixture()
        windowed = filter_observations_by_obstime(obs, obstime_max="2017-01-01")
        # The window keeps only station 568's 8 observations...
        assert set(windowed.stn.to_pylist()) == {"568"}
        # ...so holding out the single remaining tuple leaves 0 observations:
        # eligibility (min obs remaining / arc / fraction), re-evaluated on
        # the windowed data, rejects every unit and the object contributes
        # nothing to the in-window catalog.
        hold_in_sets = _run_looo_with_mock_fitter(windowed, list(DEFAULT_GROUP_BY))
        assert hold_in_sets == []


# ---------------------------------------------------------------------------
# Profile definitions + resolution precedence
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_profile_table_matches_v2_scope(self):
        # The three named profiles from docs/v2-scope.md / bead tcu.
        expected = {
            "v2_full": (("stn", "prog", "band"), None, None, "_v2_full"),
            "v2_full_pre_2017": (
                ("stn", "prog", "band"), None, "2017-01-01", "_v2_full_pre_2017",
            ),
            "v2_full_post_2017": (
                ("stn", "prog", "band"), "2017-01-01", None, "_v2_full_post_2017",
            ),
        }
        assert set(PROFILES) == set(expected)
        for name, (gb, tmin, tmax, suffix) in expected.items():
            p = get_profile(name)
            assert p.group_by == gb
            assert p.obstime_min == tmin
            assert p.obstime_max == tmax
            assert p.output_dir_suffix == suffix

    def test_unknown_profile_raises_with_available_names(self):
        with pytest.raises(ValueError, match="v2_full"):
            get_profile("v1_replication")

    def test_resolve_defaults_without_profile(self):
        options = resolve_run_options()
        assert options.group_by == tuple(DEFAULT_GROUP_BY)
        assert options.obstime_min is None
        assert options.obstime_max is None
        assert options.profile is None
        assert options.output_dir_suffix == ""

    def test_profile_resolves_to_same_options_as_explicit_flags(self):
        # --profile v2_full ≡ the same run spelled out with individual flags.
        from_profile = resolve_run_options(profile="v2_full")
        from_flags = resolve_run_options(group_by=["stn", "prog", "band"])
        assert from_profile.group_by == from_flags.group_by
        assert from_profile.obstime_min == from_flags.obstime_min
        assert from_profile.obstime_max == from_flags.obstime_max

    def test_explicit_obstime_max_overrides_profile(self):
        options = resolve_run_options(
            profile="v2_full_pre_2017", obstime_max="2015-06-30"
        )
        assert options.obstime_max == "2015-06-30"
        # Non-overridden options still come from the profile.
        assert options.obstime_min is None
        assert options.group_by == ("stn", "prog", "band")
        assert options.profile == "v2_full_pre_2017"

    def test_explicit_group_by_overrides_profile(self):
        options = resolve_run_options(profile="v2_full", group_by=["stn"])
        assert options.group_by == ("stn",)
        assert options.profile == "v2_full"

    def test_resolve_validates_group_by(self):
        with pytest.raises(ValueError, match="Unknown group_by"):
            resolve_run_options(group_by=["stn", "bogus"])


# ---------------------------------------------------------------------------
# Profile run ≡ explicit-flag run (pipeline behavior, small fixture)
# ---------------------------------------------------------------------------


def _run_with_options(mpc_obs, options):
    """Apply the resolved options the way the entrypoints do: filter, then LOOO."""
    windowed = filter_observations_by_obstime(
        mpc_obs, obstime_min=options.obstime_min, obstime_max=options.obstime_max
    )
    return windowed, _run_looo_with_mock_fitter(windowed, list(options.group_by))


class TestProfileEquivalence:
    def test_v2_full_profile_run_identical_to_explicit_flags(self):
        obs = _equiv_fixture()
        _, via_profile = _run_with_options(obs, resolve_run_options(profile="v2_full"))
        _, via_flags = _run_with_options(
            obs, resolve_run_options(group_by=["stn", "prog", "band"])
        )
        assert via_profile == via_flags
        assert len(via_profile) == 3  # one eligible unit per station tuple

    def test_pre_2017_profile_run_identical_to_explicit_flags(self):
        obs = _equiv_fixture()
        windowed, via_profile = _run_with_options(
            obs, resolve_run_options(profile="v2_full_pre_2017")
        )
        _, via_flags = _run_with_options(
            obs,
            resolve_run_options(
                group_by=["stn", "prog", "band"], obstime_max="2017-01-01"
            ),
        )
        assert via_profile == via_flags
        # Both pre-2017 stations survive the window and are eligible units.
        assert len(via_profile) == 2
        # Everything the in-window LOOO sees is on or before 2017-01-01.
        assert (_mjds(windowed) <= MJD_2017_01_01).all()
