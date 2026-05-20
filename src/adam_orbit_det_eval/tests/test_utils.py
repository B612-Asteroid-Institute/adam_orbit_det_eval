from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from adam_core.time import Timestamp
from mpcq import MPCObservations

from ..utils import (
    VERES2017_FALLBACK_SIGMA,
    get_spacebased_stns,
    get_veres2017_sigma,
    mpc_to_od_observations,
)


def test_get_spacebased_stn() -> None:
    stns = get_spacebased_stns()
    assert "250" in stns  # Hubble Space Telescope
    assert "X05" not in stns  # Rubin Observatory


def test_mpc_to_od_observations() -> None:
    data_path = Path("data/orbit_fit_eval/mpc_observations.parquet")
    mpc = MPCObservations.from_parquet(data_path)
    mpc_valid_var_mask = pc.and_(pc.is_valid(mpc.rmsra), pc.is_valid(mpc.rmsdec))
    mpc_valid_var = mpc.apply_mask(mpc_valid_var_mask)
    # test data sanity check
    assert len(mpc) > 100
    assert len(mpc_valid_var) > 100
    assert len(mpc_valid_var) < len(mpc)
    assert len(mpc_valid_var) > 100

    # Keep NaNs, use actual RMS where available
    od_nans = mpc_to_od_observations(mpc, prevent_nans=False)
    assert len(od_nans) == len(mpc)
    # Only looking at ra and dec sigmas for nans, they should be for the same records as MPC
    od_valid_var_mask = pc.invert(
        np.any(np.isnan(od_nans.coordinates.covariance.sigmas[:, 1:3]), axis=1)
    )
    od_valid_var = od_nans.apply_mask(od_valid_var_mask)
    assert od_valid_var_mask == mpc_valid_var_mask

    # Now with NaN suppression
    od_no_nans = mpc_to_od_observations(mpc, prevent_nans=True, diag_nan=36.0)
    nans_mask = np.any(
        np.isnan(od_no_nans.coordinates.covariance.sigmas[:, 1:3]), axis=1
    )
    assert not np.any(
        nans_mask
    ), "There should be no NaN sigmas in the converted observations"
    same_values = od_no_nans.apply_mask(od_valid_var_mask)
    # Sigmas that were not NaN before should stay the same
    assert np.all(
        same_values.coordinates.covariance.sigmas[:, 1:3]
        == od_valid_var.coordinates.covariance.sigmas[:, 1:3]
    )
    # Sigmas that were NaN before should be sqrt(36)
    assert np.all(
        pc.equal(
            od_no_nans.apply_mask(
                pc.invert(od_valid_var_mask)
            ).coordinates.covariance.sigmas[:, 1],
            6.0,
        )
    )
    assert np.all(
        pc.equal(
            od_no_nans.apply_mask(
                pc.invert(od_valid_var_mask)
            ).coordinates.covariance.sigmas[:, 2],
            6.0,
        )
    )


def _make_synthetic_obs(
    ra_deg: float,
    dec_deg: float,
    stn: str,
    obs_id: str = "synthetic-obs-1",
) -> MPCObservations:
    """Build a single-row MPCObservations table for bias-hook testing."""
    obs_time = Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc")
    return MPCObservations.from_kwargs(
        requested_provid=[obs_id],
        primary_designation=[obs_id],
        obsid=[obs_id],
        trksub=["trk1"],
        provid=[obs_id],
        permid=[None],
        submission_id=[None],
        obssubid=[None],
        obstime=obs_time,
        ra=[ra_deg],
        dec=[dec_deg],
        rmsra=[0.5],
        rmsdec=[0.5],
        rmscorr=[0.0],
        mag=[20.0],
        rmsmag=[0.1],
        band=["V"],
        stn=[stn],
        updated_at=Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc"),
        created_at=Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc"),
        status=["valid"],
        astcat=["UCAC4"],
        mode=["CCD"],
    )


def test_mpc_to_od_observations_bias_subtract_round_trip() -> None:
    """Legacy subtract mode: a known bias is subtracted from observed RA/Dec.

    Kept for reproducibility/regression coverage; the default application mode
    is now 'sigma_floor' (see test_mpc_to_od_observations_bias_sigma_floor).
    """
    ra_deg = 100.123456789
    dec_deg = -42.987654321
    stn = "Z99"
    bias_ra_arcsec = 0.150
    bias_dec_arcsec = -0.230

    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)

    od_no_bias = mpc_to_od_observations(obs, prevent_nans=False)
    assert od_no_bias is not None
    np.testing.assert_allclose(
        od_no_bias.coordinates.lon.to_numpy(zero_copy_only=False), [ra_deg], atol=1e-12
    )
    np.testing.assert_allclose(
        od_no_bias.coordinates.lat.to_numpy(zero_copy_only=False), [dec_deg], atol=1e-12
    )

    od_biased = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (bias_ra_arcsec, bias_dec_arcsec)},
        bias_application="subtract",
    )
    assert od_biased is not None

    cos_dec = np.cos(np.deg2rad(dec_deg))
    expected_ra = ra_deg - (bias_ra_arcsec / 3600.0) / cos_dec
    expected_dec = dec_deg - bias_dec_arcsec / 3600.0

    np.testing.assert_allclose(
        od_biased.coordinates.lon.to_numpy(zero_copy_only=False),
        [expected_ra],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        od_biased.coordinates.lat.to_numpy(zero_copy_only=False),
        [expected_dec],
        atol=1e-9,
    )


def test_mpc_to_od_observations_bias_sigma_floor() -> None:
    """Default sigma_floor mode: positions are unchanged; per-axis sigmas are
    floored at |bias| for listed stations."""
    ra_deg = 12.34
    dec_deg = -25.0
    stn = "Z99"
    # MPC reported sigma is 0.5"; bias is large in RA and small in Dec so that
    # only the RA axis gets floored.
    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)
    bias_ra_arcsec = -1.20  # |bias| > 0.5 → RA sigma gets floored to 1.20
    bias_dec_arcsec = 0.10  # |bias| < 0.5 → Dec sigma stays at 0.5

    od_floor = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (bias_ra_arcsec, bias_dec_arcsec)},
        # bias_application defaults to 'sigma_floor'
    )
    assert od_floor is not None

    # Positions untouched.
    np.testing.assert_allclose(
        od_floor.coordinates.lon.to_numpy(zero_copy_only=False), [ra_deg], atol=1e-12
    )
    np.testing.assert_allclose(
        od_floor.coordinates.lat.to_numpy(zero_copy_only=False), [dec_deg], atol=1e-12
    )

    # Sigmas: coords.covariance.sigmas index 1 is lon (RA without cos(dec)) and
    # index 2 is lat (Dec). The lon sigma in degrees is (cos(dec) sigma) / cos(dec).
    cos_dec = np.cos(np.deg2rad(dec_deg))
    sigmas_deg = od_floor.coordinates.covariance.sigmas
    sigma_ra_cosdec_arcsec = sigmas_deg[0, 1] * cos_dec * 3600.0
    sigma_dec_arcsec = sigmas_deg[0, 2] * 3600.0
    np.testing.assert_allclose(sigma_ra_cosdec_arcsec, 1.20, rtol=1e-10)
    np.testing.assert_allclose(sigma_dec_arcsec, 0.5, rtol=1e-10)


def test_mpc_to_od_observations_bias_passthrough_unknown_station() -> None:
    """A station absent from the bias table is left unchanged (positions and sigmas)."""
    obs = _make_synthetic_obs(
        ra_deg=12.5, dec_deg=30.0, stn="Z99", obs_id="passthrough-1"
    )
    od_biased = mpc_to_od_observations(
        obs, prevent_nans=False, bias_table={"568": (1.0, -1.0)}
    )
    assert od_biased is not None
    np.testing.assert_allclose(
        od_biased.coordinates.lon.to_numpy(zero_copy_only=False), [12.5], atol=1e-12
    )
    np.testing.assert_allclose(
        od_biased.coordinates.lat.to_numpy(zero_copy_only=False), [30.0], atol=1e-12
    )
    cos_dec = np.cos(np.deg2rad(30.0))
    sigmas_deg = od_biased.coordinates.covariance.sigmas
    np.testing.assert_allclose(sigmas_deg[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-10)
    np.testing.assert_allclose(sigmas_deg[0, 2] * 3600.0, 0.5, rtol=1e-10)


def test_get_veres2017_sigma_lookup() -> None:
    """Spot-check Veres lookup order: override > catalog default > fallback."""
    # Per-(stn, catalog) override
    assert get_veres2017_sigma("F51", "Gaia2") == (0.15, 0.15)
    # Per-catalog default (no stn override)
    assert get_veres2017_sigma("999", "Gaia2") == (0.18, 0.18)
    # Unknown catalog → fallback
    assert get_veres2017_sigma("999", "MADEUP") == (
        VERES2017_FALLBACK_SIGMA,
        VERES2017_FALLBACK_SIGMA,
    )
    # Both None → fallback
    assert get_veres2017_sigma(None, None) == (
        VERES2017_FALLBACK_SIGMA,
        VERES2017_FALLBACK_SIGMA,
    )


def test_mpc_to_od_observations_sigma_model_veres2017_fills_missing() -> None:
    """When MPC sigmas are missing, sigma_model='veres2017' fills from the lookup."""
    obs_time = Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc")
    # rmsra/rmsdec are explicit NaN to trigger the fill-in
    obs = MPCObservations.from_kwargs(
        requested_provid=["missing-sigma-1"],
        primary_designation=["missing-sigma-1"],
        obsid=["missing-sigma-1"],
        trksub=["trk1"],
        provid=["missing-sigma-1"],
        permid=[None],
        submission_id=[None],
        obssubid=[None],
        obstime=obs_time,
        ra=[100.0],
        dec=[0.0],
        rmsra=[float("nan")],
        rmsdec=[float("nan")],
        rmscorr=[0.0],
        mag=[20.0],
        rmsmag=[0.1],
        band=["V"],
        stn=["Z99"],
        updated_at=Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc"),
        created_at=Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc"),
        status=["valid"],
        astcat=["Gaia2"],
        mode=["CCD"],
    )

    od = mpc_to_od_observations(obs, prevent_nans=False, sigma_model="veres2017")
    assert od is not None
    sigmas_deg = od.coordinates.covariance.sigmas
    # At dec=0, cos(dec)=1 so the cos(dec) factor is a no-op.
    np.testing.assert_allclose(sigmas_deg[0, 1] * 3600.0, 0.18, rtol=1e-10)
    np.testing.assert_allclose(sigmas_deg[0, 2] * 3600.0, 0.18, rtol=1e-10)
