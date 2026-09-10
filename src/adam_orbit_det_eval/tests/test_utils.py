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
    # The star catalog must ride along verbatim, row for row: adam_core's
    # catalog-aware models (EFCC18 debias, Veres sigmas) key on it and treat
    # null as "unknown catalog" (pass-through).
    assert od_nans.astcat.to_pylist() == mpc.astcat.to_pylist()
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


def test_mpc_to_od_observations_catalog_debias_round_trip() -> None:
    """A per-observation catalog-debias correction is subtracted from RA/Dec."""
    ra_deg = 200.0
    dec_deg = 35.0
    stn = "Z99"
    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)

    bias_ra_cosdec_arcsec = 0.40  # tangent-plane arcsec
    bias_dec_arcsec = -0.25
    debias = np.array([[bias_ra_cosdec_arcsec, bias_dec_arcsec]], dtype=np.float64)
    od = mpc_to_od_observations(
        obs, prevent_nans=False, catalog_debias_arcsec=debias
    )
    assert od is not None
    cos_dec = np.cos(np.deg2rad(dec_deg))
    expected_ra = ra_deg - (bias_ra_cosdec_arcsec / 3600.0) / cos_dec
    expected_dec = dec_deg - bias_dec_arcsec / 3600.0
    np.testing.assert_allclose(
        od.coordinates.lon.to_numpy(zero_copy_only=False), [expected_ra], atol=1e-12
    )
    np.testing.assert_allclose(
        od.coordinates.lat.to_numpy(zero_copy_only=False), [expected_dec], atol=1e-12
    )

    # Zero correction = no change to position
    od_zero = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        catalog_debias_arcsec=np.zeros((1, 2), dtype=np.float64),
    )
    np.testing.assert_allclose(
        od_zero.coordinates.lon.to_numpy(zero_copy_only=False), [ra_deg], atol=1e-12
    )
    np.testing.assert_allclose(
        od_zero.coordinates.lat.to_numpy(zero_copy_only=False), [dec_deg], atol=1e-12
    )

    # Wrong shape raises
    try:
        mpc_to_od_observations(
            obs,
            prevent_nans=False,
            catalog_debias_arcsec=np.zeros((2, 2), dtype=np.float64),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for shape mismatch")


def test_mpc_to_od_observations_bias_rss_additive() -> None:
    """rss_additive mode: σ_used = sqrt(σ_baseline² + bias²) per axis."""
    ra_deg = 50.0
    dec_deg = 0.0  # cos(dec)=1 so RA σ math is direct
    stn = "Z99"
    # baseline σ_ra = σ_dec = 0.5"
    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)
    bias_ra_arcsec = 1.2
    bias_dec_arcsec = 0.3

    od = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (bias_ra_arcsec, bias_dec_arcsec)},
        bias_application="rss_additive",
    )
    assert od is not None
    sigmas_deg = od.coordinates.covariance.sigmas
    cos_dec = np.cos(np.deg2rad(dec_deg))
    expected_ra_arcsec = float(np.sqrt(0.5**2 + bias_ra_arcsec**2))
    expected_dec_arcsec = float(np.sqrt(0.5**2 + bias_dec_arcsec**2))
    np.testing.assert_allclose(
        sigmas_deg[0, 1] * cos_dec * 3600.0, expected_ra_arcsec, rtol=1e-9
    )
    np.testing.assert_allclose(
        sigmas_deg[0, 2] * 3600.0, expected_dec_arcsec, rtol=1e-9
    )
    # Position untouched
    np.testing.assert_allclose(
        od.coordinates.lon.to_numpy(zero_copy_only=False), [ra_deg], atol=1e-12
    )
    np.testing.assert_allclose(
        od.coordinates.lat.to_numpy(zero_copy_only=False), [dec_deg], atol=1e-12
    )


def test_mpc_to_od_observations_bias_performance_weighted() -> None:
    """performance_weighted: σ_used = σ_baseline × sqrt(max(chi2, 1.0))."""
    ra_deg = 50.0
    dec_deg = 0.0
    stn = "Z99"
    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)

    # chi2 = 9.0 → factor = 3.0 → σ_used = 0.5 × 3 = 1.5"
    od = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (0.0, 0.0)},  # unused for performance_weighted
        bias_application="performance_weighted",
        station_chi2_per_obs={stn: 9.0},
    )
    assert od is not None
    sigmas_deg = od.coordinates.covariance.sigmas
    cos_dec = np.cos(np.deg2rad(dec_deg))
    np.testing.assert_allclose(
        sigmas_deg[0, 1] * cos_dec * 3600.0, 1.5, rtol=1e-9
    )
    np.testing.assert_allclose(sigmas_deg[0, 2] * 3600.0, 1.5, rtol=1e-9)

    # chi2 < 1 → factor clipped at 1 → σ unchanged at 0.5"
    od_low_chi2 = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table=None,
        bias_application="performance_weighted",
        station_chi2_per_obs={stn: 0.25},
    )
    assert od_low_chi2 is not None
    sigmas_low = od_low_chi2.coordinates.covariance.sigmas
    np.testing.assert_allclose(
        sigmas_low[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-9
    )
    np.testing.assert_allclose(sigmas_low[0, 2] * 3600.0, 0.5, rtol=1e-9)

    # Station not in chi2 dict → factor = 1
    od_absent = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table=None,
        bias_application="performance_weighted",
        station_chi2_per_obs={"OTHER": 100.0},
    )
    sigmas_absent = od_absent.coordinates.covariance.sigmas
    np.testing.assert_allclose(
        sigmas_absent[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-9
    )

    # Missing station_chi2_per_obs → ValueError
    try:
        mpc_to_od_observations(
            obs, prevent_nans=False, bias_application="performance_weighted"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing station_chi2_per_obs")


def test_mpc_to_od_observations_sigma_model_uniform() -> None:
    """sigma_model='uniform': every obs gets σ_ra=σ_dec=uniform_sigma_arcsec."""
    obs = _make_synthetic_obs(ra_deg=12.0, dec_deg=-20.0, stn="Z99")
    od = mpc_to_od_observations(
        obs, prevent_nans=False, sigma_model="uniform", uniform_sigma_arcsec=0.75
    )
    assert od is not None
    sigmas_deg = od.coordinates.covariance.sigmas
    cos_dec = np.cos(np.deg2rad(-20.0))
    np.testing.assert_allclose(
        sigmas_deg[0, 1] * cos_dec * 3600.0, 0.75, rtol=1e-9
    )
    np.testing.assert_allclose(sigmas_deg[0, 2] * 3600.0, 0.75, rtol=1e-9)

    # Default uniform_sigma_arcsec is 0.5
    od_default = mpc_to_od_observations(obs, prevent_nans=False, sigma_model="uniform")
    sd = od_default.coordinates.covariance.sigmas
    np.testing.assert_allclose(sd[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-9)
    np.testing.assert_allclose(sd[0, 2] * 3600.0, 0.5, rtol=1e-9)


def test_mpc_to_od_observations_bias_bayes_shrinkage() -> None:
    """bayes_shrinkage: σ_used = sqrt(σ_base² + (|bias|·s)²), s = σ_base²/(σ_base² + SEM²)."""
    ra_deg = 50.0
    dec_deg = 0.0  # cos(dec)=1 keeps the σ math direct
    stn = "Z99"
    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)
    # Baseline σ_base = 0.5"; bias = 1.0", SEM = 0.5" → s = 0.25/(0.25+0.25) = 0.5
    # → σ_used = sqrt(0.25 + (1.0·0.5)²) = sqrt(0.25 + 0.25) = sqrt(0.5)
    od = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (1.0, 1.0)},
        bias_application="bayes_shrinkage",
        station_sem_arcsec={stn: (0.5, 0.5)},
    )
    assert od is not None
    sigmas_deg = od.coordinates.covariance.sigmas
    cos_dec = np.cos(np.deg2rad(dec_deg))
    expected = float(np.sqrt(0.5))
    np.testing.assert_allclose(
        sigmas_deg[0, 1] * cos_dec * 3600.0, expected, rtol=1e-9
    )
    np.testing.assert_allclose(sigmas_deg[0, 2] * 3600.0, expected, rtol=1e-9)

    # SEM = 0 (perfect knowledge of bias) → s = 1 → matches rss_additive.
    od_perfect = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (1.2, 0.3)},
        bias_application="bayes_shrinkage",
        station_sem_arcsec={stn: (0.0, 0.0)},
    )
    sd = od_perfect.coordinates.covariance.sigmas
    np.testing.assert_allclose(
        sd[0, 1] * cos_dec * 3600.0, np.sqrt(0.25 + 1.44), rtol=1e-9
    )
    np.testing.assert_allclose(
        sd[0, 2] * 3600.0, np.sqrt(0.25 + 0.09), rtol=1e-9
    )

    # SEM >> σ (no information about bias) → s → 0 → σ_used → σ_base.
    od_noinfo = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (5.0, 5.0)},
        bias_application="bayes_shrinkage",
        station_sem_arcsec={stn: (100.0, 100.0)},
    )
    sd = od_noinfo.coordinates.covariance.sigmas
    np.testing.assert_allclose(sd[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-3)
    np.testing.assert_allclose(sd[0, 2] * 3600.0, 0.5, rtol=1e-3)

    # Missing station_sem_arcsec → ValueError
    try:
        mpc_to_od_observations(
            obs,
            prevent_nans=False,
            bias_table={stn: (1.0, 1.0)},
            bias_application="bayes_shrinkage",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing station_sem_arcsec")


def test_mpc_to_od_observations_bias_veres_v1_max_floor() -> None:
    """veres_v1_max_floor: σ = max(Veres σ_per_(stn,cat), |bias|); ignores MPC report."""
    # Use stn='F51', astcat='Gaia2' which has a (per-stn, cat) Veres override = 0.15"
    obs_time = Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc")
    obs = MPCObservations.from_kwargs(
        requested_provid=["vmf-1"],
        primary_designation=["vmf-1"],
        obsid=["vmf-1"],
        trksub=["trk1"],
        provid=["vmf-1"],
        permid=[None],
        submission_id=[None],
        obssubid=[None],
        obstime=obs_time,
        ra=[100.0],
        dec=[0.0],
        # MPC-reported σ = 0.4" is LARGER than Veres 0.15"; veres_v1_max_floor
        # must override this with Veres (and then floor at |bias|).
        rmsra=[0.4],
        rmsdec=[0.4],
        rmscorr=[0.0],
        mag=[20.0],
        rmsmag=[0.1],
        band=["V"],
        stn=["F51"],
        updated_at=Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc"),
        created_at=Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc"),
        status=["valid"],
        astcat=["Gaia2"],
        mode=["CCD"],
    )
    # |bias_ra|=0.3 > Veres 0.15 → σ_ra = 0.3.  |bias_dec|=0.1 < Veres 0.15 → σ_dec = 0.15.
    od = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={"F51": (-0.3, 0.1)},
        bias_application="veres_v1_max_floor",
    )
    assert od is not None
    sigmas_deg = od.coordinates.covariance.sigmas
    # At dec=0, cos(dec)=1.
    np.testing.assert_allclose(sigmas_deg[0, 1] * 3600.0, 0.30, rtol=1e-9)
    np.testing.assert_allclose(sigmas_deg[0, 2] * 3600.0, 0.15, rtol=1e-9)

    # Station absent from bias_table → just Veres σ (no floor).
    od_nobias = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={"OTHER": (1.0, 1.0)},
        bias_application="veres_v1_max_floor",
    )
    sd = od_nobias.coordinates.covariance.sigmas
    np.testing.assert_allclose(sd[0, 1] * 3600.0, 0.15, rtol=1e-9)
    np.testing.assert_allclose(sd[0, 2] * 3600.0, 0.15, rtol=1e-9)


def test_mpc_to_od_observations_bias_covar_inflation() -> None:
    """covar_inflation: 2×2 cov += outer(b, b); off-diagonal couples through rmscorr."""
    ra_deg = 50.0
    dec_deg = 0.0
    stn = "Z99"
    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)
    bias_ra_arcsec = 1.0
    bias_dec_arcsec = -0.5

    od = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (bias_ra_arcsec, bias_dec_arcsec)},
        bias_application="covar_inflation",
    )
    assert od is not None

    cos_dec = np.cos(np.deg2rad(dec_deg))
    sigmas_deg = od.coordinates.covariance.sigmas
    # Inflated diagonals (in arcsec).
    expected_sigma_ra = float(np.sqrt(0.5**2 + bias_ra_arcsec**2))
    expected_sigma_dec = float(np.sqrt(0.5**2 + bias_dec_arcsec**2))
    np.testing.assert_allclose(
        sigmas_deg[0, 1] * cos_dec * 3600.0, expected_sigma_ra, rtol=1e-9
    )
    np.testing.assert_allclose(
        sigmas_deg[0, 2] * 3600.0, expected_sigma_dec, rtol=1e-9
    )
    # Off-diagonal: cov_rd_arcsec² = bias_ra * bias_dec = -0.5
    # Reconstruct from the 6×6 covariance: cov[1, 2] is in deg² (with σ_ra in
    # deg, not cos(dec)-corrected). At dec=0 this is identical to the
    # cos(dec)-corrected frame value, so cov_rd_arcsec² = cov[1,2] * 3600².
    cov_mat = od.coordinates.covariance.to_matrix()
    cov_rd_deg2 = cov_mat[0, 1, 2]
    cov_rd_arcsec2 = cov_rd_deg2 * (3600.0**2)
    np.testing.assert_allclose(
        cov_rd_arcsec2,
        bias_ra_arcsec * bias_dec_arcsec,
        rtol=1e-9,
    )

    # Station absent from bias_table → unchanged (diagonal, σ=0.5").
    od_pass = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={"OTHER": (1.0, 1.0)},
        bias_application="covar_inflation",
    )
    sd = od_pass.coordinates.covariance.sigmas
    np.testing.assert_allclose(sd[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-9)
    np.testing.assert_allclose(sd[0, 2] * 3600.0, 0.5, rtol=1e-9)


def test_mpc_to_od_observations_bias_at_ct_floor() -> None:
    """at_ct_floor: floor σ in AT/CT basis; rotate back to RA/Dec."""
    ra_deg = 50.0
    dec_deg = 0.0
    stn = "Z99"
    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)

    # Case 1 — motion along RA: u_ra=1, u_dec=0. AT = RA axis, CT = Dec axis.
    # Baseline σ_RA = σ_Dec = 0.5". |bias_AT|=1.0, |bias_CT|=0.0:
    #   σ_AT_used = max(0.5, 1.0) = 1.0; σ_CT_used = max(0.5, 0.0) = 0.5.
    # Rotated back: σ_RA = 1.0, σ_Dec = 0.5, no cross-term.
    od = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_application="at_ct_floor",
        atct_bias_table={stn: (1.0, 0.0)},
        atct_unit_vectors=np.array([[1.0, 0.0]], dtype=np.float64),
    )
    assert od is not None
    cos_dec = np.cos(np.deg2rad(dec_deg))
    sigmas_deg = od.coordinates.covariance.sigmas
    np.testing.assert_allclose(sigmas_deg[0, 1] * cos_dec * 3600.0, 1.0, rtol=1e-9)
    np.testing.assert_allclose(sigmas_deg[0, 2] * 3600.0, 0.5, rtol=1e-9)

    # Case 2 — motion at 45°: u_ra=u_dec=1/sqrt(2). |bias_AT|=2.0, |bias_CT|=0.
    # σ_AT² = max(0.5·(0.25 + 0.25), 4.0) = 4.0;  σ_CT² = max(0.5·(0.25 + 0.25), 0) = 0.25.
    # Rotated back: σ_RA² = 0.5·(4 + 0.25) = 2.125;  σ_Dec² = same = 2.125.
    # cov_RD² = 0.5·(4 - 0.25) = 1.875.
    od2 = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_application="at_ct_floor",
        atct_bias_table={stn: (2.0, 0.0)},
        atct_unit_vectors=np.array(
            [[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]], dtype=np.float64
        ),
    )
    sigmas2 = od2.coordinates.covariance.sigmas
    np.testing.assert_allclose(
        sigmas2[0, 1] * cos_dec * 3600.0, float(np.sqrt(2.125)), rtol=1e-9
    )
    np.testing.assert_allclose(
        sigmas2[0, 2] * 3600.0, float(np.sqrt(2.125)), rtol=1e-9
    )
    cov_mat2 = od2.coordinates.covariance.to_matrix()
    cov_rd_arcsec2 = cov_mat2[0, 1, 2] * (3600.0**2)
    np.testing.assert_allclose(cov_rd_arcsec2, 1.875, rtol=1e-9)

    # Case 3 — near-stationary (zero velocity unit vector) → pass through.
    od_stat = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_application="at_ct_floor",
        atct_bias_table={stn: (1.0, 1.0)},
        atct_unit_vectors=np.array([[np.nan, np.nan]], dtype=np.float64),
    )
    ssd = od_stat.coordinates.covariance.sigmas
    np.testing.assert_allclose(ssd[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-9)
    np.testing.assert_allclose(ssd[0, 2] * 3600.0, 0.5, rtol=1e-9)

    # Missing atct_bias_table → ValueError
    try:
        mpc_to_od_observations(
            obs,
            prevent_nans=False,
            bias_application="at_ct_floor",
            atct_unit_vectors=np.array([[1.0, 0.0]], dtype=np.float64),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing atct_bias_table")

    # Wrong-shape unit vectors → ValueError
    try:
        mpc_to_od_observations(
            obs,
            prevent_nans=False,
            bias_application="at_ct_floor",
            atct_bias_table={stn: (1.0, 1.0)},
            atct_unit_vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for shape mismatch")


def test_mpc_to_od_observations_bias_subtract_sem_inflated() -> None:
    """subtract_sem_inflated: position subtraction + diagonal σ inflation by σ_b.

    Verifies (1) the corrected position matches obs - bias to <1e-9, (2) the
    per-axis sigma matches sqrt(σ_baseline² + σ_b²) to <1e-9, and (3) NO
    off-diagonal cov term is introduced (corr stays 0).
    """
    ra_deg = 100.123456789
    dec_deg = 0.0  # cos(dec)=1 keeps the math direct
    stn = "Z99"
    bias_ra_arcsec = 0.20
    bias_dec_arcsec = -0.15
    sigma_b_ra_arcsec = 0.30  # 1-σ uncertainty on the bias estimate
    sigma_b_dec_arcsec = 0.10

    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)

    od = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={stn: (bias_ra_arcsec, bias_dec_arcsec)},
        bias_application="subtract_sem_inflated",
        station_bias_ci_arcsec={stn: (sigma_b_ra_arcsec, sigma_b_dec_arcsec)},
    )
    assert od is not None

    # (1) Position correction: obs_RA - bias_ra (in cos(dec)-corrected frame).
    cos_dec = np.cos(np.deg2rad(dec_deg))
    expected_ra = ra_deg - (bias_ra_arcsec / 3600.0) / cos_dec
    expected_dec = dec_deg - bias_dec_arcsec / 3600.0
    np.testing.assert_allclose(
        od.coordinates.lon.to_numpy(zero_copy_only=False), [expected_ra], atol=1e-9
    )
    np.testing.assert_allclose(
        od.coordinates.lat.to_numpy(zero_copy_only=False), [expected_dec], atol=1e-9
    )

    # (2) σ inflation: sqrt(0.5² + σ_b²) per axis.
    sigmas_deg = od.coordinates.covariance.sigmas
    expected_sigma_ra_arcsec = float(np.sqrt(0.5**2 + sigma_b_ra_arcsec**2))
    expected_sigma_dec_arcsec = float(np.sqrt(0.5**2 + sigma_b_dec_arcsec**2))
    np.testing.assert_allclose(
        sigmas_deg[0, 1] * cos_dec * 3600.0, expected_sigma_ra_arcsec, rtol=1e-9
    )
    np.testing.assert_allclose(
        sigmas_deg[0, 2] * 3600.0, expected_sigma_dec_arcsec, rtol=1e-9
    )

    # (3) No off-diagonal cov term. cov[1, 2] should remain 0 (baseline corr = 0).
    cov_mat = od.coordinates.covariance.to_matrix()
    np.testing.assert_allclose(cov_mat[0, 1, 2], 0.0, atol=1e-30)

    # Station absent from bias_table → no subtract, no inflation.
    od_pass = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_table={"OTHER": (1.0, 1.0)},
        bias_application="subtract_sem_inflated",
        station_bias_ci_arcsec={"OTHER": (0.5, 0.5)},
    )
    np.testing.assert_allclose(
        od_pass.coordinates.lon.to_numpy(zero_copy_only=False), [ra_deg], atol=1e-12
    )
    np.testing.assert_allclose(
        od_pass.coordinates.lat.to_numpy(zero_copy_only=False), [dec_deg], atol=1e-12
    )
    sd = od_pass.coordinates.covariance.sigmas
    np.testing.assert_allclose(sd[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-9)
    np.testing.assert_allclose(sd[0, 2] * 3600.0, 0.5, rtol=1e-9)

    # Missing station_bias_ci_arcsec → ValueError.
    try:
        mpc_to_od_observations(
            obs,
            prevent_nans=False,
            bias_table={stn: (bias_ra_arcsec, bias_dec_arcsec)},
            bias_application="subtract_sem_inflated",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing station_bias_ci_arcsec")


def _make_synthetic_obs_corr(
    ra_deg: float,
    dec_deg: float,
    stn: str,
    rmsra: float,
    rmsdec: float,
    rmscorr: float,
    obs_id: str = "synthetic-corr-1",
) -> MPCObservations:
    """Single-row MPCObservations with an explicit baseline RA/Dec correlation."""
    t = Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="utc")
    return MPCObservations.from_kwargs(
        requested_provid=[obs_id],
        primary_designation=[obs_id],
        obsid=[obs_id],
        trksub=["trk1"],
        provid=[obs_id],
        permid=[None],
        submission_id=[None],
        obssubid=[None],
        obstime=t,
        ra=[ra_deg],
        dec=[dec_deg],
        rmsra=[rmsra],
        rmsdec=[rmsdec],
        rmscorr=[rmscorr],
        mag=[20.0],
        rmsmag=[0.1],
        band=["V"],
        stn=[stn],
        updated_at=t,
        created_at=t,
        status=["valid"],
        astcat=["UCAC4"],
        mode=["CCD"],
    )


def test_mpc_to_od_observations_bias_empirical_covar() -> None:
    """empirical_covar: 2×2 cov += MEASURED [[rv_ra, rcov], [rcov, rv_dec]].

    Verifies (1) the inflated diagonals, (2) the off-diagonal cross-term to
    1e-9, (3) the resid_cov_n threshold gate, (4) passthrough for absent
    stations, (5) the cross-term adds *onto* a non-zero baseline correlation,
    and (6) the missing-arg ValueError.
    """
    ra_deg = 50.0
    dec_deg = 0.0  # cos(dec)=1 keeps the σ math direct
    stn = "Z99"
    # Baseline σ_ra = σ_dec = 0.5", baseline corr = 0.
    obs = _make_synthetic_obs(ra_deg=ra_deg, dec_deg=dec_deg, stn=stn)

    rv_ra = 0.30  # arcsec²
    rv_dec = 0.20
    rcov = 0.10
    n = 100.0

    od = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_application="empirical_covar",
        station_resid_covar={stn: (rv_ra, rv_dec, rcov, n)},
    )
    assert od is not None

    cos_dec = np.cos(np.deg2rad(dec_deg))
    sigmas_deg = od.coordinates.covariance.sigmas
    # (1) Inflated diagonals: σ² = 0.5² + resid_var.
    np.testing.assert_allclose(
        sigmas_deg[0, 1] * cos_dec * 3600.0, float(np.sqrt(0.25 + rv_ra)), rtol=1e-9
    )
    np.testing.assert_allclose(
        sigmas_deg[0, 2] * 3600.0, float(np.sqrt(0.25 + rv_dec)), rtol=1e-9
    )
    # (2) Cross-term: cov_rd_arcsec² = baseline(0) + rcov = 0.10. At dec=0 the
    # cos(dec)-corrected frame coincides with the (lon, lat) deg frame, so
    # cov[1,2] * 3600² recovers the cos(dec) cross-covariance directly.
    cov_mat = od.coordinates.covariance.to_matrix()
    np.testing.assert_allclose(cov_mat[0, 1, 2] * (3600.0**2), rcov, atol=1e-9)
    # Positions are NEVER touched by empirical_covar (HARD constraint).
    np.testing.assert_allclose(
        od.coordinates.lon.to_numpy(zero_copy_only=False), [ra_deg], atol=1e-12
    )
    np.testing.assert_allclose(
        od.coordinates.lat.to_numpy(zero_copy_only=False), [dec_deg], atol=1e-12
    )

    # (3) Below-threshold residual sample (n < 30) → pass through unchanged.
    od_lown = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_application="empirical_covar",
        station_resid_covar={stn: (rv_ra, rv_dec, rcov, 10.0)},
        resid_cov_n_threshold=30,
    )
    sd = od_lown.coordinates.covariance.sigmas
    np.testing.assert_allclose(sd[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-9)
    np.testing.assert_allclose(sd[0, 2] * 3600.0, 0.5, rtol=1e-9)
    np.testing.assert_allclose(
        od_lown.coordinates.covariance.to_matrix()[0, 1, 2], 0.0, atol=1e-30
    )

    # (4) Station absent from the dict → unchanged.
    od_absent = mpc_to_od_observations(
        obs,
        prevent_nans=False,
        bias_application="empirical_covar",
        station_resid_covar={"OTHER": (rv_ra, rv_dec, rcov, n)},
    )
    sda = od_absent.coordinates.covariance.sigmas
    np.testing.assert_allclose(sda[0, 1] * cos_dec * 3600.0, 0.5, rtol=1e-9)
    np.testing.assert_allclose(sda[0, 2] * 3600.0, 0.5, rtol=1e-9)

    # (5) Non-zero baseline correlation: the measured rcov adds onto the
    # baseline cross-term (corr_base · σ_ra · σ_dec).
    obs_corr = _make_synthetic_obs_corr(
        ra_deg=ra_deg, dec_deg=dec_deg, stn=stn,
        rmsra=0.5, rmsdec=0.5, rmscorr=0.4,
    )
    od_corr = mpc_to_od_observations(
        obs_corr,
        prevent_nans=False,
        bias_application="empirical_covar",
        station_resid_covar={stn: (rv_ra, rv_dec, rcov, n)},
    )
    base_cross = 0.4 * 0.5 * 0.5  # corr · σ_ra · σ_dec, arcsec²
    cov_corr = od_corr.coordinates.covariance.to_matrix()
    np.testing.assert_allclose(
        cov_corr[0, 1, 2] * (3600.0**2), base_cross + rcov, atol=1e-9
    )
    # Diagonals unchanged by baseline corr; still σ² = 0.25 + resid_var.
    sc = od_corr.coordinates.covariance.sigmas
    np.testing.assert_allclose(
        sc[0, 1] * cos_dec * 3600.0, float(np.sqrt(0.25 + rv_ra)), rtol=1e-9
    )
    np.testing.assert_allclose(
        sc[0, 2] * 3600.0, float(np.sqrt(0.25 + rv_dec)), rtol=1e-9
    )

    # (6) Missing station_resid_covar → ValueError.
    try:
        mpc_to_od_observations(
            obs, prevent_nans=False, bias_application="empirical_covar"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing station_resid_covar")


def test_load_v2_bias_catalog_rollup() -> None:
    """v2 catalog loader keeps one per-station rollup row and exposes Fix-2 cols."""
    from ..utils import load_v2_bias_catalog

    v2_path = (
        "/Users/kathleenkiker/beads_agent_setup/adam_orbit_det_eval/"
        "data/bias_catalog_v2_full_no_prog_20260622/bias_table.parquet"
    )
    if not Path(v2_path).exists():
        import pytest

        pytest.skip("v2 catalog not present in this environment")

    cat = load_v2_bias_catalog(v2_path, rollup_only=True)
    # One record per station (per the rollup filter).
    assert len(cat) == 1123
    # Every record carries the Fix-2 residual-covariance fields.
    sample = next(iter(cat.values()))
    for key in (
        "bias_ra", "bias_dec", "sem_ra", "sem_dec", "chi2_per_obs",
        "rms_ra", "rms_dec", "bias_significant", "high_confidence",
        "resid_var_ra", "resid_var_dec", "resid_cov_ra_dec", "resid_cov_n",
        "bias_at", "bias_ct", "resid_var_at", "resid_var_ct", "resid_cov_at_ct",
    ):
        assert key in sample, f"missing field {key}"
    # Spot-check a known station against the raw catalog (station 006).
    assert "006" in cat
    np.testing.assert_allclose(
        np.sqrt(cat["006"]["resid_var_ra"]), cat["006"]["rms_ra"], rtol=0.1
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


def test_mpc_to_od_observations_carries_astcat() -> None:
    """Regression for the silent-no-op hazard: the converter used to drop
    ``astcat``, so EFCC18DebiasModel did nothing and the Veres models fell
    back to 0.75" for every observation."""
    mpc = _make_synthetic_obs(ra_deg=10.0, dec_deg=20.0, stn="F51")
    od = mpc_to_od_observations(mpc)
    assert od is not None
    assert "astcat" in od.table.column_names
    assert od.astcat.to_pylist() == ["UCAC4"]
    assert od.astcat.null_count == 0
