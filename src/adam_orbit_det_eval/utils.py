import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    OrbitDeterminationPhotometry,
)
from mpc_obscodes import mpc_obscodes
from mpcq import MPCObservations


# ---------------------------------------------------------------------------
# Veres et al. 2017 (Icarus 296, 139-149) astrometric uncertainty model.
#
# Per-catalog default sigmas (arcsec) used when an observation has no
# reported rmsra/rmsdec. These are catalog-level averages from Table 1 of
# Veres 2017. Observatory-specific overrides (also from that table) live in
# VERES2017_STN_CATALOG_OVERRIDES below.
#
# Mirrors the same table maintained in the parent workspace's adam_orbit_det_eval
# (kk/claude/mpc-eval branch) so OD experiments stay comparable. If that
# table is updated in the parent workspace, re-sync this block.
# ---------------------------------------------------------------------------

VERES2017_CATALOG_DEFAULTS: Dict[str, Tuple[float, float]] = {
    # Catalog name        (sigma_RA_arcsec, sigma_Dec_arcsec)
    # --- Gaia family (best modern catalogs) ---
    "Gaia3E":           (0.15, 0.15),  # Gaia EDR3 / DR3
    "Gaia3":            (0.15, 0.15),
    "Gaia2":            (0.18, 0.18),  # Gaia DR2
    "Gaia1":            (0.25, 0.25),  # Gaia DR1
    # --- ATLAS family ---
    "ATLAS2":           (0.20, 0.20),
    "ATLAS":            (0.25, 0.25),
    # --- UCAC family ---
    "UCAC5":            (0.25, 0.25),
    "SSTRC4":           (0.25, 0.25),  # UCAC4-based survey catalog
    "UCAC4":            (0.30, 0.30),
    "UCAC3":            (0.30, 0.30),
    "UCAC2":            (0.40, 0.40),
    "UCAC1":            (0.50, 0.50),
    # --- 2MASS ---
    "2MASS":            (0.20, 0.20),
    # --- USNO catalogs ---
    "USNOB1":           (0.50, 0.50),
    "USNOA2":           (0.60, 0.60),
    "USNOSA2":          (0.60, 0.60),
    "USNOA1":           (0.80, 0.80),
    # --- GSC family ---
    "GSC":              (0.50, 0.50),
    "GSC1.1":           (0.50, 0.50),
    "GSC1.2":           (0.50, 0.50),
    "GSC2.2":           (0.40, 0.40),
    "GSC2.3":           (0.35, 0.35),
    "GSCACT":           (0.50, 0.50),
    # --- PPMXL / PPM ---
    "PPMXL":            (0.35, 0.35),
    "PPM":              (0.50, 0.50),
    # --- Other catalogs ---
    "SDSS8":            (0.20, 0.20),
    "SDSS7":            (0.20, 0.20),
    "NOMAD":            (0.40, 0.40),
    "CMC14":            (0.35, 0.35),
    "CMC15":            (0.30, 0.30),
    "Tycho":            (0.06, 0.06),
    "AC":               (0.80, 0.80),
    "Yale":             (1.00, 1.00),
    "UNK":              (1.00, 1.00),
}

VERES2017_STN_CATALOG_OVERRIDES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("703", "Gaia2"):   (0.34, 0.34),  # Catalina — wider PSF
    ("703", "UCAC4"):   (0.45, 0.45),
    ("703", "UCAC2"):   (0.55, 0.55),
    ("G96", "Gaia2"):   (0.25, 0.25),  # Mt. Lemmon
    ("G96", "UCAC4"):   (0.35, 0.35),
    ("704", "USNOA2"):  (0.60, 0.75),  # Spacewatch — known Dec bias
    ("F51", "Gaia2"):   (0.15, 0.15),  # Pan-STARRS 1
    ("F51", "Gaia1"):   (0.18, 0.18),
    ("F51", "2MASS"):   (0.20, 0.20),
    ("F52", "Gaia3E"):  (0.15, 0.15),  # Pan-STARRS 2
    ("F52", "Gaia1"):   (0.18, 0.18),
    ("T05", "Gaia2"):   (0.25, 0.25),  # ATLAS HKO
    ("T08", "Gaia2"):   (0.25, 0.25),  # ATLAS MLO
    ("W68", "Gaia2"):   (0.25, 0.25),  # ATLAS CHL
}

VERES2017_FALLBACK_SIGMA = 0.75


def get_veres2017_sigma(
    stn: Optional[str], astcat: Optional[str]
) -> Tuple[float, float]:
    """Return (sigma_ra_arcsec, sigma_dec_arcsec) for (stn, astcat) using Veres 2017.

    Lookup order:
      1. Per-(stn, catalog) override
      2. Per-catalog default
      3. Global fallback (0.75 arcsec)
    """
    if stn and astcat:
        override = VERES2017_STN_CATALOG_OVERRIDES.get((stn, astcat))
        if override:
            return override
    if astcat:
        default = VERES2017_CATALOG_DEFAULTS.get(astcat)
        if default:
            return default
    return (VERES2017_FALLBACK_SIGMA, VERES2017_FALLBACK_SIGMA)


def get_spacebased_stns() -> List[str]:
    """Return all known STN codes without fixed Earth coordinates"""
    with open(mpc_obscodes) as mpc_file:
        obscodes = json.load(mpc_file)
        return [k for k, v in obscodes.items() if "Longitude" not in v]


def mpc_to_od_observations(
    obs_set: MPCObservations,
    prevent_nans: bool = True,
    diag_nan: float = 1.0e-9,
    bias_table: Optional[Dict[str, Tuple[float, float]]] = None,
    sigma_model: str = "const",
    bias_application: str = "sigma_floor",
) -> Optional[OrbitDeterminationObservations]:
    """
    Convert MPC observations into OD observations.

    Parameters:
    -----------
    obs_set: MPCObservations (size N)
      set of MPC observations to convert
    prevent_nans: bool, default True
      should NaN/null values on covariance matrix be replaced with finite numbers.
      diagonal NaNs are replaced with diag_nan, off-diagonal with zeros
    diag_nan: float, default 1.0e-9
      the value to replace NaNs in covariance diagonals if prevent_nans is True
      and ``sigma_model='const'``
    bias_table: dict[str, tuple[float, float]] or None, default None
      Optional per-station bias from the LOOO bias catalog, keyed by MPC obs_code.
      Each value is ``(bias_ra_arcsec, bias_dec_arcsec)`` where ``bias_ra_arcsec``
      is in the cos(dec)-corrected tangent-plane frame (matching the LOOO residual
      convention). How it is applied is controlled by ``bias_application``.
      Stations absent from the table pass through unchanged.
    sigma_model: str, default 'const'
      How to fill missing/non-finite ``rmsra``/``rmsdec``:
        ``'const'``     — leave the missing rows as NaN and let ``prevent_nans``/
                          ``diag_nan`` substitute a tiny constant (legacy behavior).
        ``'veres2017'`` — fill missing sigmas via ``get_veres2017_sigma(stn, astcat)``
                          (per-catalog defaults + per-(stn, catalog) overrides from
                          Veres et al. 2017 Table 1). Observations with finite MPC
                          sigmas are untouched.
    bias_application: str, default 'sigma_floor'
      Only consulted when ``bias_table`` is supplied. How to use the bias:
        ``'sigma_floor'`` — *Recommended.* Treat the catalog like Veres 2017: for
                            every observation from a station in the table, floor
                            the per-axis sigma at ``|bias|`` so biased stations
                            get downweighted in the fit:
                              sigma_ra_used  = max(sigma_ra_reported,  |bias_ra|)
                              sigma_dec_used = max(sigma_dec_reported, |bias_dec|)
                            RA/Dec are NOT modified.
        ``'subtract'``    — *Legacy.* Subtract the bias from the observation
                            position, leaving sigmas untouched:
                              corrected_dec = obs_dec - bias_dec / 3600
                              corrected_ra  = obs_ra  - (bias_ra / 3600) / cos(dec)
                            This treats the bias estimate as ground truth and was
                            judged too aggressive in the YR4 baseline experiment;
                            kept for reproducibility/A-B comparison.

    Returns:
    --------
    Set of observations for orbit determination, size N, or None if the input set is
    malformed, for example, it has NULLs in the STN codes.
    """
    if bias_application not in ("sigma_floor", "subtract"):
        raise ValueError(
            f"Unknown bias_application={bias_application!r}; "
            "expected 'sigma_floor' or 'subtract'"
        )
    obs_time = obs_set.obstime
    codes = obs_set.stn
    if not np.all(codes):
        print(
            f"STN codes for {obs_set.requested_provid.unique().to_pylist()} include nulls"
        )
        return None

    # `mpcq`'s `MPCObservations` includes uncertainty columns:
    # - rmsra, rmsdec, rmscorr (and rmsmag)
    #
    # These RMS values come from the MPC database and are in arcseconds. By ADES/MPC convention,
    # `rmsra` is RA uncertainty *cos(dec). We convert into degrees and back out RA sigma
    # (so that downstream `Residuals` can apply its own cos(latitude) scaling consistently).
    dec_deg = obs_set.dec.to_numpy(zero_copy_only=False)
    cos_dec = np.cos(np.deg2rad(dec_deg))

    sigma_ra_cosdec_arcsec = obs_set.rmsra.to_numpy(zero_copy_only=False).astype(
        np.float64, copy=True
    )
    sigma_dec_arcsec = obs_set.rmsdec.to_numpy(zero_copy_only=False).astype(
        np.float64, copy=True
    )

    if sigma_model == "veres2017":
        # Fill rows where MPC's reported sigma is missing or non-positive using
        # the Veres 2017 per-(stn, catalog) lookup. Finite MPC sigmas are kept.
        stns_list = obs_set.stn.to_pylist()
        astcats_list = obs_set.astcat.to_pylist()
        for i in range(len(obs_set)):
            ra_bad = (
                not np.isfinite(sigma_ra_cosdec_arcsec[i])
                or sigma_ra_cosdec_arcsec[i] <= 0
            )
            dec_bad = (
                not np.isfinite(sigma_dec_arcsec[i]) or sigma_dec_arcsec[i] <= 0
            )
            if ra_bad or dec_bad:
                v_ra, v_dec = get_veres2017_sigma(stns_list[i], astcats_list[i])
                if ra_bad:
                    sigma_ra_cosdec_arcsec[i] = v_ra
                if dec_bad:
                    sigma_dec_arcsec[i] = v_dec
    elif sigma_model != "const":
        raise ValueError(
            f"Unknown sigma_model={sigma_model!r}; expected 'const' or 'veres2017'"
        )

    if bias_table is not None and bias_application == "sigma_floor":
        # Floor the per-axis sigma at |bias| for any station listed in the catalog.
        # If the MPC-reported sigma is missing/non-positive, use |bias| outright;
        # otherwise take the max. Stations absent from the table are unchanged.
        stns_list = obs_set.stn.to_pylist()
        for i, code in enumerate(stns_list):
            entry = bias_table.get(code)
            if entry is None:
                continue
            bra = float(abs(entry[0]))  # cos(dec)-corrected arcsec, matches rmsra
            bdec = float(abs(entry[1]))
            cur_ra = sigma_ra_cosdec_arcsec[i]
            cur_dec = sigma_dec_arcsec[i]
            if (not np.isfinite(cur_ra)) or cur_ra < bra:
                sigma_ra_cosdec_arcsec[i] = bra
            if (not np.isfinite(cur_dec)) or cur_dec < bdec:
                sigma_dec_arcsec[i] = bdec

    sigma_ra_cosdec_deg = sigma_ra_cosdec_arcsec / 3600.0
    sigma_dec_deg = sigma_dec_arcsec / 3600.0
    sigma_ra_deg = np.where(
        np.isfinite(cos_dec) & (cos_dec != 0.0),
        sigma_ra_cosdec_deg / cos_dec,
        np.nan,
    )

    # Include RA/Dec correlation if present; treat missing correlation as 0 (uncorrelated).
    corr = obs_set.rmscorr.to_numpy(zero_copy_only=False)
    corr = np.where(np.isfinite(corr), corr, 0.0)

    cov = np.full((len(obs_set), 6, 6), np.nan, dtype=np.float64)
    # Prevent 'Covariance matrix has NaNs on the diagonal' and 'Singular matrix
    cov[:, 1, 1] = sigma_ra_deg**2
    cov[:, 2, 2] = sigma_dec_deg**2
    if prevent_nans:
        cov[:, 1, 1] = np.nan_to_num(cov[:, 1, 1], nan=diag_nan)
        cov[:, 2, 2] = np.nan_to_num(cov[:, 2, 2], nan=diag_nan)
    cov[:, 1, 2] = corr * sigma_ra_deg * sigma_dec_deg
    cov[:, 2, 1] = cov[:, 1, 2]
    # Prevents 'UserWarning: Covariance matrix has NaNs on the off-diagonal (these will be assumed to be 0.0).'
    if prevent_nans:
        cov = np.nan_to_num(cov)

    lon = obs_set.ra.to_numpy(zero_copy_only=False).astype(np.float64, copy=True)
    lat = obs_set.dec.to_numpy(zero_copy_only=False).astype(np.float64, copy=True)

    if bias_table is not None and bias_application == "subtract":
        stn_codes = codes.to_pylist()
        bias_ra_arcsec = np.zeros(len(obs_set), dtype=np.float64)
        bias_dec_arcsec = np.zeros(len(obs_set), dtype=np.float64)
        for i, code in enumerate(stn_codes):
            entry = bias_table.get(code)
            if entry is None:
                continue
            bias_ra_arcsec[i] = entry[0]
            bias_dec_arcsec[i] = entry[1]
        # cos(dec) here uses the (uncorrected) observed declination — the bias
        # is small enough that evaluating cos at the corrected dec changes the
        # result by O(bias^2) which is negligible at sub-arcsec scales.
        cos_dec_for_bias = np.cos(np.deg2rad(lat))
        with np.errstate(divide="ignore", invalid="ignore"):
            ra_correction_deg = np.where(
                np.isfinite(cos_dec_for_bias) & (cos_dec_for_bias != 0.0),
                (bias_ra_arcsec / 3600.0) / cos_dec_for_bias,
                0.0,
            )
        lon = lon - ra_correction_deg
        lat = lat - bias_dec_arcsec / 3600.0

    coords = SphericalCoordinates.from_kwargs(
        lon=lon,
        lat=lat,
        time=obs_time,
        origin=Origin.from_kwargs(code=codes),
        frame="equatorial",
        covariance=CoordinateCovariances.from_matrix(cov),
    )

    observers = Observers.from_codes(codes=codes, times=obs_time)

    photometry = OrbitDeterminationPhotometry.from_kwargs(
        mag=obs_set.mag,
        rmsmag=obs_set.rmsmag,
        band=obs_set.band,
    )

    od_observations = OrbitDeterminationObservations.from_kwargs(
        id=obs_set.obsid.to_numpy(zero_copy_only=False),
        coordinates=coords,
        observers=observers,
        photometry=photometry,
    )
    return od_observations
