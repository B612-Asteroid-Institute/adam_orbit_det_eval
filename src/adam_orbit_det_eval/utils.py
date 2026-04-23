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
# reported rmsra/rmsdec.  These are catalog-level averages from Table 1
# of Veres 2017.  Observatory-specific overrides (also from that table)
# can be added to VERES2017_STN_CATALOG_OVERRIDES below.
#
# Key: MPC astcat code as stored in the observations table.
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
    "Tycho":            (0.06, 0.06),  # very precise but sparse
    "AC":               (0.80, 0.80),
    "Yale":             (1.00, 1.00),
    "UNK":              (1.00, 1.00),  # unknown catalog — large default
}

# Per-(stn, catalog) overrides for telescopes with known deviations from
# catalog averages.  Format: {(stn, astcat): (sigma_ra, sigma_dec)}.
# Populated from Table 1 of Veres 2017 for the highest-volume stations.
VERES2017_STN_CATALOG_OVERRIDES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("703", "Gaia2"):   (0.34, 0.34),  # Catalina — wider PSF
    ("703", "UCAC4"):   (0.45, 0.45),
    ("703", "UCAC2"):   (0.55, 0.55),
    ("G96", "Gaia2"):   (0.25, 0.25),  # Mt. Lemmon
    ("G96", "UCAC4"):   (0.35, 0.35),
    ("704", "USNOA2"):  (0.60, 0.75),  # Spacewatch — Dec bias known
    ("F51", "Gaia2"):   (0.15, 0.15),  # Pan-STARRS 1 — excellent
    ("F51", "Gaia1"):   (0.18, 0.18),
    ("F51", "2MASS"):   (0.20, 0.20),
    ("F52", "Gaia3E"):  (0.15, 0.15),  # Pan-STARRS 2
    ("F52", "Gaia1"):   (0.18, 0.18),
    ("T05", "Gaia2"):   (0.25, 0.25),  # ATLAS HKO
    ("T08", "Gaia2"):   (0.25, 0.25),  # ATLAS MLO
    ("W68", "Gaia2"):   (0.25, 0.25),  # ATLAS CHL
}

# Fallback sigma when catalog is unknown or not in the table (arcsec)
VERES2017_FALLBACK_SIGMA = 0.75


def get_veres2017_sigma(
    stn: Optional[str], astcat: Optional[str]
) -> Tuple[float, float]:
    """
    Return (sigma_ra_arcsec, sigma_dec_arcsec) for an observation from
    (stn, astcat) using the Veres et al. 2017 model.

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
    sigma_model: str = "veres2017",
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
      only used when sigma_model='const' — value to fill missing covariance diagonals
    sigma_model: str, default 'veres2017'
      how to fill missing rmsra/rmsdec values:
        'veres2017' — use Veres et al. 2017 per-(stn, catalog) sigma lookup (recommended)
        'const'     — fill with diag_nan (original behaviour, yields inflated chi2)

    Returns:
    --------
    Set of observations for orbit determination, size N, or None if the input set is
    malformed, for example, it has NULLs in the STN codes.
    """
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

    sigma_ra_cosdec_arcsec = obs_set.rmsra.to_numpy(zero_copy_only=False)
    sigma_dec_arcsec = obs_set.rmsdec.to_numpy(zero_copy_only=False)

    if sigma_model == "veres2017":
        # Fill missing sigmas using Veres 2017 per-(stn, catalog) lookup
        stns = obs_set.stn.to_pylist()
        astcats = obs_set.astcat.to_pylist()
        for i in range(len(obs_set)):
            if not np.isfinite(sigma_ra_cosdec_arcsec[i]) or sigma_ra_cosdec_arcsec[i] <= 0:
                v_ra, v_dec = get_veres2017_sigma(stns[i], astcats[i])
                sigma_ra_cosdec_arcsec[i] = v_ra
                sigma_dec_arcsec[i] = v_dec

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
    cov[:, 1, 1] = sigma_ra_deg**2
    cov[:, 2, 2] = sigma_dec_deg**2
    if prevent_nans:
        cov[:, 1, 1] = np.nan_to_num(cov[:, 1, 1], nan=diag_nan)
        cov[:, 2, 2] = np.nan_to_num(cov[:, 2, 2], nan=diag_nan)
    cov[:, 1, 2] = corr * sigma_ra_deg * sigma_dec_deg
    cov[:, 2, 1] = cov[:, 1, 2]
    if prevent_nans:
        cov = np.nan_to_num(cov)

    coords = SphericalCoordinates.from_kwargs(
        lon=obs_set.ra.to_numpy(zero_copy_only=False),
        lat=obs_set.dec.to_numpy(zero_copy_only=False),
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
