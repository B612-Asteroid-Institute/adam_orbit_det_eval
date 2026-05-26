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
    catalog_debias_arcsec: Optional[np.ndarray] = None,
    uniform_sigma_arcsec: float = 0.5,
    station_chi2_per_obs: Optional[Dict[str, float]] = None,
    station_sem_arcsec: Optional[Dict[str, Tuple[float, float]]] = None,
    atct_bias_table: Optional[Dict[str, Tuple[float, float]]] = None,
    atct_unit_vectors: Optional[np.ndarray] = None,
    station_bias_ci_arcsec: Optional[Dict[str, Tuple[float, float]]] = None,
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
        ``'sigma_floor'``           — *Recommended.* For each station in the
                                      table, floor the per-axis sigma at
                                      ``|bias|`` (``σ_used = max(σ_reported,
                                      |bias|)``). Downweights biased stations
                                      without modifying RA/Dec.
        ``'rss_additive'``          — σ_used = sqrt(σ_baseline² + bias²) per
                                      axis. Softer than the max-floor when
                                      σ_baseline >> |bias|; equivalent to
                                      sigma_floor when σ_baseline << |bias|.
        ``'performance_weighted'``  — σ_used = σ_baseline × sqrt(max(
                                      chi2_per_obs, 1.0)). Inflates sigma by
                                      the station's measured residual scatter
                                      (from v1's ``chi2_per_obs`` column).
                                      Requires ``station_chi2_per_obs`` to be
                                      supplied. Stations not in the chi2 dict
                                      get factor = 1 (unchanged).
        ``'bayes_shrinkage'``       — Per-axis Bayesian shrinkage of the bias
                                      magnitude when the bias estimate is
                                      itself noisy. Computes a shrinkage
                                      factor ``s = σ_base² / (σ_base² +
                                      SEM_bias²)`` and combines as
                                      ``σ_used = sqrt(σ_base² + (|bias|·s)²)``
                                      per axis. Reduces bias-inflation when
                                      SEM_bias >> σ_base (i.e. when the bias
                                      estimate is poorly determined). Requires
                                      ``station_sem_arcsec`` with per-station
                                      ``(sem_ra, sem_dec)``. Stations absent
                                      from either dict are unchanged.
        ``'veres_v1_max_floor'``    — Force-override the per-axis sigma with
                                      the Veres 2017 per-(stn, cat) lookup
                                      (ignoring MPC-reported sigmas), then
                                      floor at ``|bias_v1|``. Stacks the two
                                      principled sigma sources.
        ``'covar_inflation'``       — Inflate the full 2×2 obs covariance
                                      using ``(bias_ra, bias_dec)`` as a
                                      joint perturbation:
                                      ``Cov_used = Cov_baseline + outer(b, b)``
                                      where ``b = (bias_ra, bias_dec)``.
                                      Produces a non-diagonal effective
                                      covariance via the rmscorr coupling.
                                      Uses the joint structure of the bias
                                      estimate, not just diagonal σ.
        ``'at_ct_floor'``           — Sigma-floor applied in the along-track/
                                      cross-track basis of the object's
                                      sky-plane motion. Requires
                                      ``atct_bias_table`` (per-stn
                                      ``(|bias_AT|, |bias_CT|)`` in arcsec)
                                      and ``atct_unit_vectors`` (shape
                                      ``(N, 2)`` giving the per-obs cos(dec)-
                                      frame velocity unit vector
                                      ``(u_ra_cosdec, u_dec)``). The AT/CT
                                      baseline σ comes from rotating the
                                      diagonal RA/Dec baseline covariance.
                                      σ_AT_used = max(σ_AT_base, |bias_AT|);
                                      same for CT. The effective 2×2 cov is
                                      then rotated back to RA/Dec, producing
                                      correlated σ_RA, σ_Dec.
        ``'subtract'``              — *Legacy/reference.* Subtract the bias
                                      from the observation position. Treats
                                      the bias as ground truth; too aggressive
                                      in principle but retained as a baseline
                                      anchor.
        ``'subtract_sem_inflated'`` — Subtract the bias from the position
                                      (same as ``subtract``) AND inflate the
                                      per-axis sigma diagonally with the
                                      bias-estimate uncertainty:
                                      ``σ_used² = σ_baseline² + σ_b²``,
                                      no off-diagonal term. Pairs the
                                      ``subtract`` win on objects where the
                                      offset is well-determined with a
                                      graceful ``no_bias``-like fallback
                                      when the offset is uncertain. Requires
                                      ``station_bias_ci_arcsec`` (per-stn
                                      1-σ uncertainty on the bias estimate).
    catalog_debias_arcsec: ndarray of shape (N, 2) or None, default None
      Optional per-observation star-catalog debiasing correction. Each row is
      ``(bias_ra_cosdec_arcsec, bias_dec_arcsec)`` and is subtracted from the
      observation as:
        corrected_dec = obs_dec - bias_dec / 3600
        corrected_ra  = obs_ra  - (bias_ra / 3600) / cos(dec)
      Intended consumer: per-(RA, Dec, astcat, JD) EFCC18 corrections computed
      via :func:`adam_orbit_det_eval.efcc18.compute_efcc18_corrections`. Stations/
      catalogs with no available correction should be passed in as 0.0 rows.
    uniform_sigma_arcsec: float, default 0.5
      Used only when ``sigma_model='uniform'``: every observation gets
      ``σ_ra_cosdec = σ_dec = uniform_sigma_arcsec`` regardless of the MPC
      sigma column or any other input. Diagnostic mode for testing whether
      per-obs weighting matters at all.
    station_chi2_per_obs: dict[str, float] or None, default None
      Used only when ``bias_application='performance_weighted'``. Maps station
      ``obs_code`` to ``chi2_per_obs`` from the v1 bias catalog. Stations
      absent from the dict are not inflated (factor = 1).
    station_sem_arcsec: dict[str, tuple[float, float]] or None, default None
      Used only when ``bias_application='bayes_shrinkage'``. Maps station
      ``obs_code`` to ``(sem_ra_arcsec, sem_dec_arcsec)`` — the standard
      error of the per-axis mean-bias estimate from the v1 catalog. Stations
      absent from the dict are unchanged.
    atct_bias_table: dict[str, tuple[float, float]] or None, default None
      Used only when ``bias_application='at_ct_floor'``. Maps station
      ``obs_code`` to ``(|bias_AT_arcsec|, |bias_CT_arcsec|)`` from the v1
      AT/CT bias catalog (per-station rollup; ``program_code IS NULL``).
    atct_unit_vectors: ndarray of shape (N, 2) or None, default None
      Used only when ``bias_application='at_ct_floor'``. Per-observation
      sky-plane velocity unit vector in the cos(dec)-corrected frame:
      ``(u_ra_cosdec, u_dec)``. Rows with NaN/zero velocity (near-stationary)
      receive the unrotated (RA/Dec-diagonal) sigma_floor as a fallback.
    station_bias_ci_arcsec: dict[str, tuple[float, float]] or None, default None
      Used only when ``bias_application='subtract_sem_inflated'``. Maps
      station ``obs_code`` to ``(σ_b_ra, σ_b_dec)`` — the per-axis 1-σ
      uncertainty on the v1 bias estimate. Conventionally derived from the
      published 95% CI half-width: ``σ_b = (ci_high - ci_low) / 2 / 1.96``.
      Stations absent from the dict (or from ``bias_table``) pass through
      unchanged.

    Returns:
    --------
    Set of observations for orbit determination, size N, or None if the input set is
    malformed, for example, it has NULLs in the STN codes.
    """
    if bias_application not in (
        "sigma_floor",
        "subtract",
        "rss_additive",
        "performance_weighted",
        "bayes_shrinkage",
        "veres_v1_max_floor",
        "covar_inflation",
        "at_ct_floor",
        "subtract_sem_inflated",
    ):
        raise ValueError(
            f"Unknown bias_application={bias_application!r}; expected one of "
            "'sigma_floor', 'subtract', 'rss_additive', 'performance_weighted', "
            "'bayes_shrinkage', 'veres_v1_max_floor', 'covar_inflation', "
            "'at_ct_floor', 'subtract_sem_inflated'"
        )
    if bias_application == "performance_weighted" and station_chi2_per_obs is None:
        raise ValueError(
            "bias_application='performance_weighted' requires station_chi2_per_obs"
        )
    if bias_application == "bayes_shrinkage" and station_sem_arcsec is None:
        raise ValueError(
            "bias_application='bayes_shrinkage' requires station_sem_arcsec"
        )
    if bias_application == "at_ct_floor":
        if atct_bias_table is None or atct_unit_vectors is None:
            raise ValueError(
                "bias_application='at_ct_floor' requires both atct_bias_table"
                " and atct_unit_vectors"
            )
    if bias_application == "subtract_sem_inflated" and station_bias_ci_arcsec is None:
        raise ValueError(
            "bias_application='subtract_sem_inflated' requires station_bias_ci_arcsec"
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
    elif sigma_model == "uniform":
        # Override every observation's sigma with the uniform value, regardless
        # of MPC reports. Diagnostic mode.
        if uniform_sigma_arcsec <= 0:
            raise ValueError(
                f"uniform_sigma_arcsec must be > 0, got {uniform_sigma_arcsec}"
            )
        sigma_ra_cosdec_arcsec[:] = uniform_sigma_arcsec
        sigma_dec_arcsec[:] = uniform_sigma_arcsec
    elif sigma_model != "const":
        raise ValueError(
            f"Unknown sigma_model={sigma_model!r}; expected one of "
            "'const', 'veres2017', 'uniform'"
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
    elif bias_table is not None and bias_application == "rss_additive":
        # Combine the baseline sigma and the bias in quadrature. Acts like
        # sigma_floor when bias dominates and like a no-op when sigma_baseline
        # dominates. Missing/non-positive baseline sigmas are treated as 0 so
        # the result becomes |bias| (matching sigma_floor in that limit).
        stns_list = obs_set.stn.to_pylist()
        for i, code in enumerate(stns_list):
            entry = bias_table.get(code)
            if entry is None:
                continue
            bra = float(abs(entry[0]))
            bdec = float(abs(entry[1]))
            cur_ra = sigma_ra_cosdec_arcsec[i]
            cur_dec = sigma_dec_arcsec[i]
            base_ra = cur_ra if np.isfinite(cur_ra) and cur_ra > 0 else 0.0
            base_dec = cur_dec if np.isfinite(cur_dec) and cur_dec > 0 else 0.0
            sigma_ra_cosdec_arcsec[i] = float(np.sqrt(base_ra * base_ra + bra * bra))
            sigma_dec_arcsec[i] = float(np.sqrt(base_dec * base_dec + bdec * bdec))
    elif bias_application == "performance_weighted":
        # Per-station chi2_per_obs from the v1 catalog acts as a multiplicative
        # sigma scale: σ_used = σ_baseline × sqrt(max(chi2, 1)). Stations not in
        # the dict get factor = 1 (unchanged).
        assert station_chi2_per_obs is not None  # validated above
        stns_list = obs_set.stn.to_pylist()
        for i, code in enumerate(stns_list):
            chi2 = station_chi2_per_obs.get(code)
            if chi2 is None or not np.isfinite(chi2):
                continue
            factor = float(np.sqrt(max(float(chi2), 1.0)))
            cur_ra = sigma_ra_cosdec_arcsec[i]
            cur_dec = sigma_dec_arcsec[i]
            if np.isfinite(cur_ra) and cur_ra > 0:
                sigma_ra_cosdec_arcsec[i] = cur_ra * factor
            if np.isfinite(cur_dec) and cur_dec > 0:
                sigma_dec_arcsec[i] = cur_dec * factor
    elif bias_application == "bayes_shrinkage" and bias_table is not None:
        # Per-axis Bayesian shrinkage: combine baseline σ with a damped bias.
        #
        #   s = σ_base² / (σ_base² + SEM_bias²)
        #   σ_used = sqrt(σ_base² + (|bias|·s)²)
        #
        # When SEM_bias >> σ_base the bias is treated as poorly determined and
        # shrinks toward 0 (σ_used → σ_base). When SEM_bias << σ_base the bias
        # passes through nearly unmodified (σ_used → sqrt(σ_base² + bias²),
        # matching rss_additive). Falls back to rss_additive when SEM is
        # missing for a station (conservative default).
        assert station_sem_arcsec is not None  # validated above
        stns_list = obs_set.stn.to_pylist()
        for i, code in enumerate(stns_list):
            entry = bias_table.get(code)
            if entry is None:
                continue
            bra = float(abs(entry[0]))
            bdec = float(abs(entry[1]))
            sem = station_sem_arcsec.get(code)
            sem_ra = float(abs(sem[0])) if sem is not None else 0.0
            sem_dec = float(abs(sem[1])) if sem is not None else 0.0
            cur_ra = sigma_ra_cosdec_arcsec[i]
            cur_dec = sigma_dec_arcsec[i]
            base_ra_sq = (
                cur_ra * cur_ra if np.isfinite(cur_ra) and cur_ra > 0 else 0.0
            )
            base_dec_sq = (
                cur_dec * cur_dec if np.isfinite(cur_dec) and cur_dec > 0 else 0.0
            )
            denom_ra = base_ra_sq + sem_ra * sem_ra
            denom_dec = base_dec_sq + sem_dec * sem_dec
            s_ra = base_ra_sq / denom_ra if denom_ra > 0 else 0.0
            s_dec = base_dec_sq / denom_dec if denom_dec > 0 else 0.0
            sigma_ra_cosdec_arcsec[i] = float(
                np.sqrt(base_ra_sq + (bra * s_ra) ** 2)
            )
            sigma_dec_arcsec[i] = float(
                np.sqrt(base_dec_sq + (bdec * s_dec) ** 2)
            )
    elif bias_application == "subtract_sem_inflated" and bias_table is not None:
        # Diagonal-only σ inflation by the bias-estimate uncertainty σ_b:
        #     σ_used² = σ_baseline² + σ_b²    per axis
        # The actual position subtraction (RA -= bias_ra, Dec -= bias_dec) is
        # handled later in the position-mutation block (mirrors the
        # ``subtract`` mode). No off-diagonal contribution is injected here —
        # that is the entire point of this variant vs ``covar_inflation``.
        assert station_bias_ci_arcsec is not None  # validated above
        stns_list = obs_set.stn.to_pylist()
        for i, code in enumerate(stns_list):
            if bias_table.get(code) is None:
                # Stations absent from the v1 bias table get neither the
                # subtract correction (below) nor the σ inflation (here).
                continue
            sigma_b = station_bias_ci_arcsec.get(code)
            if sigma_b is None:
                continue
            sb_ra = float(abs(sigma_b[0]))
            sb_dec = float(abs(sigma_b[1]))
            cur_ra = sigma_ra_cosdec_arcsec[i]
            cur_dec = sigma_dec_arcsec[i]
            base_ra = cur_ra if np.isfinite(cur_ra) and cur_ra > 0 else 0.0
            base_dec = cur_dec if np.isfinite(cur_dec) and cur_dec > 0 else 0.0
            sigma_ra_cosdec_arcsec[i] = float(
                np.sqrt(base_ra * base_ra + sb_ra * sb_ra)
            )
            sigma_dec_arcsec[i] = float(
                np.sqrt(base_dec * base_dec + sb_dec * sb_dec)
            )
    elif bias_application == "veres_v1_max_floor" and bias_table is not None:
        # Force Veres σ (per-(stn, catalog) lookup) as the baseline, ignoring
        # any MPC-reported sigma; then floor at |bias_v1|. Stacks the two
        # principled σ sources rather than letting the MPC report through.
        stns_list = obs_set.stn.to_pylist()
        astcats_list = obs_set.astcat.to_pylist()
        for i, code in enumerate(stns_list):
            v_ra, v_dec = get_veres2017_sigma(code, astcats_list[i])
            entry = bias_table.get(code)
            if entry is None:
                sigma_ra_cosdec_arcsec[i] = float(v_ra)
                sigma_dec_arcsec[i] = float(v_dec)
                continue
            bra = float(abs(entry[0]))
            bdec = float(abs(entry[1]))
            sigma_ra_cosdec_arcsec[i] = float(max(float(v_ra), bra))
            sigma_dec_arcsec[i] = float(max(float(v_dec), bdec))

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

    # Working covariance in (σ_ra_cosdec arcsec, σ_dec arcsec) space; later
    # converted to deg + filled into the (N, 6, 6) covariance tensor. Two of
    # the new modes (covar_inflation, at_ct_floor) inject non-diagonal
    # covariance terms here.
    sigma_ra_cosdec_sq = sigma_ra_cosdec_arcsec * sigma_ra_cosdec_arcsec
    sigma_dec_sq = sigma_dec_arcsec * sigma_dec_arcsec
    # Cross-term in arcsec² in the cos(dec)-corrected RA / Dec frame.
    cov_ra_cosdec_dec_arcsec_sq = (
        corr * sigma_ra_cosdec_arcsec * sigma_dec_arcsec
    )

    if bias_application == "covar_inflation" and bias_table is not None:
        # Inflate the 2×2 obs covariance by the outer product of the bias
        # vector b = (bias_ra, bias_dec) (treated in the cos(dec)-corrected
        # frame to match MPC's rmsra convention). For stations absent from the
        # table, the covariance is unchanged.
        stns_list = obs_set.stn.to_pylist()
        for i, code in enumerate(stns_list):
            entry = bias_table.get(code)
            if entry is None:
                continue
            bra = float(entry[0])  # signed; outer-product sign matters
            bdec = float(entry[1])
            sigma_ra_cosdec_sq[i] = sigma_ra_cosdec_sq[i] + bra * bra
            sigma_dec_sq[i] = sigma_dec_sq[i] + bdec * bdec
            cov_ra_cosdec_dec_arcsec_sq[i] = (
                cov_ra_cosdec_dec_arcsec_sq[i] + bra * bdec
            )
    elif bias_application == "at_ct_floor":
        # Sigma-floor in the AT/CT basis defined by the per-obs sky-plane
        # velocity unit vector. Falls back to a diagonal RA/Dec sigma_floor
        # using |bias_ra|, |bias_dec| from `bias_table` when the unit vector
        # is undefined (near-stationary) and `bias_table` is provided —
        # otherwise the row passes through unchanged.
        assert atct_bias_table is not None  # validated above
        assert atct_unit_vectors is not None
        unit = np.asarray(atct_unit_vectors, dtype=np.float64)
        if unit.shape != (len(obs_set), 2):
            raise ValueError(
                f"atct_unit_vectors must have shape ({len(obs_set)}, 2); "
                f"got {unit.shape}"
            )
        stns_list = obs_set.stn.to_pylist()
        for i, code in enumerate(stns_list):
            u_ra = float(unit[i, 0])
            u_dec = float(unit[i, 1])
            entry = atct_bias_table.get(code)
            unit_finite = np.isfinite(u_ra) and np.isfinite(u_dec) and (
                abs(u_ra) > 0.0 or abs(u_dec) > 0.0
            )
            if entry is None or not unit_finite:
                # Either station has no AT/CT bias listed, or the obs is
                # near-stationary (rotation undefined): pass through.
                continue
            b_at = float(abs(entry[0]))
            b_ct = float(abs(entry[1]))
            # Baseline σ_AT² / σ_CT² obtained by rotating the diagonal RA/Dec
            # (cos(dec)-corrected) baseline covariance into the AT/CT basis.
            cur_ra2 = (
                sigma_ra_cosdec_sq[i]
                if np.isfinite(sigma_ra_cosdec_sq[i]) and sigma_ra_cosdec_sq[i] > 0
                else 0.0
            )
            cur_dec2 = (
                sigma_dec_sq[i]
                if np.isfinite(sigma_dec_sq[i]) and sigma_dec_sq[i] > 0
                else 0.0
            )
            sigma_at2_base = u_ra * u_ra * cur_ra2 + u_dec * u_dec * cur_dec2
            sigma_ct2_base = u_dec * u_dec * cur_ra2 + u_ra * u_ra * cur_dec2
            sigma_at2_used = max(sigma_at2_base, b_at * b_at)
            sigma_ct2_used = max(sigma_ct2_base, b_ct * b_ct)
            # Rotate diag(σ_AT², σ_CT²) back into the RA/Dec frame.
            #   R = [[u_ra, u_dec], [-u_dec, u_ra]]   (RA/Dec → AT/CT)
            #   C_RA/Dec = R.T @ diag(σ_AT², σ_CT²) @ R
            sigma_ra_cosdec_sq[i] = (
                u_ra * u_ra * sigma_at2_used + u_dec * u_dec * sigma_ct2_used
            )
            sigma_dec_sq[i] = (
                u_dec * u_dec * sigma_at2_used + u_ra * u_ra * sigma_ct2_used
            )
            cov_ra_cosdec_dec_arcsec_sq[i] = u_ra * u_dec * (
                sigma_at2_used - sigma_ct2_used
            )

    sigma_ra_cosdec_arcsec = np.sqrt(np.where(sigma_ra_cosdec_sq > 0, sigma_ra_cosdec_sq, np.nan))
    sigma_dec_arcsec = np.sqrt(np.where(sigma_dec_sq > 0, sigma_dec_sq, np.nan))
    sigma_ra_cosdec_deg = sigma_ra_cosdec_arcsec / 3600.0
    sigma_dec_deg = sigma_dec_arcsec / 3600.0
    sigma_ra_deg = np.where(
        np.isfinite(cos_dec) & (cos_dec != 0.0),
        sigma_ra_cosdec_deg / cos_dec,
        np.nan,
    )
    # Effective correlation matching the updated (possibly inflated) variances.
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_eff = np.where(
            (sigma_ra_cosdec_arcsec > 0) & (sigma_dec_arcsec > 0),
            cov_ra_cosdec_dec_arcsec_sq
            / (sigma_ra_cosdec_arcsec * sigma_dec_arcsec),
            0.0,
        )
    # Numerical clip — small floating-point overshoot would otherwise make the
    # 2×2 indefinite.
    corr_eff = np.clip(corr_eff, -0.999999, 0.999999)
    corr = np.where(np.isfinite(corr_eff), corr_eff, 0.0)

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

    if bias_table is not None and bias_application in (
        "subtract", "subtract_sem_inflated"
    ):
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

    if catalog_debias_arcsec is not None:
        debias = np.asarray(catalog_debias_arcsec, dtype=np.float64)
        if debias.shape != (len(obs_set), 2):
            raise ValueError(
                f"catalog_debias_arcsec must have shape ({len(obs_set)}, 2); "
                f"got {debias.shape}"
            )
        cos_dec_for_debias = np.cos(np.deg2rad(lat))
        with np.errstate(divide="ignore", invalid="ignore"):
            ra_debias_deg = np.where(
                np.isfinite(cos_dec_for_debias) & (cos_dec_for_debias != 0.0),
                (debias[:, 0] / 3600.0) / cos_dec_for_debias,
                0.0,
            )
        lon = lon - ra_debias_deg
        lat = lat - debias[:, 1] / 3600.0

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
