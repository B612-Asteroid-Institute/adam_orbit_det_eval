"""
Leave-One-Observatory-Out (LOOO) cross-validation for orbit determination.

For each (object, observatory) pair, this module:
  1. Removes the observatory's observations from the full set
  2. Refits the orbit via differential correction starting from a reference orbit
  3. Predicts sky-plane positions at the held-out observation times
  4. Returns per-observation residuals and summary metadata

The primary diagnostic is the chi2 calibration:
  chi2_per_obs = (residual / sigma)^2

If sigma estimates are correctly calibrated, the median chi2_per_obs should be ~1.
Systematic deviations indicate over- or under-estimated uncertainties for that
observatory / catalog combination.

Bias controls
-------------
An (object, observatory) pair is only evaluated when ALL of the following hold:
  - The observatory contributed >= min_obs_held_out observations
  - The remaining observations number >= min_obs_remaining
  - The remaining arc length >= min_arc_length_days
  - The held-out fraction <= max_held_out_fraction

These thresholds ensure the hold-in orbit is still well-determined, so prediction
errors reflect observatory quality, not ill-conditioning of the refitted orbit.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from adam_core.coordinates.residuals import Residuals
from adam_core.orbit_determination.differential_correction import fit_least_squares
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    evaluate_orbits,
)
from adam_core.orbit_determination.fitted_orbits import FittedOrbits
from adam_core.orbit_determination.orbit_fitter import OrbitFitter
from adam_core.orbits.orbits import Orbits
from adam_core.propagator.propagator import Propagator

from .eligibility import (
    ExclusionStats,
    check_pair_eligibility,
    is_comet,
)

logger = logging.getLogger(__name__)


@dataclass
class LOOOConfig:
    """Configuration for LOOO eligibility filters."""

    #: Minimum number of observations for an observatory to be tested
    min_obs_held_out: int = 1
    #: Minimum number of remaining observations after hold-out
    min_obs_remaining: int = 10
    #: Minimum arc length (days) that must remain after hold-out
    min_arc_length_days: float = 7.0
    #: Maximum fraction of total observations that may be held out
    #: (avoids underdetermined hold-in fits)
    max_held_out_fraction: float = 0.5
    #: scipy.optimize.least_squares kwargs for the DC step
    ls_kwargs: dict = None

    def __post_init__(self):
        if self.ls_kwargs is None:
            self.ls_kwargs = {}


_SCIPY_GATE_ENV_VAR = "LOOO_ALLOW_SCIPY"
_SCIPY_GATE_MESSAGE = (
    "scipy fit_least_squares is gated pending bead tvg root-cause; "
    "pass orbit_fitter=FindOrbOrbitFitter() explicitly "
    f"(or set {_SCIPY_GATE_ENV_VAR}=1 to bypass for local debugging)."
)


def _enforce_scipy_gate() -> None:
    """Block accidental use of the scipy fit_least_squares hold-in path.

    bead tvg documents a 10-100x chi2 regression between 2026-03-16 and 2026-04
    when ``fit_least_squares`` is run with ASSIST + an MPC warm-start orbit on
    the same inputs that previously fit cleanly (witnesses 2020 ML22, 2020
    TP96). Until that regression is root-caused, the LOOO pipeline must run
    via an explicit ``OrbitFitter`` (FindOrb in cloud). Local callers that
    still want the scipy path can set ``LOOO_ALLOW_SCIPY=1`` to bypass.
    """
    if os.environ.get(_SCIPY_GATE_ENV_VAR) == "1":
        return
    raise ValueError(_SCIPY_GATE_MESSAGE)


class LOOOResult(qv.Table):
    """
    Per-held-out-observation results from a LOOO evaluation.

    One row per (object, held-out observation). Columns are designed so that
    any stratification (by observatory, orbit class, arc length, etc.) can be
    done in downstream analysis without re-running the pipeline.
    """

    # --- Identifiers ---
    object_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()
    stn = qv.LargeStringColumn()

    # --- Residuals (arcseconds, observed − predicted) ---
    # sign convention: positive = object is north/east of prediction
    residual_ra_arcsec = qv.Float64Column(nullable=True)
    residual_dec_arcsec = qv.Float64Column(nullable=True)

    # --- Per-observation sigma from MPC (arcseconds) ---
    # sigma_ra here is sigma(RA * cos(dec)) to match ADES convention
    sigma_ra_cosdec_arcsec = qv.Float64Column(nullable=True)
    sigma_dec_arcsec = qv.Float64Column(nullable=True)

    # --- Chi2 statistics ---
    # chi2 = (residual_ra/sigma_ra)^2 + (residual_dec/sigma_dec)^2
    # Should be ~1 on average if sigmas are correctly calibrated.
    chi2 = qv.Float64Column(nullable=True)

    # --- Astrometric catalog used for this observation (from ADES astcat field) ---
    astcat = qv.LargeStringColumn(nullable=True)

    # --- MPC program code for this observation (from `prog` field) ---
    program_code = qv.LargeStringColumn(nullable=True)

    # --- Hold-out set context (for bias control / stratification) ---
    #: Number of observations held out from this object for this observatory
    n_obs_held_out = qv.Int64Column()
    #: Number of observations remaining in the hold-in fit
    n_obs_remaining = qv.Int64Column()
    #: Arc length in days of the hold-in observation set
    arc_length_remaining_days = qv.Float64Column()
    #: Fraction of total observations that were held out
    held_out_fraction = qv.Float64Column()
    #: Total number of observations for this object (hold-in + hold-out)
    n_obs_total = qv.Int64Column()

    # --- Orbit change due to hold-out (for sensitivity analysis) ---
    # Change in perihelion distance (AU) between full-data orbit and hold-out orbit
    delta_q_au = qv.Float64Column(nullable=True)
    # Change in eccentricity
    delta_e = qv.Float64Column(nullable=True)
    # Change in inclination (degrees)
    delta_i_deg = qv.Float64Column(nullable=True)

    # --- Fit quality of the hold-in orbit ---
    hold_in_reduced_chi2 = qv.Float64Column(nullable=True)
    hold_in_fit_success = qv.BooleanColumn(nullable=True)


def _arc_length_days(observations: OrbitDeterminationObservations) -> float:
    """Return the time span (days) of the observation set."""
    mjds = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    if len(mjds) < 2:
        return 0.0
    return float(mjds.max() - mjds.min())


def _orbit_delta(
    orbit_full: Orbits, orbit_holdout: FittedOrbits
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (delta_q_au, delta_e, delta_i_deg) between full-data and hold-out orbits.
    Returns (None, None, None) if conversion fails.
    """
    try:
        com_full = orbit_full.coordinates.to_cometary()
        com_hold = orbit_holdout.coordinates.to_cometary()
        dq = float(com_hold.q[0].as_py()) - float(com_full.q[0].as_py())
        de = float(com_hold.e[0].as_py()) - float(com_full.e[0].as_py())
        di = float(com_hold.i[0].as_py()) - float(com_full.i[0].as_py())
        return dq, de, di
    except Exception:
        return None, None, None


def run_looo_for_object(
    object_id: str,
    observations: OrbitDeterminationObservations,
    reference_orbit: Orbits,
    propagator: Propagator,
    config: Optional[LOOOConfig] = None,
    astcats: Optional[List[Optional[str]]] = None,
    program_codes: Optional[List[Optional[str]]] = None,
    holdout_column: Optional[np.ndarray] = None,
    exclusion_stats: Optional[ExclusionStats] = None,
    orbit_fitter: Optional[OrbitFitter] = None,
) -> LOOOResult:
    """
    Run leave-one-observatory-out cross-validation for a single object.

    For each holdout key (observatory or program code) that contributed
    observations, holds out those observations, refits the orbit, predicts
    at the held-out times, and returns the residuals.

    Parameters
    ----------
    object_id : str
        Object identifier (used only for labeling in output).
    observations : OrbitDeterminationObservations
        Full observation set for the object.
    reference_orbit : Orbits
        Starting orbit for differential correction (e.g., the MPC nominal orbit).
        Should be a single orbit (len==1).
    propagator : Propagator
        Propagator instance used for ephemeris generation and DC.
    config : LOOOConfig, optional
        Eligibility filter thresholds. Defaults to LOOOConfig().
    astcats : list of str or None, optional
        Astrometric catalog codes parallel to observations.id. If provided,
        these are stored in the output for per-catalog analysis.
    program_codes : list of str or None, optional
        MPC program codes (from the `prog` field) parallel to observations.id.
        If provided, stored per-row in the output for per-program-code analysis.
    holdout_column : np.ndarray, optional
        Array of holdout key values parallel to observations (e.g. program codes).
        Defaults to observatory codes (observations.coordinates.origin.code).
    exclusion_stats : ExclusionStats, optional
        If provided, records exclusion statistics for each pair checked.
    orbit_fitter : OrbitFitter, optional
        If provided, use this fitter's `initial_fit(object_id, observations)`
        for the hold-in fit instead of the built-in scipy-based
        `fit_least_squares`. Must implement the `OrbitFitter` ABC and be
        picklable for ProcessPoolExecutor.

    Returns
    -------
    LOOOResult
        One row per held-out observation that passed eligibility filters.
        Empty table if no observatory passes the filters.
    """
    if config is None:
        config = LOOOConfig()

    # Fail-fast before any per-pair work if scipy DC would be reached without
    # an explicit opt-in. See _enforce_scipy_gate / bead tvg.
    if orbit_fitter is None:
        _enforce_scipy_gate()

    # Comet exclusion
    if is_comet(object_id):
        logger.debug(f"{object_id}: comet excluded")
        return LOOOResult.empty()

    n_obs_total = len(observations)
    all_stns = observations.coordinates.origin.code.to_numpy(zero_copy_only=False)
    all_obs_ids = observations.id.to_numpy(zero_copy_only=False)
    all_mjds = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    # Determine holdout keys — default to observatory codes
    if holdout_column is not None:
        holdout_keys = holdout_column
    else:
        holdout_keys = all_stns
    unique_keys = np.unique(holdout_keys)

    results: List[LOOOResult] = []

    for key in unique_keys:
        held_out_mask = holdout_keys == key
        hold_in_mask = ~held_out_mask

        # --- Eligibility checks via eligibility module ---
        eligibility = check_pair_eligibility(
            observations, held_out_mask, config,
            object_id=object_id, holdout_key=str(key),
        )
        if exclusion_stats is not None:
            exclusion_stats.record(eligibility)
        if not eligibility.eligible:
            logger.debug(f"{object_id} / {key}: {eligibility.reason}, skipping")
            continue

        n_held_out = eligibility.stats["n_held_out"]
        n_remaining = eligibility.stats["n_remaining"]
        held_out_fraction = eligibility.stats["held_out_fraction"]
        arc_remaining = eligibility.stats["arc_remaining"]

        hold_in_obs = observations.apply_mask(pa.array(hold_in_mask))

        # Resolve the station code for the result — when holding out by
        # observatory the key IS the station; when holding out by program
        # code we still record the actual station per observation below.
        stn_for_key = str(key) if holdout_column is None else None

        # --- Hold-in fit: pluggable orbit fitter or scipy DC ---
        try:
            if orbit_fitter is not None:
                hold_in_orbit, hold_in_members = orbit_fitter.initial_fit(
                    object_id,
                    hold_in_obs,
                    reference_orbit=reference_orbit,
                )
            else:
                hold_in_orbit, hold_in_members = fit_least_squares(
                    reference_orbit,
                    hold_in_obs,
                    propagator,
                    **config.ls_kwargs,
                )
        except Exception as e:
            logger.warning(f"{object_id} / {key}: hold-in fit failed: {e}")
            continue

        if len(hold_in_orbit) == 0:
            logger.debug(f"{object_id} / {key}: DC returned no orbit, skipping")
            continue

        # --- Predict at held-out observation times ---
        held_out_obs = observations.apply_mask(pa.array(held_out_mask))
        try:
            _, held_out_members = evaluate_orbits(
                hold_in_orbit.to_orbits(),
                held_out_obs,
                propagator,
                parameters=6,
            )
        except Exception as e:
            logger.warning(f"{object_id} / {key}: evaluate_orbits failed: {e}")
            continue

        # --- Collect residuals ---
        residual_array = held_out_members.residuals.to_array()  # (N, 6)
        res_ra_arcsec = residual_array[:, 1] * 3600.0
        res_dec_arcsec = residual_array[:, 2] * 3600.0

        # --- Recover per-obs sigma from the held-out observations ---
        cov_matrix = held_out_obs.coordinates.covariance.to_matrix()  # (N, 6, 6)
        sigma_ra_deg = np.sqrt(np.abs(cov_matrix[:, 1, 1]))
        sigma_dec_deg = np.sqrt(np.abs(cov_matrix[:, 2, 2]))
        dec_deg = held_out_obs.coordinates.lat.to_numpy(zero_copy_only=False)
        cos_dec = np.cos(np.deg2rad(dec_deg))
        sigma_ra_cosdec_arcsec = sigma_ra_deg * cos_dec * 3600.0
        sigma_dec_arcsec = sigma_dec_deg * 3600.0

        chi2_vals = held_out_members.residuals.chi2.to_numpy(zero_copy_only=False)

        # --- Orbit delta ---
        dq, de, di = _orbit_delta(reference_orbit, hold_in_orbit)

        # --- astcat values for held-out obs (if provided) ---
        if astcats is not None:
            held_out_astcats = [astcats[i] for i, m in enumerate(held_out_mask) if m]
        else:
            held_out_astcats = [None] * n_held_out

        # --- program_code values for held-out obs (if provided) ---
        if program_codes is not None:
            held_out_programs = [program_codes[i] for i, m in enumerate(held_out_mask) if m]
        else:
            held_out_programs = [None] * n_held_out

        # --- Station codes: per-obs actual station when holding out by non-stn key ---
        if stn_for_key is not None:
            stn_values = np.full(n_held_out, stn_for_key, dtype=object)
        else:
            stn_values = all_stns[held_out_mask]

        # --- Build result rows ---
        held_out_obs_ids = all_obs_ids[held_out_mask]
        result = LOOOResult.from_kwargs(
            object_id=np.full(n_held_out, object_id, dtype=object),
            obs_id=held_out_obs_ids,
            stn=stn_values,
            residual_ra_arcsec=res_ra_arcsec,
            residual_dec_arcsec=res_dec_arcsec,
            sigma_ra_cosdec_arcsec=np.where(np.isfinite(sigma_ra_cosdec_arcsec), sigma_ra_cosdec_arcsec, None),
            sigma_dec_arcsec=np.where(np.isfinite(sigma_dec_arcsec), sigma_dec_arcsec, None),
            chi2=np.where(np.isfinite(chi2_vals), chi2_vals, None),
            astcat=pa.array(held_out_astcats, type=pa.large_utf8()),
            program_code=pa.array(held_out_programs, type=pa.large_utf8()),
            n_obs_held_out=np.full(n_held_out, n_held_out, dtype=np.int64),
            n_obs_remaining=np.full(n_held_out, n_remaining, dtype=np.int64),
            arc_length_remaining_days=np.full(n_held_out, arc_remaining),
            held_out_fraction=np.full(n_held_out, held_out_fraction),
            n_obs_total=np.full(n_held_out, n_obs_total, dtype=np.int64),
            delta_q_au=np.full(n_held_out, dq if dq is not None else np.nan),
            delta_e=np.full(n_held_out, de if de is not None else np.nan),
            delta_i_deg=np.full(n_held_out, di if di is not None else np.nan),
            hold_in_reduced_chi2=np.full(
                n_held_out,
                hold_in_orbit.reduced_chi2[0].as_py()
                if hold_in_orbit.reduced_chi2[0].is_valid
                else np.nan,
            ),
            hold_in_fit_success=np.full(
                n_held_out,
                hold_in_orbit.success[0].as_py() if hold_in_orbit.success[0].is_valid else None,
            ),
        )
        results.append(result)
        logger.info(
            f"{object_id} / {key}: {n_held_out} held-out obs, "
            f"chi2/obs={float(np.nanmean(chi2_vals)):.2f}"
        )

    if not results:
        return LOOOResult.empty()

    combined = qv.concatenate(results)
    if combined.fragmented():
        combined = qv.defragment(combined)
    return combined
