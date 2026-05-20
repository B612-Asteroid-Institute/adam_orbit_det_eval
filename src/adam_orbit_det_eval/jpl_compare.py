"""Helpers for comparing a locally-fitted orbit against the JPL/SBDB nominal.

Provides:

* :func:`fetch_jpl_orbit` — thin wrapper over ``adam_core.orbits.query.sbdb``
  that returns a single-row ``Orbits`` table with covariance when available.
* :func:`propagate_to_epoch` — propagates an orbit to a target ``Timestamp``
  using the supplied propagator.
* :func:`compute_orbit_gap` — given two single-row ``Orbits`` at the same
  epoch, returns a dict with element- and state-level gaps and (when JPL
  covariance is available) the gap in units of the JPL 1-sigma.
* :func:`build_comparison_table` — concatenates per-variant comparison dicts
  into a quivr table that can be written to parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.orbits.query.sbdb import query_sbdb_new
from adam_core.propagator.propagator import Propagator
from adam_core.time import Timestamp


@dataclass
class OrbitGap:
    """Element- and state-level discrepancies between a fit and a reference orbit."""

    variant: str
    epoch_mjd_tdb: float
    delta_a_au: float
    delta_e: float
    delta_i_deg: float
    delta_raan_deg: float
    delta_ap_deg: float
    delta_M_deg: float
    delta_q_au: float
    cartesian_dr_au: float
    cartesian_dv_au_per_day: float
    # Units of JPL 1-sigma (NaN if JPL covariance unavailable).
    delta_a_in_sigma: float
    delta_e_in_sigma: float
    delta_i_in_sigma: float


class ComparisonTable(qv.Table):
    """Tabular form of :class:`OrbitGap` for parquet output."""

    variant = qv.LargeStringColumn()
    epoch_mjd_tdb = qv.Float64Column()
    delta_a_au = qv.Float64Column()
    delta_e = qv.Float64Column()
    delta_i_deg = qv.Float64Column()
    delta_raan_deg = qv.Float64Column()
    delta_ap_deg = qv.Float64Column()
    delta_M_deg = qv.Float64Column()
    delta_q_au = qv.Float64Column()
    cartesian_dr_au = qv.Float64Column()
    cartesian_dv_au_per_day = qv.Float64Column()
    delta_a_in_sigma = qv.Float64Column(nullable=True)
    delta_e_in_sigma = qv.Float64Column(nullable=True)
    delta_i_in_sigma = qv.Float64Column(nullable=True)


def fetch_jpl_orbit(designation: str) -> Orbits:
    """Fetch a single SBDB orbit for ``designation``.

    Returns a one-row ``Orbits`` table at JPL's published epoch. Covariance is
    populated when SBDB returns one (most numbered/well-observed objects).
    """
    return query_sbdb_new([designation], orbit_id_from_input=True)


def propagate_to_epoch(
    orbit: Orbits, target_epoch: Timestamp, propagator: Propagator
) -> Orbits:
    """Propagate a single-row ``Orbits`` to ``target_epoch``."""
    assert len(orbit) == 1, "propagate_to_epoch requires a single-row orbit"
    propagated = propagator.propagate_orbits(orbit, target_epoch, max_processes=1)
    return propagated


def _angle_diff_deg(a: float, b: float) -> float:
    """Return (a - b) wrapped into (-180, 180] degrees."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    # Adjust the boundary so we return +180 rather than -180 for an exact flip.
    return float(diff if diff != -180.0 else 180.0)


def _keplerian_row(orbit: Orbits) -> dict[str, float]:
    kep = orbit.coordinates.to_keplerian()
    com = orbit.coordinates.to_cometary()
    return {
        "a": float(kep.a[0].as_py()),
        "e": float(kep.e[0].as_py()),
        "i": float(kep.i[0].as_py()),
        "raan": float(kep.raan[0].as_py()),
        "ap": float(kep.ap[0].as_py()),
        "M": float(kep.M[0].as_py()),
        "q": float(com.q[0].as_py()),
    }


def _jpl_sigma_keplerian(jpl_orbit: Orbits) -> Optional[dict[str, float]]:
    """Return JPL 1-sigma in keplerian elements, or None if no covariance.

    Treats any diagonal element that is non-finite or non-positive as "no
    covariance for this element". Tiny-but-finite SBDB covariances (typical
    diagonal entries ~1e-16 AU^2 for well-determined NEOs) are kept.
    """
    cov_matrix = jpl_orbit.coordinates.covariance.to_matrix()
    if cov_matrix is None or cov_matrix.shape[0] == 0:
        return None
    cov = cov_matrix[0]
    cart_diag = np.diag(cov)
    if not np.any(np.isfinite(cart_diag)) or np.all(cart_diag <= 0):
        return None
    try:
        kep_cov = jpl_orbit.coordinates.to_keplerian().covariance.to_matrix()[0]
    except Exception:
        return None
    if kep_cov is None:
        return None
    diag = np.diag(kep_cov)
    # Element-wise: missing/non-positive → NaN sigma; otherwise sqrt of variance.
    sigmas = np.where((diag > 0) & np.isfinite(diag), np.sqrt(np.maximum(diag, 0.0)), np.nan)
    return {
        "a": float(sigmas[0]),
        "e": float(sigmas[1]),
        "i": float(sigmas[2]),
        "raan": float(sigmas[3]),
        "ap": float(sigmas[4]),
        "M": float(sigmas[5]),
    }


def compute_orbit_gap(
    fitted_orbit: Orbits,
    jpl_orbit: Orbits,
    variant: str,
) -> OrbitGap:
    """Compute element- and state-level gaps between ``fitted_orbit`` and ``jpl_orbit``.

    Both orbits must be at the same epoch (caller is responsible for propagating
    one to match the other beforehand) and must be single-row tables.
    """
    assert len(fitted_orbit) == 1, "fitted_orbit must be single-row"
    assert len(jpl_orbit) == 1, "jpl_orbit must be single-row"

    fitted_epoch = float(fitted_orbit.coordinates.time.mjd()[0].as_py())
    jpl_epoch = float(jpl_orbit.coordinates.time.mjd()[0].as_py())
    if not np.isclose(fitted_epoch, jpl_epoch, atol=1e-6):
        raise ValueError(
            f"Epochs do not match: fitted={fitted_epoch} jpl={jpl_epoch}"
        )

    fk = _keplerian_row(fitted_orbit)
    jk = _keplerian_row(jpl_orbit)

    # Cartesian gap
    fc = fitted_orbit.coordinates
    jc = jpl_orbit.coordinates
    dr_vec = np.array(
        [
            float(fc.x[0].as_py()) - float(jc.x[0].as_py()),
            float(fc.y[0].as_py()) - float(jc.y[0].as_py()),
            float(fc.z[0].as_py()) - float(jc.z[0].as_py()),
        ]
    )
    dv_vec = np.array(
        [
            float(fc.vx[0].as_py()) - float(jc.vx[0].as_py()),
            float(fc.vy[0].as_py()) - float(jc.vy[0].as_py()),
            float(fc.vz[0].as_py()) - float(jc.vz[0].as_py()),
        ]
    )
    dr_au = float(np.linalg.norm(dr_vec))
    dv_au_per_day = float(np.linalg.norm(dv_vec))

    sigmas = _jpl_sigma_keplerian(jpl_orbit)
    if sigmas is None:
        d_a_sigma = float("nan")
        d_e_sigma = float("nan")
        d_i_sigma = float("nan")
    else:
        def _ratio(diff: float, sigma: float) -> float:
            if not np.isfinite(sigma) or sigma <= 0:
                return float("nan")
            return float(diff / sigma)

        d_a_sigma = _ratio(fk["a"] - jk["a"], sigmas["a"])
        d_e_sigma = _ratio(fk["e"] - jk["e"], sigmas["e"])
        d_i_sigma = _ratio(_angle_diff_deg(fk["i"], jk["i"]), sigmas["i"])

    return OrbitGap(
        variant=variant,
        epoch_mjd_tdb=fitted_epoch,
        delta_a_au=fk["a"] - jk["a"],
        delta_e=fk["e"] - jk["e"],
        delta_i_deg=_angle_diff_deg(fk["i"], jk["i"]),
        delta_raan_deg=_angle_diff_deg(fk["raan"], jk["raan"]),
        delta_ap_deg=_angle_diff_deg(fk["ap"], jk["ap"]),
        delta_M_deg=_angle_diff_deg(fk["M"], jk["M"]),
        delta_q_au=fk["q"] - jk["q"],
        cartesian_dr_au=dr_au,
        cartesian_dv_au_per_day=dv_au_per_day,
        delta_a_in_sigma=d_a_sigma,
        delta_e_in_sigma=d_e_sigma,
        delta_i_in_sigma=d_i_sigma,
    )


def build_comparison_table(gaps: Iterable[OrbitGap]) -> ComparisonTable:
    """Build a :class:`ComparisonTable` from an iterable of :class:`OrbitGap`."""
    gaps_list = list(gaps)
    return ComparisonTable.from_kwargs(
        variant=[g.variant for g in gaps_list],
        epoch_mjd_tdb=[g.epoch_mjd_tdb for g in gaps_list],
        delta_a_au=[g.delta_a_au for g in gaps_list],
        delta_e=[g.delta_e for g in gaps_list],
        delta_i_deg=[g.delta_i_deg for g in gaps_list],
        delta_raan_deg=[g.delta_raan_deg for g in gaps_list],
        delta_ap_deg=[g.delta_ap_deg for g in gaps_list],
        delta_M_deg=[g.delta_M_deg for g in gaps_list],
        delta_q_au=[g.delta_q_au for g in gaps_list],
        cartesian_dr_au=[g.cartesian_dr_au for g in gaps_list],
        cartesian_dv_au_per_day=[g.cartesian_dv_au_per_day for g in gaps_list],
        delta_a_in_sigma=[g.delta_a_in_sigma for g in gaps_list],
        delta_e_in_sigma=[g.delta_e_in_sigma for g in gaps_list],
        delta_i_in_sigma=[g.delta_i_in_sigma for g in gaps_list],
    )
