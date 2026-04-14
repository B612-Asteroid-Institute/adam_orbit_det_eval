"""
Eligibility validation for LOOO holdout pairs.

Checks whether a specific (object, holdout_key) pair — where holdout_key
is an observatory code or program code — is suitable for LOOO evaluation.

Criteria:
  - Minimum observations remaining after holdout
  - Minimum arc length remaining after holdout
  - Maximum fraction of observations held out
  - Comet exclusion (C/, P/, D/, A/, I/ prefixes and numbered periodic comets)

Note: This module deliberately avoids importing from .core to prevent
circular imports. It accepts LOOOConfig via TYPE_CHECKING only and uses
duck-typing at runtime.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pyarrow as pa

from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations

if TYPE_CHECKING:
    from .core import LOOOConfig

logger = logging.getLogger(__name__)


@dataclass
class EligibilityResult:
    """Result of an eligibility check for a single holdout pair."""

    eligible: bool
    reason: str
    stats: dict = field(default_factory=dict)


@dataclass
class ExclusionStats:
    """Tracks how many pairs are excluded and why."""

    total_checked: int = 0
    total_eligible: int = 0
    excluded_comet: int = 0
    excluded_min_obs_held_out: int = 0
    excluded_min_obs_remaining: int = 0
    excluded_max_held_out_fraction: int = 0
    excluded_min_arc_length: int = 0

    def record(self, result: EligibilityResult) -> None:
        self.total_checked += 1
        if result.eligible:
            self.total_eligible += 1
        else:
            reason = result.reason
            if "comet" in reason:
                self.excluded_comet += 1
            elif "held-out obs" in reason:
                self.excluded_min_obs_held_out += 1
            elif "remaining obs" in reason:
                self.excluded_min_obs_remaining += 1
            elif "held-out fraction" in reason:
                self.excluded_max_held_out_fraction += 1
            elif "arc length" in reason:
                self.excluded_min_arc_length += 1

    @property
    def total_excluded(self) -> int:
        return self.total_checked - self.total_eligible

    def summary(self) -> dict:
        return {
            "total_checked": self.total_checked,
            "total_eligible": self.total_eligible,
            "total_excluded": self.total_excluded,
            "excluded_comet": self.excluded_comet,
            "excluded_min_obs_held_out": self.excluded_min_obs_held_out,
            "excluded_min_obs_remaining": self.excluded_min_obs_remaining,
            "excluded_max_held_out_fraction": self.excluded_max_held_out_fraction,
            "excluded_min_arc_length": self.excluded_min_arc_length,
        }


_COMET_PREFIX_RE = re.compile(r"^\d+[PDCIA]/")  # e.g., 1P/, 109P/, 2I/
_COMET_SIMPLE_RE = re.compile(r"^\d+[PDCI]$")  # e.g., 1P, 109P (no slash, just number+letter)


def is_comet(object_id: str) -> bool:
    """Check if object_id represents a comet or comet-like object.

    Detects:
      - Letter-prefix designations: C/, P/, D/, A/, I/
      - Numbered periodic comets: 1P, 1P/Halley, 109P/Swift-Tuttle
      - Numbered interstellar objects: 2I/Borisov
    """
    oid = object_id.strip()
    # Letter-prefix designations: C/2023 A1, P/2024 B2, D/..., A/..., I/...
    if oid[:2] in ("C/", "P/", "D/", "A/", "I/"):
        return True
    # Numbered periodic comets: 1P, 1P/Halley, 109P, 109P/Swift-Tuttle
    if _COMET_PREFIX_RE.match(oid) or _COMET_SIMPLE_RE.match(oid):
        return True
    return False


def _arc_length_days_from_mjds(mjds: np.ndarray) -> float:
    """Return the time span (days) from an array of MJD values."""
    if len(mjds) < 2:
        return 0.0
    return float(mjds.max() - mjds.min())


def check_pair_eligibility(
    observations: OrbitDeterminationObservations,
    holdout_mask: np.ndarray,
    config: "LOOOConfig",
    object_id: Optional[str] = None,
    holdout_key: Optional[str] = None,
) -> EligibilityResult:
    """
    Check whether a specific holdout mask passes eligibility for LOOO.

    Parameters
    ----------
    observations : OrbitDeterminationObservations
        Full observation set for the object.
    holdout_mask : np.ndarray
        Boolean mask — True for observations to hold out.
    config : LOOOConfig
        Eligibility thresholds.
    object_id : str, optional
        For logging only.
    holdout_key : str, optional
        For logging only (e.g. station code or program code).

    Returns
    -------
    EligibilityResult
    """
    n_obs_total = len(observations)
    n_held_out = int(holdout_mask.sum())
    n_remaining = n_obs_total - n_held_out
    held_out_fraction = n_held_out / n_obs_total if n_obs_total > 0 else 1.0

    label = f"{object_id} / {holdout_key}" if object_id and holdout_key else "pair"

    if n_held_out < config.min_obs_held_out:
        return EligibilityResult(
            eligible=False,
            reason=f"only {n_held_out} held-out obs",
            stats={"n_held_out": n_held_out, "n_remaining": n_remaining},
        )

    if n_remaining < config.min_obs_remaining:
        return EligibilityResult(
            eligible=False,
            reason=f"only {n_remaining} remaining obs",
            stats={"n_held_out": n_held_out, "n_remaining": n_remaining},
        )

    if held_out_fraction > config.max_held_out_fraction:
        return EligibilityResult(
            eligible=False,
            reason=f"held-out fraction {held_out_fraction:.2f} > {config.max_held_out_fraction}",
            stats={"n_held_out": n_held_out, "n_remaining": n_remaining,
                    "held_out_fraction": held_out_fraction},
        )

    # Check arc length of remaining observations
    hold_in_mask = ~holdout_mask
    all_mjds = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    remaining_mjds = all_mjds[hold_in_mask]
    arc_remaining = _arc_length_days_from_mjds(remaining_mjds)

    if arc_remaining < config.min_arc_length_days:
        return EligibilityResult(
            eligible=False,
            reason=f"arc length {arc_remaining:.1f}d < {config.min_arc_length_days}d",
            stats={"n_held_out": n_held_out, "n_remaining": n_remaining,
                    "arc_remaining": arc_remaining},
        )

    return EligibilityResult(
        eligible=True,
        reason="eligible",
        stats={
            "n_held_out": n_held_out,
            "n_remaining": n_remaining,
            "held_out_fraction": held_out_fraction,
            "arc_remaining": arc_remaining,
        },
    )


def check_object_eligibility(
    observations: OrbitDeterminationObservations,
    config: "LOOOConfig",
    min_observatories: int = 3,
    holdout_column_values: Optional[np.ndarray] = None,
) -> Dict[str, EligibilityResult]:
    """
    Check eligibility for each holdout key (observatory or program code) for a single object.

    Parameters
    ----------
    observations : OrbitDeterminationObservations
        Full observation set for the object.
    config : LOOOConfig
        Eligibility thresholds.
    min_observatories : int
        Minimum number of distinct holdout keys required.
    holdout_column_values : np.ndarray, optional
        Array of holdout key values parallel to observations (e.g. station codes
        or program codes). Defaults to observatory codes from
        observations.coordinates.origin.code.

    Returns
    -------
    dict[str, EligibilityResult]
        Per-holdout-key eligibility results. If the object has fewer than
        min_observatories distinct keys, all are marked ineligible.
    """
    if holdout_column_values is None:
        holdout_column_values = observations.coordinates.origin.code.to_numpy(
            zero_copy_only=False
        )

    unique_keys = np.unique(holdout_column_values)
    results: Dict[str, EligibilityResult] = {}

    if len(unique_keys) < min_observatories:
        for key in unique_keys:
            results[str(key)] = EligibilityResult(
                eligible=False,
                reason=f"only {len(unique_keys)} distinct keys < {min_observatories}",
                stats={"n_distinct_keys": len(unique_keys)},
            )
        return results

    for key in unique_keys:
        key_str = str(key)
        holdout_mask = holdout_column_values == key
        results[key_str] = check_pair_eligibility(
            observations, holdout_mask, config, holdout_key=key_str,
        )

    return results
