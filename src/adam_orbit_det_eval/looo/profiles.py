"""
Named run profiles for the v2 LOOO pipeline (bead tcu).

A profile is a named combination of the two v2 configuration axes
(docs/v2-scope.md, "LOOO group keys + time filter — configurable
architecture"):

  - **group_by** — the LOOO hold-out unit / bias_table aggregation key
    (bead wl0).
  - **obstime window** — a pre-LOOO observation-time filter. Window
    semantics follow the v2 design's pre/post-2017 partition: an
    observation is kept when ``obstime > obstime_min`` AND
    ``obstime <= obstime_max`` (min exclusive, max inclusive), so
    ``v2_full_pre_2017`` (max = 2017-01-01) and ``v2_full_post_2017``
    (min = 2017-01-01) partition the archive with no overlap and no gap.

Each profile also carries an output-directory suffix; production catalogs
land at ``data/mpc_scale_results_<date>_<profile>/`` per v2-scope.md.

CLI integration (02_run_looo.py, 12_run_looo_cloud_shard.py):
``--profile <name>`` selects a profile; explicit ``--group-by`` /
``--obstime-min`` / ``--obstime-max`` flags override the profile's values.
Use :func:`resolve_run_options` so all entrypoints share the same
precedence: explicit flag > profile value > built-in default.

Future time cuts (per-year, per-quarter, pre/post known instrument events)
are added as new ``RunProfile`` entries — no pipeline change. Curated
per-program / per-period station-list profiles and a v1-replication profile
are deliberately NOT defined yet (out of scope for bead tcu).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .core import DEFAULT_GROUP_BY, _validate_group_by


@dataclass(frozen=True)
class RunProfile:
    """A named combination of group keys + obstime window + output suffix."""

    name: str
    #: LOOO hold-out / aggregation keys (subset of stn, prog, band, astcat).
    group_by: Tuple[str, ...]
    #: ISO-8601 lower bound (exclusive) on obstime, or None for unbounded.
    obstime_min: Optional[str]
    #: ISO-8601 upper bound (inclusive) on obstime, or None for unbounded.
    obstime_max: Optional[str]
    #: Appended to the run/output directory name, e.g.
    #: ``data/mpc_scale_results_<date>_v2_full_pre_2017/``.
    output_dir_suffix: str


#: The named v2 profiles (docs/v2-scope.md "Default profile + named variants").
PROFILES = {
    p.name: p
    for p in (
        RunProfile(
            name="v2_full",
            group_by=("stn", "prog", "band"),
            obstime_min=None,
            obstime_max=None,
            output_dir_suffix="_v2_full",
        ),
        RunProfile(
            name="v2_full_pre_2017",
            group_by=("stn", "prog", "band"),
            obstime_min=None,
            obstime_max="2017-01-01",
            output_dir_suffix="_v2_full_pre_2017",
        ),
        RunProfile(
            name="v2_full_post_2017",
            group_by=("stn", "prog", "band"),
            obstime_min="2017-01-01",
            obstime_max=None,
            output_dir_suffix="_v2_full_post_2017",
        ),
    )
}


def get_profile(name: str) -> RunProfile:
    """Return the named profile, or raise ValueError listing the valid names."""
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown run profile {name!r}; available profiles: "
            f"{sorted(PROFILES)}"
        ) from None


@dataclass(frozen=True)
class ResolvedRunOptions:
    """Effective pipeline options after profile/flag precedence is applied."""

    group_by: Tuple[str, ...]
    obstime_min: Optional[str]
    obstime_max: Optional[str]
    #: Name of the selected profile, or None for an ad-hoc (flags-only) run.
    profile: Optional[str]
    #: Output-dir suffix ("" for ad-hoc runs).
    output_dir_suffix: str


def resolve_run_options(
    profile: Optional[str] = None,
    group_by: Optional[List[str]] = None,
    obstime_min: Optional[str] = None,
    obstime_max: Optional[str] = None,
) -> ResolvedRunOptions:
    """
    Resolve a profile name + explicit per-option overrides into effective values.

    Precedence per option: explicit argument > profile value > built-in
    default (``DEFAULT_GROUP_BY``; no obstime bounds).

    Parameters
    ----------
    profile : str, optional
        Name of a profile in :data:`PROFILES`.
    group_by : list of str, optional
        Explicit group keys; overrides the profile's ``group_by``.
    obstime_min, obstime_max : str, optional
        Explicit ISO-8601 window bounds; each independently overrides the
        profile's value.
    """
    prof = get_profile(profile) if profile is not None else None

    eff_group_by = (
        list(group_by)
        if group_by is not None
        else list(prof.group_by) if prof is not None else list(DEFAULT_GROUP_BY)
    )
    eff_group_by = _validate_group_by(eff_group_by)

    eff_min = obstime_min if obstime_min is not None else (
        prof.obstime_min if prof is not None else None
    )
    eff_max = obstime_max if obstime_max is not None else (
        prof.obstime_max if prof is not None else None
    )

    return ResolvedRunOptions(
        group_by=tuple(eff_group_by),
        obstime_min=eff_min,
        obstime_max=eff_max,
        profile=prof.name if prof is not None else None,
        output_dir_suffix=prof.output_dir_suffix if prof is not None else "",
    )
