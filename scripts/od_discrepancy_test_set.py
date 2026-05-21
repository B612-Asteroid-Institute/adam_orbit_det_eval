"""
od_discrepancy_test_set.py
==========================

Frozen test set of 14 NEOs whose nominal adam_fo orbits (no bias correction)
diverge from JPL/SBDB by more than 3-σ at the JPL epoch. Produced by bead
`od_experiments_setup-9d7` from a 66-object stratified population; full inputs
and gap metrics in:

    data/od_discrepancy_population/discrepancy_ranking.parquet

These objects are the recommended input set for the bias-variant follow-on:
re-fit each with v1 per-station bias corrections applied and measure whether
Δr/σ improves. Ranked by Δr/σ_r descending.

Selection criteria
------------------
1. JPL/SBDB nominal orbit + covariance available
2. adam_fo initial fit converged (FindOrb, no bias correction)
3. Cartesian Δr / JPL position-σ_r > 3 at the JPL epoch (our orbit
   propagated to that epoch via ASSIST)

Usage
-----
    from scripts.od_discrepancy_test_set import TEST_SET, provids, as_dataframe

    for obj in TEST_SET:
        print(obj.provid, obj.delta_r_over_sigma)

Source provenance
-----------------
    Commit:  083b636  (kk/od-bias-experiments)
    Parquet: data/od_discrepancy_population/discrepancy_ranking.parquet
    Report:  data/od_discrepancy_population/REPORT.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiscrepantObject:
    provid: str               # MPC unpacked primary provisional designation
    sbdb_name: str            # Full SBDB fullname (number + name + provid)
    orbit_class: str          # SBDB orbit_class code (AMO/APO/ATE/ATI)
    arc_days: float           # Observed arc length (post space-based filter)
    n_obs: int                # Observation count used in fit (post-cap)
    n_stations: int           # Distinct ground-based stations
    delta_r_au: float         # |Δr| at JPL epoch (Cartesian, AU)
    jpl_pos_sigma_au: float   # JPL 1-σ Cartesian position (AU)
    delta_r_over_sigma: float # Δr / σ_r
    stratum: str              # Selection stratum from population builder


# Ordered by Δr/σ descending. n_obs reflects the 2000-obs cap applied by the
# population builder to bound fit time; raw nobs can be larger.
TEST_SET: tuple[DiscrepantObject, ...] = (
    DiscrepantObject(
        provid="A898 PA", sbdb_name="433 Eros (A898 PA)", orbit_class="AMO",
        arc_days=48411.6, n_obs=2000, n_stations=370,
        delta_r_au=8.649e-05, jpl_pos_sigma_au=3.175e-08,
        delta_r_over_sigma=2724.5, stratum="priority_well_known",
    ),
    DiscrepantObject(
        provid="1929 SH", sbdb_name="1627 Ivar (1929 SH)", orbit_class="AMO",
        arc_days=34961.3, n_obs=2000, n_stations=303,
        delta_r_au=3.863e-06, jpl_pos_sigma_au=3.396e-09,
        delta_r_over_sigma=1137.3, stratum="priority_well_known",
    ),
    DiscrepantObject(
        provid="1999 RQ36", sbdb_name="101955 Bennu (1999 RQ36)", orbit_class="APO",
        arc_days=9361.4, n_obs=603, n_stations=47,
        delta_r_au=2.038e-07, jpl_pos_sigma_au=2.537e-10,
        delta_r_over_sigma=803.3, stratum="impact_monitor",
    ),
    DiscrepantObject(
        provid="2004 MN4", sbdb_name="99942 Apophis (2004 MN4)", orbit_class="ATE",
        arc_days=6275.7, n_obs=2000, n_stations=237,
        delta_r_au=1.178e-06, jpl_pos_sigma_au=4.900e-09,
        delta_r_over_sigma=240.5, stratum="impact_monitor",
    ),
    DiscrepantObject(
        provid="2001 MZ7", sbdb_name="54789 (2001 MZ7)", orbit_class="AMO",
        arc_days=17221.2, n_obs=2000, n_stations=160,
        delta_r_au=4.498e-07, jpl_pos_sigma_au=3.493e-09,
        delta_r_over_sigma=128.8, stratum="priority_well_known",
    ),
    DiscrepantObject(
        provid="1985 DO2", sbdb_name="4055 Magellan (1985 DO2)", orbit_class="AMO",
        arc_days=15052.4, n_obs=2000, n_stations=169,
        delta_r_au=3.624e-07, jpl_pos_sigma_au=4.026e-09,
        delta_r_over_sigma=90.0, stratum="priority_well_known",
    ),
    DiscrepantObject(
        provid="2001 AU43", sbdb_name="138925 (2001 AU43)", orbit_class="AMO",
        arc_days=11944.7, n_obs=862, n_stations=62,
        delta_r_au=6.299e-07, jpl_pos_sigma_au=1.176e-08,
        delta_r_over_sigma=53.6, stratum="mod_arc_mod_obs",
    ),
    DiscrepantObject(
        provid="1998 OH", sbdb_name="12538 (1998 OH)", orbit_class="APO",
        arc_days=13980.7, n_obs=2000, n_stations=118,
        delta_r_au=2.191e-07, jpl_pos_sigma_au=4.141e-09,
        delta_r_over_sigma=52.9, stratum="priority_well_known",
    ),
    DiscrepantObject(
        provid="1985 PA", sbdb_name="3752 Camillo (1985 PA)", orbit_class="APO",
        arc_days=18144.6, n_obs=2000, n_stations=149,
        delta_r_au=1.671e-07, jpl_pos_sigma_au=5.247e-09,
        delta_r_over_sigma=31.8, stratum="long_arc_well_obs",
    ),
    DiscrepantObject(
        provid="2024 YR4", sbdb_name="(2024 YR4)", orbit_class="APO",
        arc_days=87.8, n_obs=492, n_stations=62,
        delta_r_au=1.162e-06, jpl_pos_sigma_au=4.761e-08,
        delta_r_over_sigma=24.4, stratum="impact_monitor",
    ),
    DiscrepantObject(
        provid="2001 SW169", sbdb_name="163000 (2001 SW169)", orbit_class="AMO",
        arc_days=10369.4, n_obs=1757, n_stations=86,
        delta_r_au=9.504e-08, jpl_pos_sigma_au=5.287e-09,
        delta_r_over_sigma=18.0, stratum="long_arc_well_obs",
    ),
    DiscrepantObject(
        provid="1994 LY", sbdb_name="85275 (1994 LY)", orbit_class="AMO",
        arc_days=11571.7, n_obs=2000, n_stations=115,
        delta_r_au=1.150e-07, jpl_pos_sigma_au=1.427e-08,
        delta_r_over_sigma=8.1, stratum="priority_well_known",
    ),
    DiscrepantObject(
        provid="1999 FQ5", sbdb_name="40263 (1999 FQ5)", orbit_class="AMO",
        arc_days=13786.2, n_obs=870, n_stations=64,
        delta_r_au=2.023e-07, jpl_pos_sigma_au=2.661e-08,
        delta_r_over_sigma=7.6, stratum="mod_arc_mod_obs",
    ),
    DiscrepantObject(
        provid="2002 NV16", sbdb_name="363305 (2002 NV16)", orbit_class="APO",
        arc_days=8226.1, n_obs=653, n_stations=51,
        delta_r_au=1.497e-07, jpl_pos_sigma_au=4.037e-08,
        delta_r_over_sigma=3.7, stratum="mod_arc_mod_obs",
    ),
)


# Convenience accessors -------------------------------------------------------


def provids() -> list[str]:
    """Just the MPC provids, in rank order. Useful for batch mpcq fetches."""
    return [obj.provid for obj in TEST_SET]


def as_dataframe():
    """Return the test set as a pandas DataFrame. Imports pandas lazily."""
    import pandas as pd
    return pd.DataFrame([asdict(obj) for obj in TEST_SET])


if __name__ == "__main__":
    # Quick smoke when run directly: print the test set.
    print(f"{len(TEST_SET)} discrepant NEOs in test set (ranked by Δr/σ):")
    for i, obj in enumerate(TEST_SET, 1):
        print(
            f"  {i:2d}. {obj.provid:<11s} {obj.orbit_class:>3s} "
            f"arc={obj.arc_days:>7.0f} d  n_obs={obj.n_obs:>4d}  "
            f"Δr/σ={obj.delta_r_over_sigma:>8.1f}  ({obj.sbdb_name})"
        )
