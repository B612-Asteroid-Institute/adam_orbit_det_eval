from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest
from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter
from adam_orbfit.orbfit_orbit_fitter import OrbfitOrbitFitter
from mpcq import MPCObservations
from mpcq.orbits import MPCOrbits

from adam_orbit_det_eval.utils import mpc_to_od_observations

# Load all relevant data. We'll split it into benchmark parameters
data_dir = Path("data/orbit_fit_eval")
mpc_observations = MPCObservations.from_parquet(data_dir / "mpc_observations.parquet")
mpc_orbits = MPCOrbits.from_parquet(data_dir / "mpc_orbits.parquet")
objects = mpc_observations.requested_provid.unique().to_pylist()
# All fitters to benchmark
find_orb_fitter = FindOrbOrbitFitter(fo_result_dir="fo_dir_benchmark")
orbfit_orb_fitter = OrbfitOrbitFitter(work_dir="orbfit_dir_benchmark", timeout=360)


@pytest.mark.benchmark(group="initial_orbit_fit")
@pytest.mark.parametrize(
    "fitter",
    [find_orb_fitter, orbfit_orb_fitter],
    ids=lambda val: f"{type(val).__name__}",
)
@pytest.mark.parametrize("object_id", objects)
def test_initial_orbit_fit_benchmark(benchmark, quality_tracker, object_id, fitter):
    """Check the orbit is fitted and reasonably close to the MPC version"""
    subset = mpc_observations.apply_mask(
        pc.equal(mpc_observations.requested_provid, object_id)
    )
    od_observations = mpc_to_od_observations(subset, prevent_nans=False)
    fitted_orbits, fitted_members = benchmark(
        fitter.initial_fit, object_id, od_observations
    )
    assert len(fitted_orbits) > 0
    assert len(fitted_members) == len(od_observations)
    used_members = fitted_members.solution.sum().as_py()

    mpc_orbit = mpc_orbits.apply_mask(
        pc.equal(mpc_orbits.requested_provid, object_id)
    ).orbits()
    mpc_comm = mpc_orbit.coordinates.to_cometary()
    fitted_comm = fitted_orbits.coordinates.to_cometary()

    # Compare the fitted orbit with MPC. No asserts here, because sometimes we are waaay off
    quality_tracker.check("#obs", used_members, len(fitted_members))
    q_d = quality_tracker.check("q", fitted_comm.q[0].as_py(), mpc_comm.q[0].as_py())[
        "percent"
    ]
    e_d = quality_tracker.check("e", fitted_comm.e[0].as_py(), mpc_comm.e[0].as_py())[
        "delta"
    ]
    i_d = quality_tracker.check("i", fitted_comm.i[0].as_py(), mpc_comm.i[0].as_py())[
        "delta"
    ]
    raan_d = quality_tracker.check(
        "raan", fitted_comm.raan[0].as_py(), mpc_comm.raan[0].as_py()
    )["delta"]
    ap_d = quality_tracker.check(
        "arg peri", fitted_comm.ap[0].as_py(), mpc_comm.ap[0].as_py()
    )["delta"]
    per_d = quality_tracker.check("period", fitted_comm.P[0], mpc_comm.P[0])["percent"]
    # Skipping time to peri, because epochs may be different. In theory, can adjust by adding periods

    # If the fitter used most of the observations, expect the fitted orbit params to be close to MPC values
    if used_members > len(fitted_members) * 0.6:
        assert q_d < 5.0  # percent
        assert e_d < 0.05  # abs
        assert i_d < 2.0  # degrees
        assert raan_d < 5.0  # degrees
        assert ap_d < 20.0  # degrees
        # don't check hyperbolic orbits
        if np.isfinite(mpc_comm.P[0]) and mpc_comm.P[0] < 100000:
            assert per_d < 10.0  # percent
