from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from mpcq import MPCObservations

from ..utils import get_spacebased_stns, mpc_to_od_observations


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
