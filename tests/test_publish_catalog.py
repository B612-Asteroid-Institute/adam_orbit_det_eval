"""
Tests for scripts/19_publish_catalog.py — the v2 "produce final catalog"
wrapper that chains AT/CT decomposition (script 16) and bias-table generation
(script 17) atomically.

The central guarantee under test is the bead d5b anti-repeat property: the
wrapper must (a) populate AT/CT in the final catalog and (b) hard-abort
*before* writing any bias table if AT/CT came out empty — so a catalog with
null AT/CT can never ship again.

The wrapper shells out to scripts 16/17, which import ``adam_orbit_det_eval``.
We prepend this checkout's ``src`` to ``PYTHONPATH`` (autouse fixture) so the
subprocesses exercise *this* tree rather than whatever happens to be in
site-packages.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
SMOKE_OBS = FIXTURES / "smoke_obs.parquet"
SMOKE_ORBITS = FIXTURES / "smoke_orbits.parquet"
SCRIPT_19 = REPO_ROOT / "scripts" / "19_publish_catalog.py"


def _load_wrapper():
    """Import scripts/19_publish_catalog.py (numeric filename → importlib)."""
    spec = importlib.util.spec_from_file_location("publish_catalog_19", SCRIPT_19)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper()


@pytest.fixture(autouse=True)
def _pin_src_for_subprocesses():
    """Make scripts 16/17 import the package from this checkout's src."""
    src = str(REPO_ROOT / "src")
    prev = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = src + (os.pathsep + prev if prev else "")
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prev


def _build_looo_fixture(out_path: Path, seed: int = 42) -> Path:
    """Synthesize a tiny LOOO results parquet from the committed smoke obs.

    Uses the real obsids/stations/objects so script 16 can compute genuine
    AT/CT velocity vectors via the two-body propagator. Residuals are
    deterministic (seeded) so the smoke test is reproducible.
    """
    obs = pq.read_table(
        SMOKE_OBS, columns=["requested_provid", "obsid", "stn"]
    ).to_pandas()
    rng = np.random.default_rng(seed)
    n = len(obs)
    looo = pd.DataFrame(
        {
            "object_id": obs["requested_provid"].astype(str),
            "obs_id": obs["obsid"].astype(str),
            "stn": obs["stn"].astype(str),
            "residual_ra_arcsec": rng.normal(0.05, 0.2, n),
            "residual_dec_arcsec": rng.normal(-0.03, 0.2, n),
            "sigma_ra_cosdec_arcsec": np.full(n, 0.3),
            "sigma_dec_arcsec": np.full(n, 0.3),
        }
    )
    looo["chi2"] = (
        (looo["residual_ra_arcsec"] / 0.3) ** 2
        + (looo["residual_dec_arcsec"] / 0.3) ** 2
    )
    looo["hold_in_reduced_chi2"] = rng.uniform(0.5, 1.5, n)
    pq.write_table(pa.Table.from_pandas(looo, preserve_index=False), out_path)
    return out_path


def _publish(tmp_path: Path, residuals: Path, **kwargs) -> dict:
    """Run the wrapper with the smoke obs/orbits and a fast bootstrap."""
    params = dict(
        residuals=residuals,
        observations=SMOKE_OBS,
        orbits=SMOKE_ORBITS,
        output_dir=tmp_path / "published",
        propagator="twobody",
        n_bootstrap=200,
        random_seed=0,
    )
    params.update(kwargs)
    return wrapper.publish_catalog(**params)


def test_publish_catalog_smoke(tmp_path):
    """End-to-end: 16 → guard → 17 produces AT/CT residuals + a bias catalog."""
    looo = _build_looo_fixture(tmp_path / "looo_results.parquet")
    summary = _publish(tmp_path, looo)

    # --- AT/CT-augmented residuals exist and are genuinely populated -------
    augmented = summary["augmented_residuals"]
    assert augmented.exists(), "augmented residuals parquet not written"
    aug = pq.read_table(
        augmented, columns=["residual_at_arcsec", "residual_ct_arcsec"]
    ).to_pandas()
    assert "residual_at_arcsec" in aug.columns
    assert "residual_ct_arcsec" in aug.columns
    # The d5b property: AT/CT must be non-null (not the empty-column v1 bug).
    assert aug["residual_at_arcsec"].notna().any(), "AT column is all-null"
    assert aug["residual_ct_arcsec"].notna().any(), "CT column is all-null"
    assert summary["atct_non_null_rows"] > 0

    # --- bias_catalog/ contains the expected files ------------------------
    catalog_dir = summary["bias_catalog_dir"]
    for fname in ("bias_table.parquet", "bias_table.csv", "bias_table_config.json"):
        assert (catalog_dir / fname).exists(), f"missing {fname} in bias_catalog/"

    # --- bias table is non-empty and carries non-null AT/CT bias ----------
    bt = pd.read_parquet(catalog_dir / "bias_table.parquet")
    assert len(bt) > 0, "bias table has no rows"
    for col in ("bias_at_arcsec", "bias_ct_arcsec"):
        assert col in bt.columns, f"bias table missing {col}"
    assert bt["bias_at_arcsec"].notna().any(), "bias_at all-null in catalog"
    assert bt["bias_ct_arcsec"].notna().any(), "bias_ct all-null in catalog"


def test_d5b_guard_aborts_before_bias_table_on_null_atct(tmp_path):
    """If AT/CT comes out empty, the wrapper must abort and write no catalog."""
    # obs_ids that do not exist in the observations → no velocity vectors →
    # AT/CT all-null, exactly the v1/d5b failure signature.
    bogus = pd.DataFrame(
        {
            "object_id": ["X"] * 3 + ["Y"] * 3,
            "obs_id": [f"missing-{i}" for i in range(6)],
            "stn": ["F51"] * 6,
            "residual_ra_arcsec": [0.1] * 6,
            "residual_dec_arcsec": [0.1] * 6,
            "chi2": [1.0] * 6,
        }
    )
    looo = tmp_path / "looo_bogus.parquet"
    pq.write_table(pa.Table.from_pandas(bogus, preserve_index=False), looo)

    with pytest.raises(wrapper.PublishError, match="d5b guard"):
        _publish(tmp_path, looo)

    # No bias catalog may exist when the guard fires.
    assert not (tmp_path / "published" / "bias_catalog").exists()


def test_overwrite_gate(tmp_path):
    """Re-running is idempotent: blocked without --overwrite, allowed with it."""
    looo = _build_looo_fixture(tmp_path / "looo_results.parquet")
    _publish(tmp_path, looo)

    with pytest.raises(wrapper.PublishError, match="already exist"):
        _publish(tmp_path, looo)

    # With overwrite it succeeds and still produces a populated catalog.
    summary = _publish(tmp_path, looo, overwrite=True)
    assert summary["bias_table"].exists()
    assert summary["atct_non_null_rows"] > 0
