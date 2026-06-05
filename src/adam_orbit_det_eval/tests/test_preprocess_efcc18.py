"""
Tests for scripts/preprocess_efcc18.py — EFCC18 catalog-debiasing preprocessing.

Covers:
  * Round-trip: a known-catalog/known-tile observation produces the documented
    EFCC18 offset (synthetic fixture + a cross-check against the real
    debias_2018 archive when present).
  * Coverage report: a tiny mixed-catalog fixture hits the expected fraction
    and classifies misses (Gaia reference frame vs unsupported) correctly.
  * Boundary: pre-2000 and post-2000 epochs yield drift corrections that are
    linear in (t - 2000) and flip sign across the J2000 epoch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pyarrow as pa
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
REAL_TGZ = Path("/Users/kathleenkiker/claude_cli/debias_2018.tgz")


def _import_script():
    """Import preprocess_efcc18.py as a module (registered for dataclasses)."""
    spec = importlib.util.spec_from_file_location(
        "preprocess_efcc18",
        SCRIPTS_DIR / "preprocess_efcc18.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preprocess_efcc18"] = mod
    spec.loader.exec_module(mod)
    return mod


pe = _import_script()


# ---------------------------------------------------------------------------
# Fixtures: a tiny NSIDE=1 (12-tile) synthetic EFCC18 archive
# ---------------------------------------------------------------------------
# Two catalogs: 'o' (USNO-B1.0) and 'L' (2MASS). Per-tile values are distinct
# and deterministic so a lookup can be checked exactly. The position offsets
# are set to ~0.1-0.3" — Veres-2017-scale catalog biases.
NSIDE = 1
NPIX = 12 * NSIDE * NSIDE
CODES = ["o", "L"]


def _o_vals(tile: int) -> tuple[float, float, float, float]:
    # (dRA["], dDEC["], pmRA[mas/yr], pmDEC[mas/yr]); dRA carries cos(Dec).
    return (0.10 + 0.01 * tile, 0.20 - 0.01 * tile, 5.0, -3.0)


def _l_vals(tile: int) -> tuple[float, float, float, float]:
    return (-0.05 * (tile + 1), 0.03 * (tile + 1), -1.0, 2.0)


def _write_synthetic_efcc18(efcc18_dir: Path) -> None:
    efcc18_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "! BIAS_VERSION= test-1.0",
        f"! NSIDE= {NSIDE}",
        f"! NPIX= {NPIX}",
        "! Catalogs in this file (MPC designation):",
        "! " + " ".join(CODES),
        "! synthetic fixture for unit tests",
    ]
    for tile in range(NPIX):
        row = list(_o_vals(tile)) + list(_l_vals(tile))
        lines.append(" ".join(f"{v:.6f}" for v in row))
    (efcc18_dir / "bias.dat").write_text("\n".join(lines) + "\n")

    # tiles.dat: real NSIDE=1 tile centers (radians) so artifacts are consistent.
    ra_deg, dec_deg = hp.pix2ang(NSIDE, np.arange(NPIX), nest=False, lonlat=True)
    tlines = ["! tile RA DEC", "! number [rad] [rad]"]
    for tile in range(NPIX):
        tlines.append(
            f"{tile} {np.radians(ra_deg[tile]):.6f} {np.radians(dec_deg[tile]):.6f}"
        )
    (efcc18_dir / "tiles.dat").write_text("\n".join(tlines) + "\n")


def _obs_table(ra, dec, astcat, mjd):
    """Build an observation table matching the mpc_observations schema subset."""
    days = np.floor(mjd).astype(np.int64)
    nanos = ((np.asarray(mjd, dtype=np.float64) - days) * 86_400_000_000_000.0).astype(
        np.int64
    )
    obstime = pa.StructArray.from_arrays(
        [pa.array(days), pa.array(nanos)], names=["days", "nanos"]
    )
    return pa.table(
        {
            "ra": pa.array(np.asarray(ra, dtype=np.float64)),
            "dec": pa.array(np.asarray(dec, dtype=np.float64)),
            "astcat": pa.array(list(astcat)),
            "obstime": obstime,
        }
    )


@pytest.fixture
def synthetic_bias(tmp_path):
    efcc18_dir = tmp_path / "efcc18"
    _write_synthetic_efcc18(efcc18_dir)
    bias = pe.build_efcc18_lookup(efcc18_dir, source_tgz=Path("/nonexistent.tgz"))
    return bias, efcc18_dir


# ---------------------------------------------------------------------------
# Parsing / lookup build
# ---------------------------------------------------------------------------
class TestBuildLookup:
    def test_parses_header_and_shape(self, synthetic_bias):
        bias, _ = synthetic_bias
        assert bias.nside == NSIDE
        assert bias.codes == CODES
        assert bias.npix == NPIX
        assert bias.bias_version == "test-1.0"
        assert bias.table.shape == (len(CODES), NPIX, 4)

    def test_table_values_match_source(self, synthetic_bias):
        bias, _ = synthetic_bias
        for tile in range(NPIX):
            np.testing.assert_allclose(
                bias.table[bias.code_to_col["o"], tile], _o_vals(tile)
            )
            np.testing.assert_allclose(
                bias.table[bias.code_to_col["L"], tile], _l_vals(tile)
            )

    def test_artifacts_written_and_cached_reload(self, synthetic_bias):
        bias, efcc18_dir = synthetic_bias
        for fname in ("bias.parquet", "tiles.parquet", "catalog_codes.csv"):
            assert (efcc18_dir / fname).exists()
        # Second call must hit the cached parquet (source_tgz unreadable) and match.
        reloaded = pe.build_efcc18_lookup(
            efcc18_dir, source_tgz=Path("/nonexistent.tgz")
        )
        assert reloaded.codes == bias.codes
        np.testing.assert_allclose(reloaded.table, bias.table)

    def test_mjd_to_julian_year(self):
        # MJD 51544.5 is J2000.0 exactly.
        np.testing.assert_allclose(pe.mjd_to_julian_year(np.array([51544.5])), [2000.0])
        np.testing.assert_allclose(
            pe.mjd_to_julian_year(np.array([51544.5 + 365.25])), [2001.0]
        )


# ---------------------------------------------------------------------------
# Round-trip / documented-offset cross-check
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_known_catalog_known_tile_offset(self, synthetic_bias):
        bias, _ = synthetic_bias
        # Place an observation; resolve its tile the same way the code does.
        ra, dec = 137.0, 12.5
        tile = int(hp.ang2pix(NSIDE, ra, dec, nest=False, lonlat=True))
        d_ra, d_dec, _, _ = _o_vals(tile)
        cos_dec = np.cos(np.radians(dec))

        # epoch == 2000.0 -> drift term vanishes, correction is the pure offset.
        result = pe.correct_observations(
            np.array([ra]), np.array([dec]), np.array([2000.0]), ["USNOB1"], bias
        )
        assert bool(result.applied[0])
        np.testing.assert_allclose(result.dra_arcsec[0], d_ra / cos_dec)
        np.testing.assert_allclose(result.ddec_arcsec[0], d_dec)
        # corrected = observed - correction (converted arcsec -> deg)
        np.testing.assert_allclose(
            result.ra_corrected[0], ra - (d_ra / cos_dec) / 3600.0
        )
        np.testing.assert_allclose(result.dec_corrected[0], dec - d_dec / 3600.0)

    def test_unsupported_and_reference_catalogs_unchanged(self, synthetic_bias):
        bias, _ = synthetic_bias
        ra, dec = np.array([137.0, 200.0]), np.array([12.5, -30.0])
        result = pe.correct_observations(
            ra, dec, np.array([2010.0, 2010.0]), ["Gaia2", "PS1_DR1"], bias
        )
        assert not result.applied.any()
        np.testing.assert_array_equal(result.ra_corrected, ra)
        np.testing.assert_array_equal(result.dec_corrected, dec)

    @pytest.mark.skipif(not REAL_TGZ.exists(), reason="debias_2018.tgz not on disk")
    def test_real_archive_offset_matches_raw_table(self, tmp_path):
        """Cross-check the applied correction against the raw EFCC18 value."""
        bias = pe.build_efcc18_lookup(tmp_path / "efcc18_real", source_tgz=REAL_TGZ)
        assert bias.nside == 64
        assert bias.bias_version.startswith("3.0")
        assert len(bias.codes) == 26

        ra, dec, epoch = 150.0, -20.0, 2005.0
        tile = int(hp.ang2pix(64, ra, dec, nest=False, lonlat=True))
        col = bias.code_to_col["o"]  # USNO-B1.0
        d_ra, d_dec, pm_ra, pm_dec = bias.table[col, tile]
        cos_dec = np.cos(np.radians(dec))
        exp_dra = (d_ra + (epoch - 2000.0) * pm_ra / 1000.0) / cos_dec
        exp_ddec = d_dec + (epoch - 2000.0) * pm_dec / 1000.0

        result = pe.correct_observations(
            np.array([ra]), np.array([dec]), np.array([epoch]), ["USNOB1"], bias
        )
        np.testing.assert_allclose(result.dra_arcsec[0], exp_dra)
        np.testing.assert_allclose(result.ddec_arcsec[0], exp_ddec)


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
class TestCoverage:
    def test_coverage_classification(self, synthetic_bias):
        bias, _ = synthetic_bias
        # 5 hit (USNOB1/2MASS), 3 reference (Gaia), 2 unsupported (PS1/None).
        astcat = [
            "USNOB1",
            "USNOB1",
            "2MASS",
            "USNOB1",
            "2MASS",
            "Gaia2",
            "Gaia3",
            "Gaia3E",
            "PS1_DR1",
            None,
        ]
        n = len(astcat)
        ra = np.full(n, 100.0)
        dec = np.linspace(-40.0, 40.0, n)
        mjd = np.full(n, 53000.0)
        table = _obs_table(ra, dec, astcat, mjd)

        out, cov = pe.correct_table(table, bias)
        assert cov["n_obs"] == n
        assert cov["n_hit"] == 5
        assert cov["n_miss"] == 5
        assert cov["n_reference_frame"] == 3  # Gaia2/Gaia3/Gaia3E
        assert cov["n_unsupported"] == 2  # PS1_DR1 + <None>
        assert cov["hit_pct"] == pytest.approx(50.0)
        assert cov["miss_by_catalog"]["Gaia2"] == 1

        # Output schema: drop-in ra/dec plus provenance columns.
        applied = out.column("efcc18_applied").to_numpy(zero_copy_only=False)
        ra_obs = out.column("ra_observed").to_numpy(zero_copy_only=False)
        ra_corr = out.column("ra").to_numpy(zero_copy_only=False)
        assert applied.sum() == 5
        # Untouched rows identical; touched rows changed.
        np.testing.assert_array_equal(ra_obs[~applied], ra_corr[~applied])
        assert np.all(ra_obs[applied] != ra_corr[applied])

    def test_empty_table_coverage(self, synthetic_bias):
        bias, _ = synthetic_bias
        table = _obs_table([], [], [], [])
        _, cov = pe.correct_table(table, bias)
        assert cov["n_obs"] == 0
        assert cov["hit_pct"] == 0.0


# ---------------------------------------------------------------------------
# Epoch boundary / drift behaviour
# ---------------------------------------------------------------------------
class TestEpochBoundary:
    def test_drift_linear_and_sign_flips_across_j2000(self, synthetic_bias):
        bias, _ = synthetic_bias
        ra, dec = 137.0, 12.5
        tile = int(hp.ang2pix(NSIDE, ra, dec, nest=False, lonlat=True))
        d_ra, d_dec, pm_ra, pm_dec = _o_vals(tile)
        cos_dec = np.cos(np.radians(dec))

        def corr(epoch):
            r = pe.correct_observations(
                np.array([ra]), np.array([dec]), np.array([epoch]), ["USNOB1"], bias
            )
            return r.dra_arcsec[0], r.ddec_arcsec[0]

        pre_ra, pre_dec = corr(1990.0)  # dt = -10
        mid_ra, mid_dec = corr(2000.0)  # dt = 0
        post_ra, post_dec = corr(2010.0)  # dt = +10

        # At J2000 the correction is the pure position offset.
        np.testing.assert_allclose(mid_ra, d_ra / cos_dec)
        np.testing.assert_allclose(mid_dec, d_dec)

        # Drift term is linear in (t-2000): pre/post straddle the J2000 value
        # symmetrically, and the RA drift sign matches pm_ra.
        drift_ra = (10.0 * pm_ra / 1000.0) / cos_dec
        drift_dec = 10.0 * pm_dec / 1000.0
        np.testing.assert_allclose(post_ra - mid_ra, drift_ra)
        np.testing.assert_allclose(mid_ra - pre_ra, drift_ra)
        np.testing.assert_allclose(post_dec - mid_dec, drift_dec)
        # Pre- and post-2000 drift contributions have opposite sign.
        assert np.sign(post_ra - mid_ra) == -np.sign(pre_ra - mid_ra)


# ---------------------------------------------------------------------------
# End-to-end shard processing (idempotent / resumable)
# ---------------------------------------------------------------------------
class TestProcessInput:
    def test_sharded_roundtrip_and_resume(self, synthetic_bias, tmp_path):
        bias, _ = synthetic_bias
        in_root = tmp_path / "obs"
        for s in range(2):
            shard = in_root / f"shard_{s:03d}"
            shard.mkdir(parents=True)
            ra = np.array([100.0, 137.0, 200.0])
            dec = np.array([10.0, 12.5, -30.0])
            astcat = ["USNOB1", "2MASS", "Gaia2"]
            mjd = np.array([53000.0, 53100.0, 53200.0])
            pa_pq = pytest.importorskip("pyarrow.parquet")
            pa_pq.write_table(_obs_table(ra, dec, astcat, mjd), shard / pe.OBS_FILENAME)

        out_root = tmp_path / "obs_efcc18"
        summary = pe.process_input(in_root, out_root, bias)
        assert summary["n_shards"] == 2
        assert summary["totals"]["n_hit"] == 4  # 2 hits x 2 shards
        assert (out_root / "_efcc18_coverage.json").exists()
        for s in range(2):
            assert (out_root / f"shard_{s:03d}" / pe.OBS_FILENAME).exists()

        # Resume: rerun without overwrite must skip and still report the same.
        summary2 = pe.process_input(in_root, out_root, bias, overwrite=False)
        assert summary2["totals"]["n_hit"] == summary["totals"]["n_hit"]
