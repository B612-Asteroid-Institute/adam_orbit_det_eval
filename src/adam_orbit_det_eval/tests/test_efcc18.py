"""Tests for the EFCC18 catalog-debiasing module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..efcc18 import (
    EFCC18_CATALOG_CODES,
    EFCC18_N_TILES,
    MPC_ASTCAT_TO_EFCC18,
    _ra_dec_to_healpix,
    compute_efcc18_corrections,
    load_efcc18_biases,
    n_observations_covered,
)


def _make_synthetic_bias_dat(tmp_path: Path) -> Path:
    """Write a synthetic 49152-row bias.dat with deterministic values."""
    out = tmp_path / "bias.dat"
    # Header: a single comment line + the catalog-code header is required for
    # textual parity with the real file. Our loader skips '!' lines.
    with out.open("w") as f:
        f.write("! synthetic bias.dat for unit tests\n")
        # 26 catalogs * 4 cols per row. Use a simple pattern: each cell is
        # filled with a known value derived from (tile_idx, cat_idx, col_idx).
        # Keep numbers small so the column format stays sane.
        for tile in range(EFCC18_N_TILES):
            row_vals = []
            for cat in range(26):
                # Encode (tile, cat, col) into the value so the test can check.
                # Use very small numbers to fit format.
                row_vals.extend(
                    [
                        0.001 * cat + 0.0001 * (tile % 7),  # dRA
                        0.002 * cat + 0.0001 * (tile % 5),  # dDec
                        0.0,  # pmRA
                        0.0,  # pmDec
                    ]
                )
            f.write("  ".join(f"{v: .4f}" for v in row_vals) + "\n")
    return out


def test_efcc18_catalog_codes_unique_and_sized() -> None:
    """The EFCC18 catalog code list has 26 unique single-char codes."""
    assert len(EFCC18_CATALOG_CODES) == 26
    assert len(set(EFCC18_CATALOG_CODES)) == 26
    # Spot-check a few well-known codes from the EFCC18 bias.dat header.
    assert "q" in EFCC18_CATALOG_CODES  # UCAC-4
    assert "U" in EFCC18_CATALOG_CODES  # Gaia-DR1
    assert "L" in EFCC18_CATALOG_CODES  # 2MASS


def test_mpc_astcat_mapping_covers_only_efcc18_catalogs() -> None:
    """Every code in the MPC → EFCC18 map is actually in EFCC18."""
    for mpc_name, code in MPC_ASTCAT_TO_EFCC18.items():
        assert code in EFCC18_CATALOG_CODES, (mpc_name, code)
    # Catalogs that postdate EFCC18 must NOT be in the map.
    for absent in ("Gaia2", "Gaia3", "Gaia3E", "ATLAS", "ATLAS2", "PS1_DR1"):
        assert absent not in MPC_ASTCAT_TO_EFCC18, absent


def test_ra_dec_to_healpix_basic() -> None:
    """RA=0/Dec=0 lands on a well-defined tile; antipode lands on a different one."""
    t = _ra_dec_to_healpix(np.array([0.0, 180.0]), np.array([0.0, 0.0]))
    assert t.shape == (2,)
    assert t[0] != t[1]
    assert 0 <= int(t[0]) < EFCC18_N_TILES
    assert 0 <= int(t[1]) < EFCC18_N_TILES


def test_load_efcc18_biases_round_trip(tmp_path: Path) -> None:
    """Loader reads the synthetic file, caches it, and re-reads from cache."""
    bias_dat = _make_synthetic_bias_dat(tmp_path)
    arr = load_efcc18_biases(bias_dat)
    assert arr.shape == (EFCC18_N_TILES, 26, 4)
    # Check the encoding for a known cell.
    expected_dRA = 0.001 * 5 + 0.0001 * (10 % 7)  # tile 10, cat 5 (= 'g')
    np.testing.assert_allclose(arr[10, 5, 0], expected_dRA, atol=1e-6)
    # Cache file exists.
    cached = bias_dat.with_suffix(".npy")
    assert cached.exists()
    arr2 = load_efcc18_biases(bias_dat)
    np.testing.assert_array_equal(arr, arr2)


def test_compute_efcc18_corrections_known_catalog(tmp_path: Path) -> None:
    """For a known (RA, Dec, catalog) row, the lookup returns the table cell."""
    bias_dat = _make_synthetic_bias_dat(tmp_path)
    arr = load_efcc18_biases(bias_dat)
    # Choose an observation; we don't care about RA/Dec specifically, just that
    # the right tile cell is read.
    ra = np.array([100.0])
    dec = np.array([30.0])
    jd = np.array([2451545.0])  # J2000 → zero proper-motion contribution
    astcats = ["UCAC4"]  # → 'q', column 13
    out = compute_efcc18_corrections(ra, dec, astcats, jd, bias_table=arr)
    assert out.shape == (1, 2)
    tile = int(_ra_dec_to_healpix(ra, dec)[0])
    expected_dRA = arr[tile, EFCC18_CATALOG_CODES.index("q"), 0]
    expected_dDec = arr[tile, EFCC18_CATALOG_CODES.index("q"), 1]
    np.testing.assert_allclose(out[0, 0], expected_dRA, atol=1e-6)
    np.testing.assert_allclose(out[0, 1], expected_dDec, atol=1e-6)


def test_compute_efcc18_corrections_unknown_catalog_is_zero(tmp_path: Path) -> None:
    """A catalog not in EFCC18 (e.g. Gaia DR2) gets zero correction."""
    bias_dat = _make_synthetic_bias_dat(tmp_path)
    arr = load_efcc18_biases(bias_dat)
    out = compute_efcc18_corrections(
        np.array([10.0, 200.0]),
        np.array([-5.0, 25.0]),
        ["Gaia2", "Gaia3E"],
        np.array([2459200.0, 2460000.0]),
        bias_table=arr,
    )
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, np.zeros((2, 2)))


def test_compute_efcc18_corrections_pm_term(tmp_path: Path) -> None:
    """Proper-motion term scales linearly with (year - 2000)."""
    # Write a synthetic bias.dat where one cell has nonzero pmRA so we can
    # verify the linear-in-time term.
    out_path = tmp_path / "bias.dat"
    with out_path.open("w") as f:
        f.write("! synthetic\n")
        for tile in range(EFCC18_N_TILES):
            cells = ["0.0000"] * (26 * 4)
            if tile == 0:
                # cat 'q' = col 13 → indices 52..55
                cells[52] = "0.0000"  # dRA at J2000
                cells[53] = "0.0000"  # dDec
                cells[54] = "100.0000"  # pmRA mas/yr → +0.1 arcsec/yr
                cells[55] = "0.0000"
            f.write("  ".join(cells) + "\n")
    arr = load_efcc18_biases(out_path)
    # Find the (RA, Dec) center of tile 0 by inverting the HEALPix mapping.
    import healpy as hp

    theta_center, phi_center = hp.pix2ang(64, 0, nest=True)
    dec = float(90.0 - np.rad2deg(theta_center))
    ra = float(np.rad2deg(phi_center))
    assert int(_ra_dec_to_healpix(np.array([ra]), np.array([dec]))[0]) == 0

    # 10 years after J2000 → expect 1.0 arcsec correction in cos(dec) RA
    jd = np.array([2451545.0 + 10.0 * 365.25])
    out = compute_efcc18_corrections(
        np.array([ra]), np.array([dec]), ["UCAC4"], jd, bias_table=arr
    )
    np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-6)


def test_n_observations_covered() -> None:
    assert n_observations_covered(["UCAC4", "Gaia2", "Gaia3", "USNOA2"]) == 2
    assert n_observations_covered(["Gaia2", "Gaia3", None]) == 0
    assert n_observations_covered([]) == 0
