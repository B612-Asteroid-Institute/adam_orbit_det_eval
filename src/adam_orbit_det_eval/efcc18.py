"""EFCC18 star-catalog debiasing.

Implements the per-observation correction described in Eggl, Farnocchia,
Chamberlin & Chesley (2020) "An observational reference frame for
astrometric asteroid surveys", Icarus 339:113596. The published table
(``bias.dat``) covers 26 catalogs over a HEALPix tessellation of the sky
(``N_side = 64`` → 49152 tiles, RING ordering). Each (tile, catalog) cell stores four
int-encoded numbers: position correction in RA·cos(Dec) at J2000 [arcsec],
position correction in Dec at J2000 [arcsec], proper-motion correction in
RA·cos(Dec) [mas/yr], proper-motion correction in Dec [mas/yr].

To debias a single observation:

    bias_ra_arcsec  = dRA  + (year - 2000) * pmRA  / 1000   # cos(dec) frame
    bias_dec_arcsec = dDec + (year - 2000) * pmDec / 1000
    corrected_RA  = obs_RA  - bias_ra_arcsec  / (3600 * cos(dec))   # deg
    corrected_Dec = obs_Dec - bias_dec_arcsec / 3600                # deg

This module loads ``bias.dat``, exposes a vectorised per-observation
lookup, and maps MPC ``astcat`` names (as returned by ``mpcq``) to the
single-character catalog codes EFCC18 indexes by.

Provenance
----------
The source archive is ``debias_2018.tgz`` published by JPL SSD at
ftp://ssd.jpl.nasa.gov/pub/ssd/debias/debias_2018.tgz (also linked from
the Eggl 2020 paper). On this workstation it lives at
``/Users/kathleenkiker/claude_cli/debias_2018.tgz``; extracting it
produces ``bias.dat``, ``tiles.dat``, and ``README.txt``. We use the
``bias.dat`` from this archive — checksum it before depending on it for
production work; the version embedded in the file header is
``BIAS_VERSION= 3.0 (September 21, 2018)``.

Reference path: ``/tmp/efcc18_extract/bias.dat`` (or pass an explicit
path to :func:`load_efcc18_biases`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import healpy as hp
import numpy as np

logger = logging.getLogger(__name__)

# Single-character EFCC18 catalog codes, in column order from bias.dat.
# Mirrors the header line "Catalogs in this file (MPC designation): ...".
EFCC18_CATALOG_CODES = list("abcdegijlmnopqrtuvwLNQRSUY")
assert len(EFCC18_CATALOG_CODES) == 26

#: Map from MPC ``astcat`` string (as returned by mpcq) to the
#: single-character EFCC18 code. Catalogs that postdate EFCC18 (Gaia DR2/
#: DR3/EDR3, ATLAS, Pan-STARRS, etc.) are intentionally absent — they have
#: no EFCC18 entry and should pass through with zero correction. The
#: dash-stripped MPC names come from
#: ``adam_orbit_det_eval.utils.VERES2017_CATALOG_DEFAULTS``.
MPC_ASTCAT_TO_EFCC18: dict[str, str] = {
    "USNOA1": "a",
    "USNOSA1": "b",
    "USNOA2": "c",
    "USNOSA2": "d",
    "UCAC1": "e",
    "Tycho": "g",      # Tycho-2
    "GSC1.1": "i",
    "GSC1.2": "j",
    "ACT": "l",
    "GSCACT": "m",
    "SDSS8": "n",      # SDSS-DR8
    "USNOB1": "o",
    "PPM": "p",
    "UCAC4": "q",
    "UCAC2": "r",
    "PPMXL": "t",
    "UCAC3": "u",
    "NOMAD": "v",
    "CMC14": "w",
    "2MASS": "L",
    "SDSS7": "N",
    "CMC15": "Q",
    "SSTRC4": "R",
    "URAT1": "S",
    "Gaia1": "U",      # Gaia-DR1
    "UCAC5": "Y",
}

#: HEALPix resolution used by EFCC18.
EFCC18_NSIDE = 64
#: 12 * NSIDE^2 = 49152 tiles.
EFCC18_N_TILES = 12 * EFCC18_NSIDE * EFCC18_NSIDE
#: ``bias.dat`` stores corrections in arcsec with 3 decimals (RA, Dec) and
#: mas/yr with 2 decimals (pmRA, pmDec). The text float values are stored
#: as-is; we keep them as float32 in memory (~5 MB total).
EFCC18_DTYPE = np.float32

DEFAULT_BIAS_DAT_PATHS = [
    Path("/tmp/efcc18_extract/bias.dat"),
    Path("/Users/kathleenkiker/claude_cli/efcc18/bias.dat"),
    Path("/Users/kathleenkiker/od_experiments_setup/data/efcc18/bias.dat"),
]


def _resolve_bias_dat(path: Optional[Path] = None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"EFCC18 bias.dat not at supplied path: {path}")
        return path
    for candidate in DEFAULT_BIAS_DAT_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "EFCC18 bias.dat not found in any default location. Extract "
        "debias_2018.tgz first (e.g. into /tmp/efcc18_extract/) or pass a "
        "path. Source: ftp://ssd.jpl.nasa.gov/pub/ssd/debias/debias_2018.tgz"
    )


def load_efcc18_biases(
    bias_dat: Optional[Path] = None, cache_npy: Optional[Path] = None
) -> np.ndarray:
    """Load EFCC18 ``bias.dat`` into a ``(N_TILES, 26, 4)`` ndarray.

    Last axis is ``(dRA_arcsec, dDec_arcsec, pmRA_mas_yr, pmDec_mas_yr)``.
    RA terms are in the cos(dec)-corrected tangent-plane frame.

    A ``.npy`` cache is created next to ``bias_dat`` (or at ``cache_npy``
    if supplied) on first call; subsequent calls re-use it.
    """
    bias_dat_path = _resolve_bias_dat(bias_dat)
    if cache_npy is None:
        cache_npy = bias_dat_path.with_suffix(".npy")
    if cache_npy.exists() and cache_npy.stat().st_mtime >= bias_dat_path.stat().st_mtime:
        arr = np.load(cache_npy)
        if arr.shape == (EFCC18_N_TILES, 26, 4):
            return arr
        logger.warning(
            "EFCC18 cache at %s had wrong shape %s; reparsing", cache_npy, arr.shape
        )

    logger.info("Parsing EFCC18 bias.dat from %s (one-time, ~5s)…", bias_dat_path)
    rows: list[list[float]] = []
    with bias_dat_path.open() as f:
        for line in f:
            if line.startswith("!") or not line.strip():
                continue
            vals = line.split()
            if len(vals) != 26 * 4:
                raise ValueError(
                    f"Unexpected bias.dat row width: {len(vals)} (expected {26*4})"
                )
            rows.append([float(v) for v in vals])
    if len(rows) != EFCC18_N_TILES:
        raise ValueError(
            f"bias.dat has {len(rows)} rows, expected {EFCC18_N_TILES}"
        )
    arr = np.asarray(rows, dtype=EFCC18_DTYPE).reshape(EFCC18_N_TILES, 26, 4)
    try:
        np.save(cache_npy, arr)
        logger.info("Cached EFCC18 biases to %s", cache_npy)
    except OSError as e:
        logger.warning("Could not write EFCC18 cache to %s: %s", cache_npy, e)
    return arr


def _ra_dec_to_healpix(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Vectorised (RA, Dec) → HEALPix tile index at N_side=64, RING order.

    The k-th data row of ``bias.dat`` is HEALPix RING pixel k: the JPL
    archive's ``tiles.dat`` (rows "sorted in the same way" as bias.dat per its
    README) lists ring-scheme pixel centres for all 49152 tiles, and Find_Orb
    reads the table in ring order (bias.cpp / healpix.cpp). Until 2026-09-10
    this function used ``nest=True`` — a correction of the right magnitude
    from an unrelated tile — so every EFCC18 result computed through this
    module before then must be redone (same bug and evidence as adam_core
    d56114ac, kk/efcc18-ring-order; the two implementations now agree to
    1e-4 mas on the walk-forward study's 100 objects).
    """
    theta = np.deg2rad(90.0 - dec_deg)  # colatitude
    phi = np.deg2rad(ra_deg)
    return hp.ang2pix(EFCC18_NSIDE, theta, phi, nest=False)


def compute_efcc18_corrections(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    astcats: list[Optional[str]],
    jd_tdb: np.ndarray,
    bias_table: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-observation EFCC18 corrections.

    Parameters
    ----------
    ra_deg, dec_deg : (N,) float arrays
        Observed RA and Dec in degrees.
    astcats : list of length N
        MPC ``astcat`` strings; entries not in :data:`MPC_ASTCAT_TO_EFCC18`
        receive zero correction.
    jd_tdb : (N,) float array
        Observation Julian Dates. Used for the proper-motion linear-in-time
        correction.
    bias_table : (N_TILES, 26, 4) float array, optional
        Pre-loaded EFCC18 table. If None, loaded via :func:`load_efcc18_biases`.

    Returns
    -------
    corrections : (N, 2) float array
        Columns are ``(bias_ra_arcsec, bias_dec_arcsec)`` in the
        cos(dec)-corrected tangent-plane frame, i.e. directly comparable
        to MPC's ``rmsra``. Caller subtracts ``bias_ra/3600/cos(dec)`` from
        observed RA and ``bias_dec/3600`` from observed Dec.
    """
    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    jd_tdb = np.asarray(jd_tdb, dtype=np.float64)
    n = ra_deg.shape[0]
    if dec_deg.shape != (n,) or jd_tdb.shape != (n,) or len(astcats) != n:
        raise ValueError(
            "ra_deg, dec_deg, jd_tdb, and astcats must all have length N"
        )
    if bias_table is None:
        bias_table = load_efcc18_biases()

    tile_idx = _ra_dec_to_healpix(ra_deg, dec_deg)
    # year offset for the proper-motion term: (JD - 2451545.0) / 365.25
    year_offset = (jd_tdb - 2451545.0) / 365.25

    out = np.zeros((n, 2), dtype=np.float64)
    code_to_col = {c: i for i, c in enumerate(EFCC18_CATALOG_CODES)}
    for i in range(n):
        ac = astcats[i]
        if ac is None:
            continue
        code = MPC_ASTCAT_TO_EFCC18.get(ac)
        if code is None:
            continue
        col = code_to_col[code]
        cell = bias_table[tile_idx[i], col, :]
        dRA, dDec, pmRA, pmDec = cell
        # bias.dat stores pm in mas/yr ⇒ multiply by year/1000 → arcsec.
        out[i, 0] = float(dRA) + year_offset[i] * float(pmRA) / 1000.0
        out[i, 1] = float(dDec) + year_offset[i] * float(pmDec) / 1000.0
    return out


def n_observations_covered(astcats: list[Optional[str]]) -> int:
    """Return how many entries of ``astcats`` are in :data:`MPC_ASTCAT_TO_EFCC18`."""
    return sum(1 for a in astcats if a is not None and a in MPC_ASTCAT_TO_EFCC18)
