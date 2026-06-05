#!/usr/bin/env python3
"""
EFCC18 catalog-debiasing as preprocessing before the LOOO pipeline.

Applies the Eggl, Farnocchia, Chamberlin & Chesley (2018) star-catalog
debiasing corrections (BIAS_VERSION 3.0, ``debias_2018.tgz``, Gaia-DR2 anchor)
to MPC astrometry *before* LOOO runs. EFCC18 publishes, per
(star-catalog, HEALPix-tile-at-NSIDE=64), a position offset and a
proper-motion drift term in both RA*cos(Dec) and Dec.

Correction (subtracted from the *observed* RA/Dec), per the EFCC18 README:

    corrected_RA  = observed_RA  - (dRA  + (t - 2000) * pmRA  / 1000) / cos(Dec)
    corrected_DEC = observed_DEC - (dDEC + (t - 2000) * pmDEC / 1000)

where t is the observation epoch in (Julian) years, dRA/dDEC are arcsec at
J2000 (dRA already carries the cos(Dec) factor) and pmRA/pmDEC are mas/yr.
The lookup is keyed on the observation's reference star catalog and the
HEALPix tile (RING ordering) containing its sky position.

This is a *standalone* preprocessing tool. Wiring it into the cloud shard
runner is a separate v2 step (see docs/v2-scope.md, bead 54t); the eventual
orchestrator calls this script per shard.

Background: LOOO is structurally blind to catalog systematics shared across
the residual network (bd memory ``looo-catalog-systematic-blindspot``).
Debiasing upstream lets the corrected re-run decompose catalog-induced from
intrinsic per-station bias.

CLI
---
    python scripts/preprocess_efcc18.py \\
        --input  data/mpc_scale_20260605/raw \\
        --output data/mpc_scale_20260605/raw_efcc18 \\
        [--efcc18-dir data/efcc18/] \\
        [--source-tgz /Users/.../debias_2018.tgz] \\
        [--overwrite]

``--input`` may be a directory of ``shard_NNN/mpc_observations.parquet`` (the
standard fetch layout) or a single observation parquet. Output mirrors the
input layout under ``<input>_efcc18/`` by default. The run is idempotent and
resumable: shards whose corrected output already exists are skipped unless
``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("preprocess_efcc18")

# Default location of the EFCC18 archive on disk (BIAS_VERSION 3.0).
DEFAULT_SOURCE_TGZ = Path("/Users/kathleenkiker/claude_cli/debias_2018.tgz")
DEFAULT_EFCC18_DIR = Path("data/efcc18")

# The corrected RA/Dec are the drop-in columns the LOOO pipeline consumes; the
# observed values are preserved alongside for provenance and delta-bias work.
OBS_FILENAME = "mpc_observations.parquet"

# ---------------------------------------------------------------------------
# Catalog-code mapping
# ---------------------------------------------------------------------------
# The observation parquets carry the ADES ``astcat`` *name* (e.g. "USNOA2",
# "Gaia2"), not the single-character MPC catalogue flag that EFCC18 keys on.
# This is the canonical MPC catalogue-code table (the same flag set used in the
# old 80-column astrometry format and documented by the MPC). Only the names
# actually observed in the v1 MPC-scale fetch are listed; extend as needed.
#
# A name maps to ``None`` when the MPC code has no EFCC18 correction. The two
# important "intentional no-op" classes:
#   * Gaia DR2/DR3/EDR3 — EFCC18 is defined *relative to* Gaia-DR2, so Gaia-era
#     astrometry is already in the reference frame (codes V/W/X are absent from
#     the file). Counting these as "corrected to zero" would be misleading, so
#     they are reported as a distinct `reference` class, not as failures.
#   * Surveys/catalogs outside EFCC18's 26 (Pan-STARRS, ATLAS2, generic GSC,
#     UNK, …) — no published correction; reported as `unsupported`.
ASTCAT_TO_MPC_CODE: dict[str, str | None] = {
    # --- catalogs WITH an EFCC18 correction (code is in debias_2018) ---
    "USNOA1": "a",  # USNO-A1.0
    "USNOSA1": "b",  # USNO-SA1.0
    "USNOA2": "c",  # USNO-A2.0
    "USNOSA2": "d",  # USNO-SA2.0
    "UCAC1": "e",  # UCAC-1
    "Tyc2": "g",  # Tycho-2
    "GSC1.1": "i",  # GSC-1.1
    "GSC1.2": "j",  # GSC-1.2
    "ACT": "l",  # ACT
    "GSCACT": "m",  # GSC-ACT
    "SDSS8": "n",  # SDSS-DR8
    "USNOB1": "o",  # USNO-B1.0
    "PPM": "p",  # PPM
    "UCAC4": "q",  # UCAC-4
    "UCAC2": "r",  # UCAC-2
    "PPMXL": "t",  # PPMXL
    "UCAC3": "u",  # UCAC-3
    "NOMAD": "v",  # NOMAD
    "CMC14": "w",  # CMC-14
    "2MASS": "L",  # 2MASS
    "SDSS7": "N",  # SDSS-DR7
    "CMC15": "Q",  # CMC-15
    "SSTRC4": "R",  # SST-RC4
    "URAT1": "S",  # URAT-1
    "Gaia1": "U",  # Gaia-DR1
    "UCAC5": "Y",  # UCAC-5
    # --- Gaia reference frame: no correction needed (intentional) ---
    "Gaia2": None,  # Gaia-DR2  (V) — the EFCC18 anchor itself
    "Gaia3": None,  # Gaia-DR3  (W)
    "GaiaDR3": None,  # Gaia-DR3 (alt spelling)
    "Gaia3E": None,  # Gaia-EDR3 (X)
    # --- catalogs outside EFCC18's 26: no published correction ---
    "GSC": None,  # ambiguous GSC generation — not keyable
    "GSC1.0": None,  # GSC-1.0 (h) absent from EFCC18
    "GSC2.2": None,  # GSC-2.2 (k) absent from EFCC18
    "USNOB2": None,  # USNO-B2.0 absent from EFCC18
    "ATLAS2": None,  # ATLAS-2 absent from EFCC18
    "PS1_DR1": None,  # Pan-STARRS DR1 absent from EFCC18
    "PS1-DR1": None,  # Pan-STARRS DR1 (alt spelling)
    "MPOSC3": None,
    "SAO": None,
    "SAO1984": None,
    "AC": None,
    "ACRS": None,
    "AGK3": None,
    "AGK3R": None,
    "Yale": None,
    "COSMOS": None,
    "LickGas": None,
    "FK4": None,
    "UNK": None,  # unknown / unspecified
}

# astcat names that are *deliberately* uncorrected because they already sit in
# (or very near) the EFCC18 reference frame. Used only to classify "misses".
REFERENCE_FRAME_ASTCATS = frozenset({"Gaia2", "Gaia3", "GaiaDR3", "Gaia3E"})


# ---------------------------------------------------------------------------
# EFCC18 bias model
# ---------------------------------------------------------------------------
@dataclass
class Efcc18Bias:
    """In-memory EFCC18 lookup.

    ``table`` has shape ``(n_codes, npix, 4)`` where the last axis is
    ``(dRA, dDEC, pmRA, pmDEC)``; dRA/dDEC in arcsec at J2000 (dRA includes
    cos(Dec)), pmRA/pmDEC in mas/yr. ``code_to_col`` maps a single-character
    MPC catalogue code to its row in ``table``.
    """

    nside: int
    codes: list[str]
    code_to_col: dict[str, int]
    table: npt.NDArray[np.float64]
    bias_version: str = ""
    reference_frame: str = "Gaia-DR2"

    @property
    def npix(self) -> int:
        return int(self.table.shape[1])


def _parse_efcc18_dat(
    bias_dat: Path, tiles_dat: Path
) -> tuple[int, list[str], str, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Parse the raw EFCC18 ``bias.dat``/``tiles.dat`` pair.

    Returns ``(nside, codes, bias_version, bias_table, tiles)`` where
    ``bias_table`` is ``(npix, n_codes, 4)`` and ``tiles`` is ``(npix, 2)`` of
    (RA, Dec) tile-center radians. Header-driven so the same parser ingests
    small fixtures with a different NSIDE.
    """
    nside = 0
    codes: list[str] = []
    bias_version = ""
    saw_catalog_header = False

    bias_rows: list[list[float]] = []
    with bias_dat.open() as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("!"):
                body = stripped[1:].strip()
                if body.startswith("BIAS_VERSION="):
                    bias_version = body.split("=", 1)[1].strip()
                elif body.startswith("NSIDE="):
                    nside = int(body.split("=", 1)[1].strip())
                elif body.startswith("Catalogs in this file"):
                    saw_catalog_header = True
                elif saw_catalog_header and not codes:
                    # The line immediately after the "Catalogs in this file"
                    # banner is the space-separated list of single-char codes.
                    codes = body.split()
                continue
            bias_rows.append([float(x) for x in stripped.split()])

    if not codes:
        raise ValueError(f"Could not parse catalogue codes from {bias_dat}")
    if nside <= 0:
        raise ValueError(f"Could not parse NSIDE from {bias_dat}")

    n_codes = len(codes)
    arr = np.asarray(bias_rows, dtype=np.float64)
    expected_npix = 12 * nside * nside
    if arr.shape != (expected_npix, 4 * n_codes):
        raise ValueError(
            f"{bias_dat}: expected {expected_npix} rows x {4 * n_codes} cols "
            f"for NSIDE={nside}, {n_codes} catalogs; got {arr.shape}"
        )
    bias_table = arr.reshape(expected_npix, n_codes, 4)

    tiles_rows: list[list[float]] = []
    with tiles_dat.open() as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped or stripped.startswith("!"):
                continue
            parts = stripped.split()
            tiles_rows.append([float(parts[1]), float(parts[2])])
    tiles = np.asarray(tiles_rows, dtype=np.float64)
    if tiles.shape[0] != expected_npix:
        raise ValueError(
            f"{tiles_dat}: expected {expected_npix} tiles, got {tiles.shape[0]}"
        )

    return nside, codes, bias_version, bias_table, tiles


def _extract_dat_files(efcc18_dir: Path, source_tgz: Path) -> tuple[Path, Path]:
    """Extract ``bias.dat``/``tiles.dat`` from ``source_tgz`` into ``efcc18_dir``."""
    efcc18_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_tgz, "r:gz") as tar:
        for name in ("bias.dat", "tiles.dat"):
            member = tar.getmember(name)
            with tar.extractfile(member) as src:  # type: ignore[union-attr]
                (efcc18_dir / name).write_bytes(src.read())
    return efcc18_dir / "bias.dat", efcc18_dir / "tiles.dat"


def _provenance(bias: Efcc18Bias, source: str) -> dict[bytes, bytes]:
    meta = {
        "efcc18_bias_version": bias.bias_version,
        "efcc18_nside": str(bias.nside),
        "efcc18_npix": str(bias.npix),
        "efcc18_reference_frame": bias.reference_frame,
        "efcc18_codes": "".join(bias.codes),
        "efcc18_source": source,
    }
    return {k.encode(): v.encode() for k, v in meta.items()}


def _write_artifacts(
    efcc18_dir: Path, bias: Efcc18Bias, tiles: npt.NDArray[np.float64], source: str
) -> None:
    """Persist the parsed model as parquet/csv with a provenance header."""
    prov = _provenance(bias, source)

    # bias.parquet — wide: tile + <code>_{dRA,dDEC,pmRA,pmDEC}
    bias_cols: dict[str, pa.Array] = {
        "tile": pa.array(np.arange(bias.npix, dtype=np.int32))
    }
    for col, code in enumerate(bias.codes):
        for k, field in enumerate(("dRA", "dDEC", "pmRA", "pmDEC")):
            bias_cols[f"{code}_{field}"] = pa.array(bias.table[col, :, k])
    bias_tbl = pa.table(bias_cols).replace_schema_metadata(prov)
    pq.write_table(bias_tbl, efcc18_dir / "bias.parquet")

    # tiles.parquet — tile centers in rad and deg
    ra_deg = np.degrees(tiles[:, 0])
    dec_deg = np.degrees(tiles[:, 1])
    tiles_tbl = pa.table(
        {
            "tile": pa.array(np.arange(bias.npix, dtype=np.int32)),
            "ra_rad": pa.array(tiles[:, 0]),
            "dec_rad": pa.array(tiles[:, 1]),
            "ra_deg": pa.array(ra_deg),
            "dec_deg": pa.array(dec_deg),
        }
    ).replace_schema_metadata(prov)
    pq.write_table(tiles_tbl, efcc18_dir / "tiles.parquet")

    # catalog_codes.csv — the code -> astcat-name mapping actually in use
    lines = ["mpc_code,astcat_names,in_efcc18"]
    code_to_names: dict[str, list[str]] = {}
    for name, code in ASTCAT_TO_MPC_CODE.items():
        if code is not None:
            code_to_names.setdefault(code, []).append(name)
    for code in bias.codes:
        names = "|".join(code_to_names.get(code, []))
        lines.append(f"{code},{names},yes")
    (efcc18_dir / "catalog_codes.csv").write_text("\n".join(lines) + "\n")


def _load_from_parquet(efcc18_dir: Path) -> Efcc18Bias:
    """Load a previously-built ``bias.parquet`` into an :class:`Efcc18Bias`."""
    tbl = pq.read_table(efcc18_dir / "bias.parquet")
    meta = tbl.schema.metadata or {}

    def _m(key: str, default: str = "") -> str:
        val = meta.get(key.encode())
        return val.decode() if val is not None else default

    codes = list(_m("efcc18_codes"))
    nside = int(_m("efcc18_nside", "64"))
    npix = tbl.num_rows
    table = np.empty((len(codes), npix, 4), dtype=np.float64)
    for col, code in enumerate(codes):
        for k, field in enumerate(("dRA", "dDEC", "pmRA", "pmDEC")):
            table[col, :, k] = tbl.column(f"{code}_{field}").to_numpy(
                zero_copy_only=False
            )
    return Efcc18Bias(
        nside=nside,
        codes=codes,
        code_to_col={c: i for i, c in enumerate(codes)},
        table=table,
        bias_version=_m("efcc18_bias_version"),
        reference_frame=_m("efcc18_reference_frame", "Gaia-DR2"),
    )


def build_efcc18_lookup(
    efcc18_dir: Path = DEFAULT_EFCC18_DIR,
    source_tgz: Path = DEFAULT_SOURCE_TGZ,
) -> Efcc18Bias:
    """Build (or load a cached) EFCC18 lookup keyed on (catalog code, tile).

    Resolution order, making the call idempotent and the on-disk
    ``data/efcc18/`` a regenerable cache:

    1. ``efcc18_dir/bias.parquet`` exists  -> load it (fast path).
    2. ``efcc18_dir/{bias,tiles}.dat`` exist -> parse, write artifacts, load.
    3. otherwise extract ``source_tgz`` into ``efcc18_dir`` first, then parse.
    """
    efcc18_dir = Path(efcc18_dir)
    if (efcc18_dir / "bias.parquet").exists():
        logger.info("Loading cached EFCC18 lookup from %s", efcc18_dir / "bias.parquet")
        return _load_from_parquet(efcc18_dir)

    bias_dat = efcc18_dir / "bias.dat"
    tiles_dat = efcc18_dir / "tiles.dat"
    source = str(source_tgz)
    if bias_dat.exists() and tiles_dat.exists():
        logger.info("Parsing EFCC18 .dat files in %s", efcc18_dir)
        source = str(bias_dat)
    else:
        logger.info("Extracting EFCC18 archive %s -> %s", source_tgz, efcc18_dir)
        bias_dat, tiles_dat = _extract_dat_files(efcc18_dir, source_tgz)

    nside, codes, version, bias_table, tiles = _parse_efcc18_dat(bias_dat, tiles_dat)
    bias = Efcc18Bias(
        nside=nside,
        codes=codes,
        code_to_col={c: i for i, c in enumerate(codes)},
        table=bias_table.transpose(1, 0, 2).copy(),  # -> (n_codes, npix, 4)
        bias_version=version,
    )
    _write_artifacts(efcc18_dir, bias, tiles, source)
    logger.info(
        "Built EFCC18 lookup: BIAS_VERSION=%s NSIDE=%d %d catalogs %d tiles",
        bias.bias_version,
        bias.nside,
        len(bias.codes),
        bias.npix,
    )
    return bias


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------
def mjd_to_julian_year(mjd: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert MJD to Julian epoch in years (J2000.0 = 2000.0)."""
    return 2000.0 + (mjd - 51544.5) / 365.25


def _astcat_to_columns(
    astcat: list[str | None], bias: Efcc18Bias
) -> npt.NDArray[np.int64]:
    """Map each astcat name to its column in ``bias.table`` (-1 if no correction)."""
    cols = np.full(len(astcat), -1, dtype=np.int64)
    cache: dict[str | None, int] = {}
    for i, name in enumerate(astcat):
        if name in cache:
            cols[i] = cache[name]
            continue
        code = ASTCAT_TO_MPC_CODE.get(name) if name is not None else None
        col = bias.code_to_col.get(code, -1) if code is not None else -1
        cache[name] = col
        cols[i] = col
    return cols


@dataclass
class CorrectionResult:
    """Per-observation correction output (arrays are parallel to the input)."""

    ra_corrected: npt.NDArray[np.float64]
    dec_corrected: npt.NDArray[np.float64]
    applied: npt.NDArray[np.bool_]
    dra_arcsec: npt.NDArray[np.float64]  # correction subtracted from RA (arcsec)
    ddec_arcsec: npt.NDArray[np.float64]  # correction subtracted from Dec (arcsec)


def correct_observations(
    ra_deg: npt.NDArray[np.float64],
    dec_deg: npt.NDArray[np.float64],
    epoch_year: npt.NDArray[np.float64],
    astcat: list[str | None],
    bias: Efcc18Bias,
) -> CorrectionResult:
    """Apply EFCC18 debiasing to arrays of observations (all degrees / years).

    Rows whose catalog has no EFCC18 correction are returned unchanged with
    ``applied=False`` and zero correction.
    """
    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    epoch_year = np.asarray(epoch_year, dtype=np.float64)
    n = ra_deg.shape[0]

    cols = _astcat_to_columns(astcat, bias)
    valid_pos = np.isfinite(ra_deg) & np.isfinite(dec_deg)
    applied = (cols >= 0) & valid_pos

    dra_arcsec = np.zeros(n, dtype=np.float64)
    ddec_arcsec = np.zeros(n, dtype=np.float64)

    if applied.any():
        idx = np.flatnonzero(applied)
        tile = hp.ang2pix(
            bias.nside,
            ra_deg[idx],
            dec_deg[idx],
            nest=False,
            lonlat=True,
        ).astype(np.int64)
        col = cols[idx]
        d_ra = bias.table[col, tile, 0]
        d_dec = bias.table[col, tile, 1]
        pm_ra = bias.table[col, tile, 2]
        pm_dec = bias.table[col, tile, 3]
        dt = epoch_year[idx] - 2000.0
        cos_dec = np.cos(np.radians(dec_deg[idx]))
        # dRA already carries cos(Dec); divide it out to get the RA-coordinate
        # correction. Both terms are in arcsec.
        dra_arcsec[idx] = (d_ra + dt * pm_ra / 1000.0) / cos_dec
        ddec_arcsec[idx] = d_dec + dt * pm_dec / 1000.0

    ra_corrected = ra_deg - dra_arcsec / 3600.0
    dec_corrected = dec_deg - ddec_arcsec / 3600.0
    return CorrectionResult(
        ra_corrected=ra_corrected,
        dec_corrected=dec_corrected,
        applied=applied,
        dra_arcsec=dra_arcsec,
        ddec_arcsec=ddec_arcsec,
    )


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------
def _coverage_breakdown(
    astcat: list[str | None], applied: npt.NDArray[np.bool_]
) -> dict[str, object]:
    """Build a per-shard coverage report, classifying misses."""
    n = len(astcat)
    n_hit = int(applied.sum())
    reference = 0
    unsupported = 0
    miss_by_cat: Counter[str] = Counter()
    for name, hit in zip(astcat, applied.tolist()):
        if hit:
            continue
        label = name if name is not None else "<None>"
        miss_by_cat[label] += 1
        if name in REFERENCE_FRAME_ASTCATS:
            reference += 1
        else:
            unsupported += 1
    return {
        "n_obs": n,
        "n_hit": n_hit,
        "n_miss": n - n_hit,
        "hit_pct": round(100.0 * n_hit / n, 3) if n else 0.0,
        "n_reference_frame": reference,  # Gaia DR2/DR3 — correct no-op
        "n_unsupported": unsupported,  # outside EFCC18's 26 catalogs
        "miss_by_catalog": dict(miss_by_cat.most_common(20)),
    }


def correct_table(
    table: pa.Table, bias: Efcc18Bias
) -> tuple[pa.Table, dict[str, object]]:
    """Apply EFCC18 to an observation table; return (corrected table, coverage)."""
    ra = table.column("ra").to_numpy(zero_copy_only=False).astype(np.float64)
    dec = table.column("dec").to_numpy(zero_copy_only=False).astype(np.float64)
    astcat = table.column("astcat").to_pylist()

    obstime = table.column("obstime")
    if isinstance(obstime, pa.ChunkedArray):
        obstime = obstime.combine_chunks()
    days = obstime.field("days").to_numpy(zero_copy_only=False).astype(np.float64)
    nanos = obstime.field("nanos").to_numpy(zero_copy_only=False).astype(np.float64)
    mjd = days + nanos / 86_400_000_000_000.0
    epoch_year = mjd_to_julian_year(mjd)

    result = correct_observations(ra, dec, epoch_year, astcat, bias)
    coverage = _coverage_breakdown(astcat, result.applied)

    # Drop-in: ra/dec become the corrected values; observed preserved alongside.
    cols = {name: table.column(name) for name in table.column_names}
    cols["ra"] = pa.array(result.ra_corrected)
    cols["dec"] = pa.array(result.dec_corrected)
    cols["ra_observed"] = pa.array(ra)
    cols["dec_observed"] = pa.array(dec)
    cols["efcc18_applied"] = pa.array(result.applied)
    cols["efcc18_dra_arcsec"] = pa.array(result.dra_arcsec)
    cols["efcc18_ddec_arcsec"] = pa.array(result.ddec_arcsec)

    meta = dict(table.schema.metadata or {})
    meta[b"efcc18_bias_version"] = bias.bias_version.encode()
    meta[b"efcc18_reference_frame"] = bias.reference_frame.encode()
    out = pa.table(cols).replace_schema_metadata(meta)
    return out, coverage


def _discover_shards(input_path: Path) -> list[tuple[str, Path]]:
    """Return (relative-key, observation-parquet) pairs for an input path.

    Handles three shapes: a directory of ``shard_NNN/mpc_observations.parquet``,
    a single ``shard_NNN`` directory, or a single parquet file.
    """
    if input_path.is_file():
        return [(input_path.stem, input_path)]
    shard_obs = sorted(input_path.glob(f"shard_*/{OBS_FILENAME}"))
    if shard_obs:
        return [(p.parent.name, p) for p in shard_obs]
    direct = input_path / OBS_FILENAME
    if direct.exists():
        return [(input_path.name, direct)]
    raise FileNotFoundError(f"No observation parquet found under {input_path}")


def _output_path_for(
    input_path: Path, output_root: Path, key: str, obs_parquet: Path
) -> Path:
    """Mirror the input layout under ``output_root``."""
    if input_path.is_file():
        return output_root / f"{input_path.stem}_efcc18.parquet"
    if obs_parquet.parent != input_path:
        return output_root / key / OBS_FILENAME
    return output_root / OBS_FILENAME


def process_input(
    input_path: Path,
    output_root: Path,
    bias: Efcc18Bias,
    overwrite: bool = False,
) -> dict[str, object]:
    """Process every shard under ``input_path``, writing corrected parquets.

    Idempotent/resumable: a shard whose output parquet already exists is skipped
    unless ``overwrite`` is set. Per-shard coverage is logged and a roll-up
    ``_efcc18_coverage.json`` is written under ``output_root``.
    """
    shards = _discover_shards(input_path)
    output_root.mkdir(parents=True, exist_ok=True)

    per_shard: dict[str, object] = {}
    totals = {"n_obs": 0, "n_hit": 0, "n_reference_frame": 0, "n_unsupported": 0}
    for key, obs_parquet in shards:
        out_path = _output_path_for(input_path, output_root, key, obs_parquet)
        if out_path.exists() and not overwrite:
            logger.info("[%s] skip (exists): %s", key, out_path)
            existing = pq.read_table(out_path)
            # Recompute coverage cheaply from the persisted flag for the roll-up.
            applied = existing.column("efcc18_applied").to_numpy(zero_copy_only=False)
            astcat = existing.column("astcat").to_pylist()
            cov = _coverage_breakdown(astcat, np.asarray(applied, dtype=bool))
        else:
            table = pq.read_table(obs_parquet)
            corrected, cov = correct_table(table, bias)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(corrected, out_path)
        per_shard[key] = cov
        for k in totals:
            totals[k] += int(cov[k])  # type: ignore[arg-type]
        logger.info(
            "[%s] %d obs  hit=%.2f%%  reference=%d  unsupported=%d -> %s",
            key,
            cov["n_obs"],
            cov["hit_pct"],
            cov["n_reference_frame"],
            cov["n_unsupported"],
            out_path,
        )

    n_obs = totals["n_obs"]
    summary = {
        "input": str(input_path),
        "output": str(output_root),
        "bias_version": bias.bias_version,
        "n_shards": len(shards),
        "totals": totals,
        "overall_hit_pct": round(100.0 * totals["n_hit"] / n_obs, 3) if n_obs else 0.0,
        "per_shard": per_shard,
    }
    (output_root / "_efcc18_coverage.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "DONE: %d shards, %d obs, overall hit=%.2f%% "
        "(reference-frame=%.2f%%, unsupported=%.2f%%)",
        summary["n_shards"],
        n_obs,
        summary["overall_hit_pct"],
        100.0 * totals["n_reference_frame"] / n_obs if n_obs else 0.0,
        100.0 * totals["n_unsupported"] / n_obs if n_obs else 0.0,
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply EFCC18 catalog debiasing to MPC astrometry."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Sharded observation parquet path (dir or file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (default: <input>_efcc18 alongside the input).",
    )
    parser.add_argument(
        "--efcc18-dir", type=Path, default=DEFAULT_EFCC18_DIR, help="EFCC18 cache dir."
    )
    parser.add_argument(
        "--source-tgz",
        type=Path,
        default=DEFAULT_SOURCE_TGZ,
        help="EFCC18 archive (debias_2018.tgz).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess shards even if output exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    output_root = args.output or args.input.with_name(args.input.name + "_efcc18")
    bias = build_efcc18_lookup(args.efcc18_dir, args.source_tgz)
    process_input(args.input, output_root, bias, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
