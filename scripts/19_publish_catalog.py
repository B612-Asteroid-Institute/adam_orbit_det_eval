#!/usr/bin/env python3
"""
19_publish_catalog.py
=====================
Single-command "produce the final v2 catalog" entrypoint.

Chains the two previously-manual post-run steps into one atomic, idempotent
operation:

    1. AT/CT decomposition          (scripts/16_atct_real_data.py)
    2. per-observatory bias table   (scripts/17_generate_bias_table.py)

and emits, under ``--output-dir``:

    <output-dir>/looo_results_atct.parquet   # AT/CT-augmented residuals
    <output-dir>/bias_catalog/               # bias_table.parquet + .csv + config

Why this exists (bead 54t / anti-d5b)
-------------------------------------
v1 shipped a published catalog with **empty AT/CT columns** because steps 16
and 17 were two separate manual invocations and the AT/CT step was simply
forgotten (bead d5b).  This wrapper makes "forget to run 16" structurally
impossible: 17 only ever sees the AT/CT-augmented parquet, and the wrapper
hard-aborts if step 16 produced all-null AT/CT columns *before* the bias
table is generated.  So either the catalog has real AT/CT or there is no
catalog at all.

This wrapper does **not** modify scripts 16 or 17 — it shells out to them
unmodified (their CLI is the contract).  It also does not touch the cloud
``LOOOResult`` schema; the anti-forget property is achieved at the publish
boundary, not in the pipeline.

Usage
-----
    python scripts/19_publish_catalog.py \\
        --residuals    data/looo_results/run_001/looo_results.parquet \\
        --observations data/looo_sample/mpc_observations.parquet \\
        --orbits       data/looo_sample/mpc_orbits.parquet \\
        --output-dir   data/bias_catalog/v2_published

Re-running with the same ``--output-dir`` is a no-op unless ``--overwrite``
is passed (idempotent / overwrite-gated).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("publish_catalog")

# Names of the chained scripts, resolved next to this file.
SCRIPTS_DIR = Path(__file__).resolve().parent
ATCT_SCRIPT = SCRIPTS_DIR / "16_atct_real_data.py"
BIAS_TABLE_SCRIPT = SCRIPTS_DIR / "17_generate_bias_table.py"

# Output layout under --output-dir.
AUGMENTED_RESIDUALS_NAME = "looo_results_atct.parquet"
BIAS_CATALOG_DIRNAME = "bias_catalog"

# Columns that step 16 must populate; the d5b guard checks these.
ATCT_COLUMNS = ("residual_at_arcsec", "residual_ct_arcsec")


class PublishError(RuntimeError):
    """Raised when a publish step fails or produces an unusable artifact."""


# ---------------------------------------------------------------------------
# Step wrappers (subprocess delegation to the unmodified scripts 16 / 17)
# ---------------------------------------------------------------------------


def _run_script(script: Path, argv: list[str]) -> None:
    """Invoke a sibling script with the current interpreter, streaming output.

    Raises PublishError on a non-zero exit so the chain fails fast.
    """
    if not script.exists():
        raise PublishError(f"required script not found: {script}")
    cmd = [sys.executable, str(script), *argv]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise PublishError(
            f"{script.name} exited with code {result.returncode} "
            f"(see log above)"
        )


def run_atct_decomposition(
    residuals: Path,
    observations: Path,
    orbits: Path,
    output_path: Path,
    propagator: str = "twobody",
) -> Path:
    """Step 1 — delegate to 16_atct_real_data.py. Returns the augmented path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_script(
        ATCT_SCRIPT,
        [
            "--looo-results", str(residuals),
            "--observations", str(observations),
            "--orbits", str(orbits),
            "--output", str(output_path),
            "--propagator", propagator,
        ],
    )
    if not output_path.exists():
        raise PublishError(
            f"AT/CT step reported success but produced no output: {output_path}"
        )
    return output_path


def assert_atct_populated(augmented_path: Path) -> int:
    """The d5b anti-repeat guard.

    Aborts (PublishError) if the AT/CT columns are missing or entirely null,
    i.e. exactly the failure mode that shipped v1 with empty AT/CT columns.
    Returns the number of rows with at least one non-null AT/CT value.
    """
    schema = pq.read_schema(augmented_path)
    missing = [c for c in ATCT_COLUMNS if c not in schema.names]
    if missing:
        raise PublishError(
            f"d5b guard: augmented residuals {augmented_path} is missing AT/CT "
            f"columns {missing}. Refusing to publish a catalog without AT/CT."
        )

    table = pq.read_table(augmented_path, columns=list(ATCT_COLUMNS))
    n_rows = table.num_rows
    non_null = 0
    for i in range(n_rows):
        if any(table.column(c)[i].is_valid for c in ATCT_COLUMNS):
            non_null += 1

    if non_null == 0:
        raise PublishError(
            "d5b guard: AT/CT columns are present but ALL NULL in "
            f"{augmented_path}. This is the exact v1 failure (bead d5b). "
            "Refusing to generate the bias table. Check that step 16's "
            "propagator/ephemeris ran (e.g. all stations space-based, or "
            "orbits not matching observations)."
        )

    logger.info(
        "d5b guard: AT/CT populated on %d/%d rows (%.1f%%) — OK",
        non_null, n_rows, 100.0 * non_null / max(n_rows, 1),
    )
    return non_null


def run_bias_table(
    augmented_path: Path,
    observations: Path,
    output_dir: Path,
    *,
    n_bootstrap: int | None = None,
    random_seed: int | None = None,
    min_obs_per_group: int | None = None,
    min_objects_per_group: int | None = None,
    max_chi2: float | None = None,
) -> Path:
    """Step 2 — delegate to 17_generate_bias_table.py. Returns the catalog dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--looo-results", str(augmented_path),
        "--observations", str(observations),
        "--output-dir", str(output_dir),
    ]
    if n_bootstrap is not None:
        argv += ["--n-bootstrap", str(n_bootstrap)]
    if random_seed is not None:
        argv += ["--random-seed", str(random_seed)]
    if min_obs_per_group is not None:
        argv += ["--min-obs-per-group", str(min_obs_per_group)]
    if min_objects_per_group is not None:
        argv += ["--min-objects-per-group", str(min_objects_per_group)]
    if max_chi2 is not None:
        argv += ["--max-chi2", str(max_chi2)]

    _run_script(BIAS_TABLE_SCRIPT, argv)

    bias_table = output_dir / "bias_table.parquet"
    if not bias_table.exists():
        raise PublishError(
            f"bias-table step reported success but produced no "
            f"bias_table.parquet in {output_dir}"
        )
    return output_dir


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def publish_catalog(
    residuals: Path,
    observations: Path,
    orbits: Path,
    output_dir: Path,
    *,
    propagator: str = "twobody",
    overwrite: bool = False,
    n_bootstrap: int | None = None,
    random_seed: int | None = None,
    min_obs_per_group: int | None = None,
    min_objects_per_group: int | None = None,
    max_chi2: float | None = None,
) -> dict:
    """Run 16 → (d5b guard) → 17 atomically and return the output paths.

    Returns a dict with ``augmented_residuals``, ``bias_catalog_dir``,
    ``bias_table`` and ``atct_non_null_rows``.
    """
    residuals = Path(residuals)
    observations = Path(observations)
    orbits = Path(orbits)
    output_dir = Path(output_dir)

    for path, label in [
        (residuals, "residuals (LOOO results)"),
        (observations, "observations"),
        (orbits, "orbits"),
    ]:
        if not path.exists():
            raise PublishError(f"{label} not found: {path}")

    augmented_path = output_dir / AUGMENTED_RESIDUALS_NAME
    bias_catalog_dir = output_dir / BIAS_CATALOG_DIRNAME
    bias_table_path = bias_catalog_dir / "bias_table.parquet"

    # --- Idempotency / overwrite gate ---------------------------------
    existing = [p for p in (augmented_path, bias_table_path) if p.exists()]
    if existing and not overwrite:
        raise PublishError(
            "outputs already exist:\n  "
            + "\n  ".join(str(p) for p in existing)
            + "\nPass --overwrite to regenerate the catalog in place."
        )
    if existing:
        logger.info("--overwrite set; regenerating over existing outputs.")

    logger.info("=== Step 1/2: AT/CT decomposition (script 16) ===")
    run_atct_decomposition(
        residuals=residuals,
        observations=observations,
        orbits=orbits,
        output_path=augmented_path,
        propagator=propagator,
    )

    logger.info("=== d5b guard: verifying AT/CT columns are populated ===")
    atct_non_null = assert_atct_populated(augmented_path)

    logger.info("=== Step 2/2: bias-table generation (script 17) ===")
    run_bias_table(
        augmented_path=augmented_path,
        observations=observations,
        output_dir=bias_catalog_dir,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        min_obs_per_group=min_obs_per_group,
        min_objects_per_group=min_objects_per_group,
        max_chi2=max_chi2,
    )

    summary = {
        "augmented_residuals": augmented_path,
        "bias_catalog_dir": bias_catalog_dir,
        "bias_table": bias_table_path,
        "atct_non_null_rows": atct_non_null,
    }
    logger.info("=== Published v2 catalog ===")
    logger.info("  AT/CT-augmented residuals: %s", augmented_path)
    logger.info("  bias catalog directory:    %s", bias_catalog_dir)
    logger.info("  bias_table.parquet:        %s", bias_table_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--residuals", type=Path, required=True,
                   help="LOOO results parquet (output of 02_run_looo.py / "
                        "13_collect_cloud_results.py).")
    p.add_argument("--observations", type=Path, required=True,
                   help="Source MPC observations parquet.")
    p.add_argument("--orbits", type=Path, required=True,
                   help="Original MPC catalog orbits parquet (for AT/CT "
                        "velocity vectors).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for the augmented residuals + bias_catalog/.")
    p.add_argument("--propagator", choices=["twobody", "assist"],
                   default="twobody",
                   help="Propagator backend for AT/CT ephemeris (default: "
                        "twobody).")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate even if outputs already exist.")
    # Pass-through bias-table knobs (defaults live in script 17).
    p.add_argument("--n-bootstrap", type=int, default=None,
                   help="Forwarded to script 17 (default: its own 2000).")
    p.add_argument("--random-seed", type=int, default=None,
                   help="Forwarded to script 17 (default: its own 42).")
    p.add_argument("--min-obs-per-group", type=int, default=None,
                   help="Forwarded to script 17 (default: its own 10).")
    p.add_argument("--min-objects-per-group", type=int, default=None,
                   help="Forwarded to script 17 (default: its own 3).")
    p.add_argument("--max-chi2", type=float, default=None,
                   help="Forwarded to script 17 (default: its own 100).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        publish_catalog(
            residuals=args.residuals,
            observations=args.observations,
            orbits=args.orbits,
            output_dir=args.output_dir,
            propagator=args.propagator,
            overwrite=args.overwrite,
            n_bootstrap=args.n_bootstrap,
            random_seed=args.random_seed,
            min_obs_per_group=args.min_obs_per_group,
            min_objects_per_group=args.min_objects_per_group,
            max_chi2=args.max_chi2,
        )
    except PublishError as e:
        logger.error("PUBLISH FAILED: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
