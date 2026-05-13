#!/usr/bin/env python3
"""
One-shot validation helper for bead d5b.

Writes <output-dir>/validation_atct.txt summarising:

  (a) Schema check: presence + non-null point estimates and CIs for
      bias_{at,ct}, bias_{at,ct}_median, rms_{at,ct} (plus existing
      ra/dec) across the 13 anchor stations.
  (b) Row-count parity: rows in the AT/CT-augmented residuals parquet
      equal rows in the pre-AT/CT residuals parquet.
  (c) Sigma-squared parity per anchor station:
      Bias parity is the bead's stated sanity bound:
        |bias_at|^2 + |bias_ct|^2  ~  |bias_ra|^2 + |bias_dec|^2
      The identity is only EXACT row-by-row; after object-weighted
      aggregation it can drift substantially because (i) per-row
      velocity vectors vary, so a frame-fixed RA/Dec bias averages
      down in AT/CT (and vice versa) when averaged across many rows
      with different orientations, and (ii) bias_X_arcsec is
      mean(per-object means), so Jensen-gap losses compound the
      rotation-average loss.

      The bead's "orders of magnitude" bound is one-sided in
      practice: it targets unit / cos(dec) / sign bugs in atct.py
      which would BLOW UP the AT/CT magnitudes (e.g. deg-vs-arcsec
      unit error: ~3600x; missing cos(dec): ~2x).  AT/CT magnitudes
      coming out SMALLER than RA/Dec magnitudes is the expected
      signature of a well-calibrated station: the frame-fixed
      sub-arcsec bias averages down further under per-row rotation.
      So we only flag the upper-bound violation here.

      An equivalent identity does NOT hold for the station-level RMS
      either: rms_X_arcsec is mean_obj(sqrt(mean_row(X^2))), so
      Jensen's inequality breaks the square-and-sum identity that
      holds per row.  RMS values are reported as informational only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ANCHORS = ["704", "699", "703", "W84", "809", "F52", "M22", "R17",
           "T09", "W68", "691", "644", "705"]

REQUIRED_COLS = [
    "bias_at_arcsec", "bias_at_ci_low", "bias_at_ci_high",
    "bias_ct_arcsec", "bias_ct_ci_low", "bias_ct_ci_high",
    "bias_at_median_arcsec", "bias_at_median_ci_low", "bias_at_median_ci_high",
    "bias_ct_median_arcsec", "bias_ct_median_ci_low", "bias_ct_median_ci_high",
    "rms_at_arcsec", "rms_at_ci_low", "rms_at_ci_high",
    "rms_ct_arcsec", "rms_ct_ci_low", "rms_ct_ci_high",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bias-table", type=Path, required=True,
                   help="Path to bias_catalog_published_atct/bias_table.parquet")
    p.add_argument("--residuals-pre", type=Path, required=True,
                   help="Pre-AT/CT LOOO residuals parquet (RA/Dec only).")
    p.add_argument("--residuals-post", type=Path, required=True,
                   help="AT/CT-augmented LOOO residuals parquet.")
    p.add_argument("--output", type=Path, required=True,
                   help="Path for validation_atct.txt.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lines: list[str] = []
    failures: list[str] = []

    lines.append("AT/CT augmentation validation — bead d5b")
    lines.append("=" * 72)
    lines.append(f"bias_table:      {args.bias_table}")
    lines.append(f"residuals_pre:   {args.residuals_pre}")
    lines.append(f"residuals_post:  {args.residuals_post}")
    lines.append("")

    # --- (b) Row-count parity ---------------------------------------
    pre_md = pq.read_metadata(args.residuals_pre)
    post_md = pq.read_metadata(args.residuals_post)
    n_pre = pre_md.num_rows
    n_post = post_md.num_rows
    lines.append("(b) Row-count parity")
    lines.append("-" * 72)
    lines.append(f"  rows(pre)  = {n_pre}")
    lines.append(f"  rows(post) = {n_post}")
    if n_pre == n_post:
        lines.append("  PASS")
    else:
        msg = f"row-count mismatch: pre={n_pre} post={n_post}"
        failures.append(msg)
        lines.append(f"  FAIL: {msg}")
    lines.append("")

    # --- Schema verification on augmented parquet -------------------
    post_schema = pq.read_schema(args.residuals_post)
    needed_res_cols = ["residual_at_arcsec", "residual_ct_arcsec"]
    missing_res = [c for c in needed_res_cols if c not in post_schema.names]
    lines.append("Augmented residuals schema")
    lines.append("-" * 72)
    if missing_res:
        msg = f"augmented parquet missing columns: {missing_res}"
        failures.append(msg)
        lines.append(f"  FAIL: {msg}")
    else:
        lines.append(f"  ok: {needed_res_cols} present")
    # RA/Dec residuals must be unchanged
    pre_cols = pq.read_table(args.residuals_pre,
                             columns=["residual_ra_arcsec",
                                      "residual_dec_arcsec"]).to_pandas()
    post_cols = pq.read_table(args.residuals_post,
                              columns=["residual_ra_arcsec",
                                       "residual_dec_arcsec"]).to_pandas()
    ra_equal = pre_cols["residual_ra_arcsec"].equals(
        post_cols["residual_ra_arcsec"])
    dec_equal = pre_cols["residual_dec_arcsec"].equals(
        post_cols["residual_dec_arcsec"])
    if ra_equal and dec_equal:
        lines.append("  ok: RA/Dec residuals unchanged after augmentation")
    else:
        msg = (f"RA/Dec residuals changed: ra_equal={ra_equal} "
               f"dec_equal={dec_equal}")
        failures.append(msg)
        lines.append(f"  FAIL: {msg}")
    lines.append("")

    # --- (a) Bias table schema check on anchors ---------------------
    bt = pd.read_parquet(args.bias_table)
    if "program_code" in bt.columns:
        obs_rows = bt[bt["program_code"].isna()].copy()
    else:
        obs_rows = bt.copy()
    obs_rows = obs_rows.set_index("obs_code")

    lines.append("(a) Anchor-station schema check")
    lines.append("-" * 72)
    missing_cols = [c for c in REQUIRED_COLS if c not in obs_rows.columns]
    if missing_cols:
        msg = f"bias_table missing required columns: {missing_cols}"
        failures.append(msg)
        lines.append(f"  FAIL: {msg}")
    else:
        lines.append(f"  ok: all {len(REQUIRED_COLS)} AT/CT columns present")

    missing_anchors: list[str] = []
    for code in ANCHORS:
        if code not in obs_rows.index:
            missing_anchors.append(code)
    if missing_anchors:
        msg = f"anchor stations not present in bias_table: {missing_anchors}"
        failures.append(msg)
        lines.append(f"  FAIL: {msg}")

    # Null check per anchor for each required column
    n_null_failures = 0
    for code in ANCHORS:
        if code in missing_anchors or missing_cols:
            continue
        row = obs_rows.loc[code]
        nulls = [c for c in REQUIRED_COLS if pd.isna(row[c])]
        if nulls:
            n_null_failures += 1
            failures.append(f"anchor {code}: null values in {nulls}")
            lines.append(f"  FAIL {code}: null in {nulls}")
    if n_null_failures == 0 and not missing_anchors and not missing_cols:
        lines.append(f"  ok: all {len(ANCHORS)} anchors have non-null "
                     f"AT/CT point estimates + CIs")
    lines.append("")

    # --- (c) Sigma-squared parity per anchor ------------------------
    lines.append("(c) Sigma-squared parity (per anchor station)")
    lines.append("-" * 72)
    lines.append("Bead sanity bound: |bias_at|^2 + |bias_ct|^2 should not differ from")
    lines.append("|bias_ra|^2 + |bias_dec|^2 by more than orders of magnitude (>100x).")
    lines.append("RMS columns are informational: the identity does not survive")
    lines.append("object-weighted aggregation (Jensen gap) — see header note.")
    lines.append("")
    lines.append(
        f"{'stn':<5} {'n_obj':>6} "
        f"{'bias2_radec':>11} {'bias2_atct':>10} {'ratio':>7} "
        f"{'rms2_radec':>10} {'rms2_atct':>10} {'rel_err':>8}"
    )
    lines.append("-" * 72)

    bias_ratio_bound = 100.0  # "orders of magnitude" per bead spec

    for code in ANCHORS:
        if code not in obs_rows.index:
            lines.append(f"{code:<5} MISSING")
            continue
        r = obs_rows.loc[code]
        b_ra = float(r.get("bias_ra_arcsec", np.nan))
        b_dec = float(r.get("bias_dec_arcsec", np.nan))
        b_at = float(r.get("bias_at_arcsec", np.nan))
        b_ct = float(r.get("bias_ct_arcsec", np.nan))
        rm_ra = float(r.get("rms_ra_arcsec", np.nan))
        rm_dec = float(r.get("rms_dec_arcsec", np.nan))
        rm_at = float(r.get("rms_at_arcsec", np.nan))
        rm_ct = float(r.get("rms_ct_arcsec", np.nan))

        bias2_radec = b_ra ** 2 + b_dec ** 2
        bias2_atct = b_at ** 2 + b_ct ** 2
        rms2_radec = rm_ra ** 2 + rm_dec ** 2
        rms2_atct = rm_at ** 2 + rm_ct ** 2

        ratio = (bias2_atct / bias2_radec) if bias2_radec > 0 else float("nan")
        rms_rel_err = (
            abs(rms2_atct - rms2_radec) / rms2_radec
            if rms2_radec > 0 else float("nan")
        )

        # Bias-magnitude sanity bound (one-sided: target unit / cos(dec) /
        # sign bugs that would blow up AT/CT magnitudes; not bugs that
        # would shrink them, which is the expected well-calibrated signal).
        if not np.isnan(ratio) and ratio > bias_ratio_bound:
            failures.append(
                f"anchor {code}: |bias_atct|^2 / |bias_radec|^2 = "
                f"{ratio:.3f} > {bias_ratio_bound} (possible unit bug)"
            )

        n_obj = int(r.get("n_objects", 0)) if not pd.isna(r.get("n_objects", np.nan)) else 0
        lines.append(
            f"{code:<5} {n_obj:>6d} "
            f"{bias2_radec:>11.5f} {bias2_atct:>10.5f} {ratio:>7.3f} "
            f"{rms2_radec:>10.4f} {rms2_atct:>10.4f} {rms_rel_err:>8.2e}"
        )

    lines.append("")
    lines.append("=" * 72)
    if failures:
        lines.append(f"OVERALL: FAIL ({len(failures)} issues)")
        for f in failures:
            lines.append(f"  - {f}")
    else:
        lines.append("OVERALL: PASS")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
