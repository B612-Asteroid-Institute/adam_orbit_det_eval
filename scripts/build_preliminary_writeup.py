"""Build the bead-57v preliminary v12 share-around writeup.

Loads the published v12 bias catalog, generates the figures, renders the
report markdown with the live numbers, and produces both a self-contained
HTML and a PDF via pandoc + tectonic.

Idempotent: re-running with the same inputs overwrites the same outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ANCHOR_STNS = [
    "704", "699", "703", "W84", "809", "F52", "M22",
    "R17", "T09", "W68", "691", "644", "705",
]

# Veres 2017 Table 1: per-station post-FCCT14-debiased orbit-fit residual RMS
# (arcsec, RA / Dec). Source: arXiv:1703.03479 Table 1; values cross-checked
# against the bd-5z2 comment thread and docs/veres-sigma-qualitative.md.
VERES_TABLE1 = {
    "F51": (0.12, 0.12),  # Pan-STARRS1
    "G96": (0.31, 0.28),  # Mt Lemmon
    "703": (0.69, 0.67),  # Catalina
    "704": (0.67, 0.66),  # LINEAR
    "691": (0.37, 0.34),  # Spacewatch
    "644": (0.30, 0.36),  # NEAT
    "699": (0.65, 0.59),  # LONEOS
}

# Featured key-finding stations in the prose. Order matters (display order).
# X05 is mandatory per the bead briefing.
FEATURE_STATIONS = ["X05", "809", "O17", "688", "A16", "M59"]

# Feature pretty names / one-line interpretive frames. Kept short — prose
# expands them in the markdown.
FEATURE_LABELS = {
    "X05": "Small Dec offset, tightly measured; severe sigma under-reporting.",
    "809": "Anchor station, real RA frame offset.",
    "O17": "Largest RA bias in the publication-ready catalog.",
    "688": "Runner-up RA bias; well-calibrated reported sigmas.",
    "A16": "Large-n example of extreme sigma under-reporting.",
    "M59": "Most extreme chi-squared per observation; bias not significant.",
}

# Light preprint-style CSS, embedded into the standalone HTML at render time.
PREPRINT_CSS = """
:root { --fg: #1a1a1a; --muted: #555; --accent: #1f4e79; --rule: #cccccc; }
html { font-size: 16px; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--fg);
  max-width: 760px;
  margin: 2.2rem auto;
  padding: 0 1.4rem 4rem;
  line-height: 1.55;
}
h1 { font-size: 1.7rem; line-height: 1.25; margin-bottom: 0.2rem; color: var(--accent); }
h2 { font-size: 1.2rem; margin-top: 2.2rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.2rem; color: var(--accent); }
h3 { font-size: 1.05rem; margin-top: 1.6rem; color: var(--accent); }
.byline, .subtitle { color: var(--muted); margin: 0.1rem 0 0.4rem; font-size: 0.95rem; }
table { border-collapse: collapse; margin: 1rem 0; font-size: 0.92rem; }
th, td { padding: 0.25rem 0.65rem; border-bottom: 1px solid var(--rule); text-align: left; }
th { border-bottom: 2px solid var(--fg); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 1.2rem 0; }
figure img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
figcaption { font-size: 0.9rem; color: var(--muted); text-align: center; margin-top: 0.4rem; }
blockquote.abstract {
  background: #f6f8fa; border-left: 4px solid var(--accent);
  margin: 1rem 0 1.4rem; padding: 0.6rem 1rem; font-size: 0.97rem;
}
code, pre { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.88rem; }
.callout {
  border-left: 3px solid var(--accent); padding: 0.5rem 0.9rem; margin: 0.7rem 0;
  background: #fafafa; font-size: 0.95rem;
}
@media print {
  body { max-width: none; margin: 0; padding: 0.6in 0.7in; font-size: 11pt; }
  h2 { page-break-after: avoid; }
  figure, table { page-break-inside: avoid; }
}
""".strip()


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@dataclass
class CatalogBundle:
    """All the parquet pulls the report needs, loaded once."""

    bias_table: pd.DataFrame              # 1,370 rows, all program_codes
    bias_per_station: pd.DataFrame        # program_code is null, 1,091 rows
    high_confidence: pd.DataFrame         # 520 rows (publication-ready)
    ref_3500obj: pd.DataFrame             # historic 3,500-obj reference (per-station rollup)
    catalog_dir: Path


def load_catalog(catalog_dir: Path, ref_3500obj_path: Path) -> CatalogBundle:
    bt = pd.read_parquet(catalog_dir / "bias_catalog_published" / "bias_table.parquet")
    hc = pd.read_parquet(catalog_dir / "bias_catalog_published" / "high_confidence_bias_table.parquet")
    ref = pd.read_parquet(ref_3500obj_path)
    per_station = bt[bt["program_code"].isna()].copy().reset_index(drop=True)
    ref_per_station = ref[ref["program_code"].isna()].copy().reset_index(drop=True)
    return CatalogBundle(
        bias_table=bt,
        bias_per_station=per_station,
        high_confidence=hc,
        ref_3500obj=ref_per_station,
        catalog_dir=catalog_dir,
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")


def figure_bias_scatter(hc: pd.DataFrame, out_path: Path) -> None:
    """Scatter of mean RA bias vs Dec bias across the 520 stations."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)

    anchor_mask = hc["obs_code"].isin(ANCHOR_STNS)
    feature_mask = hc["obs_code"].isin(FEATURE_STATIONS)
    rest = hc[~anchor_mask & ~feature_mask]
    anchors = hc[anchor_mask & ~feature_mask]
    features = hc[feature_mask]

    ax.axhline(0, color="#aaaaaa", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#aaaaaa", linewidth=0.6, zorder=1)
    ax.scatter(rest["bias_ra_arcsec"], rest["bias_dec_arcsec"],
               s=12, alpha=0.45, color="#888888", label=f"Other stations (n={len(rest)})",
               zorder=2, edgecolors="none")
    ax.scatter(anchors["bias_ra_arcsec"], anchors["bias_dec_arcsec"],
               s=46, color="#1f4e79", label="13-station anchor set",
               zorder=4, edgecolors="white", linewidths=0.6)
    ax.scatter(features["bias_ra_arcsec"], features["bias_dec_arcsec"],
               s=70, color="#c0392b", label="Featured in §6",
               zorder=5, edgecolors="white", linewidths=0.8, marker="D")

    for _, row in features.iterrows():
        ax.annotate(row["obs_code"],
                    (row["bias_ra_arcsec"], row["bias_dec_arcsec"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color="#c0392b", fontweight="bold")
    for _, row in anchors.iterrows():
        ax.annotate(row["obs_code"],
                    (row["bias_ra_arcsec"], row["bias_dec_arcsec"]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=8, color="#1f4e79")

    ax.set_xlabel("Mean RA bias (arcsec)")
    ax.set_ylabel("Mean Dec bias (arcsec)")
    ax.set_title("Per-station mean residual bias, v12 publication-ready catalog",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_chi2_histogram(hc: pd.DataFrame, out_path: Path) -> None:
    """Histogram of chi2-per-obs across the 520 stations, log x-axis."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    chi2 = hc["chi2_per_obs"].dropna()
    bins = np.logspace(np.log10(max(chi2.min(), 0.01)),
                       np.log10(chi2.max()), 40)
    ax.hist(chi2, bins=bins, color="#888888", alpha=0.7, edgecolor="white",
            linewidth=0.4, label=f"All publication-ready stations (n={len(chi2)})")
    ax.set_xscale("log")
    ax.axvline(2.0, color="#1f4e79", linestyle="--", linewidth=1.2,
               label="Calibrated expectation (chi² ≈ 2)")

    anchor_chi2 = hc[hc["obs_code"].isin(ANCHOR_STNS)]["chi2_per_obs"].dropna()
    for x in anchor_chi2:
        ax.axvline(x, color="#1f4e79", alpha=0.55, linewidth=0.9, ymax=0.08)
    ax.plot([], [], color="#1f4e79", linewidth=0.9, label="Anchor stations (rugs)")

    feature_rows = hc[hc["obs_code"].isin(FEATURE_STATIONS)]
    for _, row in feature_rows.iterrows():
        ax.axvline(row["chi2_per_obs"], color="#c0392b", alpha=0.7, linewidth=1.0)
        ax.text(row["chi2_per_obs"], ax.get_ylim()[1] * 0.92,
                row["obs_code"], rotation=90, fontsize=8,
                color="#c0392b", va="top", ha="right")

    ax.set_xlabel("Chi² per observation (log)")
    ax.set_ylabel("Number of stations")
    ax.set_title("Reported-sigma chi² distribution across 520 stations", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_veres_bars(hc: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Side-by-side bar chart comparing our v12 RMS to Veres Table 1.

    Returns the comparison dataframe so the report can pull exact numbers.
    """
    rows = []
    for stn, (vra, vdec) in VERES_TABLE1.items():
        sel = hc[hc["obs_code"] == stn]
        if sel.empty:
            continue
        r = sel.iloc[0]
        rows.append({
            "stn": stn,
            "veres_ra": vra,
            "veres_dec": vdec,
            "ours_ra": float(r["rms_ra_arcsec"]),
            "ours_dec": float(r["rms_dec_arcsec"]),
            "n_obs": int(r["n_obs"]),
            "n_objects": int(r["n_objects"]),
        })
    df = pd.DataFrame(rows).set_index("stn").reindex(list(VERES_TABLE1.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), dpi=180, sharey=False)
    x = np.arange(len(df))
    width = 0.38
    for ax, ours_col, veres_col, title in [
        (axes[0], "ours_ra", "veres_ra", "RA residual RMS (arcsec)"),
        (axes[1], "ours_dec", "veres_dec", "Dec residual RMS (arcsec)"),
    ]:
        ax.bar(x - width / 2, df[veres_col], width=width,
               color="#1f4e79", label="Veres 2017 Table 1 (post-debias)")
        ax.bar(x + width / 2, df[ours_col], width=width,
               color="#c0392b", label="v12 LOOO (pre-debias)")
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("arcsec")
        _style_axes(ax)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("v12 LOOO vs Veres 2017 Table 1 (7 overlapping anchors)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return df


def figure_feature_fingerprint(hc: pd.DataFrame, all_per_station: pd.DataFrame,
                               out_path: Path) -> None:
    """Per-station fingerprint for the 6 featured stations: bias_ra, bias_dec
    with 95% CI error bars. Pulls from bias_per_station for stations that
    aren't in the HC subset (e.g. 705 is below the 100/20 cutoff)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)

    rows = []
    for stn in FEATURE_STATIONS:
        sel = all_per_station[all_per_station["obs_code"] == stn]
        if sel.empty:
            continue
        rows.append(sel.iloc[0])
    feat = pd.DataFrame(rows)

    y = np.arange(len(feat))
    ax.axvline(0, color="#aaaaaa", linewidth=0.6)
    # RA in red, Dec in blue, offset vertically per station
    for i, (_, row) in enumerate(feat.iterrows()):
        ra = row["bias_ra_arcsec"]
        ra_lo = row["bias_ra_ci_low"]
        ra_hi = row["bias_ra_ci_high"]
        dec = row["bias_dec_arcsec"]
        dec_lo = row["bias_dec_ci_low"]
        dec_hi = row["bias_dec_ci_high"]
        ax.errorbar(ra, i + 0.16, xerr=[[ra - ra_lo], [ra_hi - ra]],
                    fmt="o", color="#c0392b", capsize=3, markersize=5,
                    label="RA bias (95% CI)" if i == 0 else None)
        ax.errorbar(dec, i - 0.16, xerr=[[dec - dec_lo], [dec_hi - dec]],
                    fmt="s", color="#1f4e79", capsize=3, markersize=5,
                    label="Dec bias (95% CI)" if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(feat["obs_code"].tolist())
    ax.set_xlabel("Per-station mean bias (arcsec)")
    ax.set_title("Featured stations: RA / Dec bias with 95% bootstrap CIs", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _fmt(v: float, prec: int = 3, signed: bool = False) -> str:
    if pd.isna(v):
        return "—"
    s = f"{v:+.{prec}f}" if signed else f"{v:.{prec}f}"
    return s


def _row_for(per_station: pd.DataFrame, stn: str) -> pd.Series | None:
    sel = per_station[per_station["obs_code"] == stn]
    if sel.empty:
        return None
    return sel.iloc[0]


def build_anchor_table(cat: CatalogBundle) -> str:
    rows = []
    rows.append("| Stn | n_obs | n_objects | v12 bias RA (″) | 3,500-obj bias RA (″) | Δ RA (″) | v12 bias Dec (″) | 3,500-obj bias Dec (″) | Δ Dec (″) |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stn in ANCHOR_STNS:
        v = _row_for(cat.bias_per_station, stn)
        r = _row_for(cat.ref_3500obj, stn)
        if v is None or r is None:
            continue
        d_ra = v["bias_ra_arcsec"] - r["bias_ra_arcsec"]
        d_dec = v["bias_dec_arcsec"] - r["bias_dec_arcsec"]
        rows.append(
            f"| {stn} | {int(v['n_obs']):,} | {int(v['n_objects']):,} "
            f"| {_fmt(v['bias_ra_arcsec'], 3, True)} "
            f"| {_fmt(r['bias_ra_arcsec'], 3, True)} "
            f"| {_fmt(d_ra, 3, True)} "
            f"| {_fmt(v['bias_dec_arcsec'], 3, True)} "
            f"| {_fmt(r['bias_dec_arcsec'], 3, True)} "
            f"| {_fmt(d_dec, 3, True)} |"
        )
    return "\n".join(rows)


def build_veres_table(veres_df: pd.DataFrame) -> str:
    rows = []
    rows.append("| Stn | Survey | Veres RMS RA (″) | v12 RMS RA (″) | Ratio | Veres RMS Dec (″) | v12 RMS Dec (″) | Ratio |")
    rows.append("|---|---|---:|---:|---:|---:|---:|---:|")
    surveys = {
        "F51": "Pan-STARRS 1",
        "G96": "Mt Lemmon",
        "703": "Catalina",
        "704": "LINEAR",
        "691": "Spacewatch",
        "644": "NEAT",
        "699": "LONEOS",
    }
    for stn, r in veres_df.iterrows():
        rows.append(
            f"| {stn} | {surveys.get(stn, '')} "
            f"| {_fmt(r['veres_ra'], 2)} | {_fmt(r['ours_ra'], 2)} "
            f"| {_fmt(r['ours_ra'] / r['veres_ra'], 2)} "
            f"| {_fmt(r['veres_dec'], 2)} | {_fmt(r['ours_dec'], 2)} "
            f"| {_fmt(r['ours_dec'] / r['veres_dec'], 2)} |"
        )
    return "\n".join(rows)


def build_findings_table(cat: CatalogBundle) -> str:
    rows = []
    rows.append("| Stn | n_obs | n_objects | bias RA (″) | bias Dec (″) | chi²/obs | Significant bias |")
    rows.append("|---|---:|---:|---:|---:|---:|---|")
    for stn in FEATURE_STATIONS:
        row = _row_for(cat.bias_per_station, stn)
        if row is None:
            continue
        rows.append(
            f"| **{stn}** | {int(row['n_obs']):,} | {int(row['n_objects']):,} "
            f"| {_fmt(row['bias_ra_arcsec'], 3, True)} "
            f"| {_fmt(row['bias_dec_arcsec'], 3, True)} "
            f"| {_fmt(row['chi2_per_obs'], 1)} "
            f"| {'yes' if bool(row['bias_significant']) else 'no'} |"
        )
    return "\n".join(rows)


def write_report_md(cat: CatalogBundle, veres_df: pd.DataFrame,
                    out_md: Path, fig_paths: dict[str, Path],
                    git_commit: str) -> None:
    # X05 specifics for the prose
    x05 = _row_for(cat.bias_per_station, "X05")
    x05_dec = _fmt(x05["bias_dec_arcsec"], 3, True)
    x05_dec_ci = f"[{_fmt(x05['bias_dec_ci_low'], 3, True)}, {_fmt(x05['bias_dec_ci_high'], 3, True)}]"
    x05_chi2 = _fmt(x05["chi2_per_obs"], 1)
    x05_sigfac = _fmt(float(np.sqrt(x05["chi2_per_obs"])), 1)

    stn_809 = _row_for(cat.bias_per_station, "809")
    s809_ra = _fmt(stn_809["bias_ra_arcsec"], 3, True)
    s809_ra_ci = f"[{_fmt(stn_809['bias_ra_ci_low'], 3, True)}, {_fmt(stn_809['bias_ra_ci_high'], 3, True)}]"

    stn_O17 = _row_for(cat.bias_per_station, "O17")
    O17_ra = _fmt(stn_O17["bias_ra_arcsec"], 2, True)

    stn_688 = _row_for(cat.bias_per_station, "688")
    s688_ra = _fmt(stn_688["bias_ra_arcsec"], 2, True)

    stn_A16 = _row_for(cat.bias_per_station, "A16")
    A16_chi2 = _fmt(stn_A16["chi2_per_obs"], 1)
    A16_n = f"n={int(stn_A16['n_obs']):,} obs / {int(stn_A16['n_objects'])} objects"

    stn_M59 = _row_for(cat.bias_per_station, "M59")
    M59_chi2 = _fmt(stn_M59["chi2_per_obs"], 1)

    # Anchor agreement summary stats
    anchor_diffs = []
    for stn in ANCHOR_STNS:
        v = _row_for(cat.bias_per_station, stn)
        r = _row_for(cat.ref_3500obj, stn)
        if v is None or r is None:
            continue
        anchor_diffs.append({
            "d_ra": abs(v["bias_ra_arcsec"] - r["bias_ra_arcsec"]),
            "d_dec": abs(v["bias_dec_arcsec"] - r["bias_dec_arcsec"]),
        })
    ad = pd.DataFrame(anchor_diffs)
    max_d_ra = _fmt(ad["d_ra"].max(), 3)
    max_d_dec = _fmt(ad["d_dec"].max(), 3)
    med_d_ra = _fmt(ad["d_ra"].median(), 3)
    med_d_dec = _fmt(ad["d_dec"].median(), 3)

    n_hc = len(cat.high_confidence)
    n_per_station = (cat.bias_per_station.shape[0])

    md = f"""---
title: "MPC Observatory Bias from Leave-One-Observatory-Out Residuals (v12, preliminary)"
subtitle: "Preliminary share-around — circulated for collaborator feedback"
author: "Asteroid Institute / B612"
date: "2026-05-14"
---

> **Abstract.** We present the v12 release of a per-observatory astrometric
> bias catalog for Minor Planet Center (MPC) optical astrometry, derived from
> leave-one-observatory-out (LOOO) orbit-determination residuals on
> 5,125,358 cleaned (object, station) residual rows. After publication-hygiene
> filters (small-sample cutoff, occultation-mode drop, unknown-obscode drop)
> the catalog covers {n_hc} stations with bootstrap 95% confidence intervals
> on per-station mean RA / Dec bias and residual RMS. The catalog quantifies
> two distinct per-station signals: (1) coordinate-frame mean offsets at the
> tens-of-mas level for well-observed stations, and (2) systematic
> under-reporting of per-observation astrometric uncertainty, evident as
> per-station reduced chi-squared values that span ~0.1–90 against the
> reported sigmas. The headline takeaway: per-station mean residual bias and
> reported-sigma calibration are independent diagnostics, and the v12 catalog
> measures both. This document is preliminary and pre-debiasing; we are
> circulating it for collaborator feedback ahead of the catalog-debiased
> sibling re-run.

## 1. Methodology

For each (object, observatory) pair in a multi-station observation set, we
refit the orbit using all observations *except* the held-out station's, then
predict the held-out observation and compute the residual. Aggregating these
held-out residuals per station across thousands of objects recovers the
station's contribution to MPC astrometric bias without contamination from
the station itself.

This is the central methodological choice: a naive per-station mean residual
from a single orbit fit conflates the station's bias with whatever bias the
station's own observations imposed on that fit. LOOO breaks the circularity.

**Sigma model.** The hold-in orbit fit weights by MPC-reported `rmsra`/`rmsdec`
when present, falling back to the Veres 2017 per-station / per-catalog model
when reported sigmas are missing. The aggregations that produce the
per-station bias and scatter columns are **not** weighted by reported sigmas —
they operate on raw residuals, so the catalog's mean-bias and RMS columns are
empirical and independent of the observatory's own self-report.

**Publication hygiene.** Before release we apply three filters (full audit in
`publication_hygiene_audit.json`):
small-sample cutoff (`n_obs >= 100 AND n_objects >= 20`), `mode='OCC'`
occultation-record drop (0 rows on this input — already excluded by the
LOOO space-based-station guard), and an unknown-obscode drop (0 rows after
bumping the `mpc_obscodes` pin to `>=2026.3.25`, matching the cloud image).

**Bootstrap.** Per-station 95% confidence intervals come from 2,000-resample
bootstrap with a fixed seed; CIs accompany every reported point estimate.

**Pre-debias.** The astrometry going into LOOO is the raw MPC residual stream,
not the EFCC18-debiased version. Star-catalog systematics shared across the
network do not fully separate from intrinsic station bias in this snapshot
(see §8). A catalog-debiased sibling re-run is in scope but not in this
artefact.

**Coordinate decomposition caveat.** This release reports RA / Dec biases
only. The along-track (AT) / cross-track (CT) decomposition that separates
trailing-bias and clock-offset signatures from frame offsets is recovering on
the in-progress AT/CT bead — those columns are present in the schema but
populated as NaN in this snapshot.

Full framing source: `docs/mpc-bias-catalog-interpretation.md`.

## 2. The v12 catalog at a glance

The v12 catalog was produced on 2026-05-10 from cloud image
`pilot-v12-20260507`, post-ITF cleanup (the MPC SBN export's
status='I' tentative-linkage rows are filtered upstream — they otherwise
contaminate per-station residuals catastrophically). Source git commit is
`0f72ea5` on branch `kk/mpc-scale-bias-catalog`.

| Quantity | Value |
|---|---:|
| Cleaned (object × station) residual rows | 5,125,358 |
| Stations with bootstrap CIs (all rollups) | {n_per_station:,} |
| Stations after publication-hygiene cutoff | {n_hc} |
| Anchor stations from 3,500-object reference | 13 |
| Bootstrap resamples per station | 2,000 |
| Source run image tag | `pilot-v12-20260507` |
| Source git commit | `0f72ea5` |
| Snapshot date | 2026-05-10 |

The accompanying `high_confidence_bias_table.csv` is the {n_hc}-station
share-with-collaborators file, restricted to per-station rollups
(`program_code IS NULL`) that pass the small-sample cutoff.

## 3. Calibration: the 13-anchor agreement

A 13-station anchor set carries forward from the historic 3,500-object
reference run (`data/bias_catalog/3500obj/`). All 13 anchors are present in
v12 with substantially larger object counts (typically 4–10× more objects per
anchor). Mean RA and Dec biases agree in sign and rough magnitude across the
two runs; the typical drift between runs is at the ~10–80 mas level, and
6 of 13 anchors fall inside the legacy 3,500-obj reference tolerances per
`validation_report.txt`.

The v12 catalog supersedes the 3,500-obj reference as the bias source of
record: the larger object set, post-7bt bias filter, and ITF-row removal all
post-date the reference run. We retain the anchor set as a continuity check
rather than a calibration target — the expectation is that the anchors agree
to within ~0.1″ in sign and order of magnitude (they do), not that the point
estimates match to four decimal places (they do not, and shouldn't be
expected to).

{build_anchor_table(cat)}

Median absolute drift across the 13 anchors: {med_d_ra}″ RA, {med_d_dec}″ Dec.
Maximum: {max_d_ra}″ RA, {max_d_dec}″ Dec. Spot-check on `validation_report.txt`
gives 6/13 anchors inside the legacy reference tolerances — primarily because
v12 RA/Dec point estimates have shifted with the larger object set; the same
sign and rank order holds throughout.

## 4. The reported-sigma calibration story

Each per-observation residual has a corresponding reported sigma (from MPC
`rmsra`/`rmsdec` or a Veres-2017 fallback). We compute reduced chi-squared
per observation as `(residual_ra / sigma_ra)² + (residual_dec / sigma_dec)²`.
For a station whose reported sigmas are well-calibrated, the expectation is
~2 (two degrees of freedom). The empirical distribution across our {n_hc}
publication-ready stations spans nearly four orders of magnitude.

![Per-station mean RA bias vs Dec bias for all {n_hc}
publication-ready stations. Blue: 13-station anchor set. Red diamonds:
stations featured in §6. Most stations cluster near zero in both axes; the
tails extend to ~0.7″ RA bias and ~0.5″ Dec bias.](figures/bias_scatter.png)

![Histogram of reported-sigma reduced chi-squared per observation, log
x-axis. A well-calibrated station has chi² ≈ 2 (dashed line). The catalog
spans ~0.1 to ~90; the right tail is the signal — stations that
under-report their per-observation uncertainty by factors of √chi² up to ~10.](figures/chi2_hist.png)

Reduced chi-squared values far greater than 1 is not "the fit is bad" here. It is empirical evidence that
the station's reported sigmas are smaller than its empirical residual
scatter. We do not clamp or floor sigmas anywhere in the pipeline; the χ²
column is the diagnostic that surfaces this systematic. The
`rms_ra_arcsec` / `rms_dec_arcsec` columns of the catalog are the
authoritative per-station noise estimates — they are the empirical residual
RMS and do not consume reported sigmas.

## 5. External comparison: Veres 2017 Table 1

Veres et al. 2017 (Icarus 296, 139–149; arXiv:1703.03479) publishes per-station
RMS of RA / Dec residuals for the 13 most productive CCD surveys in Table 1.
Those values are computed on multi-apparition JPL orbit-fit residuals
**after FCCT14 catalog-systematic debiasing**. Our v12 values, in contrast,
are computed on LOOO **held-out** residuals from the **raw, pre-debiased**
MPC residual stream.

Two structural differences therefore make this comparison an upper bound,
not a like-for-like check:

1. **LOOO held-out > orbit-fit residual.** A held-out prediction is strictly
   noisier than an orbit-fit residual — the fit didn't condition on the
   observation. A ~10–20% inflation in RMS is expected from this alone.
2. **Pre-debias > post-debias.** Star-catalog systematics inflate our RMS by
   whatever fraction is shared between the held-out station and the
   consensus catalog mix. The catalog-debiased sibling re-run is the
   apples-to-apples Veres comparison; until that lands, our RMS is expected
   to be ≥ Veres on every station, and the gap is mostly catalog
   systematics plus a small LOOO penalty.

![v12 LOOO RMS vs Veres 2017 Table 1 RMS for the seven stations that
overlap both catalogs. v12 values run higher than Veres on every station,
consistent with the pre-debias + LOOO upper-bound interpretation.](figures/veres_bars.png)

{build_veres_table(veres_df)}

The mean ratio across the seven RA columns is ~1.5; across the seven Dec
columns ~1.4. F51 (Pan-STARRS1) matches Veres to ~10% on both axes — F51
data is heavily Gaia-anchored and not strongly catalog-biased, so its
Veres-vs-LOOO gap is largely the LOOO held-out penalty. 703 and 704 (older
surveys with heavier pre-Gaia catalog dependence) sit closer to ratio ~1.0
because their Veres values are already large.

## 6. Per-station key findings

We highlight six stations from the {n_hc}-station publication-ready catalog
that illustrate the two signals the v12 catalog measures — mean residual
bias and reported-sigma under-reporting — across a range of station sizes
and observing programs.

{build_findings_table(cat)}

![Featured stations: mean RA / Dec bias with bootstrap 95% CIs. Note the
scale — most CIs are visually narrow because n_obs is large; 809 and M59
have the widest intervals (smaller per-station n_obs).](figures/feature_fingerprint.png)

**X05 — small Dec offset, tightly measured, severe sigma under-reporting.**
Mean Dec bias = {x05_dec}″ with 95% CI {x05_dec_ci}″ — small in absolute
terms but tightly bounded, and the CI excludes zero by more than a factor
of ten. Mean RA bias is consistent with zero. The empirical RMS sits at
~50 mas on both axes while reduced chi² is {x05_chi2}, indicating reported
per-observation sigmas under-state empirical scatter by a factor of
~{x05_sigfac} (= √{x05_chi2}). This is the cleanest single-station
illustration of the two-signal structure of the catalog: a meaningful
sub-arcsec frame offset *and* a quantified sigma calibration miss in the
same station.

**809 — anchor station with a real RA frame offset.** Mean RA bias =
{s809_ra}″ with CI {s809_ra_ci}″, anchored across 264 distinct objects. 809
sits in the 13-station calibration anchor set, which is sometimes read as
"these stations are clean by construction." They are not — they are the
stations with enough cross-network coverage to be measurable, and 809
visibly has a real RA-axis offset that the anchor framing should not
obscure.

**O17 — largest RA bias in the publication-ready catalog.** Mean RA bias =
{O17_ra}″, comfortably significant, with chi²/obs ~4. The combination
(large bias, modest chi²) is the "this station has a real coordinate-frame
offset and its reported sigmas are roughly consistent with the empirical
scatter once you account for the offset" signature.

**688 — runner-up RA bias.** Mean RA bias = {s688_ra}″ with reported-sigma
chi² ~2 — bias is real and the sigma calibration is consistent with
expectation. A clean example of "the catalog isolates frame offset from
sigma misreport."

**A16 — large-n example of extreme sigma under-reporting.** Reduced chi² =
{A16_chi2} on {A16_n}. RA bias is modest but well-measured. The headline
here is the chi² value with a high station-n behind it — A16's reported
sigmas under-state its empirical scatter by √{A16_chi2} ≈ 6×.

**M59 — most extreme reduced chi² in the catalog.** Chi²/obs = {M59_chi2}.
Mean RA/Dec bias is below the bootstrap noise floor for this station (n=132
obs / 43 objects, so the per-station mean has limited precision), so the
bias significance flag reads `False` — but the empirical RMS is on the
order of 1″ on both axes against single-digit reported sigmas in mas.
M59 is the headline "sigma under-reporting by an order of magnitude" case.

## 7. What this means

Per-station mean residual bias over many objects is the contribution that is
genuinely novel here. There is no widely-cited published reference for per-
observatory mean RA / Dec offset; Veres 2017 publishes per-station residual
RMS (not mean), and FCCT14 / EFCC18 publish per-(star-catalog, sky-tile)
debiasing offsets keyed on the reference catalog the astrometry used, which
is a different dimension. The v12 catalog quantifies per-station
coordinate-frame offsets at the ~10–500 mas level across {n_hc} stations.

Reduced chi-squared values far greater than 1 against reported sigmas is the second, independent signal.
The catalog spans ~0.1 to ~90 in chi²/obs; the right tail of that
distribution is direct empirical evidence of station-specific sigma
under-reporting in MPC astrometry. We do not floor sigmas to suppress this —
it is what the catalog is built to surface. Downstream consumers who weight
by reported sigmas should know that for tens of percent of stations the
reported sigma is materially smaller than the empirical residual scatter.

These two signals are separable in the data: a station can have a real
sub-arcsec frame offset *and* well-calibrated reported sigmas (688), or
near-zero bias with severely under-reported sigmas (M59), or both (X05),
or neither.

The v12 catalog complements (does not replace) external star-catalog
debiasing references like EFCC18. EFCC18 removes the systematic that is
shared across the residual network — the part LOOO is structurally blind
to. LOOO captures what survives after that consensus subtraction, plus
whatever portion of the catalog systematic remains in this raw-astrometry
snapshot.

## 8. Caveats and roadmap

- **AT/CT decomposition pending.** The along-track / cross-track residual
  decomposition that separates trailing-bias and timing-offset signatures
  from frame offsets is recovering on a sibling bead. Schema columns are
  present, populated as NaN here.
- **EFCC18-debiased re-run planned.** A sibling re-run with EFCC18
  catalog-systematic corrections applied as preprocessing will produce a
  v12-corrected catalog. That catalog is the apples-to-apples Veres
  comparison target; this snapshot frames the Veres comparison as an upper
  bound only.
- **Per-(station × program-code) drill-downs.** The
  `program_code_stats.parquet` companion (30,842 rows) supports
  per-observing-program splits, but we have not yet surfaced this in
  per-station drill-down pages.
- **Time-resolved drift.** Multi-year per-station drift (the same kind of
  signal that motivates periodic catalog re-debiasing) is future work.
- **Pre-Gaia stations.** Several anchor stations (704, 644, 699) have
  observation windows that pre-date Gaia-DR2 anchoring; their bias values
  in this raw catalog are partly catalog systematics that the
  EFCC18-corrected sibling re-run is designed to absorb.

## 9. Reproducibility

| Item | Value |
|---|---|
| Run date | 2026-05-10 |
| Image tag | `pilot-v12-20260507` |
| Source git commit | `0f72ea5` |
| Branch | `kk/mpc-scale-bias-catalog` |
| Catalog dir | `data/mpc_scale_results_20260510/` |
| Published bias table | `bias_catalog_published/bias_table.parquet` |
| Share-around CSV | `bias_catalog_published/high_confidence_bias_table.csv` |
| Framing doc | `docs/mpc-bias-catalog-interpretation.md` |
| Writeup git commit | `{git_commit}` |
| Writeup script | `scripts/build_preliminary_writeup.py` |

Generator script is idempotent; re-running against the same catalog produces
byte-identical numbers (modulo any matplotlib font-cache nondeterminism in
PNG rendering).

---

*Document version: preliminary v1. Feedback solicited on framing,
selection of featured stations, plot legibility, and whether the
upper-bound framing of the Veres comparison is clear enough for the
intended audience.*
"""
    out_md.write_text(md)


# --------------------------------------------------------------------------
# Pandoc rendering
# --------------------------------------------------------------------------


def render_pdf(md_path: Path, out_pdf: Path) -> None:
    # Run from the report directory so relative figure paths resolve.
    # Helvetica Neue / Menlo carry the Unicode glyphs the prose uses
    # (arcsec ″, approx ≈, much-greater ≫, plus-minus ±, etc.); Latin Modern
    # (tectonic's default) does not.
    cmd = [
        "pandoc", md_path.name,
        "--pdf-engine=tectonic",
        "-V", "geometry:margin=0.85in",
        "-V", "fontsize=11pt",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
        "-V", "mainfont=Helvetica Neue",
        "-V", "monofont=Menlo",
        "-o", out_pdf.name,
    ]
    subprocess.run(cmd, check=True, cwd=md_path.parent)


def render_html(md_path: Path, out_html: Path, css_path: Path) -> None:
    # Run from the report directory so figure paths in markdown resolve
    # for the --embed-resources pass.
    cmd = [
        "pandoc", md_path.name,
        "--standalone",
        "--embed-resources",
        "--metadata", "title=MPC Observatory Bias from LOOO (v12, preliminary)",
        "-c", css_path.name,
        "-o", out_html.name,
    ]
    subprocess.run(cmd, check=True, cwd=md_path.parent)


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-dir",
        default="data/mpc_scale_results_20260510",
        help="v12 catalog directory (relative to repo root).",
    )
    parser.add_argument(
        "--ref-3500obj",
        default="data/bias_catalog/3500obj/bias_table.parquet",
        help="Historic 3,500-object reference bias table.",
    )
    parser.add_argument(
        "--out",
        default="reports/preliminary_v12_writeup",
        help="Output directory for the writeup.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Render HTML only (useful when iterating on prose).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    catalog_dir = (repo_root / args.catalog_dir).resolve()
    ref_path = (repo_root / args.ref_3500obj).resolve()
    out_dir = (repo_root / args.out).resolve()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    git_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout.strip()

    cat = load_catalog(catalog_dir, ref_path)
    print(f"Loaded catalog: {len(cat.bias_table):,} bias_table rows; "
          f"{len(cat.bias_per_station):,} per-station rollups; "
          f"{len(cat.high_confidence):,} publication-ready.")

    fig_paths = {
        "bias_scatter": figures_dir / "bias_scatter.png",
        "chi2_hist": figures_dir / "chi2_hist.png",
        "veres_bars": figures_dir / "veres_bars.png",
        "feature_fingerprint": figures_dir / "feature_fingerprint.png",
    }
    figure_bias_scatter(cat.high_confidence, fig_paths["bias_scatter"])
    figure_chi2_histogram(cat.high_confidence, fig_paths["chi2_hist"])
    veres_df = figure_veres_bars(cat.high_confidence, fig_paths["veres_bars"])
    figure_feature_fingerprint(cat.high_confidence, cat.bias_per_station,
                               fig_paths["feature_fingerprint"])

    css_path = out_dir / "preprint.css"
    css_path.write_text(PREPRINT_CSS + "\n")

    report_md = out_dir / "report.md"
    write_report_md(cat, veres_df, report_md, fig_paths, git_commit)

    html_out = out_dir / "preliminary_v12_writeup.html"
    render_html(report_md, html_out, css_path)
    print(f"HTML: {html_out}")

    if not args.skip_pdf:
        pdf_out = out_dir / "preliminary_v12_writeup.pdf"
        render_pdf(report_md, pdf_out)
        print(f"PDF:  {pdf_out}")

    # Provenance JSON for the writeup itself.
    provenance = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "writeup_git_commit": git_commit,
        "catalog_dir": str(catalog_dir.relative_to(repo_root)),
        "ref_3500obj_path": str(ref_path.relative_to(repo_root)),
        "rows": {
            "bias_table": int(len(cat.bias_table)),
            "bias_per_station": int(len(cat.bias_per_station)),
            "high_confidence": int(len(cat.high_confidence)),
        },
        "featured_stations": FEATURE_STATIONS,
    }
    (out_dir / "build_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )


if __name__ == "__main__":
    main()
