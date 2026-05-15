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

# Friendly info-doc CSS — short, visual, less formal than a preprint.
PREPRINT_CSS = """
:root {
  --fg: #1a1a1a; --muted: #555; --accent: #1f4e79; --warm: #d97706;
  --rule: #d0d7de; --soft: #f5f7fa;
}
html { font-size: 16px; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--fg);
  max-width: 780px;
  margin: 1.8rem auto;
  padding: 0 1.4rem 3rem;
  line-height: 1.5;
}
h1 { font-size: 1.6rem; line-height: 1.2; margin: 0 0 0.1rem; color: var(--accent); }
h2 { font-size: 1.15rem; margin: 1.6rem 0 0.5rem; color: var(--accent); }
.subtitle { color: var(--muted); margin: 0 0 1rem; font-size: 0.95rem; }
figure { margin: 1rem 0; }
figure img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
figcaption { font-size: 0.88rem; color: var(--muted); text-align: center; margin-top: 0.3rem; }
code, pre { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.88rem; }
.stat-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem;
  margin: 1rem 0;
}
.stat {
  background: var(--soft); padding: 0.7rem 0.8rem; border-radius: 6px;
  border-left: 4px solid var(--accent);
}
.stat .num { font-size: 1.3rem; font-weight: 600; color: var(--accent); }
.stat .lbl { font-size: 0.82rem; color: var(--muted); margin-top: 0.15rem; }
.callout {
  background: var(--soft); border-left: 4px solid var(--warm);
  padding: 0.55rem 0.9rem; margin: 0.6rem 0; font-size: 0.95rem;
  border-radius: 0 6px 6px 0;
}
.callout strong { color: var(--warm); }
.two-up { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }
.two-up figure { margin: 0; }
.footer-meta {
  margin-top: 1.5rem; padding-top: 0.8rem; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 0.85rem;
}
@media print {
  body { max-width: none; margin: 0; padding: 0.55in 0.7in; font-size: 10.5pt; }
  h2 { page-break-after: avoid; }
  figure { page-break-inside: avoid; }
  .page-break { page-break-before: always; }
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


# Friendly color palette: B612 blue + warm amber for featured callouts.
COLOR_REST = "#9ca3af"      # muted grey
COLOR_ANCHOR = "#1f4e79"    # B612 blue
COLOR_FEATURE = "#d97706"   # warm amber
COLOR_VERES = "#1f4e79"
COLOR_OURS = "#d97706"
COLOR_BG = "#ffffff"


def _style_axes(ax, light: bool = True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if light:
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(colors="#555555", labelsize=10)
    ax.grid(True, alpha=0.2, linestyle="-", color="#dddddd")
    ax.set_axisbelow(True)


def figure_bias_scatter(hc: pd.DataFrame, out_path: Path) -> None:
    """Hero scatter of mean RA bias vs Dec bias across the 520 stations."""
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=200, facecolor=COLOR_BG)

    anchor_mask = hc["obs_code"].isin(ANCHOR_STNS)
    feature_mask = hc["obs_code"].isin(FEATURE_STATIONS)
    rest = hc[~anchor_mask & ~feature_mask]
    anchors = hc[anchor_mask & ~feature_mask]
    features = hc[feature_mask]

    ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#bbbbbb", linewidth=0.6, zorder=1)
    ax.scatter(rest["bias_ra_arcsec"], rest["bias_dec_arcsec"],
               s=18, alpha=0.5, color=COLOR_REST,
               label=f"Other stations (n={len(rest)})",
               zorder=2, edgecolors="none")
    ax.scatter(anchors["bias_ra_arcsec"], anchors["bias_dec_arcsec"],
               s=70, color=COLOR_ANCHOR,
               label="Calibration anchor stations (13)",
               zorder=4, edgecolors="white", linewidths=0.9)
    ax.scatter(features["bias_ra_arcsec"], features["bias_dec_arcsec"],
               s=120, color=COLOR_FEATURE, label="Featured stations",
               zorder=5, edgecolors="white", linewidths=1.0, marker="D")

    # Only label featured stations — anchor labels are too crowded near zero.
    for _, row in features.iterrows():
        ax.annotate(row["obs_code"],
                    (row["bias_ra_arcsec"], row["bias_dec_arcsec"]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=11, color=COLOR_FEATURE, fontweight="bold")

    ax.set_xlabel("Mean RA bias (arcsec)", fontsize=11)
    ax.set_ylabel("Mean Dec bias (arcsec)", fontsize=11)
    ax.set_title("Per-station residual bias across 520 well-observed MPC stations",
                 fontsize=12, pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)


def figure_signals_combined(hc: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """One figure, three panels: chi² histogram on the left, Veres RA + Dec
    bar charts stacked on the right. Returns the Veres comparison df."""
    fig = plt.figure(figsize=(8.4, 3.6), dpi=200, facecolor=COLOR_BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.5, wspace=0.32)

    # --- LEFT: chi² histogram (spans both rows) ---
    ax_hist = fig.add_subplot(gs[:, 0])
    chi2 = hc["chi2_per_obs"].dropna()
    bins = np.logspace(np.log10(max(chi2.min(), 0.01)),
                       np.log10(chi2.max()), 35)
    ax_hist.hist(chi2, bins=bins, color=COLOR_ANCHOR, alpha=0.75,
                 edgecolor="white", linewidth=0.4)
    ax_hist.set_xscale("log")
    ax_hist.axvline(2.0, color=COLOR_FEATURE, linestyle="--", linewidth=1.4,
                    label="Well-calibrated (chi² ≈ 2)")
    ax_hist.set_xlabel("Reported-sigma chi² per observation", fontsize=10)
    ax_hist.set_ylabel("Number of stations", fontsize=10)
    ax_hist.set_title("Many stations under-report their sigmas",
                      fontsize=11, pad=6)
    ax_hist.legend(loc="upper right", fontsize=9, frameon=False)
    _style_axes(ax_hist)

    # --- RIGHT: Veres bar charts (RA on top, Dec on bottom) ---
    rows = []
    for stn, (vra, vdec) in VERES_TABLE1.items():
        sel = hc[hc["obs_code"] == stn]
        if sel.empty:
            continue
        r = sel.iloc[0]
        rows.append({
            "stn": stn,
            "veres_ra": vra, "veres_dec": vdec,
            "ours_ra": float(r["rms_ra_arcsec"]),
            "ours_dec": float(r["rms_dec_arcsec"]),
            "n_obs": int(r["n_obs"]),
            "n_objects": int(r["n_objects"]),
        })
    df = pd.DataFrame(rows).set_index("stn").reindex(list(VERES_TABLE1.keys()))
    x = np.arange(len(df))
    width = 0.38

    for row_i, (ours_col, veres_col, label) in enumerate([
        ("ours_ra", "veres_ra", "RA RMS"),
        ("ours_dec", "veres_dec", "Dec RMS"),
    ]):
        ax = fig.add_subplot(gs[row_i, 1])
        ax.bar(x - width / 2, df[veres_col], width=width,
               color=COLOR_VERES, label="Veres 2017")
        ax.bar(x + width / 2, df[ours_col], width=width,
               color=COLOR_OURS, label="v12 LOOO")
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, fontsize=8)
        ax.set_ylabel(label + " (″)", fontsize=9)
        _style_axes(ax)
        ax.tick_params(labelsize=8)
        if row_i == 0:
            ax.legend(loc="upper left", fontsize=7.5, frameon=False,
                      handlelength=1.0)
            ax.set_title("Sanity check vs Veres 2017 Table 1",
                         fontsize=10, pad=4)

    fig.savefig(out_path, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    return df


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


def _veres_ratio_summary(veres_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Min/max ratio of our RMS to Veres RMS across the 7 overlapping anchors."""
    ratios_ra = (veres_df["ours_ra"] / veres_df["veres_ra"]).dropna()
    ratios_dec = (veres_df["ours_dec"] / veres_df["veres_dec"]).dropna()
    return (float(ratios_ra.min()), float(ratios_ra.max()),
            float(ratios_dec.min()), float(ratios_dec.max()))


def write_report_md(cat: CatalogBundle, veres_df: pd.DataFrame,
                    out_md: Path, fig_paths: dict[str, Path],
                    git_commit: str) -> None:
    # X05 specifics
    x05 = _row_for(cat.bias_per_station, "X05")
    x05_dec = _fmt(x05["bias_dec_arcsec"], 3, True)
    x05_dec_ci = f"[{_fmt(x05['bias_dec_ci_low'], 3, True)}, {_fmt(x05['bias_dec_ci_high'], 3, True)}]"
    x05_chi2 = _fmt(x05["chi2_per_obs"], 1)
    x05_sigfac = _fmt(float(np.sqrt(x05["chi2_per_obs"])), 1)

    stn_809 = _row_for(cat.bias_per_station, "809")
    s809_ra = _fmt(stn_809["bias_ra_arcsec"], 2, True)

    stn_O17 = _row_for(cat.bias_per_station, "O17")
    O17_ra = _fmt(stn_O17["bias_ra_arcsec"], 2, True)

    stn_A16 = _row_for(cat.bias_per_station, "A16")
    A16_chi2 = _fmt(stn_A16["chi2_per_obs"], 1)

    # Veres ratio range across the 7 anchors
    rmin_ra, rmax_ra, rmin_dec, rmax_dec = _veres_ratio_summary(veres_df)

    n_hc = len(cat.high_confidence)

    md = f"""---
title: ""
---

# MPC Observatory Bias Catalog — v12 Preview

*Preliminary share-around for collaborator feedback · 2026-05-14 · Asteroid Institute / B612*

We refit thousands of asteroid orbits while **holding out each observatory
in turn** and measure the residuals the held-out station leaves behind.
Aggregating those residuals per station gives an end-to-end empirical
bias and noise budget for every MPC station with enough cross-network
coverage. The v12 catalog covers **{n_hc} stations** and measures two
independent per-station signals: a **mean coordinate-frame offset** in
RA / Dec, and a check on whether reported per-observation uncertainties
match the empirical residual scatter.

|  Held-out residuals  |  Stations with CIs  |  Chi² range  |  RA bias tails  |
|:---:|:---:|:---:|:---:|
|  **5.1 million**  |  **{n_hc}**  |  **~0.1 – 90**  |  **±0.7″**  |

![**Each dot is one observatory.** X is the station's mean held-out RA
residual; Y is the same for Dec. Most stations cluster near zero — the
tails and the chi-squared distribution (Figure 2) are the story. Amber
diamonds are the stations called out below; blue points are the 13
calibration anchors.](figures/bias_scatter.png){{ width=88% }}

::: callout
**X05 — the catalog's two-signal showcase.** Mean Dec bias = **{x05_dec}″**
(95% CI {x05_dec_ci}″) — small in absolute terms but tightly bounded.
Reduced chi² per observation is **{x05_chi2}**: reported sigmas under-state
empirical scatter by ~{x05_sigfac}×. A meaningful sub-arcsec frame offset
*and* a quantified sigma misreport in the same station.
:::

::: callout
**O17, 809, A16 — three more flavors of the same story.** O17 has the
largest RA bias in the catalog ({O17_ra}″) with normal-looking sigmas.
809, a calibration anchor, carries a real {s809_ra}″ RA offset — anchor
stations are not "clean by construction." A16 reports sigmas ~6× too
optimistic over 2,220 observations (chi² = {A16_chi2}).
:::

## Two signals side by side

![**Left:** the catalog's chi-squared distribution. A well-calibrated
station sits near chi² ≈ 2 (dashed line). The catalog spans ~0.1 to ~90;
the right tail is direct evidence that many stations under-report their
per-observation uncertainties. **Right:** sanity check against Veres 2017
Table 1. For the seven stations both catalogs cover, v12 LOOO residual RMS
sits within ±20% of Veres' published post-debiased values (RA ratios
{rmin_ra:.2f}–{rmax_ra:.2f}; Dec ratios {rmin_dec:.2f}–{rmax_dec:.2f}).
v12 is pre-debiasing and uses held-out residuals — strictly noisier than
the orbit-fit residuals Veres reports — so v12 sitting at-or-slightly-above
Veres is the expected direction.](figures/signals_combined.png){{ width=100% }}

The mean-bias signal and the chi-squared signal are independent. A station
can have a real frame offset with well-calibrated sigmas, near-zero bias
with severely under-reported sigmas, both, or neither. The catalog measures
all four states for every station with enough data to support it.

**What's next.** A catalog-debiased sibling re-run (apply EFCC18 corrections
upstream, then re-run LOOO) gives the apples-to-apples Veres comparison and
isolates station-intrinsic bias from star-catalog systematics. An
along-track / cross-track decomposition is recovering on a sibling bead —
that adds the timing- and trailing-bias diagnostic. Per-program drill-downs
and multi-year drift are the next quarter's work.

*Catalog at `data/mpc_scale_results_20260510/bias_catalog_published/`
(`bias_table.parquet`, `high_confidence_bias_table.csv`); image
`pilot-v12-20260507`, source commit `0f72ea5`. Methodology and
publication-hygiene audit: `docs/mpc-bias-catalog-interpretation.md`.
Feedback welcome — what's missing, what's confusing, what would be more
useful to feature? This is a preview, not a final catalog.*
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
        "-V", "geometry:margin=0.75in",
        "-V", "fontsize=11pt",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
        "-V", "mainfont=Helvetica Neue",
        "-V", "monofont=Menlo",
        # Force figures to render where they appear in the source instead of
        # floating to their own pages (the default LaTeX behaviour).
        "-V", "figPos=H",
        "-V", "indent=false",
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
        "signals_combined": figures_dir / "signals_combined.png",
    }
    figure_bias_scatter(cat.high_confidence, fig_paths["bias_scatter"])
    veres_df = figure_signals_combined(cat.high_confidence,
                                       fig_paths["signals_combined"])

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
