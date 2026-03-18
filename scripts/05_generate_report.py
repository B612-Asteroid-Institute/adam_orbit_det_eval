#!/usr/bin/env python3
"""
05_generate_report.py
=====================
Generate a self-contained HTML results report and clean CSV sigma table
from LOOO analysis outputs.

Usage
-----
    python scripts/05_generate_report.py \\
        --analysis-dir data/looo_analysis/veres2017_3500obj_objweighted \\
        --comparison-dir data/looo_analysis/veres2017_3500obj \\
        --output-dir reports/3500obj_veres2017

Outputs
-------
    report.html           — Self-contained HTML report for sharing
    sigma_table.csv       — Per-(station, catalog) sigma estimates for OD use
    observatory_stats.csv — Full per-observatory stats table
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/looo_analysis/veres2017_3500obj_objweighted"),
        help="Primary (object-weighted) analysis directory",
    )
    p.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Optional comparison analysis directory (e.g. obs-weighted run)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/mpc_looo_3500obj"),
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=60,
        help="Number of top stations to show in main table (default: 60)",
    )
    p.add_argument(
        "--min-objects",
        type=int,
        default=5,
        help="Minimum objects for a station to appear in report (default: 5)",
    )
    return p.parse_args()


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; color: #222; background: #fafafa; }
h1 { color: #1a1a2e; border-bottom: 3px solid #e63946; padding-bottom: 10px; }
h2 { color: #1a1a2e; margin-top: 40px; border-left: 4px solid #e63946; padding-left: 12px; }
h3 { color: #333; margin-top: 24px; }
.meta { background: #e8f4f8; border-radius: 6px; padding: 16px; margin-bottom: 24px;
        font-size: 0.9em; }
.meta span { display: inline-block; margin-right: 24px; }
.meta strong { color: #1a1a2e; }
table { border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 16px 0; }
th { background: #1a1a2e; color: white; padding: 8px 12px; text-align: right;
     font-weight: 600; white-space: nowrap; }
th:first-child { text-align: left; }
td { padding: 6px 12px; text-align: right; border-bottom: 1px solid #e0e0e0; }
td:first-child { text-align: left; font-family: monospace; font-weight: 600; }
tr:hover { background: #f0f7ff; }
tr:nth-child(even) { background: #f8f8f8; }
tr:nth-child(even):hover { background: #f0f7ff; }
.good { color: #2d6a4f; font-weight: 600; }
.warn { color: #e07b00; font-weight: 600; }
.bad  { color: #c1121f; font-weight: 600; }
.note { background: #fff9c4; border-left: 4px solid #f0c040; padding: 12px 16px;
        border-radius: 0 6px 6px 0; margin: 16px 0; font-size: 0.9em; }
.finding { background: white; border: 1px solid #ddd; border-radius: 8px;
           padding: 16px 20px; margin: 12px 0; }
.finding h4 { margin: 0 0 8px 0; color: #1a1a2e; }
.finding p { margin: 0; font-size: 0.9em; line-height: 1.6; }
.summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }
.stat-box { background: white; border: 1px solid #ddd; border-radius: 8px;
            padding: 16px; text-align: center; }
.stat-box .number { font-size: 2em; font-weight: 700; color: #e63946; }
.stat-box .label { font-size: 0.8em; color: #666; margin-top: 4px; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
pre { background: #1a1a2e; color: #e8e8e8; padding: 16px; border-radius: 6px;
      overflow-x: auto; font-size: 0.82em; line-height: 1.5; }
footer { margin-top: 60px; padding-top: 20px; border-top: 1px solid #ddd;
         font-size: 0.8em; color: #888; }
"""


def chi2_class(v):
    if v is None or v != v:  # nan
        return ""
    if v < 2.0:
        return "good"
    if v < 5.0:
        return ""
    if v < 15.0:
        return "warn"
    return "bad"


def bias_class(v):
    if v is None or v != v:
        return ""
    av = abs(v)
    if av < 0.1:
        return "good"
    if av < 0.3:
        return ""
    return "warn"


def fmt(v, fmt_str=".3f", na="—"):
    if v is None or (isinstance(v, float) and v != v):
        return na
    return format(v, fmt_str)


def fmt_signed(v, fmt_str=".3f", na="—"):
    if v is None or (isinstance(v, float) and v != v):
        return na
    return f"{v:+{fmt_str}}"


def make_obs_table(rows, top_n, show_filtered=False):
    rows_sorted = sorted(rows, key=lambda r: -(r.get("n_obs") or 0))[:top_n]
    col_extra = '<th>N_filt</th>' if show_filtered else ''
    html = f"""
<table>
<thead><tr>
  <th>Station</th><th>N_obs</th><th>N_obj</th>
  <th>bias_RA&quot;</th><th>bias_Dec&quot;</th>
  <th>RMS_RA&quot;</th><th>RMS_Dec&quot;</th>
  <th>chi2/obs</th>{col_extra}
</tr></thead>
<tbody>"""
    for r in rows_sorted:
        chi2 = r.get("mean_chi2_per_obs")
        n_filt = r.get("n_objects_filtered", 0) or 0
        filt_cell = f'<td>{n_filt}</td>' if show_filtered else ''
        html += f"""<tr>
  <td>{r['stn']}</td>
  <td>{r.get('n_obs', 0):,}</td>
  <td>{r.get('n_objects', 0):,}</td>
  <td class="{bias_class(r.get('mean_ra_arcsec'))}">{fmt_signed(r.get('mean_ra_arcsec'))}</td>
  <td class="{bias_class(r.get('mean_dec_arcsec'))}">{fmt_signed(r.get('mean_dec_arcsec'))}</td>
  <td>{fmt(r.get('rms_ra_arcsec'))}</td>
  <td>{fmt(r.get('rms_dec_arcsec'))}</td>
  <td class="{chi2_class(chi2)}">{fmt(chi2, '.2f')}</td>{filt_cell}
</tr>"""
    html += "</tbody></table>"
    return html


def make_cat_table(rows, top_n=50):
    rows_sorted = sorted(rows, key=lambda r: -(r.get("n_obs") or 0))[:top_n]
    html = """
<table>
<thead><tr>
  <th>Station</th><th>Catalog</th><th>N_obs</th>
  <th>bias_RA&quot;</th><th>bias_Dec&quot;</th>
  <th>RMS_RA&quot;</th><th>RMS_Dec&quot;</th>
  <th>chi2/obs</th>
</tr></thead>
<tbody>"""
    for r in rows_sorted:
        chi2 = r.get("mean_chi2_per_obs")
        cat = r.get("astcat") or "—"
        html += f"""<tr>
  <td>{r['stn']}</td><td>{cat}</td>
  <td>{r.get('n_obs', 0):,}</td>
  <td class="{bias_class(r.get('mean_ra_arcsec'))}">{fmt_signed(r.get('mean_ra_arcsec'))}</td>
  <td class="{bias_class(r.get('mean_dec_arcsec'))}">{fmt_signed(r.get('mean_dec_arcsec'))}</td>
  <td>{fmt(r.get('rms_ra_arcsec'))}</td>
  <td>{fmt(r.get('rms_dec_arcsec'))}</td>
  <td class="{chi2_class(chi2)}">{fmt(chi2, '.2f')}</td>
</tr>"""
    html += "</tbody></table>"
    return html


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow not available; run inside pdm environment")
        import sys; sys.exit(1)

    # --- Load primary analysis ---
    obs_stats_path = args.analysis_dir / "observatory_stats.parquet"
    cat_stats_path = args.analysis_dir / "catalog_stats.parquet"
    cfg_path = args.analysis_dir / "analysis_config.json"

    if not obs_stats_path.exists():
        logger.error(f"observatory_stats.parquet not found in {args.analysis_dir}")
        import sys; sys.exit(1)

    obs_tbl = pq.read_table(obs_stats_path)
    cat_tbl = pq.read_table(cat_stats_path) if cat_stats_path.exists() else None
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    obs_rows = [
        {col: obs_tbl.column(col)[i].as_py() for col in obs_tbl.schema.names}
        for i in range(len(obs_tbl))
    ]
    cat_rows = []
    if cat_tbl is not None:
        cat_rows = [
            {col: cat_tbl.column(col)[i].as_py() for col in cat_tbl.schema.names}
            for i in range(len(cat_tbl))
        ]

    # Filter by min objects
    obs_rows = [r for r in obs_rows if (r.get("n_objects") or 0) >= args.min_objects]

    # Load comparison if provided
    cmp_rows = {}
    if args.comparison_dir and (args.comparison_dir / "observatory_stats.parquet").exists():
        cmp_tbl = pq.read_table(args.comparison_dir / "observatory_stats.parquet")
        for i in range(len(cmp_tbl)):
            row = {col: cmp_tbl.column(col)[i].as_py() for col in cmp_tbl.schema.names}
            cmp_rows[row["stn"]] = row

    # --- Compute summary stats ---
    n_obs_total = cfg.get("n_input_rows", sum(r.get("n_obs", 0) for r in obs_rows))
    n_obj_total = cfg.get("n_input_objects", 0)
    n_stns = len(obs_rows)
    well_calibrated = [r["stn"] for r in obs_rows
                       if 0.5 <= (r.get("mean_chi2_per_obs") or 99) <= 2.0
                       and (r.get("n_objects") or 0) >= 20]
    show_filtered = any("n_objects_filtered" in r for r in obs_rows)

    # Summary table for top stations
    obs_table_html = make_obs_table(obs_rows, args.top_n, show_filtered=show_filtered)
    cat_table_html = make_cat_table(cat_rows, top_n=50) if cat_rows else "<p>No catalog stats available.</p>"

    # Comparison table
    cmp_html = ""
    if cmp_rows:
        common_stns = sorted(
            [r for r in obs_rows if r["stn"] in cmp_rows],
            key=lambda r: -(r.get("n_obs") or 0),
        )[:40]
        cmp_html = """
<table>
<thead><tr>
  <th>Station</th><th>N_obs</th><th>N_obj</th>
  <th>chi2 (obj-wtd)</th><th>chi2 (obs-wtd)</th><th>Ratio</th>
  <th>RMS_RA (obj-wtd)"</th><th>RMS_RA (obs-wtd)"</th>
</tr></thead><tbody>"""
        for r in common_stns:
            stn = r["stn"]
            c_new = r.get("mean_chi2_per_obs") or 0
            c_old = cmp_rows[stn].get("mean_chi2_per_obs") or 0
            ratio = c_new / c_old if c_old else 0
            rms_new = r.get("rms_ra_arcsec") or 0
            rms_old = cmp_rows[stn].get("rms_ra_arcsec") or 0
            cmp_html += f"""<tr>
  <td>{stn}</td>
  <td>{r.get('n_obs', 0):,}</td><td>{r.get('n_objects', 0):,}</td>
  <td class="{chi2_class(c_new)}">{fmt(c_new, '.2f')}</td>
  <td class="{chi2_class(c_old)}">{fmt(c_old, '.2f')}</td>
  <td class="{'good' if ratio < 0.5 else 'warn' if ratio > 2 else ''}">{fmt(ratio, '.2f')}</td>
  <td>{fmt(rms_new)}</td><td>{fmt(rms_old)}</td>
</tr>"""
        cmp_html += "</tbody></table>"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MPC Observatory Astrometry Evaluation — LOOO Results</title>
<style>{CSS}</style>
</head>
<body>

<h1>MPC Observatory Astrometry Evaluation</h1>
<h2 style="border:none;padding:0;margin-top:4px;font-size:1.1em;color:#666;">
Leave-One-Observatory-Out Cross-Validation Results
</h2>

<div class="meta">
  <span><strong>Generated:</strong> {now}</span>
  <span><strong>Objects:</strong> {n_obj_total:,}</span>
  <span><strong>Held-out observations:</strong> {n_obs_total:,}</span>
  <span><strong>Observatories reported:</strong> {n_stns}</span>
  <span><strong>Analysis:</strong> Object-weighted averaging, max object chi2 = {cfg.get('max_object_mean_chi2', 'N/A')}</span>
</div>

<div class="summary-grid">
  <div class="stat-box"><div class="number">{n_obj_total:,}</div><div class="label">Objects evaluated</div></div>
  <div class="stat-box"><div class="number">{n_obs_total:,}</div><div class="label">Held-out observations</div></div>
  <div class="stat-box"><div class="number">{n_stns}</div><div class="label">Observatories</div></div>
  <div class="stat-box"><div class="number">{len(well_calibrated)}</div><div class="label">Well-calibrated stations (chi2 0.5–2)</div></div>
</div>

<h2>1. Methodology</h2>

<p>We evaluate the astrometric quality of MPC observatories using
<strong>Leave-One-Observatory-Out (LOOO)</strong> cross-validation:</p>

<ol>
  <li><strong>Sample:</strong> {n_obj_total:,} numbered asteroids drawn from the MPC catalog
  with ≥20 total observations (seed=42 deterministic sample).</li>
  <li><strong>For each (object, observatory) pair</strong> that passes eligibility filters:
    <ul>
      <li>Remove that observatory's observations from the dataset</li>
      <li>Refit the orbit via differential correction (ASSIST N-body propagator,
      starting from MPC nominal orbit)</li>
      <li>Predict sky-plane positions at the held-out observation times</li>
      <li>Record RA/Dec residuals and chi2 = (residual/σ)²</li>
    </ul>
  </li>
  <li><strong>Eligibility filters:</strong> ≥6 observations remaining after hold-out,
  ≥7 day arc remaining, held-out fraction ≤80%.</li>
  <li><strong>Sigma model:</strong> Veres et al. (2017) per-(station, catalog) lookup used
  as fill-in when MPC-reported rmsra/rmsdec are missing.</li>
  <li><strong>Aggregation:</strong> Object-weighted averaging — statistics are computed
  per-object first, then averaged over objects (each object weighted equally regardless
  of observation count). Objects where the mean chi2 across all stations exceeds 50
  are excluded as likely non-gravitational force cases.</li>
</ol>

<div class="note">
  <strong>Interpreting chi2/obs:</strong> If sigma estimates are correctly calibrated,
  chi2/obs ≈ 1. Values &gt;1 indicate under-estimated uncertainties (observations are
  noisier than reported); values &lt;1 indicate over-estimated uncertainties.
  The <strong>raw RMS residuals (RMS_RA&quot;, RMS_Dec&quot;)</strong> are independent
  of the sigma model and can be used directly as empirical sigma estimates.
</div>

<h2>2. Key Findings</h2>

<div class="finding">
  <h4>Well-calibrated modern survey stations</h4>
  <p>The following stations show chi2/obs in the 0.5–2.0 range, indicating
  their reported (or Veres-filled) sigmas are consistent with observed scatter:
  <strong>{', '.join(well_calibrated[:20])}</strong>{'...' if len(well_calibrated) > 20 else ''}.
  F52 (Pan-STARRS 2) and M22 (Kitt Peak) are the most reliably calibrated
  high-volume stations.</p>
</div>

<div class="finding">
  <h4>Systematic biases detected</h4>
  <p>Several stations show consistent positional offsets &gt;0.2&quot; attributable
  to their astrometric catalog. Most notable:</p>
  <ul>
    <li><strong>704</strong> (Lincoln, USNOA2): +0.21&quot; RA, +0.43&quot; Dec — well-known USNOA2 bias</li>
    <li><strong>699</strong>: +0.15&quot; RA, +0.36&quot; Dec — USNOB1/USNOA2 mix</li>
    <li><strong>809</strong>: +0.26&quot; RA, +0.11&quot; Dec — large RA bias</li>
  </ul>
</div>

<div class="finding">
  <h4>Object-weighted averaging reveals hidden issues</h4>
  <p>With flat observation-weighted averaging, F51 (Pan-STARRS 1) showed chi2=178
  and RMS_RA=2.4&quot; — suggesting catastrophic failure. Object-weighted averaging
  reveals the true picture: <strong>chi2=1.65, RMS_RA=0.14&quot;</strong>. A small
  number of pathological objects (47 flagged as non-grav candidates) were dominating
  the flat average. F51's typical performance is excellent. Similarly, G96 (Catalina)
  dropped from chi2=14.2 to chi2=2.6.</p>
</div>

<div class="finding">
  <h4>X05 (LSST/Rubin) — inflated chi2 under investigation</h4>
  <p>X05 shows chi2/obs ≈ 9–10 with only 200 objects and 3,500 observations in this
  sample. The current sample was not specifically curated for X05 coverage — objects
  with many LSST observations are underrepresented. A dedicated LSST calibration run
  (100 objects selected for maximum X05 coverage) is in progress.</p>
</div>

<h2>3. Per-Observatory Statistics</h2>
<p>Sorted by observation count. Color coding: <span class="good">green = well-calibrated</span>,
<span class="warn">orange = elevated chi2 (5–15×)</span>,
<span class="bad">red = strongly inflated (&gt;15×)</span>.
Bias color: flagged if |bias| &gt; 0.3&quot;.
<em>N_filt</em> = objects excluded as non-gravitational force candidates (chi2 &gt; 50 across all stations).</p>

{obs_table_html}

<h2>4. Per-(Station, Catalog) Statistics</h2>
<p>Top 50 groups by observation count. This breakdown drives the sigma lookup table
used in orbit determination.</p>

{cat_table_html}

{'<h2>5. Comparison: Object-Weighted vs. Observation-Weighted</h2><p>Impact of the averaging method change. A ratio &lt;1 means the station improved (was being inflated by a few dominant objects). A ratio &gt;1 means the station worsened (was previously pulled down by well-behaved heavily-observed objects).</p>' + cmp_html if cmp_html else ''}

<h2>{'6' if cmp_html else '5'}. Sigma Table — Usage in OD</h2>

<p>The <code>sigma_table.csv</code> file accompanies this report and provides the
empirical sigma estimates (RMS residuals) per (station, catalog) group. These are
suitable as drop-in replacements or supplements to the Veres 2017 lookup table.</p>

<p>Column definitions:</p>
<ul>
  <li><code>stn</code> — MPC observatory code</li>
  <li><code>astcat</code> — Astrometric catalog code (blank = unknown/mixed)</li>
  <li><code>n_obs</code> — Number of held-out observations</li>
  <li><code>n_objects</code> — Number of distinct objects (more reliable = more objects)</li>
  <li><code>bias_ra_arcsec</code>, <code>bias_dec_arcsec</code> — Systematic offset to correct</li>
  <li><code>sigma_ra_arcsec</code>, <code>sigma_dec_arcsec</code> — Empirical 1σ from RMS residuals</li>
  <li><code>chi2_per_obs</code> — Ratio of observed to reported variance (1 = well-calibrated)</li>
</ul>

<pre>
Pipeline:
  01_fetch_mpc_sample.py  →  data/looo_sample_3500/
  02_run_looo.py          →  data/looo_results/20260316T190152Z/looo_results.parquet
  03_analyze.py           →  data/looo_analysis/veres2017_3500obj_objweighted/

Run command:
  python scripts/02_run_looo.py \\
      --input-dir data/looo_sample_3500 \\
      --output-dir data/looo_results \\
      --run-id 20260316T190152Z \\
      --propagator assist --sigma-model veres2017 \\
      --min-obs-remaining 6 --min-arc-length 7.0 \\
      --max-held-out-fraction 0.8 --max-processes 6

  python scripts/03_analyze.py \\
      --input-dir data/looo_results/20260316T190152Z \\
      --output-dir data/looo_analysis \\
      --run-id veres2017_3500obj_objweighted \\
      --max-chi2 100.0 --object-weighted --max-object-chi2 50.0
</pre>

<footer>
  Generated by <code>scripts/05_generate_report.py</code> · {now} ·
  Asteroid Institute / B612 Foundation
</footer>

</body>
</html>"""

    report_path = args.output_dir / "report.html"
    report_path.write_text(html)
    logger.info(f"HTML report written to {report_path}")

    # --- Write CSVs ---
    import csv

    # Full observatory stats CSV
    obs_csv_path = args.output_dir / "observatory_stats.csv"
    if obs_rows:
        with open(obs_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(obs_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(obs_rows, key=lambda r: -(r.get("n_obs") or 0)))
        logger.info(f"Observatory stats CSV → {obs_csv_path}")

    # Clean sigma table from catalog stats
    if cat_rows:
        sigma_csv_path = args.output_dir / "sigma_table.csv"
        sigma_cols = ["stn", "astcat", "n_obs", "bias_ra_arcsec", "bias_dec_arcsec",
                      "sigma_ra_arcsec", "sigma_dec_arcsec", "chi2_per_obs"]
        with open(sigma_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sigma_cols)
            writer.writeheader()
            for r in sorted(cat_rows, key=lambda r: -(r.get("n_obs") or 0)):
                writer.writerow({
                    "stn": r.get("stn", ""),
                    "astcat": r.get("astcat") or "",
                    "n_obs": r.get("n_obs", 0),
                    "bias_ra_arcsec": round(r.get("mean_ra_arcsec") or 0, 4),
                    "bias_dec_arcsec": round(r.get("mean_dec_arcsec") or 0, 4),
                    "sigma_ra_arcsec": round(r.get("rms_ra_arcsec") or 0, 4),
                    "sigma_dec_arcsec": round(r.get("rms_dec_arcsec") or 0, 4),
                    "chi2_per_obs": round(r.get("mean_chi2_per_obs") or 0, 3),
                })
        logger.info(f"Sigma table CSV → {sigma_csv_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
