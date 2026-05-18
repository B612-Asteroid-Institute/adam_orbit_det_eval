"""Build the v1 LOOO observatory-bias catalog showcase HTML page."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Veres, Farnocchia, Chesley & Chamberlin (2017), Icarus 296, 139-149 (arXiv:1703.03479)
# Table 1: per-station residual RMS for the 13 most-productive CCD surveys,
# computed against multi-apparition JPL orbit fits after FCCT14 catalog debiasing.
# Statistics current as of 2016-02-09 (paper Section 2).
VERES_TABLE1 = {
    "704": {"name": "LINEAR",          "rms_ra": 0.67, "rms_dec": 0.66},
    "G96": {"name": "Mt. Lemmon",      "rms_ra": 0.31, "rms_dec": 0.28},
    "F51": {"name": "Pan-STARRS 1",    "rms_ra": 0.12, "rms_dec": 0.12},
    "703": {"name": "Catalina",        "rms_ra": 0.69, "rms_dec": 0.67},
    "691": {"name": "Spacewatch",      "rms_ra": 0.37, "rms_dec": 0.34},
    "G45": {"name": "SST",             "rms_ra": 0.36, "rms_dec": 0.36},
    "699": {"name": "LONEOS",          "rms_ra": 0.65, "rms_dec": 0.59},
    "644": {"name": "NEAT",            "rms_ra": 0.30, "rms_dec": 0.36},
    "D29": {"name": "Purple Mountain", "rms_ra": 0.50, "rms_dec": 0.47},
    "C51": {"name": "WISE",            "rms_ra": 0.55, "rms_dec": 0.59},
    "E12": {"name": "Siding Spring",   "rms_ra": 0.49, "rms_dec": 0.52},
    "608": {"name": "Haleakala-AMOS",  "rms_ra": 0.72, "rms_dec": 0.85},
    "J75": {"name": "La Sagra",        "rms_ra": 0.42, "rms_dec": 0.39},
}
VERES_STATIONS = list(VERES_TABLE1.keys())

STRICT_MIN_OBS = 100
STRICT_MIN_OBJECTS = 20

RUN_DATE = "2026-05-10"
IMAGE_TAG = "pilot-v12-20260507"


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()[:12]
    except Exception:
        return "(unknown)"


def _format_signed(x: float, prec: int = 4) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{prec}f}"


def _format_plain(x: float, prec: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{prec}f}"


def _format_int(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"


def load_data(catalog_dir: Path, repo_root: Path):
    bt = pd.read_parquet(catalog_dir / "bias_table.parquet")
    station_only = bt[bt["program_code"].isna()].copy().reset_index(drop=True)
    station_only["strict"] = (station_only["n_obs"] >= STRICT_MIN_OBS) & (
        station_only["n_objects"] >= STRICT_MIN_OBJECTS
    )
    station_only["is_veres"] = station_only["obs_code"].isin(VERES_STATIONS)

    obs_stats = pd.read_parquet(
        repo_root / "data" / "mpc_scale_results_20260510" / "observatory_stats_published.parquet"
    )

    merged_path = (
        repo_root / "data" / "mpc_scale_results_20260510" / "merged_looo_results_published.parquet"
    )
    if merged_path.exists():
        meta = pq.read_metadata(merged_path)
        n_residual_rows = meta.num_rows
        objs = pq.read_table(merged_path, columns=["object_id"])
        n_objects = objs.column("object_id").to_pandas().nunique()
    else:
        n_residual_rows = None
        n_objects = None

    with open(catalog_dir / "validation_report.txt") as f:
        validation_report = f.read()

    return {
        "bias_table": bt,
        "station_only": station_only,
        "obs_stats": obs_stats,
        "n_residual_rows": n_residual_rows,
        "n_objects": n_objects,
        "validation_report": validation_report,
    }


def render_scatter(station_only: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=120)
    strict = station_only[station_only["strict"]]
    weak = station_only[~station_only["strict"]]

    ax.scatter(
        weak["bias_ra_arcsec"],
        weak["bias_dec_arcsec"],
        s=8,
        c="#cccccc",
        alpha=0.55,
        edgecolors="none",
        label=f"Looser cutoff ({len(weak)})",
    )
    ax.scatter(
        strict["bias_ra_arcsec"],
        strict["bias_dec_arcsec"],
        s=14,
        c="#2c6fbb",
        alpha=0.65,
        edgecolors="none",
        label=f"Strict cutoff ({len(strict)})",
    )

    ax.axhline(0, color="#888", lw=0.6, ls="--")
    ax.axvline(0, color="#888", lw=0.6, ls="--")
    ax.set_xlabel("Mean RA bias (arcsec)")
    ax.set_ylabel("Mean Dec bias (arcsec)")
    ax.set_title("Per-station mean residual bias\n(v1 LOOO catalog, 1,091 stations)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    # Zoom in to show the bulk of the distribution
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_histograms(station_only: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)
    strict = station_only[station_only["strict"]]

    bins = np.linspace(0, 1.5, 31)

    for ax, col, label in [
        (axes[0], "bias_ra_arcsec", "|mean RA bias|"),
        (axes[1], "bias_dec_arcsec", "|mean Dec bias|"),
    ]:
        ax.hist(
            station_only[col].abs().clip(upper=1.5),
            bins=bins,
            color="#d7d7d7",
            edgecolor="#999",
            label=f"All ({len(station_only)})",
        )
        ax.hist(
            strict[col].abs().clip(upper=1.5),
            bins=bins,
            color="#2c6fbb",
            alpha=0.75,
            edgecolor="#1d4d83",
            label=f"Strict cutoff ({len(strict)})",
        )
        ax.set_xlabel(f"{label} (arcsec; clipped at 1.5)")
        ax.set_ylabel("Stations")
        ax.set_title(label)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Distribution of |mean bias| per station",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _bias_cell(value: float, lo: float, hi: float, prec: int = 4) -> str:
    if pd.isna(value):
        return "<td>—</td>"
    ci = ""
    if not (pd.isna(lo) or pd.isna(hi)):
        ci = f'<span class="ci">[{lo:+.{prec}f}, {hi:+.{prec}f}]</span>'
    return f'<td class="num">{value:+.{prec}f}{ci}</td>'


def _diff_cell(ours: float, ref: float) -> str:
    """Render the signed v1 − Veres diff, outlined in orange/red if the
    proportional deviation (ratio) is outside the methodology-gap band.
    Most v1/Veres ratios fall in ~0.7-1.4; literature stop-threshold is ~2x.
    Absolute-diff thresholds would be wrong because ±0.05″ means very
    different things for F51 (Veres 0.12″) vs 704 (Veres 0.67″).
    """
    if pd.isna(ours) or ref is None or ref == 0:
        return '<td class="num">—</td>'
    d = ours - ref
    r = ours / ref
    if 0.7 <= r <= 1.4:
        cls = "diff ok"
    elif 0.5 <= r < 0.7 or 1.4 < r <= 2.0:
        cls = "diff warn"
    else:
        cls = "diff bad"
    return f'<td class="num {cls}" data-val="{d}">{d:+.3f}</td>'


def render_veres_table(station_only: pd.DataFrame) -> str:
    """Side-by-side comparison of v1 RMS vs Veres 2017 Table 1 RMS."""
    sub = station_only[station_only["is_veres"]].set_index("obs_code")
    rows = []
    for code in VERES_STATIONS:
        v = VERES_TABLE1[code]
        if code not in sub.index:
            continue
        r = sub.loc[code]
        rms_ra = r["rms_ra_arcsec"]
        rms_dec = r["rms_dec_arcsec"]
        ci_ra = (
            f'<span class="ci">[{r["rms_ra_ci_low"]:.3f}, {r["rms_ra_ci_high"]:.3f}]</span>'
            if not pd.isna(r["rms_ra_ci_low"]) else ""
        )
        ci_dec = (
            f'<span class="ci">[{r["rms_dec_ci_low"]:.3f}, {r["rms_dec_ci_high"]:.3f}]</span>'
            if not pd.isna(r["rms_dec_ci_low"]) else ""
        )
        rows.append(
            "<tr>"
            f'<td class="code">{code}</td>'
            f'<td>{html.escape(v["name"])}</td>'
            f'<td class="num">{_format_int(r["n_obs"])}</td>'
            f'<td class="num">{_format_int(r["n_objects"])}</td>'
            f'<td class="num">{_format_plain(rms_ra)}{ci_ra}</td>'
            f'<td class="num ref">{v["rms_ra"]:.2f}</td>'
            + _diff_cell(rms_ra, v["rms_ra"])
            + f'<td class="num">{_format_plain(rms_dec)}{ci_dec}</td>'
            f'<td class="num ref">{v["rms_dec"]:.2f}</td>'
            + _diff_cell(rms_dec, v["rms_dec"])
            + "</tr>"
        )
    body = "\n".join(rows)
    return f"""
<table class="data veres-table">
  <thead>
    <tr>
      <th rowspan="2">Code</th>
      <th rowspan="2">Survey</th>
      <th rowspan="2" data-sort="num">n_obs</th>
      <th rowspan="2" data-sort="num">n_obj</th>
      <th colspan="3" class="grp">RA (″)</th>
      <th colspan="3" class="grp">Dec (″)</th>
    </tr>
    <tr>
      <th data-sort="num">v1 RMS [CI]</th>
      <th data-sort="num">Veres Table 1</th>
      <th data-sort="num">v1 − Veres</th>
      <th data-sort="num">v1 RMS [CI]</th>
      <th data-sort="num">Veres Table 1</th>
      <th data-sort="num">v1 − Veres</th>
    </tr>
  </thead>
  <tbody>
{body}
  </tbody>
</table>
""".strip()


def render_full_table(station_only: pd.DataFrame) -> str:
    rows = []
    # Sort: Veres-covered first, then strict, then by n_obs desc
    df = station_only.copy()
    df["_sort"] = (~df["is_veres"]).astype(int) * 1_000_000 + (~df["strict"]).astype(int) * 10_000
    df = df.sort_values(["_sort", "n_obs"], ascending=[True, False]).reset_index(drop=True)

    for _, r in df.iterrows():
        classes = []
        if r["is_veres"]:
            classes.append("veres")
        if r["strict"]:
            classes.append("strict")
        else:
            classes.append("loose")
        cls = " ".join(classes)
        marker = ""
        if r["is_veres"]:
            marker = '<span class="badge veres-badge" title="in Veres 2017 Table 1">V</span> '
        if r["strict"]:
            strict_badge = '<span class="badge strict-badge" title="passes n_obs ≥ 100 AND n_objects ≥ 20">✓</span>'
        else:
            strict_badge = '<span class="badge loose-badge" title="below strict cutoff">·</span>'
        rows.append(
            f'<tr class="{cls}">'
            f'<td class="code">{marker}{html.escape(str(r["obs_code"]))} {strict_badge}</td>'
            f'<td class="num" data-val="{r["n_obs"]}">{_format_int(r["n_obs"])}</td>'
            f'<td class="num" data-val="{r["n_objects"]}">{_format_int(r["n_objects"])}</td>'
            + _bias_cell(r["bias_ra_arcsec"], r["bias_ra_ci_low"], r["bias_ra_ci_high"])
            + _bias_cell(r["bias_dec_arcsec"], r["bias_dec_ci_low"], r["bias_dec_ci_high"])
            + _bias_cell(r["bias_at_arcsec"], r["bias_at_ci_low"], r["bias_at_ci_high"])
            + _bias_cell(r["bias_ct_arcsec"], r["bias_ct_ci_low"], r["bias_ct_ci_high"])
            + f'<td class="num">{_format_plain(r["rms_ra_arcsec"])}</td>'
            + f'<td class="num">{_format_plain(r["rms_dec_arcsec"])}</td>'
            + f'<td class="num">{_format_plain(r["chi2_per_obs"], 2)}</td>'
            "</tr>"
        )
    body = "\n".join(rows)
    return f"""
<table class="data full-table" id="full-table">
  <thead>
    <tr>
      <th>Code</th>
      <th data-sort="num">n_obs</th>
      <th data-sort="num">n_obj</th>
      <th data-sort="num">mean bias RA (″) [CI]</th>
      <th data-sort="num">mean bias Dec (″) [CI]</th>
      <th data-sort="num">mean bias AT (″) [CI]</th>
      <th data-sort="num">mean bias CT (″) [CI]</th>
      <th data-sort="num">RMS RA (″)</th>
      <th data-sort="num">RMS Dec (″)</th>
      <th data-sort="num">χ²/obs</th>
    </tr>
  </thead>
  <tbody>
{body}
  </tbody>
</table>
""".strip()


CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: #fafafa;
  color: #222;
  margin: 0;
  padding: 0 0 4rem;
  line-height: 1.45;
}
.container { max-width: 1180px; margin: 0 auto; padding: 1.5rem 1.75rem; }
h1 { font-size: 1.7rem; margin: 0 0 0.4rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 0.6rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
.subtitle { color: #666; font-size: 0.95rem; margin-top: 0; }
p { max-width: 78ch; }
a { color: #1d4d83; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.6rem; margin: 1.2rem 0 1rem; }
.tile { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.7rem 0.9rem; }
.tile .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }
.tile .value { font-size: 1.3rem; font-weight: 600; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
.tile .sub { font-size: 0.78rem; color: #777; margin-top: 0.15rem; }
.note {
  background: #f3f6fb; border-left: 3px solid #2c6fbb;
  padding: 0.8rem 1rem; margin: 1rem 0; border-radius: 0 4px 4px 0;
  font-size: 0.95rem;
}
.note p { margin: 0.35rem 0; }
table.data {
  width: 100%; border-collapse: collapse;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.83rem;
  background: #fff;
  border: 1px solid #ddd;
}
table.data th, table.data td { padding: 0.32rem 0.55rem; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }
table.data th { background: #f2f4f7; cursor: pointer; user-select: none; font-weight: 600; position: sticky; top: 0; }
table.data th[data-sort="num"] { text-align: right; }
table.data td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.data td.code { font-weight: 600; }
table.data .ci { display: block; font-size: 0.72rem; color: #777; }
table.data tr.veres { background: #fff7e0; }
table.data tr.veres:hover { background: #ffe9b0; }
table.data tr.missing { background: #f5f5f5; color: #888; }
table.data tr.missing em { font-style: italic; color: #a04040; }
table.data tr.strict { }
table.data tr.loose { color: #888; }
table.data tr.loose .ci { color: #aaa; }
table.data th.grp { text-align: center; background: #e8edf3; border-left: 1px solid #ddd; border-right: 1px solid #ddd; }
table.data td.ref { color: #555; }
table.data td.diff { font-variant-numeric: tabular-nums; color: #333; }
table.data td.diff.ok { color: #333; }
table.data td.diff.warn {
  color: #8a4a00;
  background: #fff1d6;
  outline: 1px solid #e0a040;
  outline-offset: -2px;
  font-weight: 600;
}
table.data td.diff.bad {
  color: #8a1010;
  background: #fde0e0;
  outline: 1px solid #c44040;
  outline-offset: -2px;
  font-weight: 600;
}
.full-table-wrap { max-height: 640px; overflow: auto; border: 1px solid #ddd; border-radius: 4px; }
.full-table-wrap table.data { border: none; }
.badge {
  display: inline-block; font-size: 0.7rem;
  padding: 0 0.35em; border-radius: 8px; vertical-align: middle;
}
.veres-badge { background: #ffe2b3; color: #7a4a00; font-weight: 600; padding: 0 0.4em; }
.strict-badge { background: #d6f0db; color: #205d2d; margin-left: 0.2em; }
.loose-badge { background: #eee; color: #999; margin-left: 0.2em; }
.legend { font-size: 0.85rem; color: #555; margin: 0.5rem 0 0.8rem; }
.legend .swatch { display: inline-block; width: 12px; height: 12px; vertical-align: middle; margin-right: 4px; border-radius: 2px; }
.plots { display: grid; gap: 1rem; grid-template-columns: 1fr; }
.plots img { width: 100%; max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; background: #fff; }
footer { color: #777; font-size: 0.82rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #ddd; }
footer code { background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
.controls { margin: 0.5rem 0; font-size: 0.88rem; }
.controls label { margin-right: 1rem; }
.print-only { display: none; }

/* Below-cutoff stations are hidden by default; toggle adds .show-loose to reveal */
#full-table tbody tr.loose { display: none; }
#full-table.show-loose tbody tr.loose { display: table-row; }

@media print {
  @page { size: Letter portrait; margin: 0.5in 0.45in; }
  body { background: #fff; color: #111; font-size: 9pt; line-height: 1.35; }
  .container { max-width: none; padding: 0; }
  h1 { font-size: 16pt; }
  h2 { font-size: 12pt; page-break-after: avoid; margin-top: 1.2em; }
  p, .note, .legend { max-width: none; }
  a { color: inherit; text-decoration: none; }

  /* Hide interactive bits — they don't work on paper */
  .controls { display: none; }
  .print-only { display: block; }

  /* Tile strip: keep on one line where possible */
  .tiles { grid-template-columns: repeat(5, 1fr); gap: 0.3rem; margin: 0.6rem 0; }
  .tile { padding: 0.35rem 0.45rem; }
  .tile .value { font-size: 11pt; }
  .tile .label, .tile .sub { font-size: 7pt; }

  /* Plots: let them stay together but reasonable */
  .plots img { max-width: 6.5in; page-break-inside: avoid; }

  /* Full table: let it flow across pages, header repeats */
  .full-table-wrap { max-height: none; overflow: visible; border: none; }
  table.data { font-size: 7.5pt; border: none; }
  table.data thead { display: table-header-group; }  /* repeat on every page */
  table.data tr { page-break-inside: avoid; }
  table.data th { position: static; background: #eee; }
  table.data .ci { font-size: 6.5pt; }
  /* Drop the looser-cutoff rows in print regardless of the on-screen toggle */
  #full-table tbody tr.loose { display: none !important; }

  /* Outline-based highlights become borders so they print on B/W too */
  table.data td.diff.warn, table.data td.diff.bad { outline: none; border: 1.5pt solid; }
  table.data td.diff.warn { border-color: #b76a00; }
  table.data td.diff.bad { border-color: #a02020; }

  footer { font-size: 8pt; }
}
"""

SORT_JS = """
(function () {
  function sortTable(table, colIdx, isNumeric, asc) {
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    rows.sort(function (a, b) {
      const aCell = a.cells[colIdx];
      const bCell = b.cells[colIdx];
      let aVal, bVal;
      if (isNumeric) {
        aVal = parseFloat(aCell.getAttribute('data-val') || aCell.textContent.replace(/[^\\d.\\-+eE]/g, ''));
        bVal = parseFloat(bCell.getAttribute('data-val') || bCell.textContent.replace(/[^\\d.\\-+eE]/g, ''));
        if (isNaN(aVal)) aVal = asc ? Infinity : -Infinity;
        if (isNaN(bVal)) bVal = asc ? Infinity : -Infinity;
        return asc ? aVal - bVal : bVal - aVal;
      }
      aVal = aCell.textContent.trim();
      bVal = bCell.textContent.trim();
      return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  }
  document.querySelectorAll('table.data').forEach(function (table) {
    const headers = table.tHead.rows[0].cells;
    Array.from(headers).forEach(function (th, idx) {
      let asc = true;
      th.addEventListener('click', function () {
        const isNumeric = th.getAttribute('data-sort') === 'num';
        sortTable(table, idx, isNumeric, asc);
        asc = !asc;
        Array.from(headers).forEach(function (h) { h.removeAttribute('data-sorted'); });
        th.setAttribute('data-sorted', asc ? 'desc' : 'asc');
      });
    });
  });
  // Toggle: show below-cutoff stations (hidden by default via CSS)
  const toggle = document.getElementById('toggle-loose');
  if (toggle) {
    toggle.addEventListener('change', function () {
      document.getElementById('full-table').classList.toggle('show-loose', toggle.checked);
    });
  }
})();
"""


def build_html(data: dict, git_sha: str, out_path: Path) -> None:
    station_only = data["station_only"]
    n_strict = int(station_only["strict"].sum())
    n_loose = len(station_only)
    n_veres_present = int(station_only["is_veres"].sum())
    n_obs_stats = len(data["obs_stats"])  # 544 strict-cutoff stations in obs_stats

    headline = f"""
<div class="tiles">
  <div class="tile">
    <div class="label">Published stations (strict)</div>
    <div class="value">{n_obs_stats:,}</div>
    <div class="sub">n_obs ≥ 100 AND n_objects ≥ 20</div>
  </div>
  <div class="tile">
    <div class="label">Stations in bias table</div>
    <div class="value">{n_loose:,}</div>
    <div class="sub">looser cutoff (n_obs ≥ 10, n_obj ≥ 3)</div>
  </div>
  <div class="tile">
    <div class="label">Cleaned residual rows</div>
    <div class="value">{(data["n_residual_rows"] or 0):,}</div>
    <div class="sub">held-out RA/Dec per observation</div>
  </div>
  <div class="tile">
    <div class="label">Distinct objects</div>
    <div class="value">{(data["n_objects"] or 0):,}</div>
    <div class="sub">all hold-in fits cross-validated</div>
  </div>
  <div class="tile">
    <div class="label">Run</div>
    <div class="value">{RUN_DATE}</div>
    <div class="sub">{IMAGE_TAG} · <code>{git_sha}</code></div>
  </div>
</div>
""".strip()

    intro = """
<p>
  This page presents the <strong>v1 Leave-One-Observatory-Out (LOOO)</strong>
  observatory-bias catalog: for each minor-planet object in the source set,
  every observatory is held out in turn, the orbit is re-fit on the remaining
  observations, and the held-out residuals are aggregated per station. The
  per-station <em>mean bias</em> over thousands of objects is the novel
  contribution — it is an end-to-end empirical measurement of each
  observatory's astrometric offset that does <em>not</em> depend on the
  station's own reported uncertainty.
</p>
<div class="note">
  <p><strong>How to read this page.</strong> Mean RA / Dec biases are in
  arcseconds. Bootstrap 95% CIs are shown beneath each value. The reduced
  χ²/obs column is computed against each station's <em>reported</em>
  sigmas; values much greater than 1 indicate the station under-reports its
  astrometric uncertainty — this is the <em>signal</em> of the study, not
  noise. The 13 stations covered by <strong>Veres 2017 Table 1</strong>
  (highlighted) are shown in a side-by-side comparison below. See
  <a href="../../docs/mpc-bias-catalog-interpretation.md">docs/mpc-bias-catalog-interpretation.md</a>
  for the full framing.</p>
</div>
""".strip()

    veres_block = render_veres_table(station_only)

    legend = f"""
<div class="legend">
  <span><span class="swatch" style="background:#fff7e0;border:1px solid #ddd"></span>Veres Table 1 station ({n_veres_present}/{len(VERES_STATIONS)} present)</span>
  &nbsp;·&nbsp;
  <span><span class="badge strict-badge">✓</span> Strict cutoff, shown by default ({n_strict} stations)</span>
  &nbsp;·&nbsp;
  <span><span class="badge loose-badge">·</span> Below strict cutoff, hidden by default ({n_loose - n_strict} stations)</span>
</div>
<div class="controls">
  <label><input type="checkbox" id="toggle-loose"> Also show below-cutoff stations (+{n_loose - n_strict} rows)</label>
  <span style="color:#777">· click any column header to sort</span>
</div>
""".strip()

    full_table = render_full_table(station_only)

    footer = f"""
<footer>
  <strong>Provenance.</strong>
  v1 LOOO catalog produced by the cloud pipeline on <strong>{RUN_DATE}</strong>
  with image <code>{IMAGE_TAG}</code>; catalog assembled on branch
  <code>kk/mpc-scale-bias-catalog</code> at <code>{git_sha}</code>.
  Source files under
  <code>adam_orbit_det_eval/data/mpc_scale_results_20260510/bias_catalog_published_atct/</code>
  (rebuilt with <code>dims_populated = ["ra", "dec", "at", "ct"]</code>).
  Hygiene filters (occultation mode, unknown obscodes, small-sample cutoff)
  are documented in <code>publication_hygiene_audit.json</code>.
  Generated by <code>scripts/build_v1_showcase.py</code>.
</footer>
""".strip()

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v1 LOOO observatory-bias catalog</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="container">
  <h1>v1 LOOO observatory-bias catalog</h1>
  <p class="subtitle">Per-station mean astrometric residual bias measured by leave-one-observatory-out cross-validation.</p>
  {intro}

  {headline}

  <h2>Comparison with Veres 2017 Table 1 ({n_veres_present} of {len(VERES_STATIONS)} stations)</h2>
  <p>
    Veres, Farnocchia, Chesley &amp; Chamberlin (2017,
    <em>Icarus</em> 296, 139–149) Table 1 reports per-station residual RMS
    for the 13 most-productive CCD surveys, computed against multi-apparition
    JPL orbit-fits after Farnocchia-Chesley-Chamberlin-Tholen (FCCT14)
    star-catalog debiasing. Of those 13, {n_veres_present} appear in v1 —
    shown below with our per-station RMS placed next to the Veres value and
    the ratio (v1 ÷ Veres). Only <strong>C51 (WISE)</strong> is absent: it
    is a space-based infrared survey and was excluded at the LOOO eligibility
    step (zero rows even in the raw 5.1M-row residual file), consistent with
    the pipeline's space-based-station exclusion.
  </p>
  <div class="note">
    <p><strong>Caveat — partially apples-to-oranges.</strong> Veres residuals
    are <em>orbit-fit</em> residuals from a multi-apparition fit that
    conditioned on the observation, with FCCT14 catalog debiasing applied as
    preprocessing. v1 residuals are <em>held-out</em> LOOO residuals from a
    fit that excluded the observation, with no catalog debiasing. The
    methodology gap predicts v1 RMS should be slightly <em>larger</em> than
    Veres (held-out residuals exceed orbit-fit residuals; skipping FCCT14
    leaves catalog systematics in the residual). Ratios in the ~1.0–1.3 band
    are consistent with that prediction.
    <strong>v1 ratios significantly below 1.0 are unexpected</strong> and
    may indicate the bead-7bt bias filter trimming catastrophic-tail rows
    that Veres retained, station calibration improvements since 2016 (e.g.
    Gaia-anchored re-reductions), or sample-composition effects. The
    <strong>v1 − Veres</strong> column is highlighted only when the
    proportional deviation falls outside the methodology-gap band (v1/Veres
    in ~0.7–1.4): orange for ~0.5–0.7 or 1.4–2.0, red for &lt;0.5 or &gt;2.0.
    Absolute-diff thresholds aren't used because ±0.05″ means very different
    things for F51 (Veres 0.12″) vs 704 (Veres 0.67″).</p>
  </div>
  {veres_block}

  <h2>Overview plots</h2>
  <div class="plots">
    <img src="assets/scatter_ra_vs_dec.png" alt="Scatter of mean RA bias vs mean Dec bias per station">
    <img src="assets/hist_abs_bias.png" alt="Histograms of |mean RA bias| and |mean Dec bias| per station">
  </div>

  <h2>Published stations ({n_strict:,} strict-cutoff &middot; {n_loose:,} total)</h2>
  <p class="print-only" style="margin: 0 0 0.4rem; font-style: italic; color: #555;">
    Print view: showing {n_strict:,} strict-cutoff stations only; the {n_loose - n_strict:,} below-cutoff rows are hidden.
  </p>
  {legend}
  <div class="full-table-wrap">
    {full_table}
  </div>

  {footer}
</div>
<script>
{SORT_JS}
</script>
</body>
</html>
"""
    out_path.write_text(page)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_catalog = (
        repo_root
        / "data"
        / "mpc_scale_results_20260510"
        / "bias_catalog_published_atct"
    )
    default_out = repo_root / "reports" / "v1_showcase"

    p = argparse.ArgumentParser(
        description=(
            "Build a self-contained static HTML showcase page for the v1 LOOO "
            "observatory-bias catalog. Reads bias_table.parquet and writes "
            "index.html + assets/*.png to the output directory."
        )
    )
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=default_catalog,
        help=f"Directory containing bias_table.parquet, default: {default_catalog}",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output directory for index.html and assets/, default: {default_out}",
    )
    args = p.parse_args()

    catalog_dir = args.catalog_dir.resolve()
    out_dir = args.out.resolve()
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v1-showcase] reading catalog: {catalog_dir}")
    data = load_data(catalog_dir, repo_root)
    print(
        f"[v1-showcase]   bias_table rows: {len(data['bias_table'])}; "
        f"station-only rows: {len(data['station_only'])}; "
        f"residual rows: {data['n_residual_rows']}; "
        f"distinct objects: {data['n_objects']}"
    )

    sha = _git_sha(repo_root)
    print(f"[v1-showcase] git sha: {sha}")

    scatter_path = assets_dir / "scatter_ra_vs_dec.png"
    hist_path = assets_dir / "hist_abs_bias.png"
    print(f"[v1-showcase] rendering plots -> {assets_dir}")
    render_scatter(data["station_only"], scatter_path)
    render_histograms(data["station_only"], hist_path)

    out_html = out_dir / "index.html"
    print(f"[v1-showcase] rendering HTML -> {out_html}")
    build_html(data, sha, out_html)
    print(f"[v1-showcase] done. Open: {out_html}")


if __name__ == "__main__":
    main()
