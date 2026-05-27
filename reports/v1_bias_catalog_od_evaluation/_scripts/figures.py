#!/usr/bin/env python
"""Build results_summary.parquet + figures for the OD-bias-experiments summary report (bead h46).

Synthesis only: reads gitignored per-bead data/ parquets, writes
reports/v1_bias_catalog_od_evaluation/{results_summary.parquet, figures/*.png}. No fits run.

Metric conventions (validated to reproduce each bead's published REPORT.md):
  * Discrepant / stratum medians: converged fits with chi2_in in [0.3, 3]
    (== ~chi2_pathological) for 7en/si3/8d9 (they filter pathological fits).
    cph predates the pathological-flag machinery and reported over ALL converged
    fits, so cph cohorts use apply_filter=False (reproduces its 1.61x headline;
    a couple of cph fits have chi2 up to 5.36 that the filter would otherwise drop).
  * Control regressions: converged fits only (NO chi2 filter), counting objects
    whose variant Dr/sigma > 2x the per-object baseline (veres_only for cph/7en,
    no_bias for si3/8d9).

8d9's published REPORT.md declared GO on a degenerate n=1 control set; this script
recomputes the si3 short-arc regressions over the full n=30 stratum (matching the
user's project-8d9 memory): v1_subtract_sem_inflated regresses 12/30 == v1_subtract.
NO-GO. 8d9 committed at 97a3a60.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # adam_orbit_det_eval/
DATA = ROOT / "data"
OUT = ROOT / "reports" / "v1_bias_catalog_od_evaluation"
FIG = OUT / "figures"

SHA = {
    "zgf": "17026ca", "9d7": "083b636", "9f2": "8ddbcc1", "cph": "359c92c",
    "qsd": "33ff24c", "7en": "aaa7243", "si3": "7e9b9de", "8d9": "97a3a60",
}

CHI2_LO, CHI2_HI = 0.3, 3.0


def _path_flag(df):
    if "chi2_pathological" in df.columns:
        return df["chi2_pathological"].fillna(False)
    return ~df["hold_in_reduced_chi2"].between(CHI2_LO, CHI2_HI)


def _median_block(df, apply_filter=True):
    """median/p95 dr_over_sigma + median chi2 over converged fits.

    apply_filter=True restricts to non-pathological (chi2_in in [0.3,3]); used for
    7en/si3/8d9. cph reported over all converged fits -> apply_filter=False.
    """
    d = df[df["converged"]].copy()
    nonpath = d[~_path_flag(d)] if apply_filter else d
    out = {}
    for v, g in nonpath.groupby("variant"):
        out[v] = dict(
            n_objects=int(g["object_id"].nunique()),
            median_dr_over_sigma=float(g["dr_over_sigma"].median()),
            p95_dr_over_sigma=float(g["dr_over_sigma"].quantile(0.95)),
            median_chi2_in=float(g["hold_in_reduced_chi2"].median()),
        )
    # n_chi2_pathological per variant (over converged)
    npath = d[_path_flag(d)].groupby("variant")["object_id"].nunique().to_dict()
    for v in out:
        out[v]["n_chi2_pathological"] = int(npath.get(v, 0))
    return out


def _regressions(df, baseline, factor=2.0):
    d = df[df["converged"]].copy()
    base = d[d["variant"] == baseline].set_index("object_id")["dr_over_sigma"]
    out = {}
    for v, g in d.groupby("variant"):
        vv = g.set_index("object_id")["dr_over_sigma"]
        common = vv.index.intersection(base.index)
        out[v] = int((vv.loc[common] > factor * base.loc[common]).sum())
    return out


# ---------------------------------------------------------------------------
# Load per-bead data
# ---------------------------------------------------------------------------
cph = pd.read_parquet(DATA / "od_discrepancy_population/variant_comparison.parquet")
en = pd.read_parquet(DATA / "wide_variant_sweep/variant_comparison.parquet")
si3 = pd.read_parquet(DATA / "validation_sweep/variant_comparison.parquet")
d8 = pd.read_parquet(DATA / "sem_inflated_sweep/variant_comparison.parquet")
qsd = pd.read_parquet(DATA / "yr4_experiment/comparison_summary_expanded.parquet")

rows = []


def add_cohort(cohort, bead, subdf, baseline_for_reg=None, apply_filter=True):
    med = _median_block(subdf, apply_filter=apply_filter)
    reg = _regressions(subdf, baseline_for_reg) if baseline_for_reg else {}
    for v, m in med.items():
        rows.append(dict(
            cohort_name=cohort, variant=v, bead_id=bead, commit_sha=SHA[bead],
            n_control_regressions=(reg.get(v) if baseline_for_reg else np.nan),
            **m,
        ))


# YR4 (qsd lineage: zgf->9f2->qsd). Single object: median==value, no regressions.
for _, r in qsd.iterrows():
    rows.append(dict(
        cohort_name="yr4", variant=r["variant"], bead_id="qsd", commit_sha=SHA["qsd"],
        n_objects=1, median_dr_over_sigma=float(r["dr_over_sigma"]),
        p95_dr_over_sigma=float(r["dr_over_sigma"]),
        median_chi2_in=float(r["hold_in_reduced_chi2"]),
        n_control_regressions=np.nan, n_chi2_pathological=np.nan,
    ))

# cph (no chi2 filter -- reproduces published 1.61x discrepant headline)
add_cohort("cph_discrepant", "cph", cph[cph["discrepant_in_9d7"]], apply_filter=False)
add_cohort("cph_controls", "cph", cph[cph["is_control"]], baseline_for_reg="veres_only", apply_filter=False)
# 7en
add_cohort("7en_discrepant", "7en", en[en["is_discrepant_in_cph"]])
add_cohort("7en_controls", "7en", en[en["stratum"] == "short_arc_mod_obs"], baseline_for_reg="veres_only")
add_cohort("7en_mba", "7en", en[en["stratum"] == "main_belt"])
add_cohort("7en_impact_monitor", "7en", en[en["stratum"] == "impact_monitor"])
add_cohort("7en_long_arc", "7en", en[en["stratum"] == "long_arc_well_obs"])
# si3
add_cohort("si3_discrepant", "si3", si3[si3["stratum"].isin(["impact_monitor", "long_arc_well_obs"])])
add_cohort("si3_controls", "si3", si3[si3["stratum"] == "short_arc_mod_obs"], baseline_for_reg="no_bias")
# 8d9 (recomputed over full n=30 control set, not the REPORT.md's degenerate n=1)
add_cohort("8d9_cph_discrepant", "8d9", d8[(d8["source_cohort"] == "cph") & (d8["is_discrepant_in_cph"])])
add_cohort("8d9_si3_controls", "8d9", d8[(d8["source_cohort"] == "si3") & (d8["stratum"] == "short_arc_mod_obs")], baseline_for_reg="no_bias")
add_cohort("8d9_joint", "8d9", d8)

summary = pd.DataFrame(rows)[[
    "cohort_name", "variant", "n_objects", "median_dr_over_sigma", "p95_dr_over_sigma",
    "median_chi2_in", "n_control_regressions", "n_chi2_pathological", "bead_id", "commit_sha",
]]
summary.to_parquet(OUT / "results_summary.parquet", index=False)
print("WROTE results_summary.parquet rows=", len(summary))
print(summary.to_string())

# ===========================================================================
# FIGURES
# ===========================================================================
PRINCIPLED = ["v1_subtract", "v1_performance_weighted", "v1_covar_inflation", "v1_subtract_sem_inflated"]
COLORS = {
    "v1_subtract": "#444444",
    "v1_performance_weighted": "#d62728",
    "v1_covar_inflation": "#1f77b4",
    "v1_subtract_sem_inflated": "#2ca02c",
}
LABEL = {
    "v1_subtract": "v1_subtract (legacy)",
    "v1_performance_weighted": "v1_performance_weighted",
    "v1_covar_inflation": "v1_covar_inflation",
    "v1_subtract_sem_inflated": "v1_subtract_sem_inflated (8d9)",
}


def med(df, variant, baseline="no_bias", apply_filter=True):
    """improvement factor vs no_bias on a discrepant subset."""
    m = _median_block(df, apply_filter=apply_filter)
    if variant not in m or baseline not in m:
        return np.nan
    return m[baseline]["median_dr_over_sigma"] / m[variant]["median_dr_over_sigma"]


# --- FIG 1: improvement factor (vs no_bias) across discrepant cohorts -- winners flip
# tuple: (label, df, is_qsd, apply_chi2_filter)
cohorts_disc = [
    ("YR4\n(qsd, n=1)", qsd, True, False),
    ("cph_disc\n(n=14)", cph[cph["discrepant_in_9d7"]], False, False),
    ("7en_disc\n(n=14)", en[en["is_discrepant_in_cph"]], False, True),
    ("si3_disc\n(n=69)", si3[si3["stratum"].isin(["impact_monitor", "long_arc_well_obs"])], False, True),
    ("8d9 cph_disc\n(n=14)", d8[(d8["source_cohort"] == "cph") & (d8["is_discrepant_in_cph"])], False, True),
]
fig, ax = plt.subplots(figsize=(10, 6))
xs = np.arange(len(cohorts_disc))
for v in PRINCIPLED:
    ys = []
    for label, df, is_qsd, filt in cohorts_disc:
        if is_qsd:
            q = df.set_index("variant")["cartesian_dr_au"]
            ys.append(q["no_bias"] / q[v] if v in q.index else np.nan)
        else:
            ys.append(med(df, v, apply_filter=filt))
    ax.plot(xs, ys, "o-", color=COLORS[v], label=LABEL[v], lw=2, ms=9)
ax.axhline(1.0, color="grey", ls="--", lw=1, label="no_bias (no improvement)")
ax.set_xticks(xs)
ax.set_xticklabels([c[0] for c in cohorts_disc])
ax.set_ylabel("improvement factor vs no_bias\n(no_bias median / variant median; >1 = closer to JPL)")
ax.set_title("The principled 'winner' flips across cohorts\n(YR4: perf_weighted leads; 7en: covar_inflation; si3: nobody)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig1_winner_flips_across_cohorts.png", dpi=130)
plt.close(fig)
print("\nFIG1 improvement factors (cohorts:", [c[0].replace(chr(10), ' ') for c in cohorts_disc], "):")
for v in PRINCIPLED:
    ys = []
    for label, df, is_qsd, filt in cohorts_disc:
        if is_qsd:
            q = df.set_index("variant")["cartesian_dr_au"]
            ys.append(round(q["no_bias"] / q[v], 3) if v in q.index else np.nan)
        else:
            ys.append(round(med(df, v, apply_filter=filt), 3))
    print(" ", v, ys)

# --- FIG 2: control-regression RATE per principled variant across cohorts
reg_cohorts = [
    ("cph\n(n=10)", cph[cph["is_control"]], "veres_only"),
    ("7en\n(n=45)", en[en["stratum"] == "short_arc_mod_obs"], "veres_only"),
    ("si3\n(n=30)", si3[si3["stratum"] == "short_arc_mod_obs"], "no_bias"),
    ("8d9 si3\n(n=30)", d8[(d8["source_cohort"] == "si3") & (d8["stratum"] == "short_arc_mod_obs")], "no_bias"),
]
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
for i, v in enumerate(PRINCIPLED):
    rates = []
    for label, df, base in reg_cohorts:
        reg = _regressions(df, base)
        n = df[df["variant"] == "no_bias"]["object_id"].nunique()
        rates.append(100.0 * reg.get(v, 0) / n if (v in df["variant"].values and n) else np.nan)
    ax.bar(np.arange(len(reg_cohorts)) + i * width, rates, width, color=COLORS[v], label=LABEL[v])
ax.set_xticks(np.arange(len(reg_cohorts)) + 1.5 * width)
ax.set_xticklabels([c[0] for c in reg_cohorts])
ax.set_ylabel("short-arc control regression rate (%)\n(Dr/sigma > 2x per-object baseline)")
ax.set_title("Control-regression rate: v1_subtract consistently damages short arcs;\ncovar_inflation's low 7en rate (4%) does NOT hold on si3 (20%);\nsem_inflated (8d9) regresses identically to v1_subtract (12/30 = 40%)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.text(0.5, 0.005, "8d9 si3 bars recomputed over the full n=30 short-arc stratum (the published 8d9 REPORT.md used a degenerate n=1 set).", ha="center", fontsize=7, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIG / "fig2_control_regression_rates.png", dpi=130)
plt.close(fig)

# --- FIG 3: PS1/PS2/H21 failure-mode demo -- per-variant Dr/sigma on the two regressed controls.
# 2026 DX from 8d9 (all 5 variants incl sem_inflated); 2025 UA3 from 7en (sem_inflated absent there).
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=False)
panels = [
    ("2026 DX", d8[(d8["designation"] == "2026 DX") & (d8["converged"])],
     ["no_bias", "v1_subtract", "v1_subtract_sem_inflated", "v1_covar_inflation", "v1_performance_weighted"],
     "8d9, F51/290/F52/H21"),
    ("2025 UA3", en[(en["designation"] == "2025 UA3") & (en["converged"])],
     ["no_bias", "v1_subtract", "v1_covar_inflation", "v1_performance_weighted"],
     "7en, F52/F51/691/807/G96"),
]
short = {"no_bias": "no_bias", "v1_subtract": "v1_subtract", "v1_subtract_sem_inflated": "sem_inflated",
         "v1_covar_inflation": "covar_infl", "v1_performance_weighted": "perf_wt"}
for ax, (tgt, sub, order, srcs) in zip(axes, panels):
    sub = sub[sub["variant"].isin(order)].set_index("variant").reindex(order)
    cols = ["#999999" if v == "no_bias" else COLORS.get(v, "#888888") for v in order]
    ax.bar(range(len(order)), sub["dr_over_sigma"].values, color=cols)
    nb = sub["dr_over_sigma"].get("no_bias", np.nan)
    ax.axhline(2 * nb, color="red", ls="--", lw=1, label="2x no_bias (regression threshold)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[v] for v in order], rotation=20, ha="right", fontsize=8)
    ax.set_title(f"{tgt}\n({srcs})")
    ax.set_ylabel("Dr/sigma vs JPL")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("PS1/PS2/H21 short-arc failure mode (two distinct mechanisms)\n"
             "v1_subtract & sem_inflated: geometric coherent shift (sem_inflated == subtract, sigma_b doesn't engage).  "
             "covar_inflation: off-diagonal RA/Dec over-coupling.", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / "fig3_ps1ps2h21_failure_mode.png", dpi=130)
plt.close(fig)
print("\nFIG3 2026 DX (8d9) per-variant dr_over_sigma:")
sub = d8[(d8["designation"] == "2026 DX") & d8["converged"]].set_index("variant")["dr_over_sigma"]
print("  ", {k: round(sub.get(k, np.nan), 4) for k in ["no_bias", "v1_subtract", "v1_subtract_sem_inflated", "v1_covar_inflation", "v1_performance_weighted"]})
print("FIG3 2025 UA3 (7en) per-variant dr_over_sigma:")
sub = en[(en["designation"] == "2025 UA3") & en["converged"]].set_index("variant")["dr_over_sigma"]
print("  ", {k: round(sub.get(k, np.nan), 4) for k in ["no_bias", "v1_subtract", "v1_covar_inflation", "v1_performance_weighted"]})

# --- FIG 4: chronological leader timeline
fig, ax = plt.subplots(figsize=(12, 4.2))
beads = [
    ("zgf/9f2\nYR4 pilot", "v1_subtract\n(only mover)", "#444444"),
    ("qsd\nYR4 triage", "v1_performance_weighted\n(top principled)", "#d62728"),
    ("cph\n24-obj pop", "v1_subtract\n(principled flat)", "#444444"),
    ("7en\n149-obj sweep", "v1_covar_inflation\n(threads needle)", "#1f77b4"),
    ("si3\n100-obj holdout", "NO-GO\n(covar doesn't generalise)", "#cc0000"),
    ("8d9\nSEM-inflated", "NO-GO\n(sem_inflated == subtract:\n12/30 controls)", "#cc0000"),
]
xs = np.arange(len(beads))
ax.plot(xs, [0] * len(beads), "-", color="grey", lw=2, zorder=0)
for x, (b, leader, c) in zip(xs, beads):
    ax.scatter([x], [0], s=160, color=c, zorder=3)
    ax.annotate(b, (x, 0), xytext=(0, 22), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.annotate(leader, (x, 0), xytext=(0, -40), textcoords="offset points", ha="center", fontsize=8, color=c)
ax.set_ylim(-1, 1)
ax.set_xlim(-0.6, len(beads) - 0.4)
ax.axis("off")
ax.set_title("Leading principled bias-application variant after each experiment\n(every principled winner was refuted by the next, harder cohort -- program ends NO-GO)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig4_chronological_leader.png", dpi=130)
plt.close(fig)

# --- FIG 5: MBA vs NEO sigma-source split (veres_v1_max_floor)
strata = ["impact_monitor", "long_arc_well_obs", "main_belt", "short_arc_mod_obs"]
fig, ax = plt.subplots(figsize=(10, 5.5))
width = 0.38
nb, vm = [], []
for s in strata:
    m = _median_block(en[en["stratum"] == s])
    nb.append(m.get("no_bias", {}).get("median_dr_over_sigma", np.nan))
    vm.append(m.get("veres_v1_max_floor", {}).get("median_dr_over_sigma", np.nan))
x = np.arange(len(strata))
ax.bar(x - width / 2, nb, width, label="no_bias", color="#999999")
ax.bar(x + width / 2, vm, width, label="veres_v1_max_floor", color="#ff7f0e")
ax.set_xticks(x)
ax.set_xticklabels(strata, rotation=15, ha="right")
ax.set_yscale("log")
ax.set_ylabel("median Dr/sigma (log scale)")
ax.set_title("MBA-NEO sigma-source split (7en): force-overriding MPC sigma with Veres\nhelps MBAs (best MBA variant) but penalises modern-survey impact monitors")
ax.legend()
ax.grid(alpha=0.3, axis="y", which="both")
fig.tight_layout()
fig.savefig(FIG / "fig5_mba_neo_sigma_split.png", dpi=130)
plt.close(fig)

print("\nWROTE figures:", sorted(p.name for p in FIG.glob("*.png")))
