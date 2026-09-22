# v2_sigma_fill.csv — provenance

Per-observation astrometric sigma FILL-IN table derived from the Asteroid Institute
MPC observatory bias study (v2), for use in `mpc_to_od_observations(sigma_model="v2_rms")`
exactly the way the Veres et al. 2017 Table 1 values are used by `sigma_model="veres2017"`:
only where the MPC reports no `rmsra`/`rmsdec`; reported sigmas are never touched.

Source: published, bias-filtered leave-one-object-out residuals of run
`mpc_scale_results_v2_full_20260622` (`published/looo_results_atct.parquet`,
5,148,283 residuals, 16,960 objects; observations EFCC18-debiased upstream, so
these are post-catalog-debias residuals — the same status as the Veres 2017
numbers, which were computed on FCCT14-debiased residuals). Built 2026-09-22
(bead od_experiments_setup-d2f).

Statistic: pooled per-observation RMS about zero, per axis, in arcsec
(RA cos(dec)-corrected) — the direct analogue of the Veres Table 1 columns.
The mean residual (station bias) is therefore INCLUDED in the RMS, as in
Veres; nothing here modifies positions.

Rows (lookup order in `get_v2rms_sigma`):
  level=stn_astcat  (station, star catalog) — HIGH-CONFIDENCE stations
                    (catalog `high_confidence`, confidence_score >= 0.5; 671
                    stations) with >= 30 residuals and >= 3 objects in the
                    group: 1762 rows
  level=stn         every high-confidence station, all catalogs pooled:
                    671 rows (Veres 2017 has 14 station overrides)
  level=astcat      per-catalog default over all high-confidence stations
                    (>= 100 residuals): 40 rows (Veres has 34)
  level=global      all high-confidence residuals: RA 0.365", Dec 0.349"
                    (Veres fallback 0.75")

Non-high-confidence stations deliberately fall through to the catalog / global
defaults. Program codes are not a key: the MPC observation caches the OD
pipeline fits carry no program field, and this residual set predates the
`prog` column.
