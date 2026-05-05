# FindOrb warm-start mechanism — investigation for bead 39j

**Date:** 2026-05-05
**Branch:** `kk/mpc-scale-bias-catalog`
**Bead:** `beads_agent_setup-39j` → blocks `beads_agent_setup-dez` (image rebuild)
**FindOrb commit examined:** `294bd5d` (the version baked into pilot v11 image
`us-west1-docker.pkg.dev/moeyens-thor-dev/ai/mpc-real-data-looo:pilot-v11-20260429`)

## Bottom line

**The premise of bead dez is wrong.** The catastrophic-residual tail in pilot
v11 cannot be fixed by shipping a published-orbit catalog (MPCORB.DAT or
mpcorb.sof) into the v12 image alone. At the FindOrb commit baked into v11
(`294bd5d`), the catalog-warm-start code path is gated behind `n_obs == 1`,
and the LOOO hold-in fits always run with `n_obs > 1` (≥6 by the LOOO
config). Shipping the catalog by itself has no effect — verified empirically
on the F51-holdout case for 2020 ML22 (bit-identical converged orbits with
and without the catalog installed) and by reading the FindOrb source.

**mxw is NOT load-bearing for dez** in the conventional sense, because there
is no FindOrb-recompile path that solves this problem alone. To enable
warm-start, the work has to be plumbed through `adam_fo` (Python wrapper) —
not through a FindOrb rebuild. dez should be rescoped before it is
unblocked. Recommendations are at the end of this memo.

## What file FindOrb reads at runtime

FindOrb's only auto-warm-start lookup uses **`mpcorb.sof`** — a binary,
header-indexed Standard Orbit File (~140 bytes/record fixed-width text in a
binary container). NOT `MPCORB.DAT` (the 313 MB plain-text catalog MPC
publishes). MPCORB.DAT is the input to a one-shot conversion tool
(`mpc2sof`); FindOrb itself never opens MPCORB.DAT.

The reader is `get_orbit_from_mpcorb_sof()` in `elem_out.cpp`:

```c
// find_orb/find_orb/elem_out.cpp:3191-3201
int get_orbit_from_mpcorb_sof( const char *object_name, double *orbit,
             ELEMENTS *elems, const double full_arc_len, double *max_resid)
{
   const char *mpcorb_dot_sof_filename = get_environment_ptr( "MPCORB_SOF_FILENAME");
   int rval = 0;

   if( *mpcorb_dot_sof_filename)
      rval = get_orbit_from_sof( mpcorb_dot_sof_filename,
                             object_name, orbit, elems, full_arc_len, max_resid);
   return( rval);
}
```

Two preconditions for warm-start to fire:

1. **`MPCORB_SOF_FILENAME` must be set non-empty** in the FindOrb config
   (`environ.def` or `environ.dat`). Default in both the shipped FindOrb
   `environ.def` AND `adam_fo`'s `environ.dat.tpl` is unset, so
   `*mpcorb_dot_sof_filename` is `'\0'` and the body of the `if` is skipped
   entirely. The catalog file could be sitting next to FindOrb and would
   never be opened.
2. **The call site must actually be reached.** This is the bigger problem.

### Where the call site lives — and why our use case never hits it

Two call sites reference `get_orbit_from_mpcorb_sof()`:

- `elem_out.cpp:3258-3263`, inside `fetch_previous_solution()` — the
  initial-orbit-load path used by `fo` (the headless binary `adam_fo` wraps):

  ```c
  if( n_obs == 1)
     got_vectors = get_orbit_from_mpcorb_sof( object_name, orbit, &elems,
                              full_arc_len, &rms_resid);
  else
     got_vectors = get_orbit_from_sof( "orbits.sof", obs->packed_id,
                                      orbit, &elems, full_arc_len, &rms_resid);
  ```

  **The `n_obs == 1` gate is the showstopper.** For multi-obs cases (every
  LOOO hold-in fit), FindOrb does NOT consult `mpcorb.sof`. It looks at
  `orbits.sof` in its working directory — a per-session cache of previously
  saved local solutions, written by FindOrb only when the `-X`
  ("save elements for re-use") flag is passed (which `adam_fo` does not
  pass). So `orbits.sof` is empty by default in our pipeline.

- `findorb.cpp:6732`, inside the interactive curses keystroke handler for
  `ALT_Y` — this is a manual feature of the interactive `find_orb` UI, not
  reached by the headless `fo` binary that the LOOO worker invokes via
  `subprocess.run`.

If neither path returns a vector, `fetch_previous_solution()` falls through
to `initial_orbit()` (Gauss/Vaisala cold-start) at `elem_out.cpp:3279-3285`
— the path that produces the catastrophic chi² in pilot v11.

### Where FindOrb expects the file

When `MPCORB_SOF_FILENAME` is set, the value is passed as-is to
`get_orbit_from_sof()` and opened via `fopen_ext()`. `fopen_ext` searches
the FindOrb config dir (`~/.find_orb/` on Linux, or `/software/.find_orb/`
or `/root/.find_orb/` per `miscell.cpp:127-216`) AND the current working
directory. So either an absolute path or a filename that lives in the
config dir / cwd will work.

In our pipeline `fo` runs with `cwd=fo_tmp_dir` (a fresh temp dir per
LOOO hold-in fit, populated by `_populate_fo_directory()` in
`adam_fo/run_fo.py:20-70`), so the practical options are:

- Absolute path (e.g. `MPCORB_SOF_FILENAME=/data/mpcorb.sof`), with
  `mpcorb.sof` shipped at that location in the image, OR
- Add `mpcorb.sof` to the list `_populate_fo_directory()` copies into each
  per-fit working dir, with a relative-path setting.

## Is mxw load-bearing for dez? **No.**

mxw is the tracking bead for "FindOrb fails to compile under GCC 14."
FindOrb would only need to be recompiled if the FindOrb C++ source itself
had to change to fix the warm-start problem. **It does not.**

The two viable fixes are entirely Python-wrapper changes inside `adam_fo`
or `adam_orbit_det_eval`:

- **Inject the seed orbit through the existing `-v` cmdline flag.** `fo`
  already accepts `-v "x y z vx vy vz"` and parses it via
  `state_vect_text` / `extract_state_vect_from_text()`
  (`elem_out.cpp:3244-3251`). This bypasses the IOD entirely and starts the
  least-squares fit from the supplied state vector. Plumbing the MPC seed
  orbit (which `LOOOConfig.reference_orbit` already has on hand for the
  scipy fitter — see `looo/core.py:277-282`) into `FindOrbOrbitFitter.initial_fit`
  and forwarding via the `-v` flag would give per-object warm-start
  without touching FindOrb's source.

- **Pre-populate `orbits.sof`** in each `fo_tmp_dir` with a single record
  for the object being fit, derived from the MPC seed orbit. This hits the
  `n_obs > 1` branch at `elem_out.cpp:3262` and would warm-start every LOOO
  hold-in fit. Slightly more wiring than the `-v` approach but no FindOrb
  recompile.

Either approach makes mxw orthogonal to the warm-start fix. The only
scenario in which mxw becomes load-bearing is if we decide to extend
FindOrb's `fetch_previous_solution()` to consult `mpcorb.sof` for
`n_obs > 1` — i.e., a FindOrb source patch — which is a much larger,
upstream-maintainer-level change and is not the right scope for dez.

## Local repro

### Setup

- v11 image: `us-west1-docker.pkg.dev/moeyens-thor-dev/ai/mpc-real-data-looo:pilot-v11-20260429`
- Witness: 2020 ML22, 40 observations in `data/looo_sample_3500/mpc_observations.parquet`.
  - F51: 29 obs from 2015-05 to 2024-04 (the historical/discovery arc)
  - F52: 3 obs on 2025-09-15 (a single tracklet)
  - V00: 8 obs split — 4 on 2025-09-15, 4 on 2025-10-03
- Worst-case holdout: **F51 station held out → held-in = 11 obs (V00+F52),
  all in a single ~18-day window in 2025-09/10**. This is the
  short-held-in-arc cluster B configuration that produced cold-start
  chi²/obs ≈ 8.86e+11 in baseline_comparison.md (Cluster B = "5–30 day
  held-in arcs, 5–10 obs"; ours is right in the middle). The LOOO pipeline
  then propagates that hold-in fit forward and back to evaluate against
  F51's 2015-2024 holdouts, which is what produces the ~250,000" max
  residual at the F51 obs times.

### Catalog source confirmation

- `https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT` (HTTP HEAD
  returns Content-Length 312,960,752 → ~313 MB, Content-Type
  `text/plain`, Last-Modified daily). This is the file FindOrb's
  conversion tool `mpc2sof` consumes.
- 2020 ML22's record found at the expected position:
  `K20M22L 18.65  0.15 K25BL  84.04635   34.46416 ... 2020 ML22 20251003`
  (epoch 2025-Nov-21).
- Built `mpcorb.sof` (222 MB, 1,541,667 records) by running
  `mpc2sof MPCORB.DAT i` (the trailing `i` is required to skip
  `ELEMENTS.COMET`, which the v11 image does not ship and which causes
  `mpc2sof` to abort before writing if invoked without it). `mpc2sof` had
  to be built from source inside the image — the v11 image ships
  `find_orb` and `fo` binaries but not `mpc2sof`. (The build itself is
  clean — no GCC 14 issues for `mpc2sof`; only certain other lunar/
  targets exhibit those.)

### Standalone `fo` call: cold-start vs mpcorb.sof installed

I bypassed the LOOO machinery (`02_run_looo.py` deadlocks under
Apple-Silicon Rosetta amd64 emulation: JAX is multithreaded, then
`ProcessPoolExecutor` forks → known-bad pattern logged by Python's
`multiprocessing` itself, and irrecoverable under Rosetta). Instead I
called `fo` directly — same binary, same ADES inputs, same environ.dat
the LOOO pipeline produces — through the `_observations_to_ades` helper
in `adam_fo`. This isolates the warm-start question from the
LOOO/JAX/ASSIST infrastructure.

Run A (cold-start): default v11-image config, no `MPCORB_SOF_FILENAME`,
no `mpcorb.sof` anywhere in the working dir.

Run B (mpcorb.sof installed): full 222 MB `mpcorb.sof` (1,541,667
records, 2020 ML22 present and verified at byte offset matching MPC) is
copied into `fo`'s working dir, AND `environ.dat` has

```
MPCORB_SOF_FILENAME=/tmp/repro2/run_B/mpcorb.sof
```

appended. Same ADES file (11 obs, V00+F52) for both. Both invocations
exit 0 and produce identical convergence:

```
$ diff /tmp/repro2/run_A/elements.txt /tmp/repro2/run_B/elements.txt
< # Elements written:  5 May 2026 17:31:06 (JD 2461166.229940)
> # Elements written:  5 May 2026 17:31:10 (JD 2461166.229980)
(only the timestamp differs — every numerical field matches bit-for-bit)
```

| | Run A (cold) | Run B (mpcorb.sof + env var) |
|---|---|---|
| `a` (semi-major axis, AU) | 2.61903477 | 2.61903477 |
| `e` (eccentricity) | 0.1113619 | 0.1113619 |
| `i` (inclination, deg) | 6.41684 | 6.41684 |
| State vec X (AU) | +2.441293763520 | +2.441293763520 |
| State vec Y (AU) | +0.344635434079 | +0.344635434079 |
| State vec Z (AU) | +0.391068151746 | +0.391068151746 |
| State vec Vx (mAU/d) | -0.930193534071 | -0.930193534071 |
| State vec Vy (mAU/d) | +10.426498211689 | +10.426498211689 |
| State vec Vz (mAU/d) | +3.810016234451 | +3.810016234451 |
| Mean residual | 0".13 | 0".13 |
| Score | -0.030969 | -0.030969 |

**Bit-identical orbits.** `mpcorb.sof` had zero effect — exactly as the
source code predicts. If the warm-start path had fired in Run B, the
seed orbit would either have differed enough to nudge the converged
solution (and the state vectors would not be bit-identical) or the
"perturbers turned on at warm-start time" branch (line 3270: `perturbers
= 0x7fe`) would have fired and changed the perturber set printed in the
elements file. Neither happened.

(Note: I attempted `strace -e openat -f` on the `fo` invocations to
catalog every file fo opens. The trace files came back empty —
strace cannot follow Rosetta-translated x86_64 binaries on Apple
Silicon. This is a known limitation, not evidence of anything. The
bit-identical-output check above is more compelling anyway.)

### Note on the LOOO 8.86e+11 chi²/obs

The bead's exit criterion was: "If chi²/obs drops by 10+ orders of
magnitude, warm-start works. If it doesn't, … the assumption is wrong."

In our experiment, the held-in rms residual went from 0".13 to 0".13 —
**0 orders of magnitude change**, which is exactly the "warm-start did
not engage" signal that the source-code analysis predicted.

What we did not reproduce here is the 8.86e+11 chi²/obs that
`baseline_comparison.md` reports for the same witness/holdout. That
number comes from the LOOO pipeline running
`evaluate_orbits(fo_returned_orbit, held_in_obs, ASSIST_propagator,
parameters=6)` AFTER `fo` returns — i.e., from re-propagating fo's
returned state vector through `adam_assist`'s ASSIST and computing
residuals at the obs times. fo itself converged cleanly in both runs.
That fo-vs-ASSIST evaluation mismatch is a separate problem (would
persist unchanged whether the catalog ships or not) and is out of scope
for this bead. Reproducing it inline would have required running the
full LOOO pipeline, which deadlocked under the JAX+fork pattern in
amd64-via-Rosetta on the host (see above) and did not seem worth fixing
just to put a second number on a question already decisively answered
by the bit-identical state vectors.

## Concrete instructions for dez (image rebuild)

### What dez should NOT do

Do NOT just `COPY MPCORB.DAT` into the image and call it shipped. The file
will be inert because:

1. FindOrb does not read `MPCORB.DAT` — it reads `mpcorb.sof`. There is no
   filesystem-search fallback. (FindOrb has a separate code path for
   reading `mpcorb.dat` lines, at `lunar/mpcorb.cpp:82-119`, but that is
   used only by standalone tools like `astcheck`, not by `fo`.)
2. Even with `mpcorb.sof` shipped, the `n_obs == 1` gate prevents the
   LOOO worker from ever consulting it.

Adding a 313 MB or 222 MB inert file to the image would just inflate the
image size with no measurable downstream effect, repeating the v11
mpcorb.hdr-only mistake at larger scale.

### What dez should do — three options, in increasing order of work

**Option A (smallest change, recommended for v12 first cut): switch
fitter back to scipy**, baseline-correctness comparison run only.
Rebuild image with `12_run_looo_cloud_shard.py` defaulting
`--orbit-fitter scipy` (or pin in the Job spec). This was the configuration
of the Mar 16 reference run that produced the well-behaved bias table.
No catalog file shipped, no FindOrb changes. Confirms the catastrophic
tail is FindOrb-cold-start-specific and gives a clean baseline-comparison
sample.

**Option B: ship mpcorb.sof + plumb a `-v` warm-start through adam_fo.**
The v12 image then has both the catalog and the `adam_fo` change that
actually uses it.

Dockerfile additions (in `infra/exp-research/Dockerfile.pilot-overlay`):

```dockerfile
# Install mpcorb.sof for FindOrb warm-start lookups.
# Source: MPCORB.DAT from MPC, converted via FindOrb's mpc2sof tool.
# Refresh cadence: monthly (MPC publishes daily; weekly/monthly is fine
# for warm-start purposes — initial orbit accuracy of ~1 arcmin is plenty).
ARG MPCORB_BUILD_DATE=20260504
RUN mkdir -p /data && cd /tmp \
    && curl -sfL https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz -o MPCORB.DAT.gz \
    && gunzip MPCORB.DAT.gz \
    && /root/.local/share/adam_fo/find_orb/lunar/mpc2sof MPCORB.DAT i \
    && mv mpcorb.sof /data/mpcorb.sof \
    && rm MPCORB.DAT \
    && stat -c "%n %s bytes built %y" /data/mpcorb.sof

# Make FindOrb consult /data/mpcorb.sof when warm-starting.
RUN sed -i '$a MPCORB_SOF_FILENAME=/data/mpcorb.sof' \
    /usr/local/lib/python3.11/site-packages/adam_fo/environ.dat.tpl \
    /app/adam_fo/src/adam_fo/environ.dat.tpl
```

Note that `mpc2sof` needs to be **built once** before the above `RUN` (the
v11 image ships only the source). Add to the build stage that already
compiles FindOrb:

```dockerfile
RUN cd /root/.local/share/adam_fo/find_orb/lunar && make mpc2sof
```

(Confirmed clean build under GCC 13 inside v11 — runs in ~2 s, no
warnings. Not affected by mxw.)

**This dockerfile change alone is necessary but not sufficient.** Without
the matching `adam_fo` Python change to forward a seed orbit (either via
`-v` or via a pre-populated `orbits.sof`), the shipped `mpcorb.sof` is
inert for LOOO multi-obs fits — same as Run B above will show.

**Option C: also ship an `orbits.sof` pre-populated per object before
running `fo`.** This requires `adam_fo` to be patched so that
`_populate_fo_directory()` writes the seed orbit (already in
`hold_in_obs.metadata` or available via `MPCOrbits` from the input
parquet) as a 1-record `orbits.sof` in the working dir. With this in
place, the existing FindOrb at 294bd5d will hit the
`get_orbit_from_sof("orbits.sof", ...)` branch at `elem_out.cpp:3262` and
warm-start. **No image rebuild required for this part — it's an
`adam_fo` PR.**

The cleanest combo is **Option B + Option C's adam_fo plumbing** — image
ships the catalog (so single-obs ephemeris-only callers can use it too),
and the Python wrapper does the per-call warm-start that's actually
needed for LOOO.

## Action items

1. **Reopen dez with rescoped acceptance criteria.** Current scope ("ship
   MPCORB warm-start") is undefined; should be split into:
   - dez-1: Image ships `/data/mpcorb.sof` + `MPCORB_SOF_FILENAME` env
     line in `environ.dat.tpl`, plus `mpc2sof` built (Option B above).
     Verifiable: image build green, `/data/mpcorb.sof` size > 200 MB,
     `grep MPCORB_SOF environ.dat.tpl` matches in installed adam_fo.
   - dez-2: `adam_fo` PR adds seed-orbit warm-start through `-v` or
     `orbits.sof`. Verifiable: pilot v12 LOOO on the same 1,349 objects
     produces |residual_RA| > 60" rate ≤ 1% (vs v11's 8.82%).
2. **Do NOT add mxw as a blocker on dez.** It is unrelated. (If the v12
   image build hits its own GCC 14 issue independently of warm-start,
   that's an mxw or new-bead concern; nothing in the warm-start work
   depends on a FindOrb recompile.)
3. **Smallest immediate value:** pivot the v12 baseline-correctness
   comparison run to `--orbit-fitter scipy` (Option A). That alone
   replicates the Mar 16 reference configuration and produces a clean
   bias-table comparison. Image ships unchanged.

## Appendix — how I checked

- `docker run -d --name findorb-inspect-v11 --entrypoint sleep
  us-west1-docker.pkg.dev/moeyens-thor-dev/ai/mpc-real-data-looo:pilot-v11-20260429 14400`
- FindOrb source at `/root/.local/share/adam_fo/find_orb/`, git tip
  `294bd5d`. Grepped for `MPCORB|mpcorb|get_orbit_from_mpcorb` across
  `find_orb/` and `lunar/`; only call sites listed in the body above.
- `adam_fo` Python wrapper at `/app/adam_fo/src/adam_fo/`, examined
  `find_orb_orbit_fitter.py:initial_fit` (calls `fo()`) and
  `run_fo.py:fo()` (subprocess invocation; cmdline does not include
  `-v` or `-X`).
- environ.dat: shipped `environ.def` (FindOrb's defaults) and
  `adam_fo/environ.dat.tpl` neither contains a `MPCORB_SOF_FILENAME=`
  line.
- MPC catalog confirmed via `curl -sI`. Conversion verified by running
  `mpc2sof MPCORB.DAT i` inside the image and inspecting the resulting
  `mpcorb.sof` (1,541,667 records, 2020 ML22 present).
- LOOO repro: ran `fo` directly with `-c -d 2 -D environ.dat -O .`
  flags matching `adam_fo/run_fo.py:147`, on an ADES file generated by
  `FindOrbOrbitFitter._observations_to_ades` from the 11 V00+F52 obs.
  Reasonable proxy for the LOOO subprocess call (which is the same `fo`
  binary, same flags, same `_observations_to_ades` ADES output).
