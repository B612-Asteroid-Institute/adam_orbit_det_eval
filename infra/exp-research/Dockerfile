# Dockerfile for MPC-scale real-data LOOO cloud run
#
# Builds a self-contained image with:
#   - adam_core, adam-assist, adam_fo (with Find_Orb binary), adam_orbit_det_eval
#   - google-cloud-storage for GCS I/O
#   - JPL ephemeris files needed by Find_Orb and ASSIST
#
# Usage:
#   docker build -t mpc-real-data-looo:latest -f infra/exp-research/Dockerfile .
#
# Note: The exp-research job spec uses build.method=cloud, so this Dockerfile
# is primarily for local testing and reference. Cloud builds use the spec's
# build.projects definitions. This Dockerfile is the fallback if you need a
# pre-built image.

FROM python:3.12-slim

# System deps for building Find_Orb (C/C++ libraries) and general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# Copy only pyproject.toml files first for better layer caching
COPY adam_core/pyproject.toml /app/adam_core/pyproject.toml
COPY adam_assist/pyproject.toml /app/adam_assist/pyproject.toml
COPY adam_fo/pyproject.toml /app/adam_fo/pyproject.toml
COPY adam_orbit_det_eval/pyproject.toml /app/adam_orbit_det_eval/pyproject.toml

# Install packages in dependency order
COPY adam_core/ /app/adam_core/
RUN cd /app/adam_core && PDM_BUILD_SCM_VERSION=99.0.0 pip install --no-cache-dir .

COPY adam_assist/ /app/adam_assist/
RUN cd /app/adam_assist && pip install --no-cache-dir .

COPY adam_fo/ /app/adam_fo/
RUN cd /app/adam_fo && pip install --no-cache-dir .

# Build Find_Orb binary and download ephemeris files
RUN build-fo

COPY adam_orbit_det_eval/ /app/adam_orbit_det_eval/
RUN cd /app/adam_orbit_det_eval && pip install --no-cache-dir .

# Install GCS client for cloud I/O
RUN pip install --no-cache-dir google-cloud-storage

# Smoke test: verify the LOOO pipeline and Find_Orb are importable
RUN python -c "from adam_orbit_det_eval.looo.pipeline import run_looo_pipeline; print('pipeline OK')"
RUN python -c "from adam_fo.find_orb_orbit_fitter import FindOrbOrbitFitter; print('findorb OK')"
RUN python -c "from adam_assist import ASSISTPropagator; print('assist OK')"

# Default entrypoint is the shard runner
ENTRYPOINT ["python", "/app/adam_orbit_det_eval/scripts/12_run_looo_cloud_shard.py"]
