FROM python:3.11-slim-bookworm

LABEL maintainer="thibault.ghesquiere-dierickx@epfl.ch" \
    paper="The semi-explicit Nonsmooth Newmark time integrator for robust unilateral contact in dynamic fragmentation simulations" \
    akantu_commit="22adc1e143ca74fdb70af185536d16ff4a3396de" \
    uv_version="0.4.30" \
    build_date="2026-05-26"

COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /bin/uv

WORKDIR /app
COPY . .
RUN bash workflows/install.sh

ENTRYPOINT ["/bin/bash", "-c", "source /app/workflows/env.sh && exec \"$@\"", "--"]

CMD ["python", "src/run_fragmentation.py", "--help"]
