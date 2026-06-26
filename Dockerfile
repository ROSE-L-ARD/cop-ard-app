# Copyright 2023 European Union
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ghcr.io/astral-sh/uv:0.9.7-python3.11-bookworm AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Configure the Python directory so it is consistent
ENV UV_PYTHON_INSTALL_DIR=/python

# Only use the managed Python version
ENV UV_PYTHON_PREFERENCE=only-managed

COPY . /cop_ard_app
WORKDIR /cop_ard_app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=secret,id=netrc,target=/root/.netrc \
    uv sync --locked --no-install-project --no-editable --python 3.11


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# Final stage
#
# Use the EOPF Dask image as the base of the Dask runtime environment.
# The 'latest' tag is used by the template to test the current version.
# Moreover, the usage of the 'latest' tag causes a warning to be raised by
# the docker-linter CI job.
# However, user projects are encouraged to fix the version of the base image.
FROM registry.eopf.copernicus.eu/sde/dask-container-images/dask-scheduler-worker:latest

# OCI annotations
# See https://github.com/opencontainers/image-spec/blob/main/annotations.md#pre-defined-annotation-keys
ARG CI_COMMIT_SHA
LABEL org.opencontainers.image.revision="$CI_COMMIT_SHA"
LABEL org.opencontainers.image.vendor="ESA"
LABEL org.opencontainers.image.authors="B-Open"

# Copy the Python version
COPY --from=builder --chown=dask:dask /python /python
# Copy the application from the builder
COPY --from=builder --chown=dask:dask /cop_ard_app /cop_ard_app

# Place executables in the environment at the front of the path
ENV PATH="/cop_ard_app/.venv/bin:$PATH"
