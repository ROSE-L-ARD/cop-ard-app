<!--
  Copyright 2023 European Union

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# COP ARD APP

Application layer of the COP ARD processing chain.

`cop-ard-app` sits between the `cop-ard-core` algorithm kernels and the
`cop-ard-processor` EOPF pipeline: mission readers, co-registration workflow,
interferogram generation, CMLP/GMLP building blocks, and distributed
computation orchestration on Dask.

Stack position:
`cop-ard-processor` → **`cop-ard-app`** → `cop-ard-core`.

## Project Structure

This project is organized as follows:

* `cop_ard_app/`: source code of the project's main Python package —
  `readers/` (`sarpy`-based readers for Sentinel-1 SAFE  and
  ROSE-L), `coreg.py`, `interf.py`, `matching.py`, `esd.py`,
  `split_spectrum.py`, `cmlp.py`, `gmlp.py`, `area_projection.py`, `dem.py`,
  `etad.py`, `orbit.py`, `slc.py`, `io.py`, `model.py`, `computation.py`,
  `simulation.py`, `utils.py`.
* `tests/`: unit and integration tests covering every module, including the
  mission readers (`test_sarpy_rosel.py`, `test_sarpy_sentinel_safe.py`).
* `notebooks/`: analysis notebooks.
* `makefile`: installation targets (`install`, `proj` geoid-grid download)
  and quality-assurance targets (`format`, `lint`, `security`, `complexity`,
  `typecheck`, `check`, `test`, `check-secrets`, `protect-secrets`).
* `Dockerfile`: container image build.
* `pyproject.toml` / `uv.lock`: package metadata and pinned dependency
  lockfile.

## Installation

The project requires Python 3.13 (`>=3.13, <3.14`) and is managed with
[`uv`](https://docs.astral.sh/uv/). `cop-ard-app` pulls `cop-ard-core` from
its private GitHub repository (via `[tool.uv.sources]`), so authenticated
GitHub access is required:

```bash
gh auth login && gh auth setup-git   # or configure a ~/.netrc entry for github.com
uv sync --frozen
```

This creates the local `.venv` from `pyproject.toml` and the pinned
`uv.lock`; `uv` provisions Python 3.13 automatically.

## Development

Quality gates are driven by the `makefile`: `make all` runs `format`
(`ruff format` and import sorting over the package and tests) followed by
`check` (`ruff check`, `bandit`, `xenon`, and `mypy`); `make test` runs
the test suite.

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push
to `main` and on every pull request against `main`. It authenticates to
GitHub with the `GIT_PAT` secret (to fetch the private `cop-ard-core`
dependency), then enforces:

* Linting (`ruff check --no-fix`) and formatting check (`ruff format --check`)
* Typing (`mypy`)
* Unit tests (`pytest`) with coverage gate (`pytest --cov`, fail below 70%)
* Docstring coverage (`docstr-coverage`) and comment coverage
  (`radon raw` + `comment_coverage.awk`, threshold 20%)
* Complexity (`xenon`) and vulnerability scanning (`bandit` with `bandit.yml`)

## Run tests

To execute the tests locally using `pytest`, run:

```bash
uv run --extra tests pytest tests/
```

## Updating core dependencies

`uv.lock` pins `cop-ard-core` to a specific commit of its GitHub
repository. To advance the pin, run:

```bash
uv lock -P cop-ard-core
```

To add updated dependencies to the local `uv` virtual environment, run:

```bash
uv sync --frozen
```
