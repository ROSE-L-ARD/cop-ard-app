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

This repository contains the COP ARD APP project.

COP ARD APP

## Project Structure

This project is organized as follows:

* `cop_ard_app` contains the source code of the
  project's main Python package.
* `tests` contains the unit and integration tests of the `COP ARD APP`
  project.

## Tools

The following tools are used by this project:

* `pyproject.toml`: Build system requirements for Python.
* `uv.lock`: `uv` lock file containing specific versions for all dependencies.

## Documentation



## Run tests

To execute the tests locally using `pytests`, run:

``` python
uv run --extra tests pytest tests/
```

The GitLab CI pipeline will also automatically run the unit and
integration tests of the project.



## Updating core dependencies
To update core dependencies in the `uv.lock` file, run:

``` python
uv lock -P cop-ard-core
```

To add updated dependencies to the local `uv` virtual environment, run:

``` python
uv sync --frozen
```
