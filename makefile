S1_ARD_DIR = .
SOURCES=${S1_ARD_DIR}/cop_ard_app
TESTS=${S1_ARD_DIR}/tests
NOTEBOOKS=${S1_ARD_DIR}/notebooks

format:
	set -e
	uv run --extra formatter --frozen ruff format ${SOURCES} ${TESTS} ${NOTEBOOKS}
	uv run --extra formatter --frozen ruff check --select I --fix ${SOURCES} ${TESTS}
	uv run --extra linter --frozen ruff check ${SOURCES} ${TESTS}
	uv run --extra security --frozen python -m bandit -c ${S1_ARD_DIR}/bandit.yml -r ${SOURCES}
	uv run --extra complexity --frozen python -m xenon --max-average b --max-modules c --max-absolute d ${SOURCES}
	uv run --extra typing --frozen ty check ${SOURCES} ${TESTS}
fix:
	uv run --extra linter --frozen ruff check --fix ${SOURCES} ${TESTS}
test:
	uv run --extra tests --frozen pytest ${TESTS}
