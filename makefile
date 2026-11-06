SOURCES := $(CURDIR)/cop_ard_app
TESTS := $(CURDIR)/tests

GRID_FILE ?= us_nga_egm08_25.tif

.PHONY: all install proj format lint security complexity typecheck check restore check-secrets protect-secrets

all: format check

install:
	uv sync
	$(MAKE) proj

proj:
	@echo "Ensuring geoid grid '$(GRID_FILE)' is installed..."
	@uv run python -c "from cop_ard_app.utils import download_geoid_grid; download_geoid_grid(grid_name='$(GRID_FILE)')"
	@echo "Grid verification completed."

format:
	uv run --extra formatter --frozen ruff format $(SOURCES) $(TESTS)
	uv run --extra formatter --frozen ruff check --select I --fix $(SOURCES) $(TESTS)

lint:
	uv run --extra linter --frozen ruff check $(SOURCES) $(TESTS)

security:
	uv run --extra security --frozen python -m bandit -c $(CURDIR)/bandit.yml -r $(SOURCES)

complexity:
	uv run --extra complexity --frozen python -m xenon --max-average b --max-modules c --max-absolute d $(SOURCES)

typecheck:
	uv run --extra typing --frozen python -m mypy $(SOURCES) --allow-redefinition

check: lint security complexity typecheck

check-secrets:
	@echo "==> Scanning entire Git history for leaked secrets..."
	gitleaks detect --source . --verbose --redact

protect-secrets:
	@echo "==> Scanning uncommitted changes for secrets..."
	gitleaks protect --staged --verbose --redact

test:
	uv run --extra tests --frozen pytest ${TESTS}
