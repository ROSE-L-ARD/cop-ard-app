SOURCES := $(CURDIR)/cop_ard_app
TESTS := $(CURDIR)/tests

GRID_FILE ?= us_nga_egm08_25.tif

.PHONY: all install proj format lint security complexity typecheck check restore check-secrets protect-secrets fix-alerts

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
	uv run --extra typing --frozen python -m mypy $(SOURCES) $(TESTS) --allow-redefinition

check: lint security complexity typecheck

check-secrets:
	@echo "==> Scanning entire Git history for leaked secrets..."
	gitleaks detect --source . --verbose --redact

protect-secrets:
	@echo "==> Scanning uncommitted changes for secrets..."
	gitleaks protect --staged --verbose --redact

test:
	uv run --extra tests --frozen pytest -s -vv ${TESTS}


fix-alerts:
	@set -e; \
	if [ -n "$$(git status --porcelain -uno)" ]; then \
		echo "Error: Tracked files have uncommitted changes. Commit or stash them first."; \
		exit 1; \
	fi; \
	echo "==> Checking open Dependabot alerts..."; \
	packages=$$(gh api "repos/{owner}/{repo}/dependabot/alerts?state=open&ecosystem=pip&per_page=100" \
		--jq '[.[].dependency.package.name] | unique | .[]'); \
	if [ -z "$$packages" ]; then \
		echo "No open Dependabot alerts found."; \
		exit 0; \
	fi; \
	echo "==> Vulnerable packages found:"; \
	echo "$$packages"; \
	branch="fix/dependabot-alerts-$$(date +%Y%m%d-%H%M%S)"; \
	git fetch origin main; \
	git checkout -b "$$branch" origin/main; \
	args=""; \
	for pkg in $$packages; do \
		args="$$args --upgrade-package $$pkg"; \
	done; \
	uv lock $$args; \
	if git diff --quiet uv.lock; then \
		echo "==> No changes in uv.lock (packages may be constrained). Cleaning up..."; \
		git checkout -; \
		git branch -D "$$branch"; \
		exit 0; \
	fi; \
	pkg_list=$$(echo "$$packages" | sed 's/^/- /'); \
	git add uv.lock; \
	git commit -m "fix(deps): patch open Dependabot alerts in uv.lock" -m "$$pkg_list"; \
	git push -u origin "$$branch"; \
	gh pr create \
		--base main \
		--head "$$branch" \
		--title "fix(deps): patch open Dependabot alerts" \
		--body "$$(printf 'Automated update of vulnerable packages reported by Dependabot alerts:\n\n%s' "$$pkg_list")"
