.DEFAULT_GOAL := help

PYTHON ?= python3
POETRY ?= poetry
RUFF ?= ruff
MYPY ?= mypy
export PYTHONDONTWRITEBYTECODE := 1
PYTHON_FILES := $(shell find . \
	-path '*/.git' -prune -o \
	-path '*/.venv' -prune -o \
	-path '*/venv' -prune -o \
	-path '*/__pycache__' -prune -o \
	-type f -name '*.py' -print | sort)

.PHONY: \
	help smoke smoke-checks quality-checks full-quality-checks poetry-check \
	ruff-format-check ruff-check mypy-check \
	python-syntax-check chapter-coverage-check chapter-coverage-self-test \
	chapter-tests experiment-checks release-hygiene-check python-readability-check \
	release-inventory-check

help:
	@printf '%s\n' 'Available companion-code checks:'
	@printf '%s\n' '  smoke                     Conventional alias for smoke-checks'
	@printf '%s\n' '  smoke-checks              Run all dependency-free standalone checks'
	@printf '%s\n' '  quality-checks            Run standalone smoke, Ruff, and mypy checks'
	@printf '%s\n' '  full-quality-checks       Alias for the standalone quality gate'
	@printf '%s\n' '  poetry-check              Validate the standalone pyproject.toml'
	@printf '%s\n' '  python-syntax-check       Compile maintained Python without bytecode'
	@printf '%s\n' '  chapter-coverage-check    Validate the chapter coverage manifest'
	@printf '%s\n' '  chapter-coverage-self-test  Test the coverage validator'
	@printf '%s\n' '  chapter-tests             Run all tests in registered chapter directories'
	@printf '%s\n' '  experiment-checks         Test selected portable experiments and recorded-data analysis'
	@printf '%s\n' '  release-hygiene-check     Reject secrets, local paths, caches, and dirty notebooks'
	@printf '%s\n' '  release-inventory-check   Validate export inventory, links, and available release hashes'
	@printf '%s\n' '  python-readability-check  Enforce standalone Python size and complexity budgets'

smoke: smoke-checks
	@:

smoke-checks: \
	python-syntax-check chapter-coverage-self-test chapter-coverage-check \
	release-hygiene-check release-inventory-check python-readability-check chapter-tests experiment-checks
	@:

quality-checks: smoke-checks poetry-check ruff-format-check ruff-check mypy-check
	@:

full-quality-checks: quality-checks
	@:

poetry-check:
	@$(POETRY) check

ruff-format-check:
	@$(RUFF) format --check .

ruff-check:
	@$(RUFF) check .

mypy-check:
	@$(MYPY) .

python-syntax-check:
	@$(PYTHON) -B -c 'import pathlib,sys; [compile(pathlib.Path(path).read_text(encoding="utf-8"), path, "exec", dont_inherit=True) for path in sys.argv[1:]]' $(PYTHON_FILES)

chapter-coverage-check:
	@$(PYTHON) -B -m _maintenance.chapter_coverage

chapter-coverage-self-test:
	@$(PYTHON) -B -m unittest discover -s _maintenance -t . -p 'test_*.py'

chapter-tests:
	@$(PYTHON) -B -m _maintenance.chapter_coverage --run-tests

experiment-checks:
	@$(PYTHON) -B _maintenance/chapter_test_runner.py experiments/test_experiments.py

release-hygiene-check:
	@$(PYTHON) -B -m _maintenance.release_hygiene

release-inventory-check:
	@$(PYTHON) -B -m _maintenance.release --check

python-readability-check:
	@$(PYTHON) -B -m _maintenance.readability
