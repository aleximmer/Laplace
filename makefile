# Test tools come in through `--with`, so the CPU-only torch in the venv stays as is.
# `--no-sync` keeps uv from reinstalling the CUDA build pinned in `uv.lock`.
PYTEST = uv run --no-sync --with pytest --with scipy --with pytest-mock --with pytest-xdist pytest

TESTS ?= tests
MARKS ?= not public_api and not cuda
ARGS ?=

.PHONY: ruff test

ruff:
	-ruff check --fix
	@ruff format

# Examples:
#   make test
#   make test TESTS=tests/test_lllaplace.py ARGS='-k sample'
#   make test MARKS=public_api
test:
	$(PYTEST) $(TESTS) -n auto -m "$(MARKS)" $(ARGS)
