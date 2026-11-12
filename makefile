# Locally, the test tools and the ASDL backend (PyPI `asdfghjkl==0.1a5`, the `asdl`
# extra) come in through `uv run --with`, so the torch in the venv stays as is.
# `--no-sync` keeps uv from reinstalling the CUDA build pinned in `uv.lock`.
# CI installs everything into `.venv` and runs `make <target> PYTHON=.venv/bin/python`.
TEST_DEPS = --with pytest --with scipy --with pytest-mock --with pytest-xdist \
	--with dill --with "asdfghjkl==0.1a5"
PYTHON ?= uv run --no-sync $(TEST_DEPS) python

# The old asdfghjkl backend, as installed by CI's old-asdfghjkl job.
OLD_ASDFGHJKL = asdfghjkl-old @ git+https://github.com/wiseodd/asdl@9c7640f8a870e82d5b439e6da59d753b207003b0

TESTS ?= tests
MARKS ?= not public_api and not cuda and not asdfghjkl
ARGS ?=

.PHONY: ruff test test-asdfghjkl test-public-api

ruff:
	-ruff check --fix
	@ruff format

# Examples:
#   make test
#   make test TESTS=tests/test_lllaplace.py ARGS='-k sample'
#   make test MARKS=cuda
test:
	$(PYTHON) -m pytest $(TESTS) -n auto -m "$(MARKS)" $(ARGS)

# Only the tests that need the old asdfghjkl backend.
test-asdfghjkl: TEST_DEPS += --with "$(OLD_ASDFGHJKL)"
test-asdfghjkl:
	$(PYTHON) -m pytest $(TESTS) -n auto -m "asdfghjkl and not public_api and not cuda" $(ARGS)

# The public API contract, see "Public API" in docs/devs_guide.md.
test-public-api:
	$(PYTHON) -m pytest $(TESTS) -m public_api -v $(ARGS)
