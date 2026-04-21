IMAGE      := bw-r-pkg
IMAGE_BASE := bw-r-base
FILE       ?=

.PHONY: base build build-no-cache test test-fast test-file shell clean \
       py-test parity-ref parity parity-smoke parity-full _parity-prereqs

# Build only the base image (system libs + R packages). Rarely changes.
base:
	docker build --target base -t $(IMAGE_BASE) .

# Full build: base + install the bw package
build:
	docker build -t $(IMAGE) .

build-no-cache:
	docker build --no-cache -t $(IMAGE) .

# Full rebuild + test
test: build
	docker run --rm $(IMAGE) Rscript -e "testthat::test_local()" 2>&1

# Fast test: mount source into cached base image, install + test in one shot.
# Skips Docker rebuild entirely — just re-installs the package and runs tests.
test-fast: base
	docker run --rm -m 1g -v "$(CURDIR):/pkg" -w /pkg $(IMAGE_BASE) \
		sh -c "R CMD INSTALL --no-docs --no-multiarch . >/dev/null 2>&1 && Rscript -e 'testthat::test_local()'" 2>&1

# Run a single test file: make test-file FILE=test_adult_weight.R
test-file: base
	docker run --rm -m 1g -v "$(CURDIR):/pkg" -w /pkg $(IMAGE_BASE) \
		sh -c "R CMD INSTALL --no-docs --no-multiarch . >/dev/null 2>&1 && Rscript -e 'options(warn=-1); testthat::test_file(\"tests/testthat/$(FILE)\")'" 2>&1

shell: build
	docker run --rm -it $(IMAGE) R

clean:
	docker rmi -f $(IMAGE) $(IMAGE_BASE)

# ─── Python targets ───────────────────────────────────────────────────

# Stamp file: uv sync runs once, subsequent targets skip it.
bw-python/.venv/.synced: bw-python/pyproject.toml bw-python/uv.lock
	cd bw-python && uv sync --quiet
	@touch $@

# Run Python unit tests in parallel (no R reference values needed)
py-test: bw-python/.venv/.synced
	cd bw-python && uv run python -m pytest tests/ \
		--ignore=tests/test_parity.py \
		--ignore=tests/test_parity_smoke.py -v

# Generate R reference values for parity testing.
# Stamp-guarded: only regenerates when R source or the generation script changes.
# Docker writes ONLY to bw-python/tests/reference_values/ — no overlap with py-test.
R_SOURCES := $(wildcard src/*.cpp src/*.h R/*.R scripts/generate_reference_values.R)

bw-python/tests/reference_values/.generated: $(R_SOURCES) Dockerfile
	docker build -t $(IMAGE) .
	docker run --rm -v "$(CURDIR)/scripts:/scripts:ro" \
		-v "$(CURDIR)/bw-python/tests/reference_values:/out" \
		-w /pkg $(IMAGE) \
		Rscript -e "install.packages('jsonlite', repos='https://cloud.r-project.org', quiet=TRUE); outdir <- '/out'; source('/scripts/generate_reference_values.R')"
	@touch $@

parity-ref: bw-python/tests/reference_values/.generated

# Full parity test: Python vs R (requires reference values from parity-ref)
parity: bw-python/.venv/.synced
	cd bw-python && uv run python -m pytest tests/test_parity.py -v

# Smoke parity test: 1 adult + 1 child + 1 energy (requires at least 3 reference files)
parity-smoke: bw-python/.venv/.synced
	cd bw-python && uv run python -m pytest tests/test_parity_smoke.py -v

# Full pipeline: generate R refs and run Python unit tests concurrently,
# then run parity comparison once refs are ready.
#
# Safe for make -j: py-test and parity-ref are independent (different dirs,
# different runtimes). py-sync is order-only so uv sync completes before
# any pytest invocation. parity waits for both via _parity-prereqs.
parity-full: _parity-prereqs
	cd bw-python && uv run python -m pytest tests/test_parity.py -v

# With make -j, parity-ref (Docker/R) and py-test (Python) run concurrently.
# Both complete before parity-full's recipe fires.
_parity-prereqs: parity-ref py-test
