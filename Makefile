IMAGE      := bw-r-pkg
IMAGE_BASE := bw-r-base
FILE       ?=

.PHONY: base build build-no-cache test test-fast test-file shell clean

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
