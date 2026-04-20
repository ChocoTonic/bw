#!/usr/bin/env Rscript
# Generate R reference values for Python parity testing.
# Run via: make parity-ref
#
# Requires: bw package installed, jsonlite package
#
# Output: bw-python/tests/reference_values/*.json

library(bw)
library(jsonlite)

# Use outdir if already set (e.g. from Docker mount), otherwise default
if (!exists("outdir")) outdir <- "bw-python/tests/reference_values"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# Helper: serialize model output to JSON-friendly format
# Matrices become lists of row vectors; scalars and vectors pass through.
serialize_model <- function(model) {
  lapply(model, function(x) {
    if (is.matrix(x)) {
      # Each row = one individual's time series
      lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))
    } else if (is.character(x) && length(x) == 1) {
      x  # scalar string like Model_Type
    } else if (is.logical(x) && length(x) == 1) {
      x  # scalar bool like Correct_Values
    } else {
      as.numeric(x)  # numeric vector like Time
    }
  })
}

# Debug: print dimensions of first model
debug_dims <- function(model, name) {
  for (k in names(model)) {
    v <- model[[k]]
    if (is.matrix(v)) {
      cat(sprintf("  %s: %s matrix %dx%d\n", name, k, nrow(v), ncol(v)))
    } else {
      cat(sprintf("  %s: %s length %d\n", name, k, length(v)))
    }
  }
}

cat("Generating adult model references...\n")

# Case 1: Male, baseline steady state (zero EIchange)
r1 <- adult_weight(76, 1.73, 36, "male")
debug_dims(r1, "adult_male")
write_json(serialize_model(r1),
           file.path(outdir, "adult_male_baseline.json"),
           auto_unbox = TRUE, digits = 17)

# Case 2: Male with explicit EI
r2 <- adult_weight(76, 1.73, 36, "male", EI = rep(2400, 1))
write_json(serialize_model(r2),
           file.path(outdir, "adult_male_ei2400.json"),
           auto_unbox = TRUE, digits = 17)

# Case 3: Female, baseline
r3 <- adult_weight(65, 1.65, 30, "female")
write_json(serialize_model(r3),
           file.path(outdir, "adult_female_baseline.json"),
           auto_unbox = TRUE, digits = 17)

# Case 4: Male with negative energy change (weight loss)
# R format: nrow = individuals, ncol = days
ei_change <- matrix(rep(-500, 365), nrow = 1, ncol = 365)
na_change <- matrix(rep(0, 365), nrow = 1, ncol = 365)
r4 <- adult_weight(90, 1.80, 40, "male", EIchange = ei_change, NAchange = na_change)
write_json(serialize_model(r4),
           file.path(outdir, "adult_male_deficit.json"),
           auto_unbox = TRUE, digits = 17)

# Case 5: Multiple individuals
r5 <- adult_weight(c(76, 65), c(1.73, 1.65), c(36, 30), c("male", "female"))
write_json(serialize_model(r5),
           file.path(outdir, "adult_multi.json"),
           auto_unbox = TRUE, digits = 17)

cat("Generating child model references...\n")

# Case 6: Male child, age 6, reference intake, 1 year
r6 <- child_weight(6, "male", days = 365)
debug_dims(r6, "child_male")
write_json(serialize_model(r6),
           file.path(outdir, "child_male_365d.json"),
           auto_unbox = TRUE, digits = 17)

# Case 7: Female child, age 10, 2 years
r7 <- child_weight(10, "female", days = 730)
write_json(serialize_model(r7),
           file.path(outdir, "child_female_730d.json"),
           auto_unbox = TRUE, digits = 17)

# Case 8: Male child, long run (5 years)
r8 <- child_weight(6, "male", days = 1825)
write_json(serialize_model(r8),
           file.path(outdir, "child_male_5yr.json"),
           auto_unbox = TRUE, digits = 17)

cat("Generating energy interpolation references...\n")

# Case 9: All 5 deterministic methods
energy <- matrix(c(2000, 2200, 1800), nrow = 1)
time_vec <- c(0, 5, 10)
for (method in c("Linear", "Exponential", "Logarithmic", "Stepwise_L", "Stepwise_R")) {
  # Call C++ EnergyBuilder directly via ::: (not the R wrapper which drops column 1)
  result <- bw:::EnergyBuilder(energy, time_vec, method)
  write_json(list(energy = list(as.numeric(result[1, ])),
                  method = method),
             file.path(outdir, paste0("energy_", tolower(method), ".json")),
             auto_unbox = TRUE, digits = 17)
}

cat(sprintf("Done. %d reference files written to %s/\n",
            length(list.files(outdir, pattern = "\\.json$")), outdir))
