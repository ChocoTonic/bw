#!/usr/bin/env Rscript
# Generate reference values from the R bw package for parity testing.
# Output: 13 JSON files in /output/ (mounted volume)
#
# 5 adult, 3 child, 5 energy interpolation

library(bw)
library(jsonlite)

outdir <- "/output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Helper: convert a model result list to a JSON-friendly structure.
# Matrices -> nested lists (row-major), vectors stay as vectors.
to_json_obj <- function(result) {
  obj <- list()
  for (nm in names(result)) {
    val <- result[[nm]]
    if (is.matrix(val)) {
      # Store as list of rows (each row is one individual or one timestep)
      obj[[nm]] <- lapply(seq_len(nrow(val)), function(i) as.numeric(val[i, ]))
    } else if (is.numeric(val) || is.integer(val)) {
      obj[[nm]] <- as.numeric(val)
    } else if (is.logical(val)) {
      obj[[nm]] <- val
    } else if (is.character(val)) {
      obj[[nm]] <- val
    } else if (is.list(val)) {
      # BMI_Category is a list of character vectors
      obj[[nm]] <- val
    } else {
      obj[[nm]] <- val
    }
  }
  obj
}

write_ref <- function(obj, filename) {
  path <- file.path(outdir, filename)
  writeLines(toJSON(obj, auto_unbox = FALSE, digits = 15, pretty = TRUE), path)
  cat("Wrote:", path, "\n")
}

# =========================================================================
# ADULT WEIGHT (5 scenarios)
# =========================================================================

cat("=== Adult scenarios ===\n")

# 1. Male baseline (76 kg, 1.73 m, age 36, defaults, 365 days)
res <- adult_weight(76, 1.73, 36, "male", days = 365)
write_ref(to_json_obj(res), "adult_male_baseline.json")

# 2. Female baseline (60 kg, 1.60 m, age 30, defaults, 365 days)
res <- adult_weight(60, 1.60, 30, "female", days = 365)
write_ref(to_json_obj(res), "adult_female_baseline.json")

# 3. Male with custom EI (76 kg, 1.73 m, age 36, EI=2500, 365 days)
res <- adult_weight(76, 1.73, 36, "male", EI = 2500, days = 365)
write_ref(to_json_obj(res), "adult_male_ei2500.json")

# 4. Male with custom fat (76 kg, 1.73 m, age 36, fat=15, 365 days)
res <- adult_weight(76, 1.73, 36, "male", fat = 15, days = 365)
write_ref(to_json_obj(res), "adult_male_fat15.json")

# 5. Male with custom EI and fat (76 kg, 1.73 m, age 36, EI=2500, fat=15, 365 days)
res <- adult_weight(76, 1.73, 36, "male", EI = 2500, fat = 15, days = 365)
write_ref(to_json_obj(res), "adult_male_ei2500_fat15.json")

# =========================================================================
# CHILD WEIGHT (3 scenarios)
# =========================================================================

cat("=== Child scenarios ===\n")

# 6. Male age 6 (default EI, 365 days, dt=1)
res <- child_weight(6, "male", days = 365, dt = 1)
write_ref(to_json_obj(res), "child_male_age6.json")

# 7. Female age 10 (default EI, 365 days, dt=1)
res <- child_weight(10, "female", days = 365, dt = 1)
write_ref(to_json_obj(res), "child_female_age10.json")

# 8. Male age 6 with Richardson logistic params
res <- child_weight(6, "male", days = 365, dt = 1,
                    richardsonparams = list(K = 2700, Q = 10, B = 12, A = 3, nu = 4, C = 1))
write_ref(to_json_obj(res), "child_male_richardson.json")

# =========================================================================
# ENERGY INTERPOLATION (5 scenarios)
# =========================================================================

cat("=== Energy interpolation scenarios ===\n")

energy_vals <- c(100, 200)
time_vals <- c(0, 10)

# 9. Linear
res <- energy_build(energy_vals, time_vals, "Linear")
write_ref(list(values = as.numeric(res), method = "Linear",
               energy = energy_vals, time = time_vals), "energy_linear.json")

# 10. Exponential
res <- energy_build(energy_vals, time_vals, "Exponential")
write_ref(list(values = as.numeric(res), method = "Exponential",
               energy = energy_vals, time = time_vals), "energy_exponential.json")

# 11. Logarithmic
res <- energy_build(energy_vals, time_vals, "Logarithmic")
write_ref(list(values = as.numeric(res), method = "Logarithmic",
               energy = energy_vals, time = time_vals), "energy_logarithmic.json")

# 12. Stepwise_L
res <- energy_build(energy_vals, time_vals, "Stepwise_L")
write_ref(list(values = as.numeric(res), method = "Stepwise_L",
               energy = energy_vals, time = time_vals), "energy_stepwise_l.json")

# 13. Brownian (seeded for reproducibility)
set.seed(42)
res <- energy_build(energy_vals, time_vals, "Brownian")
write_ref(list(values = as.numeric(res), method = "Brownian",
               energy = energy_vals, time = time_vals, seed = 42), "energy_brownian.json")

cat("\nDone! Generated 13 reference files.\n")
