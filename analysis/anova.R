# Robust Procrustes ANOVA script
library(geomorph)

# Helper Script to call path when using as subprocess
get_script_dir <- function() {
  # 1) If run with Rscript, commandArgs contains --file=...
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  
  # 2) If in RStudio, try rstudioapi
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    path <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(path)) return(dirname(normalizePath(path)))
  }
  
  # 3) When sourced, sys.frame may contain $ofile
  ofile <- tryCatch({
    env <- sys.frames()
    # iterate frames to find 'ofile' if present
    for (f in rev(env)) {
      if (!is.null(f$ofile)) return(dirname(normalizePath(f$ofile)))
    }
    NULL
  }, error = function(e) NULL)
  
  if (!is.null(ofile)) return(ofile)
  
  # 4) Last resort: current working directory
  return(getwd())
}

script_dir <- get_script_dir()
setwd(script_dir)

# Load the dataset
rawdat <- read.csv("temp/input.csv", header = TRUE, sep = ",", dec = ".")

# Identify coordinate columns ending in .X or .Y
coord_cols_raw <- grep("\\.[XY]$", names(rawdat), value = TRUE)
if (length(coord_cols_raw) == 0) stop("No coordinate columns found. Columns must end with .X or .Y")

# Derive landmark base names and order columns as X then Y per landmark
coord_base <- sub("\\.(X|Y)$", "", coord_cols_raw)
unique_bases <- unique(coord_base)
n_landmarks <- length(unique_bases)

ordered_cols <- unlist(lapply(unique_bases, function(b) c(paste0(b, ".X"), paste0(b, ".Y"))), use.names = FALSE)

# Check that ordered columns actually exist in the data
missing_cols <- setdiff(ordered_cols, names(rawdat))
if (length(missing_cols) > 0) {
  stop("Missing coordinate columns when ordering X/Y pairs: ", paste(missing_cols, collapse = ", "))
}

# Build coords array (p x k x n) for geomorph
coords_mat <- as.matrix(rawdat[, ordered_cols])
coords_array <- arrayspecs(coords_mat, p = n_landmarks, k = 2)

# Identify metadata columns (i.e., non-coordinate columns)
metadata_cols <- setdiff(names(rawdat), ordered_cols)

# Build model formula
if (length(metadata_cols) == 0) {
  model_formula <- coords_array ~ 1
} else {
  model_formula <- as.formula(paste("coords_array ~", paste(metadata_cols, collapse = " + ")))
}

# Perform Procrustes ANOVA
anova_result <- procD.lm(model_formula, data = rawdat, iter = 999)

# Extract ANOVA table and save to CSV
anova_table <- as.data.frame(summary(anova_result)$table)
write.csv(anova_table, file = "temp/output.csv", row.names = TRUE)