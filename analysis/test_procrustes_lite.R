# Mimic procrustes.R behavior without geomorph
args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
script_dir <- dirname(script_path)
cat("Script dir:", script_dir, "\n")
setwd(script_dir)
cat("CWD after setwd:", getwd(), "\n")

# Try reading the input file
tryCatch({
  rawdat <- read.csv("temp/input.csv", header = TRUE, sep = ",")
  cat("Read input.csv: rows =", nrow(rawdat), "cols =", ncol(rawdat), "\n")
  cat("First few column names:", paste(head(colnames(rawdat)), collapse=", "), "\n")
  cat("Any NA values:", any(is.na(rawdat)), "\n")
  rawdat[] <- lapply(rawdat, as.numeric)
  coords_matrix <- as.matrix(rawdat)
  n_landmarks <- length(colnames(coords_matrix)) / 2
  cat("n_landmarks:", n_landmarks, "\n")
  cat("Matrix dimensions:", nrow(coords_matrix), "x", ncol(coords_matrix), "\n")
  cat("SUCCESS: Data loaded and processed correctly\n")
}, error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
})
