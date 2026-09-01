# Load necessary package
library(geomorph)

# Set working directory to the directory of the R script
args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
script_dir <- dirname(script_path)
setwd(script_dir)

# Optional trailing argument: path to a stored consensus shape (switches GPA -> OPA)
user_args <- commandArgs(trailingOnly = TRUE)

# Load the input dataframe
coords_array <- readland.tps("temp/input.tps", specID = "ID")

# Procrustes Analysis
if (length(user_args) >= 1) {
  # OPA: align each specimen to the stored consensus (center, scale, rotate).
  # Semilandmarks are treated as fixed here; sliding only happens during the
  # reference GPA that produced the consensus.
  mean_shape <- as.matrix(read.csv(user_args[1]))
  proc_coords <- coords_array
  # two.d.array builds its "1.X, 1.Y, ..." column names from dimnames (gpagen
  # normally sets these)
  dimnames(proc_coords)[[1]] <- 1:dim(proc_coords)[1]
  dimnames(proc_coords)[[2]] <- c("X", "Y")
  for (i in 1:dim(coords_array)[3]) {
    spec <- scale(coords_array[, , i], scale = FALSE)
    spec <- spec / sqrt(sum(spec^2))  # unit centroid size, matching gpagen
    s <- svd(t(spec) %*% mean_shape)
    if (det(s$u %*% t(s$v)) < 0) s$u[, 2] <- -s$u[, 2]  # forbid reflection
    proc_coords[, , i] <- spec %*% (s$u %*% t(s$v))
  }
} else {
  # Load the sliding semilandmarks definition
  sliders_matrix <- as.matrix(read.csv("temp/sliders.csv", header = FALSE))

  # GPA with Sliding
  # curves: the matrix of sliding triplets
  # ProcD = TRUE: slides landmarks to minimize Procrustes distance (recommended)
  gproc <- gpagen(coords_array,
                  curves = sliders_matrix,
                  ProcD = TRUE,
                  print.progress = FALSE)
  proc_coords <- gproc$coords
  mean_shape <- mshape(gproc$coords)
}

# Extract the aligned coordinates
gpa_coors <- two.d.array(proc_coords)

# Calculate Euclidean distances to mean per landmark per specimen
distances_per_specimen <- apply(proc_coords, 3, function(specimen) {
  sqrt(rowSums((specimen - mean_shape)^2))
})

# distances_per_specimen is a matrix (n_landmarks x n_specimens)
avg_distances <- colMeans(distances_per_specimen)
max_distances <- apply(distances_per_specimen, 2, max)

# Export Results
output_data <- cbind(gpa_coors,
                     Avg_Procrustes_Dist = avg_distances,
                     Max_Procrustes_Dist = max_distances)

write.csv(output_data,
          file = file.path("temp/output.csv"),
          row.names = FALSE)