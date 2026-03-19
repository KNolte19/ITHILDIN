# Load necessary package
library(geomorph)

# Set working directory to the directory of the R script
args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
script_dir <- dirname(script_path)
setwd(script_dir)

# Load the input dataframe 
rawdat <- read.csv("temp/input.csv", header = TRUE, sep = ",")

# Transform all rawdat to numeric 
rawdat[] <- lapply(rawdat, as.numeric)

# Extract coordinate data
coords_matrix <- as.matrix(rawdat)

# Assume half of coord_cols are X, half Y
n_landmarks <- length(colnames(coords_matrix)) / 2
coords_array <- arrayspecs(coords_matrix, p = n_landmarks, k = 2)

# Procrustes Analysis
gproc <- gpagen(coords_array, print.progress = FALSE)
gpa_coors <- two.d.array(gproc$coords)

# Calculate mean shape
mean_shape <- mshape(gproc$coords)

# Calculate distances to mean per landmark per specimen
distances_per_specimen <- apply(gproc$coords, 3, function(specimen) {
  sqrt(rowSums((specimen - mean_shape)^2))
})

# distances_per_specimen is a matrix (n_landmarks x n_specimens)
# Compute average and max distance per specimen
avg_distances <- colMeans(distances_per_specimen)
max_distances <- apply(distances_per_specimen, 2, max)

# Combine data
output_data <- cbind(gpa_coors,
                     Avg_Procrustes_Dist = avg_distances,
                     Max_Procrustes_Dist = max_distances)

# Save to File
write.csv(output_data,
          file = file.path("temp/output.csv"),
          row.names = FALSE)
