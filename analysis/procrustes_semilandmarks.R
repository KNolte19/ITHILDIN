# Load necessary package
library(geomorph)

# Set working directory to the directory of the R script
args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
script_dir <- dirname(script_path)
setwd(script_dir)

# Load the input dataframe 
coords_array <- readland.tps("temp/input.tps", specID = "ID")

# Load the sliding semilandmarks definition
sliders_matrix <- as.matrix(read.csv("temp/sliders.csv", header = FALSE))

# Procrustes Analysis with Sliding
# curves: the matrix of sliding triplets
# ProcD = TRUE: slides landmarks to minimize Procrustes distance (recommended)
gproc <- gpagen(coords_array, 
                curves = sliders_matrix, 
                ProcD = TRUE, 
                print.progress = FALSE)

# Extract the aligned (slid) coordinates
gpa_coors <- two.d.array(gproc$coords)

# Calculate mean shape and distances
mean_shape <- mshape(gproc$coords)

# Calculate Euclidean distances to mean per landmark per specimen
distances_per_specimen <- apply(gproc$coords, 3, function(specimen) {
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