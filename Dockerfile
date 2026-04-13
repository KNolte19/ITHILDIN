# Use Python 3.12 slim image as base
FROM python:3.12-slim
FROM rocker/r-ver:4.4.1 

# Set working directory
WORKDIR /app

# Install system dependencies including build tools required for compiling R packages
# build-essential (gcc, g++, make) is required to compile C++ R packages such as rrpp/Rcpp,
# which are dependencies of geomorph. Without it, install.packages() silently fails on slim images.
# gfortran is required for R packages that contain Fortran code.
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    build-essential \
    gfortran \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    liblapack-dev \
    libblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install R packages and verify the installation succeeded.
# Without the explicit stop() call, install.packages() exits 0 even on failure,
# which would allow the Docker build to succeed with geomorph missing.
RUN R -e "install.packages(c('geomorph', 'shapes'), repos='https://cloud.r-project.org/', dependencies=TRUE, verbose=TRUE, INSTALL_opts='--no-test-load')" 2>&1 | tee /tmp/r_install.log || true
RUN cat /tmp/r_install.log
RUN R -e "if (!requireNamespace('geomorph', quietly=TRUE)) stop('geomorph installation failed')"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories after copying application code.
# analysis/temp is created here (not before COPY) to avoid Docker overlay layer ordering issues
# where a directory created before COPY . . could be shadowed in the final image.
RUN mkdir -p /app/analysis/temp /app/static/requests

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
