# Use rocker/r-ver as base - provides stable R 4.4.1 with ARM64 support
FROM rocker/r-ver:4.4.1

# Set working directory
WORKDIR /app

# Install system dependencies: Python 3.12, build tools, and R compilation libraries
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    gfortran \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    liblapack-dev \
    libblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install R packages: install dependencies first, then geomorph and shapes
# dependencies=TRUE ensures all transitive dependencies are installed
RUN R -e "install.packages(c('Rcpp', 'RcppArmadillo', 'rrpp', 'shapes', 'geomorph'), repos='https://cloud.r-project.org/', dependencies=TRUE)"

# Verify R package installations succeeded
# Without explicit stop(), install.packages() exits 0 even on failure
RUN R -e "if (!requireNamespace('geomorph', quietly=TRUE)) stop('geomorph installation failed')"
RUN R -e "if (!requireNamespace('shapes', quietly=TRUE)) stop('shapes installation failed')"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

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
