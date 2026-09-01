# Use Python 3.10.12 slim image as base
FROM python:3.10.12-slim

# Set working directory
WORKDIR /app

# Create Temporary Folders
RUN mkdir -p /app/analysis/temp && chown -R 1000:1000 /app/analysis/temp

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gnupg2 \
    curl \
    cmake \
    libpng-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgfortran5 \
    libblas-dev \
    liblapack-dev \
    libuv1-dev \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R 4.4.2 from Posit (auto-detects arm64 or amd64)
RUN apt-get update \
    && ARCH=$(dpkg --print-architecture) \
    && curl -fsSL https://cdn.posit.co/r/debian-12/pkgs/r-4.4.2_1_${ARCH}.deb -o /tmp/r-4.4.2.deb \
    && apt-get install -y --no-install-recommends /tmp/r-4.4.2.deb \
    && rm /tmp/r-4.4.2.deb \
    && rm -rf /var/lib/apt/lists/*

# Add Posit R to PATH
ENV PATH="/opt/R/4.4.2/bin:${PATH}"

# Install R packages explicitly into R 4.4.2 library.
# install.packages never exits non-zero, so verify every package loads and
# fail the build otherwise.
RUN /opt/R/4.4.2/bin/Rscript -e "\
    lib <- '/opt/R/4.4.2/lib/R/library'; \
    pkgs <- c('geomorph','shapes','RRPP','rgl','ape','ggplot2','jpeg'); \
    install.packages(pkgs, lib=lib, repos='https://cloud.r-project.org/'); \
    missing <- pkgs[!sapply(pkgs, requireNamespace, quietly=TRUE)]; \
    if (length(missing)) stop('R packages failed to install: ', paste(missing, collapse=', '))"

# Build argument to select the requirements file (default: local/CPU build)
ARG REQUIREMENTS_FILE=requirements.local.txt

# Copy requirements files
COPY requirements.local.txt requirements.server.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

# Copy application code
COPY . .

# Create static/requests directory for volume mount
RUN mkdir -p /app/static/requests

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]