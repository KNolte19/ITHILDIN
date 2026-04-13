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

# Install R packages explicitly into R 4.4.2 library
RUN /opt/R/4.4.2/bin/Rscript -e "\
    lib <- '/opt/R/4.4.2/lib/R/library'; \
    install.packages(c('geomorph','shapes','RRPP','rgl','ape','ggplot2','jpeg'), \
    lib=lib, repos='https://cloud.r-project.org/')"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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