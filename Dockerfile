FROM rocker/r-ver:4.4.1

WORKDIR /app

# Python 3.12 requires deadsnakes PPA on Ubuntu 22.04 (Jammy)
# Also install all system libs needed for R package compilation and Python packages on arm64
RUN apt-get update && apt-get install -y software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-distutils \
        build-essential \
        gfortran \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        liblapack-dev \
        libblas-dev \
        libglib2.0-0 \
        libgomp1 \
        libjpeg-dev \
        zlib1g-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# deadsnakes Python 3.12 does not include pip - bootstrap it
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set python3.12 as default python3 and ensure pip tools are on PATH
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
ENV PATH="/usr/local/bin:$PATH"

# Install R packages in dependency order (geomorph depends on rrpp which depends on Rcpp)
RUN R -e "install.packages(c('Rcpp', 'RcppArmadillo', 'rrpp', 'shapes', 'geomorph'), repos='https://cloud.r-project.org/', dependencies=TRUE)"

# Verify R packages actually load (install.packages exits 0 even on failure)
RUN R -e "library(geomorph); library(shapes); cat('R packages verified OK\n')"

# Copy and install Python dependencies
# --prefer-binary uses pre-built wheels where available (important for arm64 packages like torch)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories. analysis/temp must be writable for R script I/O (input.csv, output.csv)
RUN mkdir -p /app/analysis/temp /app/static/requests && \
    chmod -R 777 /app/analysis/temp

RUN chmod +x /app/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
