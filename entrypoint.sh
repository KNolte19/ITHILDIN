#!/bin/bash

# Exit on error
set -e

echo "Starting ITHILDIN Wing Analysis Application..."

# Ensure the requests directory exists
mkdir -p /app/static/requests

# Warn if using insecure default secret key
if [ "${FLASK_SECRET_KEY:-CHANGEME-INSECURE-DEFAULT}" = "CHANGEME-INSECURE-DEFAULT" ]; then
    echo "WARNING: Using insecure default FLASK_SECRET_KEY!"
    echo "Please set a secure secret key via environment variable for production use."
fi

# Start gunicorn with the config file
exec gunicorn app:app -b 0.0.0.0:8080 --timeout 360
