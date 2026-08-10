# Docker Deployment Guide for ITHILDIN Wing Analysis

This guide explains how to deploy the ITHILDIN Wing Analysis application using Docker.

## Prerequisites

- Docker Engine 20.10 or later
- Docker Compose V2 or later
- At least 4GB of available RAM
- **Model files** placed in `training/models/` 
  - ⚠️ **Important**: The Docker image includes these model files during build. Ensure they are present before running `docker compose up`
  - If model files are not included in your repository, you'll need to download them separately

## Quick Start

### 1. Choose your setup mode

ITHILDIN supports two deployment modes:

#### Local (CPU-only, no GPU required)

```bash
docker compose up -d
```

This uses `compose.yaml` and installs CPU-only dependencies (`rembg`, `onnxruntime`). Suitable for any machine without an NVIDIA GPU.

#### Server (NVIDIA GPU)

```bash
docker compose -f compose.yaml -f compose.server.yaml up -d
```

This merges `compose.server.yaml` on top of `compose.yaml`, enabling the NVIDIA runtime and installing GPU-accelerated dependencies (`rembg[gpu]`, `onnxruntime-gpu`). Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

> **Note:** The two modes install different Python packages. Rebuild the image when switching modes:
> ```bash
> docker compose [-f compose.yaml -f compose.server.yaml] up -d --build
> ```

Both modes will:
- Build the Docker image with all dependencies
- Start the container in detached mode
- Map port 127.0.0.1:8080 to container port 8080
- Create and mount the `./static/requests` volume for storing analysis results

### 2. Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:8080
```

### 3. Stop the Container

```bash
docker compose down
```

## Configuration

### Port Configuration

The application is configured to listen on `127.0.0.1:8080` by default. This means it's only accessible from the local machine. To make it accessible from other machines on your network, modify `compose.yaml`:

```yaml
ports:
  - "0.0.0.0:8080:8080"  # Accessible from all network interfaces
```

⚠️ **Security Warning**: Only expose the application to external networks if you have proper security measures in place.

### Volume Configuration

The application stores analysis results in `./static/requests`. This directory is mounted as a Docker volume, ensuring data persists even when the container is restarted or rebuilt.

The volume mapping in `compose.yaml`:
```yaml
volumes:
  - ./static/requests:/app/static/requests
```

### Environment Variables

⚠️ **Security Warning**: Always set a secure secret key for production deployments!

Generate a secure secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Then set it as an environment variable:
```bash
export FLASK_SECRET_KEY="your-secure-generated-key-here"
docker compose up -d
```

Or create a `.env` file in the project root (copy from `.env.example`):
```bash
cp .env.example .env
# Edit .env and replace the default key with a secure one
```

Or modify it directly in `compose.yaml`:
```yaml
environment:
  - FLASK_SECRET_KEY=your-secure-generated-key-here
```

## Production Deployment

### Gunicorn Configuration

The application uses Gunicorn as the WSGI server. Configuration is in `gunicorn.config.py`:

- **Workers**: `(CPU cores * 2) + 1` for optimal performance
- **Timeout**: 300 seconds (5 minutes) to handle long-running predictions
- **Binding**: `0.0.0.0:8080` inside the container

Adjust these settings based on your server specifications and workload.

### Resource Requirements

Recommended specifications:
- **CPU**: 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB+ for application and results
- **GPU**: Optional, for faster predictions (requires NVIDIA Docker runtime)

### Health Checks

The container includes a health check that pings the root endpoint every 30 seconds. View health status:

```bash
docker ps
```

Look for the "STATUS" column showing `healthy` or `unhealthy`.

## Troubleshooting

### Container won't start

Check logs:
```bash
docker compose logs -f
```

### Permission issues with volume

Ensure the `./static/requests` directory has proper permissions:
```bash
mkdir -p ./static/requests
chmod 777 ./static/requests  # Or adjust to your security requirements
```

### Out of memory errors

Increase Docker's memory limit in Docker Desktop settings or your daemon configuration.

### Model files not found

Ensure model weights are in place before building:
- `training/models/` - Main models
- `training/models_tsetse/` - Tsetse fly models (if using)

Update paths in `config.py` if needed.

## Maintenance

### Viewing Logs

```bash
# Follow logs in real-time
docker compose logs -f

# View last 100 lines
docker compose logs --tail=100
```

### Updating the Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose up -d --build
```

### Cleaning Up

Remove all containers, images, and volumes:
```bash
docker compose down --volumes --rmi all
```

⚠️ **Warning**: This will delete all stored analysis results in the volume.

## Advanced Configuration

### GPU Support

GPU acceleration is enabled via the server deployment mode. See [Quick Start — Server](#server-nvidia-gpu) for the command.

Prerequisites:
1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Ensure your Docker daemon has the NVIDIA runtime configured
3. Use the server compose command to build and run with GPU support
