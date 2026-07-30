FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by trimesh/open3d/rtree at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libgomp1 \
        libspatialindex-dev \
        && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (swap for a CUDA wheel index if you have a GPU host)
RUN pip install --no-cache-dir torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-geometric>=2.4.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "inference/remesh.py"]
CMD ["--help"]
