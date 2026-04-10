FROM python:3.11-slim

# System deps for geospatial libs
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libffi-dev \
    libssl-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source — data/ is git-ignored so not copied
COPY . .

# Data dirs are mounted at runtime, not baked in
RUN mkdir -p data/raw data/processed data/audit figures

# Env vars loaded from .env at runtime (never baked into image)
ENV PYTHONPATH=/project
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Jupyter config — no browser, token-less for local dev
# Override in production with proper auth
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
