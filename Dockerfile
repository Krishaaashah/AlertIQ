# AlertIQ — Backend API image
#
# Builds a container that serves the FastAPI backend (api/main.py).
# The ML/RL pipeline itself (rule engine, model training, RL training) is
# expected to be run separately (see run_all.py / run_phase*.py) — this
# image serves whatever trained artifacts already exist under models/ and
# outputs/, falling back to safe defaults if they don't (see README).
#
# Build:
#   docker build -t alertiq-api .
# Run:
#   docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs -v $(pwd)/models:/app/models alertiq-api
# Or use docker-compose (recommended — also runs the frontend):
#   docker compose up --build

FROM python:3.11-slim

WORKDIR /app

# System deps needed by scientific-python wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY dataset/ ./dataset/

# outputs/ and models/ are populated by running the pipeline (run_all.py)
# or mounted as volumes in docker-compose.yml; create empty dirs so the
# app can start cleanly on first boot without them.
RUN mkdir -p outputs models

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
