# One-command image for the primer-design-suite Streamlit app.
#
#   docker build -t primer-design-suite .
#   docker run --rm -p 8501:8501 primer-design-suite      # -> http://localhost:8501
#
# The "Score manual primers" tab works with no API key; the Design (agent) tab
# needs one — pass it at run time:
#   docker run --rm -p 8501:8501 -e OPENAI_API_KEY=sk-... primer-design-suite
#
# On first launch the head-A models build automatically (~20 s) from the
# committed openPrimeR data, so the image ships no trained artifacts.

FROM python:3.11-slim

# libgomp1: LightGBM/XGBoost OpenMP runtime (required).
# ncbi-blast+: only the Specificity tab uses it (kept in so all tabs work).
# build-essential: fallback for any dependency without a manylinux wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ncbi-blast+ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (cached across code edits).
COPY pyproject.toml README.md ./
COPY primer_core ./primer_core
COPY predictor ./predictor
COPY copilot ./copilot
RUN pip install --no-cache-dir ".[copilot]"

# Committed data (feature matrix + templates) drives the build-on-launch models,
# and the helper script builds the optional BLAST demo DB.
COPY data/public ./data/public
COPY scripts ./scripts

EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "copilot/app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
