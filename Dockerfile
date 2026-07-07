# Minimal image for primer-design-suite. Heavy optional extras (torch,
# streamlit, anthropic, snakemake, mlflow) are NOT installed by default —
# add them via `pip install .[predictor,copilot]` in a derived image or at
# `docker run` time as needed.

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY primer_core ./primer_core
COPY predictor ./predictor
COPY copilot ./copilot

RUN pip install --no-cache-dir .

CMD ["python", "-c", "import primer_core; print('primer_core ready')"]
