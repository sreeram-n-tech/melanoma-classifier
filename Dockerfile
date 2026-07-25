# Container for the melanoma-classifier FastAPI demo (webapp/).
# CPU-only inference, no GPU. Runs anywhere that runs a container:
# Hugging Face Spaces (Docker SDK — it routes traffic to port 7860),
# Render, Fly.io, Cloud Run.
FROM python:3.11-slim

# opencv-python (used by the Phase-3 preprocessing toggles) needs these libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of torch first, from PyTorch's CPU index, so we never
# pull the ~2 GB CUDA wheel. Then the rest of the serving deps.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

# Only the code the app actually imports, at the same relative layout the app's
# sys.path expects: the web app, plus the research model/ and preprocessing/
# packages it reaches into. webapp/ also carries the tiny committed weights.
COPY webapp/        webapp/
COPY model/         model/
COPY preprocessing/ preprocessing/

EXPOSE 7860
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "7860"]
