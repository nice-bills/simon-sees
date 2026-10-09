# Simon Sees on an NVIDIA GPU host (e.g. a DigitalOcean GPU Droplet). See docs/deploy-digitalocean.md.
# CPU-only build: docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 simon
USER simon
ENV HOME=/home/simon PATH=/home/simon/.local/bin:$PATH HF_HOME=/home/simon/.cache/huggingface \
    PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /home/simon/app

COPY --chown=simon requirements*.txt ./
RUN pip install --no-cache-dir --user --extra-index-url ${TORCH_INDEX} -r requirements-model.txt \
    && pip install --no-cache-dir --user -r requirements-extras.txt || true

COPY --chown=simon . .
EXPOSE 8000
# weights download on first start into the mounted HF cache (~6 GB for d1-3B)
CMD ["python", "-m", "simon_sees", "serve", "--host", "0.0.0.0", "--port", "8000"]
