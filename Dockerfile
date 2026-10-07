# hrrmony with CUDA. CPU-only hosts work too (slower): torch falls back automatically.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HRRMONY_HOME=/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[gpu,web]"

VOLUME /data
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')" || exit 1
ENTRYPOINT ["hrrmony"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7860"]
