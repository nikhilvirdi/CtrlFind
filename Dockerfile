FROM python:3.11-slim

WORKDIR /app

# Use the CPU build of PyTorch for the demo.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY demo.py .
COPY src/__init__.py src/cache.py src/config.py src/data.py src/encoder.py src/evolution.py src/execute.py src/versions.py src/
COPY artifacts/versions/bugs.json artifacts/versions/log.json artifacts/versions/objects.jsonl artifacts/versions/
COPY artifacts/versions/refs/v1.json artifacts/versions/refs/v2.json artifacts/versions/refs/v3.json artifacts/versions/refs/

ENV HOST=0.0.0.0
EXPOSE 7860
CMD ["python", "demo.py"]
