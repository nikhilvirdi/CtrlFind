FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image small; the pipeline never needs a GPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
EXPOSE 7860
CMD ["python", "demo.py"]
