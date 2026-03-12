FROM python:3.11-slim

WORKDIR /app

# Install system deps for lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

EXPOSE 8000

CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}
