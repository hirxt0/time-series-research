FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY GeoMagAnalyst /app/GeoMagAnalyst

COPY GeoMagAnalyst/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/GeoMagAnalyst

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]