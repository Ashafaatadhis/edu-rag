# Gunakan Python versi ringan
FROM python:3.11-slim

# Set environment variables untuk cache & model download
ENV HOME=/app
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache/transformers
ENV TORCH_HOME=/app/huggingface_cache/torch

# Set direktori kerja
WORKDIR /app

# Install dependensi sistem yang dibutuhkan Chroma dan sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Buat cache huggingface (optional) dan pastikan writeable
RUN mkdir -p /app/huggingface_cache && chmod -R 777 /app/huggingface_cache

# Salin requirements dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh source code
COPY . .

# Port Gradio
EXPOSE 7860

# Jalankan aplikasi
CMD ["python", "app.py"]
