# Gunakan Python versi ringan
FROM python:3.11-slim

# Set environment variables untuk cache & model download
ENV HOME=/app
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache/transformers
ENV TORCH_HOME=/app/huggingface_cache/torch

# Set direktori kerja di container
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

# Buat direktori cache & direktori vektorstore Chroma + permission agar tidak permission denied
RUN mkdir -p /app/huggingface_cache && chmod -R 777 /app/huggingface_cache
RUN mkdir -p /app/chroma_data && chmod -R 777 /app/chroma_data
RUN mkdir -p /app/uploads && chmod -R 777 /app/uploads

# Salin requirements dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh source code ke image
COPY . .

# Jika pakai .env (opsional)
# COPY .env .env

# Port Gradio
EXPOSE 7860

# Jalankan aplikasinya
CMD ["python", "app.py"]
