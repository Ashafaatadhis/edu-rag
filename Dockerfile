# Gunakan Python versi ringan
FROM python:3.11-slim

# Set environment variables untuk cache directory Hugging Face & Transformers
ENV HOME=/app
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache/transformers
ENV TORCH_HOME=/app/huggingface_cache/torch

# Set working directory
WORKDIR /app

# Install dependency dasar OS
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Buat direktori cache dan beri izin akses
RUN mkdir -p /app/huggingface_cache && chmod -R 777 /app/huggingface_cache

# Copy requirements dan install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project ke container
COPY . .

# Ekspos port default Gradio (walau kamu mungkin gak pakai Gradio)
EXPOSE 7860

# Jalankan aplikasi
CMD ["python", "app.py"]
