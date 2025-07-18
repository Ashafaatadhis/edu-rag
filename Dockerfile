# Gunakan Python versi ringan
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependency dasar OS (biar bisa install wheel, sentence-transformers, chromadb)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy file ke image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file project
COPY . .

# Ekspos port default Gradio
EXPOSE 7860

# Command untuk menjalankan Gradio app
CMD ["python", "app.py"]
