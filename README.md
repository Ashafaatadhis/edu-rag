---
title: EduRAG Chatbot
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🧠 RAG Assistant - Chat dengan Dokumenmu!

**RAG Assistant** adalah aplikasi AI berbasis [RAG (Retrieval-Augmented Generation)](https://www.promptingguide.ai/techniques/rag) yang memungkinkan kamu untuk:

- Mengunggah dokumen (PDF)
- Mengajukan pertanyaan berdasarkan isi dokumen
- Mendapatkan jawaban yang relevan secara kontekstual
- Disimpan secara lokal menggunakan ChromaDB

Dibangun menggunakan:

- [LangChain](https://www.langchain.com/)
- [Gradio](https://gradio.app/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 🚀 Fitur

- ✅ Unggah file PDF
- ✅ Ekstraksi & vektorisasi isi dokumen
- ✅ Chat interaktif dengan riwayat percakapan (ConversationalRetrievalChain)
- ✅ Backend lokal dengan penyimpanan Chroma
- ✅ Multi-session user (dengan `session_id`)
- ✅ Support SQLite untuk simpan metadata (opsional)

---

## 📦 Instalasi

### 1. Clone repositori

```bash
git clone https://github.com/namamu/rag-assistant.git
cd rag-assistant
```

### 2. Buat environtment (opsional)

```bash
conda create -n rag_env python=3.11
conda activate rag_env
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

## 🐳 Menjalankan dengan Docker

```bash
docker build -t rag-assistant.
docker run -p 7860:7860 rag-assistant
```

## 🧪 Menjalankan Secara Lokal

```bash
python app.py
```

Buka broser ke: http://localhost:7860

## 📁 Struktur Proyek

```bash
rag-assistant/
├── app.py                  # Entry point Gradio
├── rag_utils.py            # Utility untuk RAG (vectorizer, retriever, memory)
├── chroma_db/              # Folder penyimpanan Chroma
├── requirements.txt
└── README.md
```

## 🙌 Kontribusi

Pull request & issue sangat terbuka!
Silakan fork repo ini dan kirim perubahanmu 💙
