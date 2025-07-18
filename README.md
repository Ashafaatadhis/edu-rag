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
