import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from db import init_db, SessionLocal, Session, Document, ChatHistory
import uuid
from dotenv import load_dotenv
import os

os.environ["HOME"] = "/app"
os.environ["HF_HOME"] = "/app/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/huggingface_cache/transformers"
os.environ["TORCH_HOME"] = "/app/huggingface_cache/torch"

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Init DB
init_db()

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True}
)
retriever_dict = {}
memory_dict = {}

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"  
)

custom_prompt = PromptTemplate.from_template("""
Kamu adalah asisten AI yang ahli membaca dokumen dan menjawab pertanyaan pengguna berdasarkan isi dokumen tersebut.

Berdasarkan konteks berikut, jawablah pertanyaan dengan jelas dan ringkas, dalam bahasa Indonesia yang formal.

### Konteks:
{context}

### Pertanyaan:
{question}

### Jawaban:
""")



# Upload handler
import shutil
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from db import SessionLocal, Session, Document
from langchain.memory import ConversationBufferMemory

def handle_upload(file, session_id):
    try:
        print("✅ Mulai proses upload dokumen...")

        # 1. Simpan file ke direktori permanen
        os.makedirs("/app/uploads", exist_ok=True)
        save_path = f"/app/uploads/{session_id}_{uuid.uuid4().hex}.pdf"
        shutil.copyfile(file.name, save_path)
        print(f"→ File disimpan ke: {save_path}")

        # 2. Load dokumen
        print("→ Membaca dokumen...")
        loader = PyPDFLoader(save_path)
        docs = loader.load()
        if not docs:
            return "❌ Gagal membaca isi dokumen."

        # 3. Split dokumen
        print("→ Memotong dokumen jadi chunk...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print(f"→ Jumlah chunk: {len(chunks)}")
        if not chunks:
            return "❌ Tidak ada konten yang bisa diproses."

        # 4. Embedding dan simpan ke Chroma
        print("→ Menyiapkan embedding dan simpan ke vectorstore...")
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True}
        )

        vectordb = Chroma.from_documents(
            chunks,
            embedding=embedding,
            persist_directory="/app/chroma_data"
        )
        vectordb.persist()
        print("✅ Dokumen berhasil disimpan ke Chroma.")

        # 5. Simpan retriever & memory
        retriever_dict[session_id] = vectordb.as_retriever()
        memory_dict[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer", k=5
        )

        # 6. Simpan metadata ke database
        db = SessionLocal()
        if not db.query(Session).filter_by(id=session_id).first():
            db.add(Session(id=session_id))
        db.add(Document(session_id=session_id, filename=file.name))
        db.commit()
        db.close()
        print("✅ Metadata dokumen disimpan ke database.")

        return "✅ File berhasil diupload dan diproses."

    except Exception as e:
        print("❌ Terjadi error:", str(e))
        return f"❌ Terjadi kesalahan saat upload: {str(e)}"


# Pertanyaan handler
def handle_question(question, session_id):
    if session_id not in retriever_dict:
        return "❌ Belum upload dokumen."

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever_dict[session_id],
        memory=memory_dict[session_id],
        return_source_documents=False,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    result = chain.invoke(question)
    answer = result["answer"]

    db = SessionLocal()
    db.add(ChatHistory(session_id=session_id, question=question, answer=answer))
    db.commit()
    db.close()

    return answer

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Chat AI Dokumen - RAG with Chroma + SQLite")
    session_id = gr.Textbox(label="Session ID", value=str(uuid.uuid4()), visible=False)

    file = gr.File(label="Upload PDF")
    status = gr.Textbox(label="Status")
    question = gr.Textbox(label="Pertanyaan", placeholder="Tanya isi dokumen...")
    answer = gr.Textbox(label="Jawaban")

    file.change(handle_upload, inputs=[file, session_id], outputs=status)
    question.submit(handle_question, inputs=[question, session_id], outputs=answer)

demo.launch(server_name="0.0.0.0", server_port=7860)
