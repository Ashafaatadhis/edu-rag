import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import shutil
import os
import uuid
from dotenv import load_dotenv

# DB
from db import init_db, SessionLocal, Session, Document, ChatHistory, get_engine

# Set cache path
os.environ["HOME"] = "/app"
os.environ["HF_HOME"] = "/app/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/huggingface_cache/transformers"
os.environ["TORCH_HOME"] = "/app/huggingface_cache/torch"

# Load .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Init DB
init_db()
engine = get_engine()

# Embedding & model init
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True}
)
retriever_dict = {}
memory_dict = {}

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
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
def handle_upload(file, session_id):
    try:
        print("✅ Mulai proses upload dokumen...")

        # Simpan PDF ke /tmp
        os.makedirs("/tmp/uploads", exist_ok=True)
        saved_filename = f"{session_id}_{uuid.uuid4().hex}.pdf"
        save_path = f"/tmp/uploads/{saved_filename}"
        shutil.copyfile(file.name, save_path)
        print(f"→ File disimpan ke: {save_path}")

        # Load & split
        loader = PyPDFLoader(save_path)
        docs = loader.load()
        if not docs:
            return "❌ Gagal membaca isi dokumen."

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print(f"→ Jumlah chunk: {len(chunks)}")
        if not chunks:
            return "❌ Tidak ada konten yang bisa diproses."

        # Embedding & vectorstore
        vectordb = Chroma.from_documents(chunks, embedding=embedding)
        retriever_dict[session_id] = vectordb.as_retriever()
        memory_dict[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer", k=5
        )
        print("✅ Dokumen berhasil disimpan ke Chroma (in-memory).")

        # DB session & document
        db = SessionLocal()
        if not db.query(Session).filter_by(id=session_id).first():
            db.add(Session(id=session_id))
            db.commit()
            print("✅ Session baru ditambahkan ke database.")

        db.add(Document(session_id=session_id, filename=saved_filename))
        db.commit()
        db.close()
        print("✅ Metadata dokumen disimpan ke database.")
        return "✅ File berhasil diupload dan diproses."

    except Exception as e:
        print("❌ Terjadi error:", str(e))
        return f"❌ Terjadi kesalahan saat upload: {str(e)}"

# Question handler
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

# Gradio UIawefaewf
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Chat AI Dokumen - RAG with Chroma + PostgreSQL")

    session_id = gr.Textbox(label="Session ID", visible=False)
    file = gr.File(label="Upload PDF")
    status = gr.Textbox(label="Status")
    question = gr.Textbox(label="Pertanyaan", placeholder="Tanya isi dokumen...")
    answer = gr.Textbox(label="Jawaban")

    # Set UUID baru saat halaman diload
    def set_session_id():
        return str(uuid.uuid4())

    demo.load(fn=set_session_id, inputs=[], outputs=[session_id])

    file.change(fn=handle_upload, inputs=[file, session_id], outputs=status)
    question.submit(fn=handle_question, inputs=[question, session_id], outputs=answer)



demo.launch(server_name="0.0.0.0", server_port=7860)
