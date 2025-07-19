import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import shutil
import os
from langchain.schema import AIMessage, HumanMessage
import uuid
from dotenv import load_dotenv
# PERUBAHAN: Mengimpor kelas Pinecone dari modul pinecone
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

# DB
from db import init_db, SessionLocal, Session, Document, ChatHistory, get_engine

# Set cache path
os.environ["HOME"] = "/app"
os.environ["HF_HOME"] = "/app/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/huggingface_cache/transformers"
os.environ["TORCH_HOME"] = "/app/huggingface_cache/torch"

# Load .env
# load_dotenv() 
groq_api_key = os.environ.get("GROQ_API_KEY")

# PERUBAHAN: Inisialisasi Pinecone menggunakan sintaks v3+
# Klien akan membaca PINECONE_API_KEY dan PINECONE_ENVIRONMENT dari env vars secara otomatis
try:
    pc = Pinecone()
except Exception as e:
    raise ValueError(f"Gagal menginisialisasi Pinecone. Pastikan PINECONE_API_KEY dan PINECONE_ENVIRONMENT sudah diatur. Error: {e}")


# Init DB
init_db()
engine = get_engine()

# Embedding & model init
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True}
)

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
        
        os.makedirs("/tmp/uploads", exist_ok=True)
        saved_filename = f"{session_id}_{uuid.uuid4().hex}.pdf"
        save_path = f"/tmp/uploads/{saved_filename}"
        shutil.copyfile(file.name, save_path)
        print(f"→ File disimpan ke: {save_path}")

        loader = PyPDFLoader(save_path)
        docs = loader.load()
        if not docs:
            return "❌ Gagal membaca isi dokumen."

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print(f"→ Jumlah chunk: {len(chunks)}")
        if not chunks:
            return "❌ Tidak ada konten yang bisa diproses."

        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not index_name:
            return "❌ PINECONE_INDEX_NAME tidak ditemukan di env."
        
        # PERUBAHAN: Menggunakan pc.Index (sintaks v3+) untuk mendapatkan objek index
        print(f"→ Mengakses Pinecone index: {index_name}")
        index = pc.Index(index_name)

        # Menggunakan from_documents untuk menambahkan data ke index yang sudah ada
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embedding,
            index_name=index_name,
            namespace=session_id
        )

        print("✅ Dokumen berhasil disimpan ke Pinecone.")

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
        print(f"❌ Terjadi error saat upload: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Terjadi kesalahan saat upload: {str(e)}"

# Question handler
def handle_question(question, session_id):
    try:
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not index_name:
            return "❌ PINECONE_INDEX_NAME tidak ditemukan di env."
        
        # Inisialisasi VectorStore untuk mengambil data yang sudah ada
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding,
            namespace=session_id
        )
        retriever = vectorstore.as_retriever()

        db = SessionLocal()
        chat_records = db.query(ChatHistory).filter_by(session_id=session_id).order_by(ChatHistory.created_at.asc()).all()
        db.close()

        chat_history = []
        for record in chat_records:
            chat_history.append(HumanMessage(content=record.question))
            chat_history.append(AIMessage(content=record.answer))

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        memory.chat_memory.messages = chat_history

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            output_key="answer",
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )

        result = chain.invoke({"question": question})
        answer = result["answer"]

        db = SessionLocal()
        db.add(ChatHistory(session_id=session_id, question=question, answer=answer))
        db.commit()
        db.close()

        return answer
    except Exception as e:
        print(f"❌ Terjadi error saat menjawab pertanyaan: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Terjadi kesalahan saat memproses pertanyaan: {str(e)}"


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📄 Chat AI Dokumen - RAG with Pinecone + PostgreSQL")
    
    with gr.Row():
        with gr.Column(scale=1):
            session_id = gr.Textbox(label="Session ID (Generated Automatically)", interactive=False)
            file = gr.File(label="Upload PDF Anda")
            status = gr.Textbox(label="Status Proses", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", height=500)
            question = gr.Textbox(label="Pertanyaan", placeholder="Tanya apa saja tentang isi dokumen...")
            
            def chat_interface(message, history):
                if not session_id.value:
                    history.append(("Error", "Session ID belum terbuat. Mohon refresh halaman."))
                    return "", history
                if not message:
                    return "", history
                ans = handle_question(message, session_id.value)
                history.append((message, ans))
                return "", history

            question.submit(chat_interface, [question, chatbot], [question, chatbot])

    def set_session_id():
        return str(uuid.uuid4())

    def upload_and_update_status(file_obj, sess_id):
        if file_obj is None:
            return "Mohon upload file terlebih dahulu."
        return handle_upload(file_obj, sess_id)

    demo.load(fn=set_session_id, inputs=[], outputs=[session_id])
    file.change(fn=upload_and_update_status, inputs=[file, session_id], outputs=status)

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
