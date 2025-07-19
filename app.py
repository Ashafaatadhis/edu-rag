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
from langchain.schema import AIMessage, HumanMessage
import uuid
from dotenv import load_dotenv
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
# groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key = os.environ["GROQ_API_KEY"]

# Init DB
init_db()
engine = get_engine()

# Embedding & model init
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
 

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

        # Simpan ke Pinecone
       
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not index_name:
            return "❌ PINECONE_INDEX_NAME tidak ditemukan di env."
        index = pc.Index(index_name)

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embedding,
            index=index,
            namespace=session_id
        )

        print("✅ Dokumen berhasil disimpan ke Pinecone.")

        # Simpan metadata ke DB
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
    # Ambil retriever dari Pinecone
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        return "❌ PINECONE_INDEX_NAME tidak ditemukan di env."
    index = pc.Index(index_name)

    retriever = PineconeVectorStore(
        index=index,
        embedding=embedding,
        namespace=session_id
    ).as_retriever()

    # Ambil chat history dari PostgreSQL
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

    # Bangun chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    # Jalankan query
    result = chain.invoke(question)
    answer = result["answer"]

    # Simpan ke DB
    db = SessionLocal()
    db.add(ChatHistory(session_id=session_id, question=question, answer=answer))
    db.commit()
    db.close()

    return answer



# Gradio UIawefaewf
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Chat AI Dokumen - RAG with Pinecone + PostgreSQL")

    index_name = os.environ.get("PINECONE_INDEX_NAME")
   

    gr.Text(index_name)
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
