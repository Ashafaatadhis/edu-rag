# db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("❌ DATABASE_URL tidak ditemukan di .env")

    print("📡 DATABASE_URL:", db_url)

    return create_engine(
        db_url,
        pool_pre_ping=True,  # ✅ reconnect kalau koneksi putus
        pool_recycle=1800,   # ✅ recycle pool tiap 30 menit
        echo=False           # 🔇 matikan log query (bisa diaktifkan kalau debug)
    )

# Global engine & session
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# === Models ===
class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    filename = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    question = Column(Text)
    answer = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# === Inisialisasi DB ===
def init_db():
    print("📦 Init PostgreSQL DB:", engine.url.database)
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ DB Migrations sukses")
    except Exception as e:
        print("❌ Gagal inisialisasi DB:", str(e))
