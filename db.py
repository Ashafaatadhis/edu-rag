# db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
# engine = create_engine("sqlite:///rag_chat.db")
# engine = create_engine("sqlite:///uploads/rag_chat.db")
engine = create_engine("sqlite:////app/rag_chat.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

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

def init_db():
    Base.metadata.create_all(bind=engine)
