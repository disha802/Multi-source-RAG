# ============================================================================
# config.py
# ============================================================================
"""
Configuration settings for the RAG application
"""
import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    DOCUMENTS_FOLDER = UPLOAD_FOLDER / 'documents'
    AUDIO_FOLDER = UPLOAD_FOLDER / 'audio'
    VECTOR_DB_PATH = BASE_DIR / 'vector_db' / 'faiss_index'
    
    # File types
    ALLOWED_DOCS = {'pdf', 'docx', 'xlsx', 'xls', 'csv', 'txt'}
    ALLOWED_AUDIO = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
    
    # RAG settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    EMBEDDING_MODEL = "BAAI/bge-large-en"
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 500
    TOP_K_RESULTS = 3
    
    # Whisper
    WHISPER_MODEL = "base"
    
    # Flask
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # API Key
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    
    @classmethod
    def create_folders(cls):
        """Create necessary folders if they don't exist"""
        for folder in [cls.DOCUMENTS_FOLDER, cls.AUDIO_FOLDER, 
                       cls.VECTOR_DB_PATH.parent]:
            folder.mkdir(parents=True, exist_ok=True)