# app.py - MAIN APPLICATION (RUN THIS FILE)
# ============================================================================
"""
Main Flask application
"""
from flask import Flask, jsonify
from flask_cors import CORS
from pathlib import Path

# Import configuration
from config import Config

# Import models
from models.document_loader import DocumentLoader
from models.vector_store import VectorStore
from models.rag_assistant import RAGAssistant

# Import utilities
from utils.audio import VoiceSystem

# Import route creators
from routes.upload import create_upload_blueprint
from routes.query import create_query_blueprint

def create_app():
    """Application factory"""
    app = Flask(__name__)
    CORS(app)
    
    # Configure
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    Config.create_folders()
    
    # Initialize components
    print("\n" + "="*60)
    print("🚀 INITIALIZING RAG SYSTEM")
    print("="*60)
    
    doc_loader = DocumentLoader()
    vector_store = VectorStore(Config)
    vector_store.initialize()
    
    rag_assistant = RAGAssistant(vector_store, Config)
    voice_system = VoiceSystem(Config.WHISPER_MODEL)
    
    print("="*60)
    print("✅ SYSTEM READY")
    print("="*60 + "\n")
    
    # Register routes
    app.register_blueprint(
        create_upload_blueprint(Config, vector_store, rag_assistant, doc_loader)
    )
    app.register_blueprint(
        create_query_blueprint(rag_assistant, voice_system, Config)
    )
    
    # Root endpoint
    @app.route('/')
    def index():
        return jsonify({
            'status': 'running',
            'message': 'RAG API is ready',
            'endpoints': {
                'query': 'POST /query',
                'voice': 'POST /query/voice',
                'upload_doc': 'POST /upload/document',
                'upload_audio': 'POST /upload/audio',
                'rebuild': 'POST /rebuild',
                'stats': 'GET /stats'
            }
        })
    
    # Stats endpoint
    @app.route('/stats')
    def stats():
        doc_count = len(list(Config.DOCUMENTS_FOLDER.glob('*.*')))
        audio_count = len(list(Config.AUDIO_FOLDER.glob('*.*')))
        
        return jsonify({
            'vector_db_loaded': vector_store.is_loaded,
            'total_vectors': vector_store.vector_db.index.ntotal if vector_store.vector_db else 0,
            'documents_count': doc_count,
            'audio_files_count': audio_count,
            'cache_size': len(rag_assistant.cache),
            'config': {
                'embedding_model': Config.EMBEDDING_MODEL,
                'llm_model': Config.LLM_MODEL,
                'chunk_size': Config.CHUNK_SIZE,
                'top_k': Config.TOP_K_RESULTS
            }
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    print("\n🌐 Starting Flask server...")
    print("📖 Frontend: Open the React app in your browser")
    print("🔗 Backend API: http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)