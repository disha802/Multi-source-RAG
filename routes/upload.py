"""
Upload route handlers - WITH DEBUG LOGGING
"""
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path

def create_upload_blueprint(config, vector_store, rag_assistant, doc_loader):
    upload_bp = Blueprint('upload', __name__)
    
    from utils.helpers import allowed_file
    
    @upload_bp.route('/upload/document', methods=['POST'])
    def upload_document():
        """Upload a document"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename, config.ALLOWED_DOCS):
            return jsonify({'error': f'Invalid format. Allowed: {config.ALLOWED_DOCS}'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = config.DOCUMENTS_FOLDER / filename
        file.save(str(filepath))
        
        print(f"✅ File saved: {filepath}")  # DEBUG
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'filename': filename,
            'note': 'Call /rebuild to update vector database'
        })
    
    @upload_bp.route('/upload/audio', methods=['POST'])
    def upload_audio():
        """Upload an audio file"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename, config.ALLOWED_AUDIO):
            return jsonify({'error': f'Invalid format. Allowed: {config.ALLOWED_AUDIO}'}), 400
        
        filename = secure_filename(file.filename)
        filepath = config.AUDIO_FOLDER / filename
        file.save(str(filepath))
        
        return jsonify({
            'message': 'Audio uploaded successfully',
            'filename': filename
        })
    
    @upload_bp.route('/rebuild', methods=['POST'])
    def rebuild():
        """Rebuild vector database from all documents"""
        print("\n" + "="*60)
        print("🔄 REBUILD REQUEST RECEIVED")
        print("="*60)
        
        # Check what files exist
        print(f"📂 Checking folder: {config.DOCUMENTS_FOLDER}")
        all_files = list(config.DOCUMENTS_FOLDER.glob('*.*'))
        print(f"📄 Found {len(all_files)} file(s): {[f.name for f in all_files]}")
        
        # Load documents
        print("📖 Loading documents...")
        try:
            docs = doc_loader.load_all_documents(
                str(config.DOCUMENTS_FOLDER),
                config.ALLOWED_DOCS
            )
            print(f"✅ Loaded {len(docs)} document(s)")
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to load documents: {str(e)}'}), 500
        
        if not docs:
            print("⚠️  No documents found!")
            return jsonify({'error': 'No documents found'}), 400
        
        # Rebuild vector store
        print("🔨 Rebuilding vector store...")
        try:
            success = vector_store.rebuild(docs)
        except Exception as e:
            print(f"❌ Error rebuilding: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Rebuild failed: {str(e)}'}), 500
        
        if success:
            print("✅ Rebuild successful!")
            rag_assistant.clear_cache()
            total_vectors = vector_store.vector_db.index.ntotal if vector_store.vector_db else 0
            print(f"📊 Total vectors: {total_vectors}")
            print("="*60 + "\n")
            
            return jsonify({
                'message': 'Vector database rebuilt successfully',
                'total_vectors': total_vectors
            })
        else:
            print("❌ Rebuild failed!")
            return jsonify({'error': 'Failed to rebuild database'}), 500
    
    return upload_bp