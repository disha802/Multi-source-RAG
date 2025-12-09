# ============================================================================
# routes/upload.py
# ============================================================================
"""
Upload route handlers
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
        docs = doc_loader.load_all_documents(
            str(config.DOCUMENTS_FOLDER),
            config.ALLOWED_DOCS
        )
        
        if not docs:
            return jsonify({'error': 'No documents found'}), 400
        
        success = vector_store.rebuild(docs)
        
        if success:
            rag_assistant.clear_cache()
            return jsonify({
                'message': 'Vector database rebuilt successfully',
                'total_vectors': vector_store.vector_db.index.ntotal
            })
        else:
            return jsonify({'error': 'Failed to rebuild database'}), 500
    
    return upload_bp