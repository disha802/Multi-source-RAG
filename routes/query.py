# ============================================================================
# routes/query.py
# ============================================================================
"""
Query route handlers
"""
from flask import Blueprint, request, jsonify
import tempfile
import os
from pathlib import Path

def create_query_blueprint(rag_assistant, voice_system, config):
    query_bp = Blueprint('query', __name__)
    
    from utils.helpers import allowed_file
    
    @query_bp.route('/query', methods=['POST'])
    def query():
        """Text query endpoint"""
        data = request.json
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question parameter'}), 400
        
        question = data['question']
        result = rag_assistant.query(question)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    @query_bp.route('/query/voice', methods=['POST'])
    def query_voice():
        """Voice query endpoint - upload audio file"""
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename, config.ALLOWED_AUDIO):
            return jsonify({'error': f'Invalid audio format'}), 400
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Transcribe
            transcription = voice_system.transcribe(tmp_path)
            
            if not transcription:
                return jsonify({'error': 'Failed to transcribe audio'}), 400
            
            # Query
            result = rag_assistant.query(transcription)
            result['transcription'] = transcription
            
            if 'error' in result:
                return jsonify(result), 400
            
            return jsonify(result)
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    return query_bp