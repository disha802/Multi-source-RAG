# ============================================================================
# README.md
# ============================================================================
# RAG Assistant - Full Stack Application

A complete RAG (Retrieval-Augmented Generation) system with document upload, 
voice queries, and intelligent question answering.

## Features

✅ **Multi-format document support** (PDF, Word, Excel, CSV, TXT)
✅ **Voice queries** using Whisper AI
✅ **Smart caching** for faster repeated queries
✅ **Vector database persistence** (FAISS)
✅ **Source citations** with relevance scores
✅ **Modern React frontend**
✅ **RESTful API backend**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from: https://console.groq.com/

### 3. Run Backend

```bash
python app.py
```

Backend runs on: http://localhost:5000

### 4. Run Frontend

The React frontend is embedded in the artifact above. 
To use it:
- Copy the React code to a local file
- Or integrate it into your existing React app
- Make sure it points to http://localhost:5000

## File Structure

```
flask_rag_app/
├── app.py                    # Main Flask app (RUN THIS)
├── config.py                 # Configuration
├── models/
│   ├── __init__.py
│   ├── document_loader.py    # Document loading
│   ├── vector_store.py       # FAISS vector DB
│   └── rag_assistant.py      # RAG logic
├── routes/
│   ├── __init__.py
│   ├── query.py              # Query endpoints
│   └── upload.py             # Upload endpoints
├── utils/
│   ├── __init__.py
│   ├── audio.py              # Whisper integration
│   └── helpers.py            # Utilities
├── uploads/
│   ├── documents/            # Uploaded docs (auto-created)
│   └── audio/                # Audio files (auto-created)
├── vector_db/
│   └── faiss_index/          # Vector DB (auto-created)
├── requirements.txt
├── .env
└── README.md
```

## API Endpoints

### Query Endpoints

**Text Query**
```bash
POST /query
Content-Type: application/json

{
  "question": "What is the main topic?"
}
```

**Voice Query**
```bash
POST /query/voice
Content-Type: multipart/form-data

audio: <audio_file>
```

### Upload Endpoints

**Upload Document**
```bash
POST /upload/document
Content-Type: multipart/form-data

file: <document_file>
```

**Rebuild Vector DB**
```bash
POST /rebuild
```

### Stats Endpoint

```bash
GET /stats
```

## Usage Flow

1. **Upload Documents**: Use the frontend or API to upload PDFs, Word docs, etc.
2. **Rebuild Database**: Click "Rebuild Database" to process documents
3. **Ask Questions**: Type questions or upload audio queries
4. **Get Answers**: Receive answers with source citations and relevance scores

## Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE`: Document chunk size (default: 500)
- `EMBEDDING_MODEL`: Sentence transformer model
- `LLM_MODEL`: Groq LLM model
- `TOP_K_RESULTS`: Number of sources to retrieve
- `WHISPER_MODEL`: Whisper model size (tiny/base/small/medium/large)

## Troubleshooting

**No module named 'models'**
```bash
# Create __init__.py files:
touch models/__init__.py routes/__init__.py utils/__init__.py
```

**API Key Error**
```bash
# Make sure .env file exists with:
GROQ_API_KEY=your_actual_key
```

**Vector DB not building**
- Upload documents first
- Click "Rebuild Database"
- Check console for errors

## Development

**Run in development mode:**
```bash
export FLASK_ENV=development
python app.py
```

**Enable debug logging:**
```python
# In app.py
app.run(debug=True)
```

## License

MIT

## Contributing

Pull requests welcome! For major changes, open an issue first.