# 🤖 Enhanced RAG Assistant - Streamlit Application

A production-ready RAG (Retrieval-Augmented Generation) system with **user authentication**, **personalized responses**, and **intelligent document analysis**.

## ✨ Key Features

### 🔐 Authentication & User Management
- **Secure login system** with password hashing (SHA-256 + salt)
- **Session management** with 24-hour token expiration
- **User roles**: Regular users and superusers
- **Profile customization** for personalized responses
- **Default credentials**: 
  - Superuser: `admin` / `admin123`

### 📚 Document Intelligence
- ✅ **Multi-format support**: PDF, Word, Excel, CSV, TXT
- ✅ **Domain detection**: Automatically identifies document type (Finance, Healthcare, Legal, Technical, etc.)
- ✅ **Smart relevance checking**: Blocks out-of-scope queries
- ✅ **Vector database persistence** (FAISS)
- ✅ **Source citations** with relevance scores

### 🎨 Personalization Features
- **Response styles**: Professional, Casual, Educational, Concise, Storytelling
- **Age-appropriate language** adaptation
- **Interest-based examples** and analogies
- **User profile persistence** across sessions

### 🚀 Performance Features
- ⚡ **Query result caching** for instant repeated queries
- ⚡ **Retry logic** for API reliability
- ⚡ **Hallucination risk detection**
- ⚡ **Confidence scoring** for each answer

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Secrets

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Or set as environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Get your API key from: https://console.groq.com/

### 3. Run Application

```bash
streamlit run streamlit.py
```

Application will open at: **http://localhost:8501**

---

## 📖 User Guide

### First Time Setup

1. **Login** with default credentials:
   - Username: `admin`
   - Password: `admin123`

2. **Upload Documents** using the sidebar:
   - Click "📚 Document Library" → "➕ Add Documents"
   - Upload one or more files
   - Click "💾 Add to Library"

3. **Build Index**:
   - Click "🔄 Rebuild Index"
   - Wait for processing to complete

4. **Customize Profile** (optional):
   - Open "👤 Personalization" in sidebar
   - Select your preferred response style
   - Add interests for better context
   - Save profile

5. **Ask Questions**:
   - Type questions in the main interface
   - Use suggested questions for quick starts
   - View sources and confidence scores

---

## 👥 User Management (Superuser Only)

Superusers have access to additional features:

### Create New User
1. Open "👥 User Management" in sidebar
2. Click "➕ Create New User"
3. Fill in username, email, password
4. Select role (user/superuser)
5. Click "Create User"

### Manage Existing Users
- **View all users** with last login times
- **Delete users** (except superusers)
- **Reset passwords** for any user

### Regular Users Can:
- Change their own password
- Customize their profile
- Upload/delete their documents
- Ask questions and get personalized answers

---

## 🎨 Personalization Options

### Response Styles

| Style | Description | Best For |
|-------|-------------|----------|
| **Professional** | Formal, precise, business terminology | Work documents, reports |
| **Casual & Friendly** | Conversational, engaging | General learning, personal docs |
| **Educational** | Thorough explanations, teaching mode | Academic papers, study materials |
| **Concise & Direct** | Brief, to-the-point | Quick lookups, summaries |
| **Storytelling** | Narrative flow, real-world examples | Complex concepts, presentations |

### Age-Based Adaptation
- **18-35**: Modern, relatable language
- **56+**: Clear explanations without jargon

### Interest-Based Examples
Add your interests (e.g., "Finance, Technology") to receive:
- Relevant analogies
- Domain-specific examples
- Contextual explanations

---

## 🔍 How It Works

### Document Processing Pipeline

```
Upload Documents
    ↓
Extract Text & Metadata
    ↓
Analyze Domain Type (Finance/Legal/Technical/etc.)
    ↓
Split into Chunks (500 chars, 100 overlap)
    ↓
Generate Embeddings (BAAI/bge-large-en)
    ↓
Build FAISS Vector Index
    ↓
Save to Disk
```

### Query Processing Pipeline

```
User Query
    ↓
Check Cache (instant if hit)
    ↓
Embed Query → Search Vector DB
    ↓
Retrieve Top-K Relevant Chunks
    ↓
Check Domain Relevance
    ↓
Build Context + User Profile
    ↓
Generate Answer with LLM (Groq)
    ↓
Assess Confidence & Risk
    ↓
Return Answer + Citations
```

### Smart Features

**Domain Detection**
- Analyzes keywords and document types
- Identifies domain (Finance, Healthcare, Legal, etc.)
- Provides domain-specific context to LLM

**Relevance Checking**
- Blocks out-of-scope queries (e.g., "latest Netflix shows")
- Checks semantic similarity to documents
- Provides helpful redirection messages

**Hallucination Prevention**
- Tracks source citations
- Monitors confidence scores
- Warns users about low-confidence answers
- Recommends verification when needed

---

## 📁 File Structure

```
streamlit_rag_app/
├── streamlit.py              # Main application (RUN THIS)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .streamlit/
│   └── secrets.toml         # API keys (create this)
├── upload/                   # Uploaded documents (auto-created)
├── vector_store/             # FAISS index (auto-created)
├── users.json               # User accounts (auto-created)
├── sessions.json            # Active sessions (auto-created)
└── domain_metadata.json     # Document analysis (auto-created)
```

---

## ⚙️ Configuration

### System Parameters

Edit `RAGConfig` class in `streamlit.py`:

```python
CHUNK_SIZE = 500              # Document chunk size
CHUNK_OVERLAP = 100           # Overlap between chunks
EMBEDDING_MODEL = "BAAI/bge-large-en"  # Sentence transformer
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq LLM
LLM_TEMPERATURE = 0.1         # Lower = more deterministic
LLM_MAX_TOKENS = 800          # Max answer length
TOP_K_RESULTS = 3             # Sources to retrieve
CACHE_SIZE = 100              # Max cached queries
```

### Session Settings

```python
SESSION_TIMEOUT_HOURS = 24    # Auto-logout after 24 hours
```

---

## 🔒 Security Features

### Password Security
- **SHA-256 hashing** with random salt
- **Minimum 6 characters** required
- Passwords never stored in plain text

### Session Management
- **Token-based authentication**
- **Automatic expiration** (24 hours)
- Secure session storage

### User Isolation
- Each user's profile stored separately
- Query cache isolated per session
- Document library shared across users

---

## 🐛 Troubleshooting

### "Session expired" Error
- Login sessions last 24 hours
- Simply login again to continue

### Documents Not Processing
1. Check file format is supported
2. Ensure file is not corrupted
3. Click "🔄 Rebuild Index"
4. Check terminal for error messages

### Low Confidence Warnings
- Normal for edge-case questions
- Verify answer against source documents
- Try rephrasing your question
- Add more relevant documents

### "Out of Scope" Message
- Question doesn't match document content
- Upload relevant documents first
- Check suggested topics in sidebar

### API Key Issues
```bash
# Verify secrets file exists
cat .streamlit/secrets.toml

# Should contain:
GROQ_API_KEY = "gsk_..."
```

---

## 💡 Best Practices

### Document Upload
- ✅ Upload related documents together
- ✅ Use descriptive filenames
- ✅ Mix formats for comprehensive coverage
- ❌ Avoid duplicate content

### Asking Questions
- ✅ Be specific and clear
- ✅ Reference document sections when relevant
- ✅ Use suggested questions as templates
- ❌ Don't ask about external/recent events

### Profile Customization
- ✅ Choose style matching document type
- ✅ Update interests for better examples
- ✅ Enable analogies for complex topics
- ❌ Don't expect factual changes (style only)

---

## 🔄 Updates & Maintenance

### Clearing Cache
- Click "Clear Cache" in sidebar
- Helps if seeing stale results
- Does not affect document index

### Rebuilding Index
- Required after adding/removing documents
- Click "🔄 Rebuild Index" button
- Takes ~10-30 seconds depending on size

### User Management
- Regularly review user list (superusers)
- Remove inactive accounts
- Reset passwords as needed

---

## 📊 Performance Metrics

### What Each Metric Means

| Metric | Good | Medium | Bad | Action |
|--------|------|--------|-----|--------|
| **Confidence** | 🟢 HIGH | 🟡 MEDIUM | 🔴 LOW | Rephrase or add docs |
| **Avg Similarity** | >70% | 50-70% | <50% | Check document relevance |
| **Hallucination Risk** | ✅ LOW | ⚡ MEDIUM | ⚠️ HIGH | Verify with sources |
| **Cache Hit Rate** | >60% | 30-60% | <30% | Normal for varied questions |

---

## 🤝 Contributing

### Planned Features
- [ ] Multi-language support
- [ ] Advanced search filters
- [ ] Document annotations
- [ ] Export answers to PDF
- [ ] Team collaboration features

### Report Issues
Create an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- System info (OS, Python version)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **LangChain** - Document processing framework
- **Groq** - Fast LLM inference
- **FAISS** - Efficient vector search
- **Streamlit** - Web application framework
- **HuggingFace** - Embedding models

---

## 📞 Support

For questions and support:
1. Check this README first
2. Review troubleshooting section
3. Open an issue on GitHub
4. Contact your system administrator

---

**Version**: 3.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready ✅
