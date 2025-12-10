"""
RAG Assistant - Streamlit Web Application
Multi-format document QA system with voice support
"""

import streamlit as st
import os
import re
import time
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# LangChain
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader,
    CSVLoader, TextLoader
)
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq
from groq import Groq

# Audio
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
import whisper

# Retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for white and sky blue theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-blue: #0EA5E9;
        --light-blue: #BAE6FD;
        --sky-blue: #7DD3FC;
        --dark-blue: #0284C7;
    }
    
    /* Main container */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F0F9FF 0%, #E0F2FE 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0284C7 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0EA5E9;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0284C7;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #F0F9FF;
        border-left: 4px solid #0EA5E9;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #0284C7;
    }
    
    /* Text inputs */
    .stTextInput>div>div>input {
        border: 2px solid #BAE6FD;
        border-radius: 8px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #0EA5E9;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #F0F9FF;
        border: 2px dashed #0EA5E9;
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #F0F9FF;
        border-radius: 8px;
        color: #0284C7;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: #DCFCE7;
        color: #166534;
    }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #BAE6FD;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
    }
    
    /* Answer box */
    .answer-box {
        background-color: #FFFFFF;
        border: 2px solid #0EA5E9;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
    }
    
    /* Source citations */
    .citation-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0EA5E9;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

class RAGConfig:
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    EMBEDDING_MODEL = "BAAI/bge-large-en"
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 500
    TOP_K_RESULTS = 3
    SIMILARITY_THRESHOLD = 0.5
    CACHE_SIZE = 100
    WHISPER_MODEL = "base"
    SAMPLE_RATE = 16000
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt']
    SUPPORTED_AUDIO = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_section_from_text(text: str) -> str:
    """Extract section headers from document text"""
    if not text or len(text.strip()) == 0:
        return "Empty Content"
    
    patterns = [
        r'^#{1,3}\s+(.+)$',
        r'^(SECTION|CHAPTER)\s+[\d\w]+[:\s]+(.+)$',
        r'^\d+\.\s+([A-Z][A-Za-z\s]{3,})$',
    ]
    
    lines = text.split('\n')[:15]
    for line in lines:
        line_clean = line.strip()
        if len(line_clean) < 3:
            continue
        
        for pattern in patterns:
            match = re.search(pattern, line_clean, re.IGNORECASE)
            if match:
                section = match.group(1) if match.lastindex == 1 else match.group(2)
                return section.strip()[:100]
    
    return "General Content"

def calculate_similarity_score(distance: float) -> float:
    """Convert FAISS L2 distance to similarity percentage"""
    return 1 / (1 + distance)

def assess_confidence(similarity_scores: List[float]) -> Tuple[str, str]:
    """Assess retrieval confidence"""
    avg_sim = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    if avg_sim > 0.7:
        return "HIGH", "🟢"
    elif avg_sim > 0.5:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🔴"

def detect_hallucination_risk(answer: str, num_sources: int, avg_similarity: float) -> Tuple[str, str]:
    """Detect potential hallucination"""
    has_citations = any(f"Source {i+1}" in answer for i in range(num_sources))
    
    uncertainty_phrases = [
        "not available", "not mentioned", "not found", "does not provide",
        "cannot find", "not specified", "not stated", "not included"
    ]
    admits_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
    
    if not has_citations and not admits_uncertainty and avg_similarity < 0.6:
        return "HIGH ⚠️", "⚠️ WARNING: Low source relevance. Answer may contain hallucinations."
    elif not has_citations and avg_similarity < 0.7:
        return "MEDIUM ⚡", "⚡ CAUTION: Answer lacks explicit source citations."
    else:
        return "LOW ✅", ""

# ============================================================================
# DOCUMENT LOADER
# ============================================================================

class DocumentLoader:
    @staticmethod
    def load_documents(uploaded_files) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                docs = DocumentLoader._load_single_file(tmp_path, uploaded_file.name, file_ext)
                documents.extend(docs)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        return documents
    
    @staticmethod
    def _load_single_file(file_path: str, file_name: str, file_ext: str) -> List[Document]:
        """Load a single file"""
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    "source_type": "PDF",
                    "document_title": file_name,
                    "page_number": i + 1,
                    "total_pages": len(docs),
                    "section": extract_section_from_text(doc.page_content)
                })
        
        elif file_ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source_type": "Word Document",
                    "document_title": file_name,
                    "page_number": "N/A",
                    "section": extract_section_from_text(doc.page_content)
                })
        
        elif file_ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    "source_type": "Excel",
                    "document_title": file_name,
                    "page_number": f"Sheet {i+1}",
                    "section": "Tabular Data"
                })
        
        elif file_ext == ".csv":
            loader = CSVLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source_type": "CSV",
                    "document_title": file_name,
                    "page_number": "Data Table",
                    "section": "Tabular Data"
                })
        
        elif file_ext == ".txt":
            loader = TextLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source_type": "Text File",
                    "document_title": file_name,
                    "page_number": "N/A",
                    "section": extract_section_from_text(doc.page_content)
                })
        
        return docs

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class AudioProcessor:
    def __init__(self):
        if 'whisper_model' not in st.session_state:
            with st.spinner("Loading Whisper model..."):
                st.session_state.whisper_model = whisper.load_model(RAGConfig.WHISPER_MODEL)
        self.model = st.session_state.whisper_model
    
    def transcribe_audio(self, audio_file) -> Optional[str]:
        """Transcribe audio file"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file.getvalue())
                tmp_path = tmp.name
            
            result = self.model.transcribe(tmp_path, language='en')
            transcription = result['text'].strip()
            
            os.unlink(tmp_path)
            return transcription
        
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return None

# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStoreManager:
    @staticmethod
    @st.cache_resource
    def get_embeddings():
        """Load embedding model (cached)"""
        return HuggingFaceEmbeddings(
            model_name=RAGConfig.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
    
    @staticmethod
    def create_vector_db(documents: List[Document]):
        """Create vector database"""
        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        progress_bar = st.progress(0)
        for i, doc in enumerate(documents):
            chunks.extend(text_splitter.split_documents([doc]))
            progress_bar.progress((i + 1) / len(documents))
        
        progress_bar.empty()
        
        # Create vector DB
        embeddings = VectorStoreManager.get_embeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        return vector_db, len(chunks)

# ============================================================================
# RAG ASSISTANT
# ============================================================================

class RAGAssistant:
    def __init__(self, vector_db, groq_api_key: str):
        self.vector_db = vector_db
        self.client = Groq(api_key=groq_api_key)
        
        if 'query_cache' not in st.session_state:
            st.session_state.query_cache = {}
        if 'cache_hits' not in st.session_state:
            st.session_state.cache_hits = 0
        if 'cache_misses' not in st.session_state:
            st.session_state.cache_misses = 0
    
    def query(self, question: str, k: int = RAGConfig.TOP_K_RESULTS) -> Dict:
        """Process query and return results"""
        start_time = time.time()
        
        # Check cache
        query_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
        
        if query_hash in st.session_state.query_cache:
            st.session_state.cache_hits += 1
            result = st.session_state.query_cache[query_hash]
            result['metrics']['latency'] = time.time() - start_time
            result['metrics']['cache_hit'] = True
            return result
        
        st.session_state.cache_misses += 1
        
        # Retrieve documents
        relevant_docs = self.vector_db.similarity_search_with_score(question, k=k)
        
        # Build context
        context_parts = []
        citations = []
        similarity_scores = []
        seen_sources = set()
        
        for i, (doc, distance) in enumerate(relevant_docs):
            similarity = calculate_similarity_score(distance)
            similarity_scores.append(similarity)
            
            meta = doc.metadata
            source_id = f"{meta.get('document_title')}|{meta.get('page_number')}"
            
            context_parts.append(
                f"[SOURCE {i+1}] (Relevance: {similarity:.1%})\n{doc.page_content}\n"
            )
            
            if source_id not in seen_sources:
                seen_sources.add(source_id)
                
                citation = {
                    'title': meta.get('document_title', 'Unknown'),
                    'page': meta.get('page_number', 'N/A'),
                    'section': meta.get('section', 'General'),
                    'similarity': similarity
                }
                citations.append(citation)
        
        context = "\n".join(context_parts)
        
        # Generate answer
        answer = self._generate_answer(context, question)
        
        # Calculate metrics
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        confidence_level, confidence_emoji = assess_confidence(similarity_scores)
        hallucination_risk, warning = detect_hallucination_risk(
            answer, len(relevant_docs), avg_similarity
        )
        
        latency = time.time() - start_time
        
        result = {
            'answer': answer,
            'citations': citations,
            'warning': warning,
            'metrics': {
                'avg_similarity': avg_similarity,
                'confidence': confidence_level,
                'confidence_emoji': confidence_emoji,
                'hallucination_risk': hallucination_risk,
                'num_sources': len(citations),
                'latency': latency,
                'cache_hit': False
            }
        }
        
        # Cache result
        st.session_state.query_cache[query_hash] = result
        if len(st.session_state.query_cache) > RAGConfig.CACHE_SIZE:
            oldest_key = next(iter(st.session_state.query_cache))
            del st.session_state.query_cache[oldest_key]
        
        return result
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer using LLM"""
        prompt = f"""You are a precise document analysis assistant. Answer questions using ONLY the provided sources.

STRICT RULES:
1. Answer ONLY with information EXPLICITLY in the sources
2. Cite sources: "According to Source 1..." or "Source 2 states..."
3. If info is missing, say: "This information is not available in the provided documents"
4. Do NOT infer, assume, or add external knowledge
5. Quote exact phrases when possible

SOURCES:
{context}

QUESTION: {question}

ANSWER (with source citations):"""
        
        response = self.client.chat.completions.create(
            model=RAGConfig.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=RAGConfig.LLM_TEMPERATURE,
            max_tokens=RAGConfig.LLM_MAX_TOKENS
        )
        
        return response.choices[0].message.content

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🤖 RAG Assistant")
    st.markdown("### Multi-format Document QA System with Voice Support")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key"
        )
        
        if not groq_api_key:
            st.warning("Please enter your Groq API key to continue")
            st.stop()
        
        st.divider()
        
        # Document Upload
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Supported: PDF, Word, Excel, CSV, TXT"
        )
        
        st.divider()
        
        # Audio Upload
        st.header("🎤 Voice Query")
        audio_file = st.file_uploader(
            "Upload audio query",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="Upload an audio file to transcribe"
        )
        
        st.divider()
        
        # Settings
        st.header("🔧 Settings")
        k_results = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=RAGConfig.TOP_K_RESULTS,
            help="Number of document chunks to retrieve"
        )
        
        st.divider()
        
        # Cache Stats
        if st.session_state.get('cache_hits', 0) + st.session_state.get('cache_misses', 0) > 0:
            st.header("📊 Cache Statistics")
            total = st.session_state.cache_hits + st.session_state.cache_misses
            hit_rate = (st.session_state.cache_hits / total * 100) if total > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cache Hits", st.session_state.cache_hits)
            with col2:
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
            
            if st.button("Clear Cache", use_container_width=True):
                st.session_state.query_cache = {}
                st.session_state.cache_hits = 0
                st.session_state.cache_misses = 0
                st.success("Cache cleared!")
    
    # Main content
    if not uploaded_files:
        st.info("👆 Please upload documents in the sidebar to get started")
        
        # Welcome message
        st.markdown("""
        ### Welcome to RAG Assistant! 🎉
        
        **Features:**
        - 📄 Multi-format document support (PDF, Word, Excel, CSV, TXT)
        - 🎤 Voice query support
        - 💾 Smart caching for faster responses
        - 📊 Detailed metrics and confidence scores
        - ⚡ Hallucination risk detection
        
        **How to use:**
        1. Upload your documents in the sidebar
        2. Wait for processing to complete
        3. Ask questions about your documents
        4. Optionally, upload audio files for voice queries
        """)
        
        st.stop()
    
    # Process documents
    if 'vector_db' not in st.session_state or st.session_state.get('last_files') != [f.name for f in uploaded_files]:
        with st.spinner("📚 Loading and processing documents..."):
            # Load documents
            documents = DocumentLoader.load_documents(uploaded_files)
            
            if not documents:
                st.error("No documents could be loaded. Please check your files.")
                st.stop()
            
            # Create vector DB
            vector_db, num_chunks = VectorStoreManager.create_vector_db(documents)
            
            st.session_state.vector_db = vector_db
            st.session_state.num_documents = len(documents)
            st.session_state.num_chunks = num_chunks
            st.session_state.last_files = [f.name for f in uploaded_files]
            
            st.success(f"✅ Processed {len(documents)} documents into {num_chunks} chunks")
    
    # Initialize assistant
    assistant = RAGAssistant(st.session_state.vector_db, groq_api_key)
    
    # Document stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📄 Documents", st.session_state.num_documents)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔍 Chunks", st.session_state.num_chunks)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        cached = len(st.session_state.get('query_cache', {}))
        st.metric("💾 Cached Queries", cached)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Query input
    query_input = ""
    
    # Handle audio transcription
    if audio_file:
        with st.spinner("🎤 Transcribing audio..."):
            audio_processor = AudioProcessor()
            transcription = audio_processor.transcribe_audio(audio_file)
            
            if transcription:
                st.success(f"✅ Transcribed: {transcription}")
                query_input = transcription
            else:
                st.error("Failed to transcribe audio")
    
    # Text query
    question = st.text_input(
        "💬 Ask a question about your documents:",
        value=query_input,
        placeholder="What is this document about?",
        help="Enter your question or upload an audio file"
    )
    
    if st.button("🔍 Search", type="primary", use_container_width=True) and question:
        with st.spinner("🔍 Searching documents..."):
            try:
                result = assistant.query(question, k=k_results)
                
                # Answer
                st.markdown("### 📝 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                if result['warning']:
                    st.warning(result['warning'])
                
                # Metrics
                st.markdown("### 📊 Metrics")
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                
                metrics = result['metrics']
                
                with met_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Confidence",
                        f"{metrics['confidence_emoji']} {metrics['confidence']}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with met_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Avg Similarity",
                        f"{metrics['avg_similarity']:.1%}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with met_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Sources",
                        metrics['num_sources']
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with met_col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Latency",
                        f"{metrics['latency']:.2f}s"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk indicators
                col_risk1, col_risk2 = st.columns(2)
                with col_risk1:
                    st.info(f"**Hallucination Risk:** {metrics['hallucination_risk']}")
                with col_risk2:
                    cache_status = "✅ Yes" if metrics['cache_hit'] else "❌ No"
                    st.info(f"**Cache Hit:** {cache_status}")
                
                # Citations
                st.markdown("### 📚 Sources")
                for i, citation in enumerate(result['citations'], 1):
                    with st.expander(f"Source {i}: {citation['title']}", expanded=False):
                        st.markdown(f'<div class="citation-box">', unsafe_allow_html=True)
                        st.markdown(f"**Page/Sheet:** {citation['page']}")
                        st.markdown(f"**Section:** {citation['section']}")
                        st.markdown(f"**Relevance:** {citation['similarity']:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()