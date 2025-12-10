"""
Enhanced RAG Assistant - Streamlit Web Application
Multi-format document QA system with persistent storage, domain awareness, and personalization
"""

import streamlit as st
import os
import re
import time
import hashlib
import tempfile
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

# Retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================

class RAGConfig:
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    EMBEDDING_MODEL = "BAAI/bge-large-en"
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 800
    TOP_K_RESULTS = 3
    SIMILARITY_THRESHOLD = 0.5
    CACHE_SIZE = 100
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt']
    UPLOAD_FOLDER = "upload"  # Persistent document storage
    VECTOR_DB_PATH = "vector_store"  # Persistent vector store
    USER_PROFILES_PATH = "user_profiles.json"  # User personalization data

# ============================================================================

def extract_section_from_text(text: str) -> str:
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
    return 1 / (1 + distance)

def assess_confidence(similarity_scores: List[float]) -> Tuple[str, str]:
    avg_sim = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    if avg_sim > 0.7:
        return "HIGH", "🟢"
    elif avg_sim > 0.5:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🔴"

def detect_hallucination_risk(answer: str, num_sources: int, avg_similarity: float) -> Tuple[str, str]:
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

class UserProfileManager:
    """Manages user personalization profiles"""
    
    @staticmethod
    def load_profiles() -> Dict:
        if os.path.exists(RAGConfig.USER_PROFILES_PATH):
            try:
                with open(RAGConfig.USER_PROFILES_PATH, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    @staticmethod
    def save_profile(user_id: str, profile: Dict):
        profiles = UserProfileManager.load_profiles()
        profiles[user_id] = profile
        with open(RAGConfig.USER_PROFILES_PATH, 'w') as f:
            json.dump(profiles, f, indent=2)
    
    @staticmethod
    def get_profile(user_id: str) -> Optional[Dict]:
        profiles = UserProfileManager.load_profiles()
        return profiles.get(user_id)

# ============================================================================

class DocumentStorage:
    """Manages persistent document storage"""
    
    @staticmethod
    def ensure_upload_folder():
        os.makedirs(RAGConfig.UPLOAD_FOLDER, exist_ok=True)
    
    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """Save uploaded file to persistent storage"""
        DocumentStorage.ensure_upload_folder()
        file_path = os.path.join(RAGConfig.UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    @staticmethod
    def get_stored_files() -> List[str]:
        """Get list of all stored documents"""
        DocumentStorage.ensure_upload_folder()
        files = []
        for file in os.listdir(RAGConfig.UPLOAD_FOLDER):
            file_path = os.path.join(RAGConfig.UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                ext = Path(file).suffix.lower()
                if ext in RAGConfig.SUPPORTED_EXTENSIONS:
                    files.append(file_path)
        return files
    
    @staticmethod
    def delete_file(filename: str):
        """Delete a file from storage"""
        file_path = os.path.join(RAGConfig.UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

# ============================================================================

class DomainDetector:
    """Detects document domain and checks query relevance"""
    
    @staticmethod
    def extract_domain_keywords(documents: List[Document]) -> List[str]:
        """Extract key terms from documents to understand domain"""
        all_text = " ".join([doc.page_content[:500] for doc in documents[:10]])
        
        # Simple keyword extraction (you could use TF-IDF or more advanced methods)
        words = re.findall(r'\b[A-Za-z]{4,}\b', all_text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:50]]
    
    @staticmethod
    def check_query_relevance(query: str, domain_keywords: List[str], context_similarity: float) -> Tuple[bool, Optional[str]]:
        """Check if query is relevant to document domain"""
        
        # If similarity is very high, query is definitely relevant
        if context_similarity > 0.6:
            return True, None
        
        # Check for common domain keywords in query
        query_words = set(re.findall(r'\b[A-Za-z]{4,}\b', query.lower()))
        domain_words = set(domain_keywords)
        
        overlap = len(query_words & domain_words)
        
        # If low similarity AND low keyword overlap, likely out of domain
        if context_similarity < 0.4 and overlap < 2:
            # Try to suggest related query
            suggestion = DomainDetector.suggest_related_query(domain_keywords)
            return False, suggestion
        
        return True, None
    
    @staticmethod
    def suggest_related_query(domain_keywords: List[str]) -> str:
        """Suggest a related query based on domain"""
        if len(domain_keywords) >= 3:
            top_terms = ", ".join(domain_keywords[:3])
            return f"It seems like these documents focus on topics related to {top_terms}. Would you like to ask about those topics instead?"
        return "This question appears to be outside the scope of the uploaded documents. Please ask questions related to the document content."

# ============================================================================

class DocumentLoader:
    @staticmethod
    def load_documents(file_paths: List[str]) -> List[Document]:
        documents = []
        
        for file_path in file_paths:
            try:
                file_name = os.path.basename(file_path)
                file_ext = Path(file_path).suffix.lower()
                
                docs = DocumentLoader._load_single_file(file_path, file_name, file_ext)
                documents.extend(docs)
                
            except Exception as e:
                st.error(f"Error loading {file_name}: {str(e)}")
        
        return documents
    
    @staticmethod
    def _load_single_file(file_path: str, file_name: str, file_ext: str) -> List[Document]:
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
        else:
            st.warning(f"Unsupported file type: {file_ext}")
            return []
        
        return docs

# ============================================================================

class VectorStoreManager:
    @staticmethod
    @st.cache_resource
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name=RAGConfig.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
    
    @staticmethod
    def create_vector_db(documents: List[Document]):
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
        
        embeddings = VectorStoreManager.get_embeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        return vector_db, len(chunks)
    
    @staticmethod
    def save_vector_db(vector_db):
        """Save vector database to disk"""
        os.makedirs(RAGConfig.VECTOR_DB_PATH, exist_ok=True)
        vector_db.save_local(RAGConfig.VECTOR_DB_PATH)
    
    @staticmethod
    def load_vector_db():
        """Load vector database from disk"""
        if os.path.exists(RAGConfig.VECTOR_DB_PATH):
            embeddings = VectorStoreManager.get_embeddings()
            return FAISS.load_local(RAGConfig.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return None

# ============================================================================

class RAGAssistant:
    def __init__(self, vector_db, groq_api_key: str, domain_keywords: List[str], user_profile: Optional[Dict] = None):
        self.vector_db = vector_db
        self.client = Groq(api_key=groq_api_key)
        self.domain_keywords = domain_keywords
        self.user_profile = user_profile or {}
        
        if 'query_cache' not in st.session_state:
            st.session_state.query_cache = {}
        if 'cache_hits' not in st.session_state:
            st.session_state.cache_hits = 0
        if 'cache_misses' not in st.session_state:
            st.session_state.cache_misses = 0
    
    def query(self, question: str, k: int = RAGConfig.TOP_K_RESULTS) -> Dict:
        start_time = time.time()
        
        query_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
        if query_hash in st.session_state.query_cache:
            st.session_state.cache_hits += 1
            result = st.session_state.query_cache[query_hash]
            result['metrics']['latency'] = time.time() - start_time
            result['metrics']['cache_hit'] = True
            return result
        
        st.session_state.cache_misses += 1
        relevant_docs = self.vector_db.similarity_search_with_score(question, k=k)
        
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
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Check domain relevance
        is_relevant, suggestion = DomainDetector.check_query_relevance(
            question, self.domain_keywords, avg_similarity
        )
        
        if not is_relevant:
            result = {
                'answer': f"🚫 **Out of Domain Query Detected**\n\n{suggestion}",
                'citations': [],
                'warning': "This question appears to be outside the scope of the uploaded documents.",
                'metrics': {
                    'avg_similarity': avg_similarity,
                    'confidence': "N/A",
                    'confidence_emoji': "🔴",
                    'hallucination_risk': "N/A",
                    'num_sources': 0,
                    'latency': time.time() - start_time,
                    'cache_hit': False,
                    'out_of_domain': True
                }
            }
            return result
        
        context = "\n".join(context_parts)
        answer = self._generate_answer(context, question, avg_similarity)
        
        confidence_level, confidence_emoji = assess_confidence(similarity_scores)
        hallucination_risk, warning = detect_hallucination_risk(
            answer, len(relevant_docs), avg_similarity
        )
        
        # Add verification reminder for low confidence
        if avg_similarity < 0.6:
            top_sources = [c['title'] for c in citations[:2]]
            verification_msg = f"\n\n⚠️ **Please verify this information** with the following sources: {', '.join(top_sources)}"
            answer += verification_msg
            if not warning:
                warning = "Low confidence answer - please cross-verify with sources"
        
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
                'cache_hit': False,
                'out_of_domain': False
            }
        }
        
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
    def _generate_answer(self, context: str, question: str, avg_similarity: float) -> str:
        # Build personalization context
        personalization = ""
        if self.user_profile:
            age = self.user_profile.get('age', '')
            gender = self.user_profile.get('gender', '')
            location = self.user_profile.get('location', '')
            interests = self.user_profile.get('interests', '')
            
            personalization = f"\n\nUSER PROFILE (adapt response style accordingly):\n"
            if age:
                personalization += f"- Age: {age}\n"
            if gender:
                personalization += f"- Gender: {gender}\n"
            if location:
                personalization += f"- Location: {location}\n"
            if interests:
                personalization += f"- Areas of Interest: {interests}\n"
        
        # Add confidence-based instruction
        confidence_instruction = ""
        if avg_similarity < 0.6:
            confidence_instruction = "\n\nIMPORTANT: This query has lower confidence (similarity < 60%). After providing the answer, explicitly recommend the user to cross-verify the information with the specific source documents mentioned."
        
        prompt = f"""You are a precise document analysis assistant. Answer questions using ONLY the provided sources.

STRICT RULES:
1. Answer ONLY with information EXPLICITLY in the sources
2. Cite sources: "According to Source 1..." or "Source 2 states..."
3. If info is missing, say: "This information is not available in the provided documents"
4. Do NOT infer, assume, or add external knowledge
5. Quote exact phrases when possible
{confidence_instruction}

SOURCES:
{context}
{personalization}

QUESTION: {question}

ANSWER (with source citations, adapted to user profile if provided):"""
        
        response = self.client.chat.completions.create(
            model=RAGConfig.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=RAGConfig.LLM_TEMPERATURE,
            max_tokens=RAGConfig.LLM_MAX_TOKENS
        )
        
        return response.choices[0].message.content

# ============================================================================

def show_personalization_form():
    """Display form to collect user personalization data"""
    with st.sidebar:
        st.divider()
        st.header("👤 Personalization")
        
        with st.expander("📝 Set Your Profile", expanded=False):
            st.markdown("Help us personalize responses for you!")
            
            user_id = st.text_input("User ID (optional)", value=st.session_state.get('user_id', ''), 
                                   help="Leave blank for anonymous")
            
            age_group = st.selectbox("Age Group", 
                                    ["Prefer not to say", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
            
            gender = st.selectbox("Gender", 
                                 ["Prefer not to say", "Male", "Female", "Non-binary", "Other"])
            
            location = st.text_input("Location (City/Country)", 
                                    help="E.g., Mumbai, India")
            
            interests = st.text_area("Areas of Interest", 
                                    help="E.g., Finance, Technology, Healthcare",
                                    placeholder="Separate multiple interests with commas")
            
            if st.button("💾 Save Profile", use_container_width=True):
                profile = {
                    'age': age_group if age_group != "Prefer not to say" else "",
                    'gender': gender if gender != "Prefer not to say" else "",
                    'location': location,
                    'interests': interests
                }
                
                if user_id:
                    UserProfileManager.save_profile(user_id, profile)
                    st.session_state.user_id = user_id
                
                st.session_state.user_profile = profile
                st.success("✅ Profile saved!")
                st.rerun()
        
        # Show current profile
        if st.session_state.get('user_profile'):
            profile = st.session_state.user_profile
            st.markdown("**Current Profile:**")
            if profile.get('age'):
                st.markdown(f"📅 Age: {profile['age']}")
            if profile.get('gender'):
                st.markdown(f"👤 Gender: {profile['gender']}")
            if profile.get('location'):
                st.markdown(f"📍 Location: {profile['location']}")
            if profile.get('interests'):
                st.markdown(f"💡 Interests: {profile['interests']}")

# ============================================================================

def manage_documents():
    """Interface for managing persistent documents"""
    with st.sidebar:
        st.divider()
        st.header("📚 Document Library")
        
        stored_files = DocumentStorage.get_stored_files()
        
        if stored_files:
            st.markdown(f"**{len(stored_files)} documents in library**")
            
            with st.expander("📋 View Documents", expanded=False):
                for file_path in stored_files:
                    filename = os.path.basename(file_path)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(filename[:30] + "..." if len(filename) > 30 else filename)
                    with col2:
                        if st.button("🗑️", key=f"del_{filename}", help="Delete"):
                            DocumentStorage.delete_file(filename)
                            st.session_state.needs_rebuild = True
                            st.rerun()
        else:
            st.info("No documents in library yet")
        
        st.divider()
        
        # Upload new documents
        st.subheader("➕ Add Documents")
        new_files = st.file_uploader(
            "Upload new documents",
            type=RAGConfig.SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help="These will be added to your library",
            key="new_doc_uploader"
        )
        
        if new_files and st.button("💾 Add to Library", use_container_width=True):
            with st.spinner("Saving documents..."):
                for file in new_files:
                    DocumentStorage.save_uploaded_file(file)
                st.session_state.needs_rebuild = True
                st.success(f"✅ Added {len(new_files)} document(s)")
                st.rerun()
        
        if st.button("🔄 Rebuild Vector Store", use_container_width=True, 
                    help="Rebuild the search index with current documents"):
            st.session_state.needs_rebuild = True
            st.rerun()

# ============================================================================

def main():
    st.title("🤖 Enhanced RAG Assistant")
    st.markdown("### Personalized Multi-format Document QA System")
    
    # Initialize session state
    if 'user_profile' not in st.session_state:
        # Try to load saved profile
        user_id = st.session_state.get('user_id', '')
        if user_id:
            profile = UserProfileManager.get_profile(user_id)
            if profile:
                st.session_state.user_profile = profile
    
    if 'needs_rebuild' not in st.session_state:
        st.session_state.needs_rebuild = False
    
    with st.sidebar:
        st.header("⚙️ Configuration")
    

        groq_api_key = st.text_input("Enter Groq API Key", type="password")
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API key")
            st.stop()
    
        st.divider()
        
        st.header("🔧 Settings")
        k_results = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=RAGConfig.TOP_K_RESULTS,
            help="Number of document chunks to retrieve"
        )
        
        st.divider()
        
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
    
    # Personalization form
    show_personalization_form()
    
    # Document management
    manage_documents()
    
    # Load or build vector store
    stored_files = DocumentStorage.get_stored_files()
    
    if not stored_files:
        st.info("📚 No documents in library. Please add documents using the sidebar.")
        st.stop()
    
    # Check if we need to rebuild the vector store
    if st.session_state.needs_rebuild or 'vector_db' not in st.session_state:
        with st.spinner("🔨 Building vector store from library documents..."):
            documents = DocumentLoader.load_documents(stored_files)
            
            if not documents:
                st.error("No documents could be loaded. Please check your files.")
                st.stop()
            
            vector_db, num_chunks = VectorStoreManager.create_vector_db(documents)
            VectorStoreManager.save_vector_db(vector_db)
            
            # Extract domain keywords
            domain_keywords = DomainDetector.extract_domain_keywords(documents)
            
            st.session_state.vector_db = vector_db
            st.session_state.num_documents = len(documents)
            st.session_state.num_chunks = num_chunks
            st.session_state.domain_keywords = domain_keywords
            st.session_state.needs_rebuild = False
            
            st.success(f"✅ Processed {len(documents)} documents into {num_chunks} chunks")
    
    # Create assistant
    assistant = RAGAssistant(
        st.session_state.vector_db, 
        groq_api_key,
        st.session_state.get('domain_keywords', []),
        st.session_state.get('user_profile')
    )
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", st.session_state.get('num_documents', 0))
    with col2:
        st.metric("🔍 Chunks", st.session_state.get('num_chunks', 0))
    with col3:
        cached = len(st.session_state.get('query_cache', {}))
        st.metric("💾 Cached Queries", cached)
    
    st.divider()
    
    # Query interface
    question = st.text_input(
        "💬 Ask a question about your documents:",
        placeholder="What is this document about?",
        help="Enter your question"
    )
    
    if st.button("🔎 Search", type="primary", use_container_width=True) and question:
        with st.spinner("🔍 Searching documents..."):
            try:
                result = assistant.query(question, k=k_results)
                
                # Check if out of domain
                if result['metrics'].get('out_of_domain', False):
                    st.error(result['answer'])
                    if result['warning']:
                        st.warning(result['warning'])
                else:
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    if result['warning']:
                        st.warning(result['warning'])
                    
                   
                    # Display sources
                    if result['citations']:
                        st.markdown("### 📚 Sources")
                        for i, citation in enumerate(result['citations'], 1):
                            with st.expander(f"Source {i}: {citation['title']}", expanded=False):
                                st.markdown(f'<div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px;">', unsafe_allow_html=True)
                                st.markdown(f"**Page/Sheet:** {citation['page']}")
                                st.markdown(f"**Section:** {citation['section']}")
                                st.markdown(f"**Relevance:** {citation['similarity']:.1%}")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display hallucination risk
                    risk = result['metrics']['hallucination_risk']
                    if "HIGH" in risk or "MEDIUM" in risk:
                        st.info(f"🔍 Hallucination Risk: {risk}")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.exception(e)
    
    # Example queries based on domain
    if st.session_state.get('domain_keywords'):
        st.divider()
        st.markdown("### 💡 Suggested Questions")
        
        domain_keywords = st.session_state.domain_keywords[:5]
        st.markdown(f"Based on your documents (topics: *{', '.join(domain_keywords[:3])}*), try asking:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 What are the main topics covered?", use_container_width=True):
                st.session_state.suggested_query = "What are the main topics covered in these documents?"
                st.rerun()
        with col2:
            if st.button("🔍 Summarize key information", use_container_width=True):
                st.session_state.suggested_query = "Can you provide a summary of the key information?"
                st.rerun()
        
        # Handle suggested query
        if st.session_state.get('suggested_query'):
            st.info(f"Running: {st.session_state.suggested_query}")
            # Clear the suggested query
            query = st.session_state.suggested_query
            st.session_state.suggested_query = None
            
            with st.spinner("🔍 Searching documents..."):
                try:
                    result = assistant.query(query, k=k_results)
                    
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    if result['warning']:
                        st.warning(result['warning'])
                    
                    if result['citations']:
                        st.markdown("### 📚 Sources")
                        for i, citation in enumerate(result['citations'], 1):
                            with st.expander(f"Source {i}: {citation['title']}", expanded=False):
                                st.markdown(f"**Page:** {citation['page']} | **Section:** {citation['section']} | **Relevance:** {citation['similarity']:.1%}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()