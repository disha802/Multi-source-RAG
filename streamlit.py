"""
Enhanced Domain-Agnostic RAG Assistant - Streamlit Web Application
Multi-format document QA system with intelligent domain detection and adaptation
"""

import streamlit as st
import os
import re
import time
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

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
    page_title="Domain-Agnostic RAG Assistant",
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
    UPLOAD_FOLDER = "upload"
    VECTOR_DB_PATH = "vector_store"
    USER_PROFILES_PATH = "user_profiles.json"
    DOMAIN_METADATA_PATH = "domain_metadata.json"

# ============================================================================

def extract_section_from_text(text: str) -> str:
    """Extract section headers intelligently from any document type"""
    if not text or len(text.strip()) == 0:
        return "Empty Content"
    
    patterns = [
        r'^#{1,3}\s+(.+)$',  # Markdown headers
        r'^(SECTION|CHAPTER|PART|ARTICLE)\s+[\d\w]+[:\s]+(.+)$',  # Formal sections
        r'^\d+\.\s+([A-Z][A-Za-z\s]{3,})$',  # Numbered sections
        r'^([A-Z][A-Z\s]{5,})$',  # ALL CAPS headers
    ]
    
    lines = text.split('\n')[:20]  # Check more lines for better detection
    for line in lines:
        line_clean = line.strip()
        if len(line_clean) < 3 or len(line_clean) > 100:
            continue
        
        for pattern in patterns:
            match = re.search(pattern, line_clean, re.IGNORECASE)
            if match:
                section = match.group(1) if match.lastindex == 1 else match.group(2)
                return section.strip()[:100]
    
    # Fallback: use first substantial line
    for line in lines:
        if len(line.strip()) > 10:
            return line.strip()[:100]
    
    return "General Content"

def calculate_similarity_score(distance: float) -> float:
    """Convert FAISS L2 distance to similarity percentage"""
    return 1 / (1 + distance)

def assess_confidence(similarity_scores: List[float]) -> Tuple[str, str]:
    """Assess retrieval confidence based on similarity"""
    avg_sim = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    if avg_sim > 0.7:
        return "HIGH", "🟢"
    elif avg_sim > 0.5:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🔴"

def detect_hallucination_risk(answer: str, num_sources: int, avg_similarity: float) -> Tuple[str, str]:
    """Detect potential hallucination in responses"""
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

class DomainAnalyzer:
    """Analyzes document collection to understand domain and context"""
    
    INTENT_EXAMPLES = {
        "document_related": [
            "summarize the document",
            "what topics are discussed",
            "what does the report say",
            "explain section 2",
            "summaries of uploaded files",
            "analyze the provided documents"
        ],
        "entertainment": [
            "latest netflix trend",
            "best movies right now",
            "what anime should I watch",
            "top tv shows",
            "celebrity news",
            "oscar winners"
        ],
        "news": [
            "latest news",
            "current events",
            "breaking news",
            "what happened today",
            "politics today"
        ],
        "personal": [
            "weather today",
            "temperature in my city",
            "how to cook pasta",
            "best restaurants near me",
            "what should I eat today"
        ],
        "general_knowledge": [
            "who is alan turing",
            "what is photosynthesis",
            "capital of france",
            "history of ai"
        ]
    }

    @staticmethod
    def semantic_intent_classify(query: str, embeddings_model) -> str:
        """Classify query intent using embedding similarity."""
        query_emb = embeddings_model.embed_query(query)

        best_category = None
        best_sim = -1

        for category, examples in DomainAnalyzer.INTENT_EXAMPLES.items():
            for ex in examples:
                ex_emb = embeddings_model.embed_query(ex)
                sim = sum(a*b for a,b in zip(query_emb, ex_emb))  # dot product since embeddings normalized
                if sim > best_sim:
                    best_sim = sim
                    best_category = category

        return best_category

    
    @staticmethod
    def analyze_document_collection(documents: List[Document]) -> Dict:
        """Perform comprehensive domain analysis"""
        
        # Extract key terms and frequencies
        all_text = " ".join([doc.page_content[:1000] for doc in documents[:20]])
        words = re.findall(r'\b[A-Za-z]{4,}\b', all_text.lower())
        word_freq = Counter(words)
        
        # Remove common stopwords
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 
                    'will', 'would', 'could', 'should', 'about', 'which', 'their',
                    'there', 'where', 'when', 'what', 'these', 'those'}
        filtered_freq = {k: v for k, v in word_freq.items() if k not in stopwords}
        
        # Get top keywords
        top_keywords = [word for word, freq in sorted(filtered_freq.items(), 
                       key=lambda x: x[1], reverse=True)[:100]]
        
        # Detect document types
        doc_types = {}
        for doc in documents:
            doc_type = doc.metadata.get('source_type', 'Unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Detect domain characteristics
        domain_indicators = DomainAnalyzer._detect_domain_type(top_keywords, documents)
        
        return {
            'keywords': top_keywords,
            'document_types': doc_types,
            'domain_type': domain_indicators['type'],
            'domain_confidence': domain_indicators['confidence'],
            'characteristics': domain_indicators['characteristics'],
            'total_documents': len(documents),
            'vocabulary_size': len(filtered_freq)
        }
    
    @staticmethod
    def _detect_domain_type(keywords: List[str], documents: List[Document]) -> Dict:
        """Detect the likely domain/field of the document collection"""
        
        # Domain keyword patterns
        domain_patterns = {
            'Finance/Business': ['revenue', 'profit', 'market', 'investment', 'financial', 
                                'sales', 'cost', 'business', 'company', 'stock'],
            'Healthcare/Medical': ['patient', 'medical', 'health', 'treatment', 'disease', 
                                  'clinical', 'hospital', 'diagnosis', 'therapy'],
            'Technology/IT': ['software', 'system', 'data', 'network', 'server', 'code',
                             'application', 'technology', 'computer', 'digital'],
            'Legal/Compliance': ['legal', 'contract', 'agreement', 'clause', 'party',
                               'compliance', 'regulation', 'law', 'rights'],
            'Academic/Research': ['study', 'research', 'analysis', 'results', 'method',
                                'theory', 'experiment', 'findings', 'hypothesis'],
            'Technical/Engineering': ['design', 'specification', 'process', 'system',
                                     'component', 'function', 'parameter', 'technical']
        }
        
        # Calculate domain scores
        domain_scores = {}
        for domain, patterns in domain_patterns.items():
            score = sum(1 for keyword in keywords[:50] if keyword in patterns)
            domain_scores[domain] = score
        
        # Determine primary domain
        if max(domain_scores.values()) > 0:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[primary_domain] / 10  # Normalize
            confidence = min(confidence, 1.0)
        else:
            primary_domain = "General/Mixed"
            confidence = 0.5
        
        # Extract characteristics
        characteristics = []
        
        # Check for tabular data
        tabular_count = sum(1 for doc in documents[:10] 
                          if doc.metadata.get('source_type') in ['CSV', 'Excel'])
        if tabular_count > len(documents) * 0.3:
            characteristics.append("Heavy tabular/numerical data")
        
        # Check for technical language
        technical_terms = ['system', 'process', 'method', 'function', 'parameter']
        if sum(1 for k in keywords[:30] if k in technical_terms) > 5:
            characteristics.append("Technical documentation")
        
        # Check document length
        avg_length = sum(len(doc.page_content) for doc in documents[:10]) / min(len(documents), 10)
        if avg_length > 3000:
            characteristics.append("Detailed/lengthy content")
        elif avg_length < 1000:
            characteristics.append("Concise/summary format")
        
        return {
            'type': primary_domain,
            'confidence': confidence,
            'characteristics': characteristics,
            'domain_scores': domain_scores
        }

    # @staticmethod
    # def check_query_relevance(query: str, domain_metadata: Dict, 
    #                          context_similarity: float) -> Tuple[bool, Optional[str]]:
    #     """Check if query is relevant to the document domain"""
        
    #     # High similarity = definitely relevant
    #     if context_similarity > 0.6:
    #         return True, None
        
    #     # Detect obviously off-topic queries (entertainment, general knowledge, etc.)
    #     off_topic_patterns = {
    #         'entertainment': ['movie', 'film', 'tv show', 'series', 'netflix', 'actor', 'actress', 
    #                         'music', 'song', 'album', 'band', 'singer', 'concert', 'game', 'video game'],
    #         'general_knowledge': ['who is', 'who was', 'when did', 'where is', 'what is the capital',
    #                             'how tall', 'how old', 'birthday', 'famous for'],
    #         'personal': ['weather', 'temperature', 'forecast', 'restaurant', 'recipe', 'cook'],
    #         'current_events': ['news today', 'latest news', 'current events', 'breaking news']
    #     }
        
    #     query_lower = query.lower()
        
    #     # Check for obviously off-topic patterns
    #     for category, patterns in off_topic_patterns.items():
    #         if any(pattern in query_lower for pattern in patterns):
    #             # Entertainment / personal / news queries are ALWAYS out-of-domain
    #             if category in ['entertainment', 'personal', 'current_events']:
    #                 return False, DomainAnalyzer._generate_gentle_refusal(query, domain_metadata)

    #             # For general-knowledge, still check overlap
    #             domain_words = set(domain_metadata.get('keywords', [])[:100])
    #             query_words = set(re.findall(r'\b[A-Za-z]{4,}\b', query_lower))
    #             overlap = len(query_words & domain_words)
    #             if overlap < 2:
    #                 return False, DomainAnalyzer._generate_gentle_refusal(query, domain_metadata)

        
    #     # Extract query terms
    #     query_words = set(re.findall(r'\b[A-Za-z]{4,}\b', query_lower))
    #     domain_words = set(domain_metadata.get('keywords', [])[:100])
        
    #     # Calculate overlap
    #     overlap = len(query_words & domain_words)
    #     overlap_ratio = overlap / len(query_words) if query_words else 0
        
    #     # Low similarity AND low overlap = likely out of domain
    #     if context_similarity < 0.4 and overlap_ratio < 0.2:
    #         return False, DomainAnalyzer._generate_gentle_refusal(query, domain_metadata)
        
    #     return True, None

    @staticmethod
    def check_query_relevance(query: str, domain_metadata: Dict, context_similarity: float) -> Tuple[bool, Optional[str]]:

        query_lower = query.lower()

        # 1. Hard-block categories (instant OOD)
        hard_block_keywords = [
            "netflix", "movie", "film", "anime", "tv show", "series",
            "celebrity", "concert", "tiktok", "instagram",
            "breaking news", "latest news", "weather", "temperature",
            "restaurant", "cook", "recipe"
        ]

        if any(k in query_lower for k in hard_block_keywords):
            return False, DomainAnalyzer._generate_gentle_refusal(query, domain_metadata)

        # 2. Semantic classification
        embeddings = VectorStoreManager.get_embeddings()
        intent = DomainAnalyzer.semantic_intent_classify(query, embeddings)

        if intent in ["entertainment", "news", "personal"]:
            return False, DomainAnalyzer._generate_gentle_refusal(query, domain_metadata)

        # 3. High-similarity override
        if context_similarity > 0.60:
            return True, None

        # 4. Low-overlap + low-similarity → OOD
        domain_words = set(domain_metadata.get('keywords', [])[:100])
        query_words = set(re.findall(r'\b[A-Za-z]{4,}\b', query_lower))
        overlap = len(query_words & domain_words)
        overlap_ratio = overlap / max(len(query_words), 1)

        if context_similarity < 0.40 and overlap_ratio < 0.20:
            return False, DomainAnalyzer._generate_gentle_refusal(query, domain_metadata)

        return True, None

    @staticmethod
    def _generate_gentle_refusal(query: str, domain_metadata: Dict) -> str:
        """Generate a kind, helpful OOD message when query is off-scope."""

        top_keywords = domain_metadata.get('keywords', [])[:5]

        msg = (
            "🔎 **Heads up!**\n"
            "It looks like you're asking something *outside the scope of the documents you've uploaded.*\n\n"
            "Your files mainly talk about topics like: "
            f"**{', '.join(top_keywords[:3])}**\n\n"
            "I can help you with things like:\n"
            "• Summarizing documents\n"
            "• Explaining sections\n"
            "• Extracting key info\n"
            "• Clarifying topics inside the uploaded content\n\n"
            "Feel free to ask anything related to your documents! 💡"
        )

        return msg


    # @staticmethod
    # def _generate_gentle_refusal(query: str, domain_metadata: Dict) -> str:
    #     """Generate a gentle, helpful refusal for out-of-domain queries"""
        
    #     # Get document scope info without explicitly mentioning "domain"
    #     top_keywords = domain_metadata.get('keywords', [])[:5]
        
    #     refusal = "I'd be happy to help, but that question appears to be outside the scope of your uploaded documents. 📄\n\n"
        
    #     if top_keywords:
    #         refusal += f"Your documents cover topics like: **{', '.join(top_keywords[:3])}**\n\n"
        
    #     refusal += "💡 **I can help you with questions about:**\n"
    #     refusal += "- Information contained in your documents\n"
    #     refusal += "- Summaries of specific sections\n"
    #     refusal += "- Details about the topics covered\n\n"
    #     refusal += "Feel free to ask me anything related to the content you've uploaded!"
        
    #     return refusal

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
        DocumentStorage.ensure_upload_folder()
        file_path = os.path.join(RAGConfig.UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    @staticmethod
    def get_stored_files() -> List[str]:
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
        file_path = os.path.join(RAGConfig.UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    @staticmethod
    def save_domain_metadata(metadata: Dict):
        """Save domain analysis metadata"""
        with open(RAGConfig.DOMAIN_METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_domain_metadata() -> Optional[Dict]:
        """Load domain analysis metadata"""
        if os.path.exists(RAGConfig.DOMAIN_METADATA_PATH):
            try:
                with open(RAGConfig.DOMAIN_METADATA_PATH, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

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
        os.makedirs(RAGConfig.VECTOR_DB_PATH, exist_ok=True)
        vector_db.save_local(RAGConfig.VECTOR_DB_PATH)
    
    @staticmethod
    def load_vector_db():
        if os.path.exists(RAGConfig.VECTOR_DB_PATH):
            embeddings = VectorStoreManager.get_embeddings()
            return FAISS.load_local(RAGConfig.VECTOR_DB_PATH, embeddings, 
                                   allow_dangerous_deserialization=True)
        return None

# ============================================================================

class RAGAssistant:
    def __init__(self, vector_db, groq_api_key: str, domain_metadata: Dict, 
                 user_profile: Optional[Dict] = None):
        self.vector_db = vector_db
        self.client = Groq(api_key=groq_api_key)
        self.domain_metadata = domain_metadata
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
        is_relevant, suggestion = DomainAnalyzer.check_query_relevance(
            question, self.domain_metadata, avg_similarity
        )
        
        if not is_relevant:
            result = {
                'answer': suggestion,
                'citations': [],
                'warning': "This question appears outside your document scope.",
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
            verification_msg = f"\n\n⚠️ **Please verify this information** with: {', '.join(top_sources)}"
            answer += verification_msg
            if not warning:
                warning = "Low confidence - please cross-verify with sources"
        
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
        # Build domain-aware context
        domain_context = self._build_domain_context()
        
        # Build personalization context
        personalization = self._build_personalization_context()
        
        # Confidence-based instruction
        confidence_instruction = ""
        if avg_similarity < 0.6:
            confidence_instruction = "\n\nIMPORTANT: Lower confidence query. Explicitly mention which sources support each claim and recommend verification."
        
        prompt = f"""You are a document analysis assistant. Answer questions using ONLY the provided sources.

CRITICAL RULES:
1. Base answers EXCLUSIVELY on the provided source material
2. Always cite sources: "According to Source 1..." or "Source 2 indicates..."
3. If information is not in the sources, clearly state: "This information is not available in the provided documents"
4. Do NOT make inferences beyond what is explicitly stated
5. Quote relevant passages when appropriate
{confidence_instruction}

DOCUMENT CONTEXT:
{domain_context}

SOURCES:
{context}
{personalization}

QUESTION: {question}

ANSWER (cite all sources used):"""
        
        response = self.client.chat.completions.create(
            model=RAGConfig.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=RAGConfig.LLM_TEMPERATURE,
            max_tokens=RAGConfig.LLM_MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    def _build_domain_context(self) -> str:
        """Build domain-aware context for the LLM"""
        domain_type = self.domain_metadata.get('domain_type', 'General')
        characteristics = self.domain_metadata.get('characteristics', [])
        doc_types = self.domain_metadata.get('document_types', {})
        
        context = f"DOMAIN: {domain_type}\n"
        
        if characteristics:
            context += f"CHARACTERISTICS: {', '.join(characteristics)}\n"
        
        if doc_types:
            doc_list = [f"{k} ({v})" for k, v in doc_types.items()]
            context += f"DOCUMENT TYPES: {', '.join(doc_list)}\n"
        
        return context
    
    def _build_personalization_context(self) -> str:
        """Build personalization context with style instructions"""
        if not self.user_profile:
            return ""
        
        style_instructions = {
            "Professional": "Use formal, precise language. Structure answers clearly with proper business terminology.",
            "Casual & Friendly": "Use conversational tone, as if explaining to a friend. Keep it engaging and approachable.",
            "Educational": "Explain concepts thoroughly, as if teaching. Break down complex ideas into simple components.",
            "Concise & Direct": "Be brief and to-the-point. Skip unnecessary details and get straight to the answer.",
            "Storytelling": "Present information in a narrative flow. Use examples and real-world scenarios to illustrate points."
        }
        
        context_parts = []
        
        # Style instruction
        style = self.user_profile.get('response_style', 'Professional')
        if style in style_instructions:
            context_parts.append(f"\nRESPONSE STYLE: {style_instructions[style]}")
        
        # Age-appropriate language
        if self.user_profile.get('age'):
            age = self.user_profile['age']
            if age in ['18-25', '26-35']:
                context_parts.append("Use modern, relatable language.")
            elif age in ['56-65', '65+']:
                context_parts.append("Use clear, well-explained language without jargon.")
        
        # Interest-based examples
        if self.user_profile.get('interests'):
            interests = self.user_profile['interests']
            context_parts.append(f"When appropriate, relate concepts to these interests: {interests}")
        
        # Analogy preference
        if self.user_profile.get('use_analogies', True):
            context_parts.append("Include helpful analogies or examples when they clarify concepts.")
        
        # Important note about facts
        if context_parts:
            context_parts.insert(0, "\n--- PERSONALIZATION (adapt presentation style, NOT facts) ---")
            context_parts.append("CRITICAL: Keep all factual information accurate. Only adapt HOW you present it, not WHAT you present.")
        
        return "\n".join(context_parts) if context_parts else ""

# ============================================================================

def show_domain_insights():
    """Display insights about the document collection domain"""
    if 'domain_metadata' not in st.session_state:
        return
    
    metadata = st.session_state.domain_metadata
    
    with st.sidebar:
        st.divider()
        st.header("📊 Document Insights")
        
        with st.expander("📈 Collection Statistics", expanded=False):
            # Show stats without explicitly mentioning "domain"
            doc_types = metadata.get('document_types', {})
            if doc_types:
                st.markdown("**Document Types:**")
                for doc_type, count in doc_types.items():
                    st.markdown(f"• {doc_type}: {count}")
            
            total_docs = metadata.get('total_documents', 0)
            vocab_size = metadata.get('vocabulary_size', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", total_docs)
            with col2:
                st.metric("Vocabulary", vocab_size)
            
            # Show top keywords without domain classification
            keywords = metadata.get('keywords', [])[:8]
            if keywords:
                st.markdown("**Key Topics:**")
                st.markdown(", ".join(keywords))

def show_personalization_form():
    """Display user personalization form"""
    with st.sidebar:
        st.divider()
        st.header("👤 Personalization")
        
        with st.expander("🎨 Customize Response Style", expanded=False):
            st.markdown("Tailor how answers are presented to you!")
            
            user_id = st.text_input("User ID (optional)", 
                                   value=st.session_state.get('user_id', ''))
            
            age_group = st.selectbox("Age Group", 
                ["Prefer not to say", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
            
            response_style = st.selectbox(
                "Response Style",
                ["Professional", "Casual & Friendly", "Educational", "Concise & Direct", "Storytelling"],
                help="Choose how you'd like answers to be presented"
            )
            
            interests = st.text_area("Areas of Interest", 
                help="E.g., Finance, Technology, Healthcare",
                placeholder="Separate with commas")
            
            use_analogies = st.checkbox("Use analogies/examples", value=True,
                                       help="Include relatable examples and comparisons")
            
            if st.button("💾 Save Profile", use_container_width=True):
                profile = {
                    'age': age_group if age_group != "Prefer not to say" else "",
                    'interests': interests,
                    'response_style': response_style,
                    'use_analogies': use_analogies
                }
                
                if user_id:
                    UserProfileManager.save_profile(user_id, profile)
                    st.session_state.user_id = user_id
                
                st.session_state.user_profile = profile
                st.success("✅ Profile saved!")
                st.rerun()
        
        if st.session_state.get('user_profile'):
            profile = st.session_state.user_profile
            st.markdown("**Active Profile:**")
            if profile.get('response_style'):
                st.markdown(f"🎨 Style: {profile['response_style']}")
            if profile.get('age'):
                st.markdown(f"📅 Age: {profile['age']}")
            if profile.get('interests'):
                st.markdown(f"💡 Interests: {profile['interests']}")

def manage_documents():
    """Document library management interface"""
    with st.sidebar:
        st.divider()
        st.header("📚 Document Library")
        
        stored_files = DocumentStorage.get_stored_files()
        
        if stored_files:
            st.markdown(f"**{len(stored_files)} documents**")
            
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
            st.info("No documents yet")
        
        st.divider()
        st.subheader("➕ Add Documents")
        new_files = st.file_uploader(
            "Upload documents",
            type=RAGConfig.SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            key="new_doc_uploader"
        )
        
        if new_files and st.button("💾 Add to Library", use_container_width=True):
            with st.spinner("Saving documents..."):
                for file in new_files:
                    DocumentStorage.save_uploaded_file(file)
                st.session_state.needs_rebuild = True
                st.success(f"✅ Added {len(new_files)} document(s)")
                st.rerun()
        
        if st.button("🔄 Rebuild Index", use_container_width=True):
            st.session_state.needs_rebuild = True
            st.rerun()

# ============================================================================

def generate_smart_suggestions(domain_metadata: Dict) -> List[str]:
    """Generate domain-appropriate suggested questions"""
    domain_type = domain_metadata.get('domain_type', 'General')
    keywords = domain_metadata.get('keywords', [])[:10]
    
    suggestions = [
        "What are the main topics covered in these documents?",
        "Can you provide a summary of the key information?"
    ]
    
    # Add domain-specific suggestions
    if 'Finance' in domain_type or 'Business' in domain_type:
        suggestions.extend([
            "What financial metrics or figures are mentioned?",
            "What are the key business insights?"
        ])
    elif 'Technical' in domain_type or 'Technology' in domain_type:
        suggestions.extend([
            "What systems or processes are described?",
            "What technical specifications are provided?"
        ])
    elif 'Healthcare' in domain_type or 'Medical' in domain_type:
        suggestions.extend([
            "What treatments or procedures are discussed?",
            "What are the key health-related findings?"
        ])
    elif 'Legal' in domain_type:
        suggestions.extend([
            "What are the main contractual obligations?",
            "What compliance requirements are mentioned?"
        ])
    else:
        # Generic suggestions based on keywords
        if keywords:
            suggestions.append(f"Tell me about {keywords[0]} mentioned in the documents")
    
    return suggestions

# ============================================================================

def main():
    st.title("🤖 Domain-Agnostic RAG Assistant")
    st.markdown("### Intelligent Document QA System")
    
    # Initialize session state
    if 'user_profile' not in st.session_state:
        user_id = st.session_state.get('user_id', '')
        if user_id:
            profile = UserProfileManager.get_profile(user_id)
            if profile:
                st.session_state.user_profile = profile
    
    if 'needs_rebuild' not in st.session_state:
        st.session_state.needs_rebuild = False
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.text_input("Groq API Key", type="password")
        
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API key")
            st.stop()
        
        st.divider()
        st.header("🔧 Settings")
        k_results = st.slider("Number of sources", 1, 10, RAGConfig.TOP_K_RESULTS)
        
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
    
    # Show forms
    show_domain_insights()
    show_personalization_form()
    manage_documents()
    
    # Load documents
    stored_files = DocumentStorage.get_stored_files()
    
    if not stored_files:
        st.info("📚 No documents in library. Please add documents using the sidebar.")
        st.stop()
    
    # Build/load vector store
    if st.session_state.needs_rebuild or 'vector_db' not in st.session_state:
        with st.spinner("🔨 Building vector store..."):
            documents = DocumentLoader.load_documents(stored_files)
            
            if not documents:
                st.error("No documents could be loaded.")
                st.stop()
            
            # Analyze domain (internal use only, not displayed)
            domain_metadata = DomainAnalyzer.analyze_document_collection(documents)
            DocumentStorage.save_domain_metadata(domain_metadata)
            
            vector_db, num_chunks = VectorStoreManager.create_vector_db(documents)
            VectorStoreManager.save_vector_db(vector_db)
            
            st.session_state.vector_db = vector_db
            st.session_state.num_documents = len(documents)
            st.session_state.num_chunks = num_chunks
            st.session_state.domain_metadata = domain_metadata
            st.session_state.needs_rebuild = False
            
            st.success(f"✅ Processed {len(documents)} documents → {num_chunks} chunks")
    
    # Create assistant
    assistant = RAGAssistant(
        st.session_state.vector_db,
        groq_api_key,
        st.session_state.get('domain_metadata', {}),
        st.session_state.get('user_profile')
    )
    
    # Display metrics (removed domain display)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Documents", st.session_state.get('num_documents', 0))
    with col2:
        st.metric("📝 Chunks", st.session_state.get('num_chunks', 0))
    
    st.divider()
    
    # Query interface
    question = st.text_input(
        "💬 Ask a question:",
        placeholder="What would you like to know about these documents?",
        help="Ask anything related to your uploaded documents"
    )
    
    if st.button("🔎 Search", type="primary", use_container_width=True) and question:
        with st.spinner("🔍 Searching documents..."):
            try:
                result = assistant.query(question, k=k_results)
                
                if result['metrics'].get('out_of_domain', False):
                    st.error(result['answer'])
                    if result['warning']:
                        st.warning(result['warning'])
                else:
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    if result['warning']:
                        st.warning(result['warning'])
                    
                    if result['citations']:
                        st.markdown("### 📚 Sources")
                        for i, citation in enumerate(result['citations'], 1):
                            with st.expander(f"Source {i}: {citation['title']}", expanded=False):
                                st.markdown(f"**Page:** {citation['page']} | **Section:** {citation['section']} | **Relevance:** {citation['similarity']:.1%}")
                    
                    risk = result['metrics']['hallucination_risk']
                    if "HIGH" in risk or "MEDIUM" in risk:
                        st.info(f"🔍 Hallucination Risk: {risk}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Smart suggestions
    if st.session_state.get('domain_metadata'):
        st.divider()
        st.markdown("### 💡 Suggested Questions")
        
        suggestions = generate_smart_suggestions(st.session_state.domain_metadata)
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:4]):
            with cols[i % 2]:
                if st.button(f"📌 {suggestion[:50]}...", key=f"sug_{i}", use_container_width=True):
                    st.session_state.suggested_query = suggestion
                    st.rerun()
        
        if st.session_state.get('suggested_query'):
            query = st.session_state.suggested_query
            st.session_state.suggested_query = None
            
            with st.spinner("🔍 Searching..."):
                try:
                    result = assistant.query(query, k=k_results)
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    if result['citations']:
                        st.markdown("### 📚 Sources")
                        for i, citation in enumerate(result['citations'], 1):
                            with st.expander(f"Source {i}: {citation['title']}"):
                                st.markdown(f"**Page:** {citation['page']} | **Relevance:** {citation['similarity']:.1%}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()