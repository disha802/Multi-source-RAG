"""
Vector database management using FAISS
"""
import os
from typing import List, Optional
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:
    """Manages FAISS vector database"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = None
        self.vector_db = None
        self.is_loaded = False
    
    def initialize(self):
        """Initialize embedding model and load/create vector store"""
        print("🔄 Initializing vector store...")
        
        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Try to load existing DB
        if os.path.exists(self.config.VECTOR_DB_PATH):
            try:
                self.vector_db = FAISS.load_local(
                    str(self.config.VECTOR_DB_PATH),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                self.is_loaded = True
                print(f"✅ Loaded existing vector DB ({self.vector_db.index.ntotal} vectors)")
                return True
            except Exception as e:
                print(f"⚠️  Could not load existing DB: {e}")
        
        print("ℹ️  No vector database found. Upload documents to create one.")
        return False
    
    def build(self, documents: List[Document]) -> bool:
        """Build vector database from documents"""
        if not documents:
            return False
        
        print(f"🔨 Building vector DB from {len(documents)} documents...")
        
        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} chunks")
        
        # Create vector DB
        self.vector_db = FAISS.from_documents(chunks, self.embedding_model)
        
        # Save
        self.vector_db.save_local(str(self.config.VECTOR_DB_PATH))
        self.is_loaded = True
        
        print(f"✅ Vector DB built and saved ({self.vector_db.index.ntotal} vectors)")
        return True
    
    def rebuild(self, documents: List[Document]) -> bool:
        """Rebuild entire vector database"""
        return self.build(documents)
    
    def search(self, query: str, k: int = None):
        """Search for relevant documents"""
        if not self.is_loaded or self.vector_db is None:
            return []
        
        k = k or self.config.TOP_K_RESULTS
        return self.vector_db.similarity_search_with_score(query, k=k)