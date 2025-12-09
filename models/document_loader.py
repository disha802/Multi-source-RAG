# ============================================================================
# models/document_loader.py
# ============================================================================
"""
Document loading and processing
"""
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader,
    CSVLoader, TextLoader
)

class DocumentLoader:
    """Handles loading documents of various formats"""
    
    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        """Load a single file and return documents with metadata"""
        ext = Path(file_path).suffix.lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.csv': CSVLoader,
            '.txt': TextLoader
        }
        
        if ext not in loaders:
            return []
        
        try:
            loader = loaders[ext](file_path)
            docs = loader.load()
            
            # Enrich metadata
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    'source': Path(file_path).name,
                    'type': ext[1:].upper(),
                    'page': i + 1
                })
            
            return docs
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    @staticmethod
    def load_all_documents(folder: str, allowed_extensions: set) -> List[Document]:
        """Load all documents from a folder"""
        all_docs = []
        
        for ext in allowed_extensions:
            for file_path in Path(folder).glob(f"*.{ext}"):
                docs = DocumentLoader.load_file(str(file_path))
                all_docs.extend(docs)
        
        return all_docs
