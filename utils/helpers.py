"""
Helper utility functions
"""
from typing import Tuple

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def calculate_similarity(distance: float) -> float:
    """Convert FAISS L2 distance to similarity score"""
    return 1 / (1 + distance)

def validate_query(query: str) -> Tuple[bool, str]:
    """Validate user query"""
    if len(query.strip()) < 3:
        return False, "Query too short (minimum 3 characters)"
    if len(query) > 500:
        return False, "Query too long (maximum 500 characters)"
    return True, ""