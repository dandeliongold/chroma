"""Utility functions for text processing and chunking."""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Source text to chunk
        chunk_size: Target size for each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks with specified overlap
        
    Implementation:
    1. Initialize empty chunks list
    2. Set starting position
    3. While text remains:
        - Calculate end position
        - Find nearest sentence boundary
        - Extract chunk
        - Move position accounting for overlap
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_len = len(text)

    # Special case for small text
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    def find_best_sentence_boundary(target_pos: int) -> int:
        """Find the best sentence boundary near the target position.
        First looks backwards from target, then forwards if no boundary found."""
        
        # First look backwards from target
        for i in range(target_pos, max(start - 1, -1), -1):
            if text[i] in '.!?' and (i + 1 >= text_len or text[i + 1].isspace()):
                return i + 1
                
        # If no boundary found before, look forward
        search_end = min(target_pos + 50, text_len)
        for i in range(target_pos, search_end):
            if text[i] in '.!?' and (i + 1 >= text_len or text[i + 1].isspace()):
                return i + 1
                
        # If no boundary found in either direction, use target
        return target_pos

    while start < text_len:
        # Calculate target end position
        target_end = min(start + chunk_size, text_len)
        
        # Find best sentence boundary
        if target_end >= text_len:
            end = text_len
        else:
            end = find_best_sentence_boundary(target_end)
            
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        if end >= text_len:
            break
            
        # Move to next position with overlap
        start = max(start + 1, end - overlap)
            
    return chunks

def generate_chunk_metadata(
    original_metadata: dict,
    doc_id: str, 
    chunk_index: int,
    total_chunks: int
) -> dict:
    """Create metadata for a chunk that links back to original document.
    
    Args:
        original_metadata: Original document metadata
        doc_id: Original document ID
        chunk_index: Position of this chunk (0-based)
        total_chunks: Total number of chunks
        
    Returns:
        Enhanced metadata dict with chunk information
        
    Implementation:
    1. Copy original metadata
    2. Add chunk-specific fields:
        - parent_doc_id
        - chunk_index 
        - total_chunks
        - chunk_type: chunk
    """
    # Create copy of original metadata
    metadata = original_metadata.copy() if original_metadata else {}
    
    # Add chunk-specific fields
    metadata.update({
        "parent_doc_id": doc_id,
        "chunk_index": str(chunk_index),
        "total_chunks": str(total_chunks),
        "chunk_type": "chunk"  # Use this for filtering instead of boolean
    })
    
    return metadata
