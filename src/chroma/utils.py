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

    while start < text_len:
        # Calculate end position
        end = start + chunk_size
        if end > text_len:
            end = text_len
            
        # Find nearest sentence boundary after target end
        # Look for ., !, ? followed by space or newline
        if end < text_len:
            for i in range(end + 50):  # Look ahead up to 50 chars
                if i >= text_len:
                    break
                if text[i] in '.!?' and (i + 1 >= text_len or text[i + 1].isspace()):
                    end = i + 1
                    break
                    
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
            
        # Move start position accounting for overlap
        start = end - overlap
        if start < 0:
            start = 0
            
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
        - is_chunk: true
    """
    # Create copy of original metadata
    metadata = original_metadata.copy() if original_metadata else {}
    
    # Add chunk-specific fields
    metadata.update({
        "parent_doc_id": doc_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "is_chunk": True
    })
    
    return metadata
