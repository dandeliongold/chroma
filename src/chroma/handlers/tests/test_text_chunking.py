"""Unit tests for utility functions."""
import pytest
from ...text_chunking import chunk_text, generate_chunk_metadata

def test_chunk_text_empty():
    """Test chunking empty text returns empty list."""
    assert chunk_text("") == []

def test_chunk_text_small():
    """Test text smaller than chunk size stays as single chunk."""
    text = "Short text."  # Simplified test case
    chunks = chunk_text(text, chunk_size=20)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()

def test_chunk_text_sentence_boundaries():
    """Test chunking respects sentence boundaries."""
    text = "First sentence. Second sentence. Third sentence."
    # Set chunk size to split after first sentence
    chunks = chunk_text(text, chunk_size=15, overlap=5)
    
    # Should split into multiple chunks
    assert len(chunks) >= 2
    
    # First chunk should be "First sentence."
    assert chunks[0] == "First sentence."
    
    # Subsequent chunks should also end with sentence boundaries
    for chunk in chunks[1:]:
        assert any(chunk.endswith(end) for end in [".","!","?"])

def test_chunk_text_overlap():
    """Test chunks have proper overlap."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    # Get two consecutive chunks and verify overlap
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        # The end of first chunk should appear at start of second
        overlap_text = chunk1[-10:]
        assert overlap_text in chunk2

def test_chunk_text_no_empty_chunks():
    """Test no empty chunks are returned."""
    text = "Short text. With some. Random. Breaks."
    chunks = chunk_text(text, chunk_size=10, overlap=5)
    assert all(chunk.strip() for chunk in chunks)

def test_generate_chunk_metadata_empty():
    """Test metadata generation with empty original metadata."""
    metadata = generate_chunk_metadata({}, "doc123", 0, 5)
    assert metadata["parent_doc_id"] == "doc123"
    assert metadata["chunk_index"] == "0"
    assert metadata["total_chunks"] == "5"
    assert metadata["chunk_type"] == "chunk"

def test_generate_chunk_metadata_preserve_original():
    """Test original metadata is preserved."""
    original = {"author": "Test", "date": "2024"}
    metadata = generate_chunk_metadata(original, "doc123", 1, 5)
    assert metadata["author"] == "Test"
    assert metadata["date"] == "2024"
    assert metadata["parent_doc_id"] == "doc123"
    assert metadata["chunk_index"] == "1"
    assert metadata["total_chunks"] == "5"

def test_generate_chunk_metadata_none():
    """Test metadata generation with None original metadata."""
    metadata = generate_chunk_metadata(None, "doc123", 0, 1)
    assert metadata["parent_doc_id"] == "doc123"
    assert metadata["chunk_index"] == "0"
    assert metadata["total_chunks"] == "1"
    assert metadata["chunk_type"] == "chunk"

def test_generate_chunk_metadata_no_modification():
    """Test original metadata dict is not modified."""
    original = {"key": "value"}
    original_copy = original.copy()
    generate_chunk_metadata(original, "doc123", 0, 1)
    assert original == original_copy
