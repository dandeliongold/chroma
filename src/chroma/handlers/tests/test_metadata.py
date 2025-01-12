import pytest
from unittest.mock import Mock

from ..metadata import handle_get_document_metadata, DocumentOperationError

@pytest.mark.asyncio
async def test_get_metadata_missing_doc_id():
    """Test that missing document_id raises error"""
    collection = Mock()
    
    with pytest.raises(DocumentOperationError, match="Missing document_id"):
        await handle_get_document_metadata(collection, {})

@pytest.mark.asyncio
async def test_get_metadata_document_not_found():
    """Test handling of nonexistent document"""
    collection = Mock()
    collection.get.return_value = {"metadatas": []}
    
    with pytest.raises(DocumentOperationError, match="Document not found"):
        await handle_get_document_metadata(collection, {"document_id": "nonexistent"})

@pytest.mark.asyncio
async def test_get_metadata_success():
    """Test successful metadata retrieval"""
    collection = Mock()
    collection.get.return_value = {
        "metadatas": [{
            "total_chunks": 3,
            "chunk_ids": ["1", "2", "3"],
            "title": "Test Document",
            "author": "Test Author"
        }]
    }
    
    result = await handle_get_document_metadata(collection, {"document_id": "test123"})
    
    assert len(result) == 1
    assert result[0].type == "text"
    content = result[0].text
    
    # Verify all expected information is present
    assert "Metadata for document 'test123'" in content
    assert "Number of chunks: 3" in content
    assert "Chunk IDs: ['1', '2', '3']" in content
    assert "'title': 'Test Document'" in content
    assert "'author': 'Test Author'" in content

@pytest.mark.asyncio
async def test_get_metadata_no_chunks():
    """Test metadata retrieval without chunk information"""
    collection = Mock()
    collection.get.return_value = {
        "metadatas": [{
            "title": "Test Document",
            "author": "Test Author"
        }]
    }
    
    result = await handle_get_document_metadata(collection, {"document_id": "test123"})
    
    assert len(result) == 1
    content = result[0].text
    
    # Verify only custom metadata is present
    assert "Number of chunks" not in content
    assert "Chunk IDs" not in content
    assert "'title': 'Test Document'" in content
    assert "'author': 'Test Author'" in content

@pytest.mark.asyncio
async def test_get_metadata_collection_error():
    """Test handling of collection errors"""
    collection = Mock()
    collection.get.side_effect = Exception("Database error")
    
    with pytest.raises(DocumentOperationError, match="Database error"):
        await handle_get_document_metadata(collection, {"document_id": "test123"})

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
