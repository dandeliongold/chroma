import os
import pytest
import asyncio
from unittest.mock import patch, mock_open

from ..utils import retry_operation, read_file_content, DocumentOperationError

# Test retry_operation decorator
@pytest.mark.asyncio
async def test_retry_operation_success():
    """Test that operation succeeds on first try"""
    
    @retry_operation("test")
    async def mock_operation():
        return "success"
    
    result = await mock_operation()
    assert result == "success"

@pytest.mark.asyncio
async def test_retry_operation_retry_success():
    """Test that operation succeeds after retries"""
    attempts = 0
    
    @retry_operation("test")
    async def mock_operation():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise DocumentOperationError("Temporary error")
        return "success"
    
    result = await mock_operation()
    assert result == "success"
    assert attempts == 2

@pytest.mark.asyncio
async def test_retry_operation_max_retries():
    """Test that operation fails after max retries"""
    
    @retry_operation("test")
    async def mock_operation():
        raise DocumentOperationError("Persistent error")
    
    with pytest.raises(DocumentOperationError, match="Persistent error"):
        await mock_operation()

@pytest.mark.asyncio
async def test_retry_operation_error_mapping():
    """Test error message mapping"""
    
    @retry_operation("test")
    async def mock_operation():
        raise Exception("not found")
    
    with pytest.raises(DocumentOperationError, match="Document not found"):
        await mock_operation()

# Test read_file_content function
def test_read_file_content_relative_path():
    """Test that relative paths are rejected"""
    with pytest.raises(DocumentOperationError, match="File path must be absolute"):
        read_file_content("relative/path.txt")

def test_read_file_content_nonexistent():
    """Test handling of nonexistent files"""
    with pytest.raises(DocumentOperationError, match="File not found"):
        read_file_content("/nonexistent/file.txt")

def test_read_file_content_success():
    """Test successful file reading"""
    test_content = "test content"
    mock_file = mock_open(read_data=test_content)
    
    with patch('builtins.open', mock_file):
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isabs', return_value=True):
                content = read_file_content("/test/file.txt")
                assert content == test_content

def test_read_file_content_error():
    """Test handling of file reading errors"""
    with patch('os.path.exists', return_value=True):
        with patch('os.path.isabs', return_value=True):
            with patch('builtins.open', side_effect=Exception("Read error")):
                with pytest.raises(DocumentOperationError, match="Error reading file: Read error"):
                    read_file_content("/test/file.txt")

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
