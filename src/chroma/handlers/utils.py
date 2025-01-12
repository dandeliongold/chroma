"""Utility functions for document operations."""

import asyncio
import os
import logging
from functools import wraps
from typing import Callable, TypeVar, Any

from ..types import DocumentOperationError
from ..config import MAX_RETRIES, RETRY_DELAY, BACKOFF_FACTOR

# Set up logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

def retry_operation(operation_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry document operations with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except DocumentOperationError as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Clean up error message
                        msg = str(e)
                        if msg.lower().startswith(operation_name.lower()):
                            msg = msg[len(operation_name):].lstrip(': ')
                        if msg.lower().startswith('failed'):
                            msg = msg[7:].lstrip(': ')
                        if msg.lower().startswith('search failed'):
                            msg = msg[13:].lstrip(': ')
                        
                        # Map error patterns to friendly messages
                        error_msg = msg.lower()
                        doc_id = kwargs.get('arguments', {}).get('document_id')
                        
                        if "not found" in error_msg:
                            error = f"Document not found{f' [id={doc_id}]' if doc_id else ''}"
                        elif "already exists" in error_msg:
                            error = f"Document already exists{f' [id={doc_id}]' if doc_id else ''}"
                        elif "invalid" in error_msg:
                            error = "Invalid input"
                        elif "filter" in error_msg:
                            error = "Invalid filter"
                        else:
                            error = "Operation failed"
                            
                        raise DocumentOperationError(error)
                    await asyncio.sleep(2 ** attempt)
            return None  # type: ignore
        return wrapper
    return decorator

def read_file_content(file_path: str) -> str:
    """Read content from a file safely."""
    if not os.path.isabs(file_path):
        raise DocumentOperationError("File path must be absolute")
    
    if not os.path.exists(file_path):
        raise DocumentOperationError(f"File not found: {file_path}")
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise DocumentOperationError(f"Error reading file: {str(e)}")
