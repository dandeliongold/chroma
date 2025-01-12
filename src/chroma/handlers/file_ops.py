"""File-based operations for document handling."""

import logging
from typing import List, Any

import mcp.types as types
from ..types import DocumentOperationError
from .utils import read_file_content

# Set up logging
logger = logging.getLogger(__name__)

async def handle_create_from_file(collection: Any, arguments: dict) -> List[types.TextContent]:
    """Handle document creation from file with retry logic"""
    doc_id = arguments.get("document_id")
    file_path = arguments.get("file_path")
    metadata = arguments.get("metadata")

    if not doc_id or not file_path:
        raise DocumentOperationError("Missing document_id or file_path")

    try:
        # Read content from file
        content = read_file_content(file_path)

        # Check if document exists
        try:
            existing = collection.get(
                ids=[doc_id],
                include=['documents']
            )
            if existing and existing.get('documents'):
                raise DocumentOperationError(f"Document already exists [id={doc_id}]")
        except Exception as e:
            if "not found" not in str(e).lower():
                raise

        # Process metadata
        if metadata:
            processed_metadata = {
                k: str(v) if isinstance(v, (int, float)) else v
                for k, v in metadata.items()
            }
            processed_metadata['doc_id'] = doc_id  # Store document ID in metadata
        else:
            processed_metadata = {"doc_id": doc_id}  # Store document ID in metadata

        # Add document
        collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[processed_metadata]
        )

        return [
            types.TextContent(
                type="text",
                text=f"Created document '{doc_id}' from file '{file_path}' successfully"
            )
        ]
    except DocumentOperationError:
        raise
    except Exception as e:
        raise DocumentOperationError(str(e))

async def handle_update_from_file(collection: Any, arguments: dict) -> List[types.TextContent]:
    """Handle document update from file with retry logic"""
    doc_id = arguments.get("document_id")
    file_path = arguments.get("file_path")
    metadata = arguments.get("metadata")

    if not doc_id or not file_path:
        raise DocumentOperationError("Missing document_id or file_path")

    try:
        # Read content from file
        content = read_file_content(file_path)

        # Check if document exists
        try:
            existing = collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            if not existing or not existing.get('documents'):
                raise DocumentOperationError(f"Document not found [id={doc_id}]")
        except Exception as e:
            if "not found" in str(e).lower():
                raise DocumentOperationError(f"Document not found [id={doc_id}]")
            raise

        # Process metadata
        if metadata:
            processed_metadata = {
                k: str(v) if isinstance(v, (int, float)) else v
                for k, v in metadata.items()
            }
            processed_metadata['doc_id'] = doc_id  # Store document ID in metadata
            collection.update(
                ids=[doc_id],
                documents=[content],
                metadatas=[processed_metadata]
            )
        else:
            collection.update(
                ids=[doc_id],
                documents=[content],
                metadatas=[{"doc_id": doc_id}]  # Store document ID in metadata
            )

        return [
            types.TextContent(
                type="text",
                text=f"Updated document '{doc_id}' from file '{file_path}' successfully"
            )
        ]
    except DocumentOperationError:
        raise
    except Exception as e:
        raise DocumentOperationError(str(e))
