"""Metadata operations for document handling."""

import logging
from typing import List, Any

import mcp.types as types
from ..types import DocumentOperationError

# Set up logging
logger = logging.getLogger(__name__)

async def handle_get_document_metadata(collection: Any, arguments: dict) -> List[types.TextContent]:
    """Get metadata and chunk information for a document without retrieving its content"""
    doc_id = arguments.get("document_id")

    if not doc_id:
        raise DocumentOperationError("Missing document_id")

    logger.info(f"Retrieving metadata for document: {doc_id}")

    try:
        # Only fetch metadata
        result = collection.get(
            ids=[doc_id],
            include=['metadatas']
        )
        
        if not result or not result.get('metadatas'):
            raise DocumentOperationError(f"Document not found [id={doc_id}]")
        
        metadata = result['metadatas'][0]
        
        # Format response
        response = [f"Metadata for document '{doc_id}':"]
        
        # Show chunk information
        total_chunks = metadata.get('total_chunks')
        chunk_ids = metadata.get('chunk_ids', [])
        if total_chunks:
            response.append(f"Number of chunks: {total_chunks}")
            if chunk_ids:
                response.append(f"Chunk IDs: {chunk_ids}")
        
        # Show other metadata (excluding internal fields)
        user_metadata = {
            k: v for k, v in metadata.items()
            if k not in ['chunk_ids', 'total_chunks', 'chunk_type', 'doc_id']
        }
        if user_metadata:
            response.append(f"Custom metadata: {user_metadata}")

        return [
            types.TextContent(
                type="text",
                text="\n".join(response)
            )
        ]

    except Exception as e:
        raise DocumentOperationError(str(e))
