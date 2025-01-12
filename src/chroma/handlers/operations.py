"""Core CRUD operations for document handling."""

import logging
from typing import List, Any

import mcp.types as types
from ..types import DocumentOperationError
from ..text_chunking import chunk_text, generate_chunk_metadata

# Set up logging
logger = logging.getLogger(__name__)

async def handle_create_document(collection: Any, arguments: dict) -> List[types.TextContent]:
    """Handle document creation with retry logic and content chunking"""
    doc_id = arguments.get("document_id")
    content = arguments.get("content")
    metadata = arguments.get("metadata")

    if not doc_id or not content:
        raise DocumentOperationError("Missing document_id or content")

    try:
        # Check if document exists using peek()
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

        # Process base metadata - ensure all values are strings
        processed_metadata = {}
        if metadata:
            processed_metadata = {
                k: str(v) for k, v in metadata.items()
            }

        # Split content into chunks
        logger.info(f"Splitting document {doc_id} into chunks...")
        chunks = chunk_text(content)  # Use default chunk_size=1000, overlap=100
        total_chunks = len(chunks)
        logger.info(f"Generated {total_chunks} chunks")

        if total_chunks == 0:
            raise DocumentOperationError("Failed to generate chunks from content")

        # Prepare chunk data
        chunk_ids = []
        chunk_contents = []
        chunk_metadatas = []

        # Process each chunk with retry logic
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = generate_chunk_metadata(
                processed_metadata,
                doc_id,
                i,
                total_chunks
            )
            
            chunk_ids.append(chunk_id)
            chunk_contents.append(chunk)
            chunk_metadatas.append(chunk_metadata)
            
            logger.info(f"Prepared chunk {i+1}/{total_chunks} for document {doc_id}")

        # Add chunks in batches
        batch_size = 10
        for i in range(0, len(chunk_ids), batch_size):
            batch_end = min(i + batch_size, len(chunk_ids))
            logger.info(f"Adding chunk batch {i//batch_size + 1}/{(len(chunk_ids) + batch_size - 1)//batch_size}")
            
            try:
                collection.add(
                    documents=chunk_contents[i:batch_end],
                    ids=chunk_ids[i:batch_end],
                    metadatas=chunk_metadatas[i:batch_end]
                )
            except Exception as e:
                logger.error(f"Failed to add chunk batch: {str(e)}")
                # Clean up any chunks that were added
                for chunk_id in chunk_ids[:i]:
                    try:
                        collection.delete(ids=[chunk_id])
                    except Exception:
                        pass
                raise DocumentOperationError(f"Failed to add chunks: {str(e)}")

        # Add original document with chunk references
        logger.info(f"Adding original document {doc_id} with chunk references")
        original_metadata = processed_metadata.copy()
        original_metadata.update({
            "chunk_ids": chunk_ids,
            "total_chunks": str(total_chunks),
            "chunk_type": "original",
            "doc_id": doc_id
        })

        collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[original_metadata]
        )

        logger.info(f"Successfully created document {doc_id} with {total_chunks} chunks")
        return [
            types.TextContent(
                type="text",
                text=f"Created document '{doc_id}' successfully with {total_chunks} chunks"
            )
        ]
    except DocumentOperationError:
        raise
    except Exception as e:
        logger.error(f"Error creating document: {str(e)}", exc_info=True)
        raise DocumentOperationError(str(e))

async def handle_read_document(collection: Any, arguments: dict) -> List[types.TextContent]:
    """Handle document reading with retry logic"""
    doc_id = arguments.get("document_id")

    if not doc_id:
        raise DocumentOperationError("Missing document_id")

    logger.info(f"Reading document with ID: {doc_id}")

    try:
        result = collection.get(
            ids=[doc_id],
            include=['documents', 'metadatas']
        )
        if not result or not result.get('documents'):
            raise DocumentOperationError(f"Document not found [id={doc_id}]")

        logger.info(f"Successfully retrieved document: {doc_id}")
        
        # Format the response
        doc_content = result['documents'][0]
        doc_metadata = result['metadatas'][0] if result.get('metadatas') else {}
        
        response = [
            f"Document ID: {doc_id}",
            f"Content: {doc_content}",
            f"Metadata: {doc_metadata}"
        ]

        return [
            types.TextContent(
                type="text",
                text="\n".join(response)
            )
        ]

    except Exception as e:
        raise DocumentOperationError(str(e))

async def handle_update_document(collection: Any, arguments: dict) -> List[types.TextContent]:
    """Handle document update with retry logic and chunk management"""
    doc_id = arguments.get("document_id")
    content = arguments.get("content")
    metadata = arguments.get("metadata")

    if not doc_id or not content:
        raise DocumentOperationError("Missing document_id or content")

    logger.info(f"Updating document: {doc_id}")
    
    try:
        # Check if document exists and get its metadata
        existing = collection.get(
            ids=[doc_id],
            include=['documents', 'metadatas']
        )
        if not existing or not existing.get('documents'):
            raise DocumentOperationError(f"Document not found [id={doc_id}]")
            
        existing_metadata = existing['metadatas'][0] if existing.get('metadatas') else {}
        
        # Process metadata
        if metadata:
            processed_metadata = {
                k: v if isinstance(v, (int, float)) else str(v)
                for k, v in metadata.items()
            }
        else:
            processed_metadata = {}

        # Delete existing chunks if this was a chunked document
        if existing_metadata.get('chunk_ids'):
            logger.info(f"Deleting existing chunks for document: {doc_id}")
            collection.delete(ids=existing_metadata['chunk_ids'])

        # Split content into chunks
        chunks = chunk_text(content)
        total_chunks = len(chunks)

        # Prepare chunk data
        chunk_ids = []
        chunk_contents = []
        chunk_metadatas = []

        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = generate_chunk_metadata(
                processed_metadata,
                doc_id,
                i,
                total_chunks
            )
            
            chunk_ids.append(chunk_id)
            chunk_contents.append(chunk)
            chunk_metadatas.append(chunk_metadata)

        # Add new chunks
        collection.add(
            documents=chunk_contents,
            ids=chunk_ids,
            metadatas=chunk_metadatas
        )

        # Update original document with new content and metadata
        original_metadata = processed_metadata.copy()
        original_metadata.update({
            "chunk_ids": chunk_ids,
            "total_chunks": str(total_chunks),
            "chunk_type": "original",
            "doc_id": doc_id
        })

        collection.update(
            ids=[doc_id],
            documents=[content],
            metadatas=[original_metadata]
        )
        
        logger.info(f"Successfully updated document: {doc_id}")
        return [
            types.TextContent(
                type="text",
                text=f"Updated document '{doc_id}' successfully with {total_chunks} chunks"
            )
        ]

    except Exception as e:
        raise DocumentOperationError(str(e))

async def handle_delete_document(collection: Any, arguments: dict) -> List[types.TextContent]:
    """Handle document deletion with retry logic and chunk cleanup"""
    doc_id = arguments.get("document_id")

    if not doc_id:
        raise DocumentOperationError("Missing document_id")

    logger.info(f"Attempting to delete document: {doc_id}")

    try:
        # First verify the document exists and get its metadata
        logger.info(f"Verifying document existence: {doc_id}")
        existing = collection.get(
            ids=[doc_id],
            include=['metadatas', 'documents']
        )
        if not existing or not existing.get('metadatas'):
            raise DocumentOperationError(f"Document not found [id={doc_id}]")
        
        # Get metadata to check for chunks
        metadata = existing['metadatas'][0]
        chunk_ids = metadata.get('chunk_ids', [])
        
        logger.info(f"Document found, proceeding with deletion: {doc_id}")
        
        # Delete chunks first if they exist
        if chunk_ids:
            logger.info(f"Deleting {len(chunk_ids)} chunks for document: {doc_id}")
            collection.delete(ids=chunk_ids)

        # Delete the original document
        collection.delete(ids=[doc_id])
        
        logger.info(f"Successfully deleted document and its chunks: {doc_id}")
        return [
            types.TextContent(
                type="text",
                text=f"Deleted document '{doc_id}' and its chunks successfully"
            )
        ]

    except Exception as e:
        if "not found" in str(e).lower():
            raise DocumentOperationError(f"Document not found [id={doc_id}]")
        raise DocumentOperationError(str(e))

async def handle_list_documents(collection: Any, arguments: dict) -> List[types.TextContent]:
    """Handle document listing with retry logic"""
    limit = arguments.get("limit", 10)
    offset = arguments.get("offset", 0)

    try:
        # Get all document IDs
        all_ids = collection.get()['ids'] if collection.count() > 0 else []
        if not all_ids:
            return [
                types.TextContent(
                    type="text",
                    text="No documents found in collection"
                )
            ]

        # Get documents in batches
        start = offset
        end = min(start + limit, len(all_ids))
        batch_ids = all_ids[start:end]
        
        if not batch_ids:
            return [
                types.TextContent(
                    type="text",
                    text="No documents found in collection"
                )
            ]

        results = collection.get(
            ids=batch_ids,
            include=['documents', 'metadatas']
        )

        # Format results
        response = [f"Documents (showing {len(batch_ids)} results):"]
        for i, doc_id in enumerate(batch_ids):
            content = results['documents'][i]
            metadata = results['metadatas'][i]
            
            response.append(f"\nDocument ID: {doc_id}")
            response.append(f"Content: {content[:100]}...")  # Show first 100 chars
            
            # Filter out internal metadata
            display_metadata = {
                k: v for k, v in metadata.items()
                if k not in ['chunk_ids', 'total_chunks', 'chunk_type', 'doc_id']
            }
            if display_metadata:
                response.append(f"Metadata: {display_metadata}")

        return [
            types.TextContent(
                type="text",
                text="\n".join(response)
            )
        ]
    except Exception as e:
        raise DocumentOperationError(str(e))
