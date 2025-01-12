"""Handlers for document operations."""

import asyncio
import os
import logging
from functools import wraps
from typing import List

import mcp.types as types
from .types import DocumentOperationError
from .utils import chunk_text, generate_chunk_metadata
from .config import MAX_RETRIES, RETRY_DELAY, BACKOFF_FACTOR

# Set up logging
logger = logging.getLogger(__name__)

def retry_operation(operation_name: str):
    """Decorator to retry document operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
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
            return None
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

class DocumentHandlers:
    """Handlers for document operations."""
    
    def __init__(self, collection, embedding_function):
        self.collection = collection
        self.embedding_function = embedding_function

    @retry_operation("create_document")
    async def handle_create_document(self, arguments: dict) -> List[types.TextContent]:
        """Handle document creation with retry logic and content chunking"""
        doc_id = arguments.get("document_id")
        content = arguments.get("content")
        metadata = arguments.get("metadata")

        if not doc_id or not content:
            raise DocumentOperationError("Missing document_id or content")

        try:
            # Check if document exists using peek()
            try:
                existing = self.collection.get(
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
                    self.collection.add(
                        documents=chunk_contents[i:batch_end],
                        ids=chunk_ids[i:batch_end],
                        metadatas=chunk_metadatas[i:batch_end]
                    )
                except Exception as e:
                    logger.error(f"Failed to add chunk batch: {str(e)}")
                    # Clean up any chunks that were added
                    for chunk_id in chunk_ids[:i]:
                        try:
                            self.collection.delete(ids=[chunk_id])
                        except Exception:
                            pass
                    raise DocumentOperationError(f"Failed to add chunks: {str(e)}")

            # Add original document with chunk references
            logger.info(f"Adding original document {doc_id} with chunk references")
            original_metadata = processed_metadata.copy()
            original_metadata.update({
                "chunk_ids": chunk_ids,
                "total_chunks": str(total_chunks),
                "chunk_type": "original",  # Use consistent field name
                "doc_id": doc_id  # Store document ID in metadata
            })

            self.collection.add(
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

    @retry_operation("read_document")
    async def handle_read_document(self, arguments: dict) -> List[types.TextContent]:
        """Handle document reading with retry logic"""
        doc_id = arguments.get("document_id")

        if not doc_id:
            raise DocumentOperationError("Missing document_id")

        logger.info(f"Reading document with ID: {doc_id}")

        try:
            result = self.collection.get(
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

    @retry_operation("update_document")
    async def handle_update_document(self, arguments: dict) -> List[types.TextContent]:
        """Handle document update with retry logic and chunk management"""
        doc_id = arguments.get("document_id")
        content = arguments.get("content")
        metadata = arguments.get("metadata")

        if not doc_id or not content:
            raise DocumentOperationError("Missing document_id or content")

        logger.info(f"Updating document: {doc_id}")
        
        try:
            # Check if document exists and get its metadata
            existing = self.collection.get(
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
                self.collection.delete(ids=existing_metadata['chunk_ids'])

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
            self.collection.add(
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
                "doc_id": doc_id  # Store document ID in metadata
            })

            self.collection.update(
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

    @retry_operation("delete_document")
    async def handle_delete_document(self, arguments: dict) -> List[types.TextContent]:
        """Handle document deletion with retry logic and chunk cleanup"""
        doc_id = arguments.get("document_id")

        if not doc_id:
            raise DocumentOperationError("Missing document_id")

        logger.info(f"Attempting to delete document: {doc_id}")

        # First verify the document exists and get its metadata
        try:
            logger.info(f"Verifying document existence: {doc_id}")
            existing = self.collection.get(
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
                self.collection.delete(ids=chunk_ids)
                
        except Exception as e:
            if "not found" in str(e).lower():
                raise DocumentOperationError(f"Document not found [id={doc_id}]")
            raise DocumentOperationError(str(e))

        # Attempt deletion of original document with exponential backoff
        max_attempts = MAX_RETRIES
        current_attempt = 0
        delay = RETRY_DELAY

        while current_attempt < max_attempts:
            try:
                logger.info(f"Delete attempt {current_attempt + 1}/{max_attempts} for document: {doc_id}")
                self.collection.delete(ids=[doc_id])
                
                # Document deletion was successful
                logger.info(f"Successfully deleted document and its chunks: {doc_id}")
                return [
                    types.TextContent(
                        type="text",
                        text=f"Deleted document '{doc_id}' and its chunks successfully"
                    )
                ]

            except Exception as e:
                current_attempt += 1
                if current_attempt < max_attempts:
                    logger.warning(
                        f"Delete attempt {current_attempt} failed for document {doc_id}. "
                        f"Retrying in {delay} seconds. Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                    delay *= BACKOFF_FACTOR
                else:
                    logger.error(
                        f"All delete attempts failed for document {doc_id}. "
                        f"Final error: {str(e)}", 
                        exc_info=True
                    )
                    raise DocumentOperationError(str(e))

        # This shouldn't be reached, but just in case
        raise DocumentOperationError("Operation failed")

    @retry_operation("list_documents")
    async def handle_list_documents(self, arguments: dict) -> List[types.TextContent]:
        """Handle document listing with retry logic"""
        limit = arguments.get("limit", 10)
        offset = arguments.get("offset", 0)

        try:
            # Get all document IDs
            all_ids = self.collection.get()['ids'] if self.collection.count() > 0 else []
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

            results = self.collection.get(
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

    @retry_operation("search_similar")
    async def handle_search_similar(self, arguments: dict) -> List[types.TextContent]:
        """Handle similarity search with chunk-aware results aggregation"""
        query = arguments.get("query")
        num_results = arguments.get("num_results", 5)
        metadata_filter = arguments.get("metadata_filter", {})
        content_filter = arguments.get("content_filter")

        if not query:
            raise DocumentOperationError("Missing query")

        try:
            # Build base query parameters
            # Increase n_results to account for multiple chunks per document
            query_params = {
                "query_texts": [query],
                "n_results": num_results * 3,  # Get more results to account for chunks
                "include": ['documents', 'metadatas', 'distances']
            }

            # Build where clause
            where_clause = {"chunk_type": "chunk"}  # Simple equality check
            
            # Add metadata filters if present
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if isinstance(value, (int, float)):
                        where_clause[key] = str(value)
                    elif isinstance(value, dict):
                        # Handle operator conditions
                        processed_value = {}
                        for op, val in value.items():
                            if isinstance(val, (list, tuple)):
                                processed_value[op] = [str(v) if isinstance(v, (int, float)) else v for v in val]
                            else:
                                processed_value[op] = str(val) if isinstance(val, (int, float)) else val
                        where_clause[key] = processed_value
                    else:
                        where_clause[key] = str(value)
            
            query_params["where"] = where_clause

            # Add content filter if specified
            if content_filter:
                query_params["where_document"] = {"$contains": content_filter}

            # Execute search
            logger.info(f"Executing search with params: {query_params}")
            results = self.collection.query(**query_params)

            if not results or not results.get('documents') or len(results['documents'][0]) == 0:
                msg = ["No documents found matching query: " + query]
                if metadata_filter:
                    msg.append(f"Metadata filter: {metadata_filter}")
                if content_filter:
                    msg.append(f"Content filter: {content_filter}")
                return [types.TextContent(type="text", text="\n".join(msg))]

            # Group results by parent document
            grouped_results = {}
            for i, (doc_id, content, metadata, distance) in enumerate(
                zip(results['documents'][0], results['documents'][0], 
                    results['metadatas'][0], results['distances'][0])
            ):
                parent_id = metadata.get('parent_doc_id')
                if not parent_id:
                    continue
                    
                if parent_id not in grouped_results:
                    grouped_results[parent_id] = {
                        'chunks': [],
                        'best_distance': float('inf')
                    }
                    
                chunk_data = {
                    'content': content,
                    'distance': distance,
                    'chunk_index': metadata.get('chunk_index', 0),
                    'total_chunks': metadata.get('total_chunks', 1)
                }
                
                grouped_results[parent_id]['chunks'].append(chunk_data)
                grouped_results[parent_id]['best_distance'] = min(
                    grouped_results[parent_id]['best_distance'],
                    distance
                )

            # Sort documents by best matching chunk
            sorted_results = sorted(
                grouped_results.items(),
                key=lambda x: x[1]['best_distance']
            )[:num_results]

            # Format results with context
            response = ["Similar documents:"]
            for i, (doc_id, result_data) in enumerate(sorted_results):
                # Get original document to show metadata
                try:
                    original_doc = self.collection.get(
                        ids=[doc_id],
                        include=['metadatas']
                    )
                    original_metadata = original_doc['metadatas'][0] if original_doc.get('metadatas') else {}
                except Exception:
                    original_metadata = {}

                # Remove internal chunk tracking from displayed metadata
                display_metadata = {
                    k: v for k, v in original_metadata.items()
                    if k not in ['chunk_ids', 'total_chunks', 'is_original', 'doc_id']
                }

                response.append(f"\n{i+1}. Document '{doc_id}' (best match distance: {result_data['best_distance']:.4f})")
                if display_metadata:
                    response.append(f"   Metadata: {display_metadata}")

                # Sort chunks by index and show relevant excerpts
                sorted_chunks = sorted(
                    result_data['chunks'],
                    key=lambda x: x['chunk_index']
                )
                for chunk in sorted_chunks[:2]:  # Show up to 2 best matching chunks
                    response.append(f"   Matching excerpt (chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}, distance: {chunk['distance']:.4f}):")
                    response.append(f"   {chunk['content']}")

            return [types.TextContent(type="text", text="\n".join(response))]

        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            raise DocumentOperationError(str(e))

    @retry_operation("create_from_file")
    @retry_operation("update_from_file")
    async def handle_update_from_file(self, arguments: dict) -> List[types.TextContent]:
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
                existing = self.collection.get(
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
                self.collection.update(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[processed_metadata]
                )
            else:
                self.collection.update(
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

    async def handle_create_from_file(self, arguments: dict) -> List[types.TextContent]:
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
                existing = self.collection.get(
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
            self.collection.add(
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
