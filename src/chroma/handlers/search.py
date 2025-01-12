"""Search operations for document handling."""

import logging
from typing import List, Any

import mcp.types as types
from ..types import DocumentOperationError

# Set up logging
logger = logging.getLogger(__name__)

async def handle_search_similar(collection: Any, arguments: dict) -> List[types.TextContent]:
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
        results = collection.query(**query_params)

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
                original_doc = collection.get(
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
