"""Document operation handlers package."""

import logging
from typing import List

import mcp.types as types
from .utils import retry_operation
from .operations import (
    handle_create_document,
    handle_read_document,
    handle_update_document,
    handle_delete_document,
    handle_list_documents
)
from .search import handle_search_similar
from .file_ops import handle_create_from_file, handle_update_from_file
from .metadata import handle_get_document_metadata

# Set up logging
logger = logging.getLogger(__name__)

class DocumentHandlers:
    """Handlers for document operations."""
    
    def __init__(self, collection, embedding_function):
        self.collection = collection
        self.embedding_function = embedding_function

    # Basic CRUD operations
    @retry_operation("create_document")
    async def handle_create_document(self, arguments: dict) -> List[types.TextContent]:
        return await handle_create_document(self.collection, arguments)

    @retry_operation("read_document")
    async def handle_read_document(self, arguments: dict) -> List[types.TextContent]:
        return await handle_read_document(self.collection, arguments)

    @retry_operation("update_document")
    async def handle_update_document(self, arguments: dict) -> List[types.TextContent]:
        return await handle_update_document(self.collection, arguments)

    @retry_operation("delete_document")
    async def handle_delete_document(self, arguments: dict) -> List[types.TextContent]:
        return await handle_delete_document(self.collection, arguments)

    @retry_operation("list_documents")
    async def handle_list_documents(self, arguments: dict) -> List[types.TextContent]:
        return await handle_list_documents(self.collection, arguments)

    # Search operations
    @retry_operation("search_similar")
    async def handle_search_similar(self, arguments: dict) -> List[types.TextContent]:
        return await handle_search_similar(self.collection, arguments)

    # File operations
    @retry_operation("create_from_file")
    async def handle_create_from_file(self, arguments: dict) -> List[types.TextContent]:
        return await handle_create_from_file(self.collection, arguments)

    @retry_operation("update_from_file")
    async def handle_update_from_file(self, arguments: dict) -> List[types.TextContent]:
        return await handle_update_from_file(self.collection, arguments)

    # Metadata operations
    @retry_operation("get_document_metadata")
    async def handle_get_document_metadata(self, arguments: dict) -> List[types.TextContent]:
        return await handle_get_document_metadata(self.collection, arguments)
