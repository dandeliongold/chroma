"""Main MCP server implementation for Chroma document operations."""

import asyncio
import os
import logging
import sys
from typing import Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

from .handlers import DocumentHandlers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Chroma client with persistence
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

client = chromadb.Client(Settings(
    persist_directory=data_dir,
    is_persistent=True
))

# Use sentence transformers for better embeddings
model_name = "all-MiniLM-L6-v2"
logger.info(f"Initializing embedding function with model: {model_name}")

try:
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    logger.info("Embedding function initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding function: {str(e)}")
    raise

# Initialize collection with embedding function
try:
    collection = client.get_collection(
        name="documents",
        embedding_function=embedding_function
    )
    logger.info("Retrieved existing collection 'documents'")
except Exception:
    collection = client.create_collection(
        name="documents",
        embedding_function=embedding_function
    )
    logger.info("Created new collection 'documents'")

# Ensure embedding function is set
collection._embedding_function = embedding_function

# Initialize handlers with collection and embedding function
handlers = DocumentHandlers(collection, embedding_function)

# Create server instance
server = Server("chroma")

# Server command options
server.command_options = {
    "update_from_file": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "file_path": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": True}
        },
        "required": ["document_id", "file_path"]
    },
    "create_from_file": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "file_path": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": True}
        },
        "required": ["document_id", "file_path"]
    },
    "create_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "content": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": True}
        },
        "required": ["document_id", "content"]
    },
    "read_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"}
        },
        "required": ["document_id"]
    },
    "update_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "content": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": True}
        },
        "required": ["document_id", "content"]
    },
    "delete_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"}
        },
        "required": ["document_id"]
    },
    "list_documents": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 10},
            "offset": {"type": "integer", "minimum": 0, "default": 0}
        }
    },
    "search_similar": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "minimum": 1, "default": 5},
            "metadata_filter": {"type": "object", "additionalProperties": True},
            "content_filter": {"type": "string"}
        },
        "required": ["query"]
    },
    "get_document_metadata": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"}
        },
        "required": ["document_id"]
    }
}

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for document operations."""
    return [
        types.Tool(
            name="update_from_file",
            description="Update an existing document in Chroma using content from a local file",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["document_id", "file_path"]
            }
        ),
        types.Tool(
            name="create_from_file",
            description="Create a new document in Chroma using content from a local file",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["document_id", "file_path"]
            }
        ),
        types.Tool(
            name="create_document",
            description="Create a new document in the Chroma vector database",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["document_id", "content"]
            }
        ),
        types.Tool(
            name="read_document",
            description="Retrieve a document from the Chroma vector database by its ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"}
                },
                "required": ["document_id"]
            }
        ),
        types.Tool(
            name="update_document",
            description="Update an existing document in the Chroma vector database",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["document_id", "content"]
            }
        ),
        types.Tool(
            name="delete_document",
            description="Delete a document from the Chroma vector database by its ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"}
                },
                "required": ["document_id"]
            }
        ),
        types.Tool(
            name="list_documents",
            description="List all documents stored in the Chroma vector database with pagination",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "default": 10},
                    "offset": {"type": "integer", "minimum": 0, "default": 0}
                }
            }
        ),
        types.Tool(
            name="search_similar",
            description="Search for semantically similar documents in the Chroma vector database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "minimum": 1, "default": 5},
                    "metadata_filter": {"type": "object", "additionalProperties": True},
                    "content_filter": {"type": "string"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_document_metadata",
            description="Get metadata and chunk information for a document without retrieving its content",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"}
                },
                "required": ["document_id"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """Handle document operations."""
    if not arguments:
        arguments = {}

    try:
        if name == "update_from_file":
            return await handlers.handle_update_from_file(arguments)
        elif name == "create_from_file":
            return await handlers.handle_create_from_file(arguments)
        elif name == "create_document":
            return await handlers.handle_create_document(arguments)
        elif name == "read_document":
            return await handlers.handle_read_document(arguments)
        elif name == "update_document":
            return await handlers.handle_update_document(arguments)
        elif name == "delete_document":
            return await handlers.handle_delete_document(arguments)
        elif name == "list_documents":
            return await handlers.handle_list_documents(arguments)
        elif name == "search_similar":
            return await handlers.handle_search_similar(arguments)
        elif name == "get_document_metadata":
            return await handlers.handle_get_document_metadata(arguments)
        
        raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )
        ]

async def main():
    """Run the MCP server."""
    # Configure UTF-8 for stdin/stdout on Windows
    if sys.platform == 'win32':
        import msvcrt
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf-8'])
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    # Use standard stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        try:
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="chroma",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
            raise
