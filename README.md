# Chroma MCP Server

A Model Context Protocol (MCP) server implementation that provides vector database capabilities through Chroma. This server enables semantic document search, metadata filtering, and document management with persistent storage.

## Features

- **Semantic Search**: Find documents based on meaning using Chroma's embeddings
- **Metadata Filtering**: Filter search results by metadata fields
- **Content Filtering**: Additional filtering based on document content
- **File-Based Operations**: Direct creation and updates from file content
- **Persistent Storage**: Data persists between server restarts
- **Error Handling**: Comprehensive error handling with clear messages

## Tools

### Document Management

#### File Operations

- `create_from_file`: Create a new document using content from a local file
  ```json
  {
    "document_id": "string",
    "file_path": "string (absolute path to the file)",
    "metadata": {
      "additional": "properties"
    }
  }
  ```

- `update_from_file`: Update an existing document using content from a local file
  ```json
  {
    "document_id": "string",
    "file_path": "string (absolute path to the file)",
    "metadata": {
      "additional": "properties"
    }
  }
  ```

#### Direct Operations

- `create_document`: Create a new document with provided content
  ```json
  {
    "document_id": "string",
    "content": "string",
    "metadata": {
      "additional": "properties"
    }
  }
  ```

- `read_document`: Retrieve a document by ID
  ```json
  {
    "document_id": "string"
  }
  ```

- `update_document`: Update an existing document
  ```json
  {
    "document_id": "string",
    "content": "string",
    "metadata": {
      "additional": "properties"
    }
  }
  ```

- `delete_document`: Remove a document
  ```json
  {
    "document_id": "string"
  }
  ```

- `list_documents`: List all documents with pagination
  ```json
  {
    "limit": "integer (minimum: 1, default: 10)",
    "offset": "integer (minimum: 0, default: 0)"
  }
  ```

- `get_document_metadata`: Get metadata and chunk information without content
  ```json
  {
    "document_id": "string"
  }
  ```

### Search Operations

- `search_similar`: Find semantically similar documents
  ```json
  {
    "query": "string",
    "num_results": "integer (minimum: 1, default: 5)",
    "metadata_filter": {
      "field": "value"
    },
    "content_filter": "string"
  }
  ```

## Usage Examples

### Creating Documents

```python
# Create from file
create_from_file({
    "document_id": "research_paper",
    "file_path": "/path/to/paper.txt",
    "metadata": {
        "author": "John Smith",
        "year": 2024,
        "category": "ML"
    }
})

# Create directly
create_document({
    "document_id": "ml_paper1",
    "content": "Convolutional neural networks improve image recognition accuracy.",
    "metadata": {
        "year": 2024,
        "field": "computer vision",
        "complexity": "advanced"
    }
})
```

### Searching Documents

```python
# Basic search
search_similar({
    "query": "machine learning applications",
    "num_results": 3
})

# Search with filters
search_similar({
    "query": "neural networks",
    "num_results": 5,
    "metadata_filter": {
        "year": 2024,
        "field": "computer vision"
    },
    "content_filter": "accuracy"
})
```

### Document Management

```python
# List documents with pagination
list_documents({
    "limit": 10,
    "offset": 0
})

# Get document metadata
get_document_metadata({
    "document_id": "ml_paper1"
})

# Update document
update_document({
    "document_id": "ml_paper1",
    "content": "Updated research findings on CNN architectures.",
    "metadata": {
        "year": 2024,
        "status": "revised"
    }
})
```

## Error Handling

The server provides clear error messages for common scenarios:
- Document already exists
- Document not found
- Invalid input parameters
- Invalid filter syntax
- File access errors
- Operation failures

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
