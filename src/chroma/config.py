"""Chroma client and collection configuration."""

import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
BACKOFF_FACTOR = 2

def initialize_chroma_client():
    """Initialize Chroma client with optimized settings."""
    logger.info("Initializing Chroma client with optimized settings...")
    client = chromadb.PersistentClient(
        path="./chroma_db",  # Persist data to disk
    )
    logger.info("Chroma client initialized")
    return client

def initialize_embedding_function():
    """Configure sentence transformer embedding function."""
    logger.info("Configuring embedding function...")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",  # Efficient, general-purpose model
        device="cpu"  # Use CPU for reliability
    )

def initialize_collection(client, embedding_function):
    """Create collection with optimized settings."""
    logger.info("Creating collection...")
    collection = client.create_collection(
        name="documents",
        get_or_create=True,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}  # Optimize for cosine similarity
    )
    logger.info("Collection ready")
    return collection

def add_sample_document(collection):
    """Add a sample document if collection is empty."""
    try:
        if collection.count() == 0:
            logger.info("Adding sample document to empty collection")
            collection.add(
                documents=[
                    "Vector databases are specialized databases designed to store and retrieve high-dimensional vectors efficiently. "
                    "In machine learning, they are crucial for similarity search, recommendation systems, and semantic search applications. "
                    "They use techniques like LSH or HNSW for fast approximate nearest neighbor search."
                ],
                ids=["sample_doc"],
                metadatas=[{
                    "topic": "vector databases",
                    "type": "sample",
                    "date": "2024-12-31"
                }]
            )
            logger.info("Sample document added successfully")
    except Exception as e:
        logger.error(f"Error adding sample document: {e}")

def initialize():
    """Initialize all Chroma components."""
    client = initialize_chroma_client()
    embedding_function = initialize_embedding_function()
    collection = initialize_collection(client, embedding_function)
    add_sample_document(collection)
    return collection

# Helper functions
def sanitize_metadata(metadata: dict) -> dict:
    """Convert metadata values to strings for Chroma compatibility"""
    if not metadata:
        return {}
    return {k: str(v) for k, v in metadata.items()}

def build_where_clause(metadata: dict) -> dict:
    """Build a valid Chroma where clause for multiple metadata conditions"""
    if not metadata:
        return {}
    
    def process_value(value):
        """Process value based on type"""
        if isinstance(value, (int, float)):
            # Keep numeric values as strings for Chroma
            return str(value)
        return str(value)
    
    conditions = []
    for key, value in metadata.items():
        if value is None:
            continue
            
        if isinstance(value, dict) and any(k.startswith('$') for k in value.keys()):
            # Handle operator conditions
            processed_value = {}
            for op, val in value.items():
                if isinstance(val, (list, tuple)):
                    # Handle array operators like $in
                    processed_value[op] = [process_value(v) for v in val]
                else:
                    # Handle single value operators
                    processed_value[op] = process_value(val)
            conditions.append({key: processed_value})
        else:
            # Simple equality condition
            conditions.append({key: {"$eq": process_value(value)}})
    
    if not conditions:
        return {}
        
    if len(conditions) == 1:
        return conditions[0]
    
    return {"$and": conditions}
