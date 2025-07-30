# This is the main pipeline that ties together chunking, embedding, storing, and retrieval.

from chunker import chunk_text
from embedder import get_embedder
from vector_store import create_schema, store_chunks, query_similar_chunks
from context_appender import append_context_to_query

def process_and_store_document(text: str):
    chunks = chunk_text(text)
    embedder = get_embedder()
    embeddings = embedder.embed_documents(chunks)
    create_schema()
    store_chunks(chunks, embeddings)

def retrieve_context_for_query(query: str, k: int = 5) -> str:
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)
    retrieved = query_similar_chunks(query_vector, k)
    return append_context_to_query(query, retrieved)
