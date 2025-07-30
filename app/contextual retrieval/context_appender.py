# context_appender.py
# This module appends relevant context from vector search to the user query.

from typing import List

def append_context_to_query(query: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n".join(retrieved_chunks)
    return f"Context:\n{context}\n\nQuery:\n{query}"
