# Contextual Retrieval Module

This module handles:
- Chunking documents
- Generating embeddings using OpenAI
- Storing and querying vectors using Weaviate
- Appending context for Retrieval-Augmented Generation (RAG)

## Files
- `chunker.py`: Text splitting
- `embedder.py`: OpenAI embedding
- `vector_store.py`: Weaviate interface
- `context_appender.py`: Combines context + query
- `pipeline.py`: Orchestrates the whole process

## Setup

> Create a virtual environment:
   ```bash
   python3 -m venv venv

1. Create `.env` file from `.env.example`

2. Install dependencies:
```bash
pip install langchain openai weaviate-client python-dotenv
