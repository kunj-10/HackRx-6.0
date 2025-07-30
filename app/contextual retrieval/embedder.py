# This module generates vector embeddings using OpenAI.

import os
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embedder():
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: List[str]):
    embedder = get_embedder()
    return embedder.embed_documents(texts)
