# This module interfaces with Weaviate for storing and querying vector embeddings.

import os
import weaviate
from weaviate.util import get_valid_uuid
from dotenv import load_dotenv
from typing import List

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
)

CLASS_NAME = "DocumentChunk"

def create_schema():
    if client.schema.exists(CLASS_NAME):
        return
    schema = {
        "classes": [{
            "class": CLASS_NAME,
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]},
            ]
        }]
    }
    client.schema.create(schema)

def store_chunks(chunks: List[str], vectors: List[List[float]]):
    for text, vector in zip(chunks, vectors):
        client.data_object.create(
            data_object={"text": text},
            class_name=CLASS_NAME,
            vector=vector
        )

def query_similar_chunks(query_vector: List[float], k: int = 5) -> List[str]:
    result = client.query.get(CLASS_NAME, ["text"]) \
        .with_near_vector({"vector": query_vector}) \
        .with_limit(k).do()
    return [item["text"] for item in result["data"]["Get"][CLASS_NAME]]
