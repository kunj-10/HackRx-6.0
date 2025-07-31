"""
Graph utilities for Neo4j/Graphiti integration using Google Gemini for both LLM and embeddings.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig as GeminiLLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class GraphitiClient:
    """Manages Graphiti knowledge graph operations using Gemini LLM and embeddings."""
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        # Neo4j configuration (fallback to environment variables)
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")

        # Gemini LLM configuration
        self.llm_choice = os.getenv("LLM_CHOICE", "gemini-1.5-flash")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_base_url = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta"
        )
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Embedding configuration using Gemini
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-gecko-001")
        self.embedding_dimensions = int(os.getenv("VECTOR_DIMENSION", "1536"))

        self.graphiti: Optional[Graphiti] = None
        self._initialized = False

    async def initialize(self):
        """Initialize Graphiti with Gemini LLM client and embedder."""
        if self._initialized:
            return

        # Configure Gemini LLM client
        llm_config = GeminiLLMConfig(
            api_key=self.gemini_api_key,
            model=self.llm_choice,
            base_url=self.gemini_base_url
        )
        llm_client = GeminiClient(config=llm_config)

        # Configure Gemini embedder
        embedder = GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=self.gemini_api_key,
                embedding_model=self.embedding_model,
                embedding_dim=self.embedding_dimensions,
                base_url=self.gemini_base_url
            )
        )

        # Use OpenAI reranker (compatible with any LLM client)
        cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)

        # Instantiate Graphiti with Gemini clients
        self.graphiti = Graphiti(
            self.neo4j_uri,
            self.neo4j_user,
            self.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder
        )

        # Create constraints and indexes
        await self.graphiti.build_indices_and_constraints()
        self._initialized = True
        logger.info(f"Graphiti initialized with Gemini model {self.llm_choice}")

    async def close(self):
        """Close the Graphiti connection."""
        if self.graphiti:
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False
            logger.info("Graphiti connection closed")

    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add an episode to the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        from graphiti_core.nodes import EpisodeType
        episode_timestamp = timestamp or datetime.now(timezone.utc)
        await self.graphiti.add_episode(
            name=episode_id,
            episode_body=content,
            source=EpisodeType.text,
            source_description=source,
            reference_time=episode_timestamp
        )
        logger.info(f"Added episode {episode_id}")

    async def search(
        self,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """Search the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        results = await self.graphiti.search(query)
        return [
            {
                "fact": r.fact,
                "uuid": str(r.uuid),
                "valid_at": getattr(r, 'valid_at', None),
                "invalid_at": getattr(r, 'invalid_at', None),
                "source_node_uuid": getattr(r, 'source_node_uuid', None)
            }
            for r in results
        ]

    async def get_related_entities(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1
    ) -> Dict[str, Any]:
        """Get related entities via semantic search."""
        if not self._initialized:
            await self.initialize()
        results = await self.graphiti.search(f"relationships involving {entity_name}")
        facts = [
            {
                "fact": r.fact,
                "uuid": str(r.uuid),
                "valid_at": getattr(r, 'valid_at', None)
            }
            for r in results
        ]
        return {"central_entity": entity_name, "related_facts": facts}

    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get timeline facts for an entity."""
        if not self._initialized:
            await self.initialize()
        results = await self.graphiti.search(f"timeline history of {entity_name}")
        timeline = [
            {
                "fact": r.fact,
                "uuid": str(r.uuid),
                "valid_at": getattr(r, 'valid_at', None),
                "invalid_at": getattr(r, 'invalid_at', None)
            }
            for r in results
        ]
        timeline.sort(key=lambda x: x.get('valid_at') or '', reverse=True)
        return timeline

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Return basic Graphiti stats."""
        if not self._initialized:
            await self.initialize()
        try:
            sample = await self.graphiti.search("test")
            return {"graph_initialized": True, "sample_count": len(sample)}
        except Exception as e:
            return {"graph_initialized": False, "error": str(e)}

    async def clear_graph(self):
        """Clear all data from the graph."""
        if not self._initialized:
            await self.initialize()
        try:
            await clear_data(self.graphiti.driver)
            logger.warning("Cleared all graph data")
        except Exception:
            await self.close()
            await self.initialize()

# Global instance
graph_client = GraphitiClient()

# Async convenience
async def initialize_graph():
    await graph_client.initialize()

async def close_graph():
    await graph_client.close()

async def add_to_knowledge_graph(content: str, source: str, episode_id: Optional[str] = None):
    if not episode_id:
        episode_id = f"episode_{datetime.now(timezone.utc).isoformat()}"
    await graph_client.add_episode(episode_id, content, source)
    return episode_id
