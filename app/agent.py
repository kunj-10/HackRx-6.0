from __future__ import annotations as _annotations

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from app.utils import RAG_AGENT_SYSTEM_PROMPT

load_dotenv()

class Deps:
    pass

agent = Agent(
    "google-gla:gemini-2.0-flash",
    system_prompt=RAG_AGENT_SYSTEM_PROMPT,
    deps_type=None,
    retries=2
)

@agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[Deps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    pass
