from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dataclasses import dataclass
from typing import List
from supabase import Client
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


openai_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@dataclass
class PDFAIAgentDeps:
    supabase: Client

pdf_ai_expert = Agent(
    "google-gla:gemini-2.0-flash",
    system_prompt="""
You are a PDF AI assistant with access to vector search on PDF chunks stored in a database.
You help users retrieve and understand knowledge extracted from technical documents.
You rely only on documentation you’ve retrieved using tools.
""",
    deps_type=PDFAIAgentDeps,
    retries=2
)

async def get_embedding(text: str) -> List[float]:
    try:
        response = await openai_client.embeddings.create(
            model="gemini-embedding-exp-03-07",
            dimensions=1536,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

@pdf_ai_expert.tool
async def retrieve_relevant_pdf_chunks(ctx: RunContext[PDFAIAgentDeps], user_query: str, source_file: str = "") -> str:
    print(f"calling retrieve_relevant_documents. User Query: {user_query}, sorce_file: {source_file}")
    embedding = await get_embedding(user_query)

    result = ctx.deps.supabase.rpc(
        'match_pdf_chunks',
        {
            'query_embedding': embedding,
            'match_count': 5,
            'source': source_file
        }
    ).execute()

    if not result.data:
        return "No relevant chunks found."

    return "\n\n---\n\n".join([
        f"# {r['title']}\n\n{r['content']}" for r in result.data
    ])

@pdf_ai_expert.tool
async def get_pdf_content(ctx: RunContext[PDFAIAgentDeps], source_file: str) -> str:
    print("Calling get_pdf_content...")
    result = ctx.deps.supabase.from_('pdf_chunks') \
        .select('title, content, chunk_number') \
        .eq('source_file', source_file) \
        .order('chunk_number') \
        .execute()

    if not result.data:
        return f"No content found for PDF: {source_file}"

    page_title = result.data[0]['title']
    return f"# {page_title}\n\n" + "\n\n".join(chunk['content'] for chunk in result.data)

async def main():
    from supabase import Client
    supabase: Client = Client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )

    deps = PDFAIAgentDeps(
        supabase=supabase,
    )

    print("getting output...")
    res = await pdf_ai_expert.run("source_file is one.pdf. user_query: what is . Congenital Anomaly?", deps=deps)
    print(res.output)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
