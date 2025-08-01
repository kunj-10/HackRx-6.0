from app.utils import RAG_AGENT_SYSTEM_PROMPT
from dotenv import load_dotenv
from pydantic_ai import Agent
from app.vector_store import (
    supabase,
    get_embedding
)

load_dotenv()

pdf_ai_expert = Agent(
    "gemini-2.5-flash",
    system_prompt=RAG_AGENT_SYSTEM_PROMPT,
    retries=2
)

@pdf_ai_expert.tool_plain
async def retrieve_relevant_pdf_chunks(user_query: str, source_file: str = "") -> str:
    """
    Queries the vector  database to get chunks relevant to the given query

    Args:
    user_query: the query asked by the user
    source_file: the file from which chunks are to be fetched

    Returns:
    Top 3 matching documents. 
    """
    print(f"calling retrieve_relevant_documents. User Query: {user_query}, sorce_file: {source_file}")

    embedding = await get_embedding(user_query)

    result = supabase.rpc(
        'match_pdf_chunks',
        {
            'query_embedding': embedding,
            'match_count': 3,
            'source': source_file
        }
    ).execute()

    if not result.data:
        return "No relevant chunks found."

    result = "\n\n---\n\n".join([
        f"# {r['title']}\n\n{r['content']}" for r in result.data
    ])

    # print(result)
    return result

@pdf_ai_expert.tool_plain
async def get_pdf_content(source_file: str) -> str:
    """
    Gives the content of whole pdfs. Pdfs are very huge so call this tool carefully and responsibly.

    Args:
    source_file: the file whose content is to be fetched.

    Returns:
    The content of the entire pdf.
    """
    
    print("Calling get_pdf_content...")
    
    result = supabase.from_('pdf_chunks') \
        .select('title, content, chunk_number') \
        .eq('source_file', source_file) \
        .order('chunk_number') \
        .execute()

    if not result.data:
        return f"No content found for PDF: {source_file}"

    page_title = result.data[0]['title']
    return f"# {page_title}\n\n" + "\n\n".join(chunk['content'] for chunk in result.data)

async def main():
    print("getting output...")

    res = await pdf_ai_expert.run("source_file is three.pdf. user_query: Explain how the Cashless Facility (Definition 5) operates in conjunction with the Network Provider definition (Definition 31), and what Condition Precedent (Definition 6) and Co-Payment (Definition 8) requirements apply when an insured avails cashless treatment. Additionally, describe how Room Rent and ICU Charges limits (Section C Part A, Table of Benefits) interact with these provisions.")
    print(res.output)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
