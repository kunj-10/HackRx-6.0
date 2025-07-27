import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.types import GPTKeywordExtractionFormat
import numpy as np

async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    
    return await openai_complete_if_cache(
        model="gemini-2.0-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def embedding_func(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    client_configs: dict[str, Any] = None,
) -> np.ndarray:
    return await openai_embed(
        texts=texts,
        model=model,
        base_url=base_url,
        api_key=api_key
        client_configs=client_configs
    )

async def main():
    # Initialize RAG instance
    rag = LightRAG(
        working_dir="data/",
        embedding_func=embedding_func,
        llm_model_func=llm_model_func
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Insert text
    await rag.ainsert("The most popular AI agent framework of all time is probably Langchain.")
    await rag.ainsert("Under the Langchain hood we also have LangGraph, LangServe, and LangSmith.")
    await rag.ainsert("Many people prefer using other frameworks like Agno or Pydantic AI instead of Langchain.")
    await rag.ainsert("It is very easy to use Python with all of these AI agent frameworks.")

    # Run the query
    result = await rag.aquery(
        "What programming language should I use for coding AI agents?",
        param=QueryParam(mode="mix")
    )

    print(result)

if __name__ == "__main__":
    asyncio.run(main())