import asyncio
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

# 🔐 Configure OpenAI‐Proxy (Gemini) client
client = OpenAI(
    api_key="<YOUR_API_KEY>",  # replace with your key
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# 📌 Gemini embedding function via OpenAI‐Proxy
async def gemini_embed(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        # wrap the blocking call in a thread so we stay async
        resp = await asyncio.to_thread(
            client.embeddings.create,
            model="models/embedding-001",
            input=text,
            task_type="retrieval_document"
        )
        embeddings.append(resp.data[0].embedding)
    return embeddings

# ✨ Inform LightRAG how large each embedding is:
gemini_embed.embedding_dim = 3072  # Gemini embeddings are 3072‑dim

# 🧠 Gemini LLM completion function via OpenAI‐Proxy
async def gemini_complete(prompt: str) -> str:
    # wrap in to_thread as well
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model="gemini-2.5-flash",  # or "gemini-2.5-flash"
        n=1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# 🚀 Main RAG setup and query flow
async def main():
    # Initialize LightRAG with our new OpenAI‐Proxy functions
    rag = LightRAG(
        working_dir="data/",
        embedding_func=gemini_embed,
        llm_model_func=gemini_complete
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Add your documents
    await rag.ainsert("The most popular AI agent framework of all time is probably Langchain.")
    await rag.ainsert("Under the Langchain hood we also have LangGraph, LangServe, and LangSmith.")
    await rag.ainsert("Many people prefer using other frameworks like Agno or Pydantic AI instead of Langchain.")
    await rag.ainsert("It is very easy to use Python with all of these AI agent frameworks.")

    # Run a query
    result = await rag.aquery(
        "What programming language should I use for coding AI agents?",
        param=QueryParam(mode="mix")
    )

    print("\n🔍 Answer from Gemini via OpenAI Proxy:\n", result)

if __name__ == "__main__":
    asyncio.run(main())


# import asyncio
# import google.generativeai as genai
# from lightrag import LightRAG, QueryParam
# from lightrag.kg.shared_storage import initialize_pipeline_status

# # 🔐 Configure Gemini API key
# genai.configure(api_key="<YOUR_API_KEY>")  # Replace with your actual API key

# # 📌 Gemini embedding function (retrieval use-case)
# async def gemini_embed(texts: list[str]) -> list[list[float]]:
#     responses = []
#     for text in texts:
#         res = genai.embed_content(
#             model="models/embedding-001",
#             content=text,
#             task_type="retrieval_document"
#         )
#         responses.append(res["embedding"])
#     return responses

# # 🧠 Gemini LLM completion function (async)
# async def gemini_complete(prompt: str) -> str:
#     model = genai.GenerativeModel("gemini-1.5-pro")  # or "gemini-2.5-flash"
#     response = await model.generate_content_async(prompt)
#     return response.text.strip()

# # 🚀 Main RAG setup and query flow
# async def main():
#     # Initialize LightRAG instance with Gemini
#     rag = LightRAG(
#         working_dir="data/",
#         embedding_func=gemini_embed,
#         llm_model_func=gemini_complete
#     )

#     await rag.initialize_storages()
#     await initialize_pipeline_status()

#     # Add documents
#     await rag.ainsert("The most popular AI agent framework of all time is probably Langchain.")
#     await rag.ainsert("Under the Langchain hood we also have LangGraph, LangServe, and LangSmith.")
#     await rag.ainsert("Many people prefer using other frameworks like Agno or Pydantic AI instead of Langchain.")
#     await rag.ainsert("It is very easy to use Python with all of these AI agent frameworks.")

#     # Run a query
#     result = await rag.aquery(
#         "What programming language should I use for coding AI agents?",
#         param=QueryParam(mode="mix")
#     )

#     print("\n🔍 Answer from Gemini:\n", result)

# if __name__ == "__main__":
#     asyncio.run(main())
# import asyncio
# import google.generativeai as genai
# from lightrag import LightRAG, QueryParam
# from lightrag.kg.shared_storage import initialize_pipeline_status

# # 1️⃣ Configure Gemini
# genai.configure(api_key="<YOUR_API_KEY>")

# # 2️⃣ Embedding function
# async def gemini_embed(texts: list[str]) -> list[list[float]]:
#     embeddings = []
#     for t in texts:
#         res = genai.embed_content(
#             model="models/embedding-001",
#             content=t,
#             task_type="retrieval_document"
#         )
#         embeddings.append(res["embedding"])
#     return embeddings

# # ← Tell LightRAG the embedding size
# gemini_embed.embedding_dim = 1536

# # 3️⃣ Completion function
# async def gemini_complete(prompt: str) -> str:
#     model = genai.GenerativeModel("gemini-2.5-flash")
#     resp = await model.generate_content_async(prompt)
#     return resp.text.strip()

# # 4️⃣ Main RAG flow
# async def main():
#     rag = LightRAG(
#         working_dir="data/",
#         embedding_func=gemini_embed,
#         llm_model_func=gemini_complete
#     )

#     await rag.initialize_storages()
#     await initialize_pipeline_status()

#     # insert your docs
#     await rag.ainsert("Langchain is the most popular AI agent framework.")
#     await rag.ainsert("Under Langchain you also have LangGraph, LangServe, and LangSmith.")
#     await rag.ainsert("Alternatives include Agno and Pydantic AI.")
#     await rag.ainsert("Python is widely used with these frameworks.")

#     # run a query
#     answer = await rag.aquery(
#         "What programming language should I use for AI agents?",
#         param=QueryParam(mode="mix")
#     )

#     print("🔍 Answer from Gemini:", answer)

# if __name__ == "__main__":
#     asyncio.run(main())
# import asyncio
# import google.generativeai as genai
# from lightrag import LightRAG, QueryParam
# from lightrag.kg.shared_storage import initialize_pipeline_status

# genai.configure(api_key="<YOUR_API_KEY>")

# # 1️⃣ Embedding—accept any kwargs, always return 1536‑d
# async def gemini_embed(texts: list[str], **kwargs) -> list[list[float]]:
#     embeddings = []
#     for t in texts:
#         res = genai.embed_content(
#             model="models/embedding-001",
#             content=t,
#             task_type="retrieval_document"
#         )
#         embeddings.append(res["embedding"])
#     return embeddings

# gemini_embed.embedding_dim = 1536

# # 2️⃣ Completion—swallow any extra kwargs
# async def gemini_complete(prompt: str, **kwargs) -> str:
#     model = genai.GenerativeModel("gemini-1.5-pro")
#     resp = await model.generate_content_async(prompt)
#     return resp.text.strip()

# async def main():
#     # Clear your data/ folder before running this script
#     rag = LightRAG(
#         working_dir="data/",
#         embedding_func=gemini_embed,
#         llm_model_func=gemini_complete,
#         # If supported:
#         # keyword_embedding_func=gemini_embed,
#     )

#     await rag.initialize_storages()
#     await initialize_pipeline_status()

#     # Insert documents
#     await rag.ainsert("Langchain is the most popular AI agent framework.")
#     await rag.ainsert("Under Langchain you also have LangGraph, LangServe, and LangSmith.")
#     await rag.ainsert("Alternatives include Agno and Pydantic AI.")
#     await rag.ainsert("Python is widely used with these frameworks.")

#     # Query
#     answer = await rag.aquery(
#         "What programming language should I use for AI agents?",
#         param=QueryParam(mode="mix")
#     )
#     print("🔍 Answer from Gemini:", answer)

# if __name__ == "__main__":
#     asyncio.run(main())
# import asyncio
# from openai import OpenAI
# from lightrag import LightRAG, QueryParam
# from lightrag.kg.shared_storage import initialize_pipeline_status

# # ── 1️⃣ Configure the OpenAI client to talk to Gemini ─────────────────────────
# client = OpenAI(
#     api_key="<YOUR_API_KEY>",
#     base_url="https://generativelanguage.googleapis.com/v1beta/"
# )

# # ── 2️⃣ Embedding function via OpenAI Proxy (Gemini) ─────────────────────────
# async def gemini_embed(texts: list[str], **kwargs) -> list[list[float]]:
#     embeddings = []
#     for t in texts:
#         # run the sync call in a thread so we can await it
#         resp = await asyncio.to_thread(
#             client.embeddings.create,
#             model="models/embedding-001",
#             input=t,
#             # you can pass any extra kwargs here if needed
#         )
#         # first item in data array
#         embeddings.append(resp.data[0].embedding)
#     return embeddings

# # inform LightRAG how large each vector is
# gemini_embed.embedding_dim = 1536

# # ── 3️⃣ Completion function via OpenAI Proxy (Gemini) ────────────────────────
# async def gemini_complete(prompt: str, **kwargs) -> str:
#     # LightRAG will stick everything into 'prompt', so we just wrap it as a user message
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user",   "content": prompt},
#     ]
#     resp = await asyncio.to_thread(
#         client.chat.completions.create,
#         model="gemini-1.5-flash",
#         n=1,
#         messages=messages,
#     )
#     # the SDK returns .choices[0].message
#     return resp.choices[0].message.content.strip()

# # ── 4️⃣ Full RAG pipeline ──────────────────────────────────────────────────────
# async def main():
#     # (Re)create data/, shared status, etc.
#     rag = LightRAG(
#         working_dir="data/",
#         embedding_func=gemini_embed,
#         llm_model_func=gemini_complete,
#         # if your LightRAG version supports it, you can also force
#         # keyword_embedding_func=gemini_embed, to keep dims consistent
#     )

#     await rag.initialize_storages()
#     await initialize_pipeline_status()

#     # insert your docs
#     await rag.ainsert("Langchain is the most popular AI agent framework.")
#     await rag.ainsert("Under Langchain you also have LangGraph, LangServe, and LangSmith.")
#     await rag.ainsert("Alternatives include Agno and Pydantic AI.")
#     await rag.ainsert("Python is widely used with these frameworks.")

#     # run a query
#     answer = await rag.aquery(
#         "What programming language should I use for AI agents?",
#         param=QueryParam(mode="mix")
#     )

#     print("🔍 Answer from Gemini:", answer)

# if __name__ == "__main__":
#     asyncio.run(main())

