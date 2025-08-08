from dotenv import load_dotenv
from google.genai import types
from google import genai
import requests
import logging
import httpx
import io

from app.services.vector_store_service import (
    supabase,
    openai_client,
    get_embedding
)
from app.services.agent import (
    ApiDependencies,
    agent
)
from app.utils import (
    RAG_AGENT_SYSTEM_PROMPT,
    PDF_AGENT_PROMPT
)
# from app.services.round_robin import RoundRobin

load_dotenv()
client = genai.Client()

async def retrieve_relevant_pdf_chunks(user_query: str, source_file: str = "") -> str:
    embedding = await get_embedding(user_query)
    retrieve = 3
     
    # if source_file.split('.')[-1] == 'xlsx':
    #     retrieve = 1
    # print(retrieve)
    result = supabase.rpc(
        'match_pdf_chunks',
        {
            'query_embedding': embedding,
            'match_count': retrieve,
            'source': source_file
        }
    ).execute()

    if not result.data:
        return "No relevant chunks found."

    result = "\n\n---\n\n".join([
        f"{r['content']}" for r in result.data
    ])

    return result

async def answer_query(user_query: str, source_file: str = None) -> str:
    try:
        if source_file:
            context = await retrieve_relevant_pdf_chunks(user_query, source_file)
            prompt = f"Retrieved Chunks: {context}. \n User Query: {user_query}."
        else: 
            prompt = user_query

        # response = await openai_client.chat.completions.create(
        #     model="gemini-2.5-pro",
        #     messages=[
        #         {"role": "system", "content": RAG_AGENT_SYSTEM_PROMPT},
        #         {"role": "user", "content": prompt}
        #     ]
        # )

        # content = response.choices[0].message.content
        # return content

        async with httpx.AsyncClient() as client:
            api_deps = ApiDependencies(http_client=client)
            result = await agent.run(prompt, deps=api_deps)

            return result.output

        # import os
        # keys = []
        # for i in range(5):
        #     keys.append(os.getenv(f"KEY{i + 1}"))

        # rr = RoundRobin(keys)
        # result  = await rr.run(prompt)
        # return result

    except Exception as e:
        logging.error(f"Error getting answer: {e}")

def read_image(url: str = None, image_bytes = None, mime_type: str = "image/jpeg") -> str:
    if url: image_bytes = requests.get(url).content
    image = types.Part.from_bytes(
        data=image_bytes, 
        mime_type=mime_type
    )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=["Give all the details of the image in text, so that the other agent can answer questions on the image based on the text you give.", image],
    )

    print(response.text)
    return response.text

async def answer_image_query(user_query: str, image_text: str) -> str:
    try:
        system_prompt = """ You are tasked to answer the question asked by the user on the basis of the image given. The image model has convertad the image into text describing the image. You will receive that description along with the query. You need to answer user's query in short. Your answer should be short and to the point. If the image does not contain answer of the query, then answer it correctly by your own. Try to identidy patterns from the image before answering by your own.  """
        prompt = f"Text description of the image given by user: {image_text}. \n User Query: {user_query}."

        response = await openai_client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content
        return content
    except Exception as e:
        logging.error(f"Error getting answer: {e}")

async def pdf_query(url: str, questions: list) -> list:
    doc_io = io.BytesIO(httpx.get(url).content)

    sample_doc = client.files.upload(
    file=doc_io,
    config=dict(
        mime_type='application/pdf')
    )
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            sample_doc, 
            PDF_AGENT_PROMPT(questions)
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[str],
        },
    )

    answers = response.parsed
    return answers

async def main():
    result = await answer_query("How are you?")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())