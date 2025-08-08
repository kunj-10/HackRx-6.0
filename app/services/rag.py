from dotenv import load_dotenv
from google.genai import types
from google import genai
import requests
import logging
import httpx
import io
from typing import Literal, Optional, Any, Dict, List
from pydantic_ai import RunContext
from dataclasses import dataclass
from crawl4ai import AsyncWebCrawler



from app.utils import RAG_AGENT_SYSTEM_PROMPT, PDF_AGENT_PROMPT
from app.services.vector_store_service import (
    supabase,
    openai_client,
    get_embedding
)

load_dotenv()
client = genai.Client()

@dataclass
class ApiDependencies:
    http_client: httpx.AsyncClient

async def api_request(
    ctx: RunContext,
    url: str,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
    payload: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Makes a generic HTTP request to a specified URL and returns the JSON response.

    Use this tool to interact with any external API to fetch or send data when
    no other more specific tool is available.

    :param url: The full, absolute URL of the API endpoint to request. Must be a valid HTTP or HTTPS URL.
    :param method: The HTTP method to use. Must be one of 'GET', 'POST', 'PUT', 'DELETE', or 'PATCH'.
    :param payload: An optional dictionary of data to send as the JSON body. Typically used with 'POST', 'PUT', or 'PATCH' methods.
    """
    try:
        response = await ctx.deps.http_client.request(
            method=method,
            url=url,
            json=payload if payload else None,
            follow_redirects=True,
            timeout=10.0,
        )
        response.raise_for_status()
        if response.status_code!= 204:
            return response.json()
        else:
            return {"status": "success", "code": response.status_code}
    except httpx.HTTPStatusError as e:
        error_body = e.response.text
        return {
            "error": "HTTP Error", 
            "status_code": e.response.status_code, 
            "details": f"The server responded with an error. Response body: {error_body[:500]}"
        }
    except httpx.RequestError as e:
        return {"error": "Request Error", "details": f"A network error occurred: {e}"}
    except Exception as e:
        return {"error": "An unexpected error occurred", "details": str(e)}

async def crawl_and_aggregate_website(
    start_url: str,
    max_pages: int = 50,
    max_depth: int = 3,
    strategy: str = "bfs"
) -> str:
    """
    Crawls a website from a starting URL and aggregates the content of all
    successfully crawled pages into a single string.

    Args:
        start_url (str): The homepage or starting URL of the website to crawl.
        max_pages (int): The maximum number of pages to crawl. Acts as a safeguard.
        max_depth (int): The maximum link depth to follow from the start_url.
        strategy (str): The deep crawl strategy to use ('bfs', 'dfs', or 'bestfirst').

    Returns:
        str: A single string containing the aggregated 'fit_markdown' from all
             crawled pages, with each page's content clearly separated.
    """
    logging.info(f"Starting crawl for {start_url} with max_pages={max_pages}, max_depth={max_depth}")
    
    aggregated_content: List[str] = []
    pages_crawled_count = 0
    pages_failed_count = 0

    # The 'async with' context manager ensures the browser is properly started and shut down.
    async with AsyncWebCrawler() as crawler:
        try:
            crawl_generator = await crawler.adeep_crawl(
                start_url=start_url,
                strategy=strategy,
                max_depth=max_depth,
                max_pages=max_pages,
                # Exclude common non-content file types to make the crawl more efficient.
                exclude_patterns=[
                    r".*\.(pdf|jpg|jpeg|png|gif|zip|rar|css|js|xml|ico)$"
                ]
            )

            async for result in crawl_generator:

                if result and result.success:
                    pages_crawled_count += 1
                    logging.info(f" Crawled: {result.url} (Depth: {result.depth})")
                    
                    page_header = f"# Page URL: {result.url}\n\n"
                    
                    page_content = result.markdown.fit_markdown
                    
                    if page_content and page_content.strip():
                        aggregated_content.append(page_header + page_content)
                else:
                    pages_failed_count += 1

                    logging.warning(f" Failed to crawl: {result.url}, Error: {result.error_message}")

        except Exception as e:
            logging.error(f"An unexpected error occurred during the crawl: {e}", exc_info=True)

    logging.info(f"Crawl finished. Successfully crawled {pages_crawled_count} pages. Failed on {pages_failed_count} pages.")
    
    return "\n\n---\n\n---\n\n".join(aggregated_content)


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

async def answer_query(user_query: str, source_file: str) -> str:
    try:
        context = await retrieve_relevant_pdf_chunks(user_query, source_file)

        prompt = f"Retrieved Chunks: {context}. \n User Query: {user_query}."

        response = await openai_client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": RAG_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content
        return content
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


# Usage

# async def main():
#     """
#     Main function to set up dependencies and run the agent.
#     """
#     prompt = "Please fetch the details for the user with ID 1 from the JSONPlaceholder API."
#     print(f"User Prompt: {prompt}\n")

#     # Use an async context manager for the client to ensure proper cleanup
#     async with httpx.AsyncClient() as client:
#         # Instantiate the dependencies
#         api_deps = ApiDependencies(http_client=client)

#         # Run the agent
#         result = await agent.run(prompt, deps=api_deps)

#         # Print the final output
#         print("--- Agent Final Output ---")
#         import json
#         print(json.dumps(result.output, indent=2))
