from typing import Literal, Optional, Any, Dict, List
from pydantic_ai import RunContext, Agent
from crawl4ai import AsyncWebCrawler
from dataclasses import dataclass
from dotenv import load_dotenv
import logging
import httpx

from app.utils import RAG_AGENT_SYSTEM_PROMPT
from app.services.vector_store_service import (
    supabase,
    get_embedding
)

load_dotenv()

@dataclass
class ApiDependencies:
    http_client: httpx.AsyncClient


agent = Agent(
    "google-gla:gemini-2.5-pro",
    system_prompt=RAG_AGENT_SYSTEM_PROMPT,
    deps=ApiDependencies,
    retries=2
)

@agent.tool
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

    Args:
    url: The full, absolute URL of the API endpoint to request. Must be a valid HTTP or HTTPS URL.
    method: The HTTP method to use. Must be one of 'GET', 'POST', 'PUT', 'DELETE', or 'PATCH'.
    payload: An optional dictionary of data to send as the JSON body. Typically used with 'POST', 'PUT', or 'PATCH' methods.
    """
    try:
        logging.info(f"Calling API request tool. URL: {url} Method: {method} Payload: {payload}")

        response = await ctx.deps.http_client.request(
            method=method,
            url=url,
            json=payload if payload else None,
            follow_redirects=True,
            timeout=10.0,
        )
        response.raise_for_status()

        logging.infO(f"{response.json()}")

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
        start_url: The homepage or starting URL of the website to crawl.
        max_pages: The maximum number of pages to crawl. Acts as a safeguard.
        max_depth: The maximum link depth to follow from the start_url.
        strategy: The deep crawl strategy to use ('bfs', 'dfs', or 'bestfirst').

    Returns:
        A single string containing the aggregated 'fit_markdown' from all crawled pages, with each page's content clearly separated.
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
    
    return "\n\n---\n\n".join(aggregated_content)


async def main():
    print("getting output...")

    res = await pdf_ai_expert.run("source_file is three.pdf. user_query: Explain how the Cashless Facility (Definition 5) operates in conjunction with the Network Provider definition (Definition 31), and what Condition Precedent (Definition 6) and Co-Payment (Definition 8) requirements apply when an insured avails cashless treatment. Additionally, describe how Room Rent and ICU Charges limits (Section C Part A, Table of Benefits) interact with these provisions.")
    print(res.output)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
