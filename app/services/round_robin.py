from pydantic_ai.providers.openai import OpenAIProvider
from typing import Literal, Optional, Any, Dict
from pydantic_ai.models.openai import OpenAIModel
from openai import RateLimitError
from dotenv import load_dotenv
from pydantic_ai import Agent
import logging
import httpx
import time

from app.utils import RAG_AGENT_SYSTEM_PROMPT

load_dotenv()

def get_agent(api_key: str):
    model = OpenAIModel(
        'gemini-2.5-pro',
        provider=OpenAIProvider(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    )
    agent = Agent(
        model,
        system_prompt=RAG_AGENT_SYSTEM_PROMPT,
        retries=2
    )

    @agent.tool_plain
    async def api_request(
        url: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Makes a generic HTTP request to a specified URL and returns the JSON response.

        Use this tool to interact with any external API to fetch or send data when
        no other more specific tool is available.
        **This tool can also be used for getting html of a website by doing a GET request to that website**

        Args:
        url: The full, absolute URL of the API endpoint to request. Must be a valid HTTP or HTTPS URL.
        method: The HTTP method to use. Must be one of 'GET', 'POST', 'PUT', 'DELETE', or 'PATCH'.
        payload: An optional dictionary of data to send as the JSON body. Typically used with 'POST', 'PUT', or 'PATCH' methods.
        """
        try:
            logging.info(f"Calling API request tool. URL: {url} Method: {method} Payload: {payload}")

            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    json=payload if payload else None,
                    follow_redirects=True,
                    timeout=10.0,
                )
                response.raise_for_status()

                logging.info(f"{response.text}")

                if response.status_code!= 204:
                    return response.text
                else:
                    return {"status": "success", "code": response.status_code}
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logging.error(error_body)
            return {
                "error": "HTTP Error", 
                "status_code": e.response.status_code, 
                "details": f"The server responded with an error. Response body: {error_body[:500]}"
            }
        except httpx.RequestError as e:
            return {"error": "Request Error", "details": f"A network error occurred: {e}"}
        except Exception as e:
            logging.error(e)
            return {"error": "An unexpected error occurred", "details": str(e)}

    return agent

class RoundRobin:
    def __init__(self, api_keys: list[str]):
        self.keys = api_keys
        self.clients = [get_agent(key) for key in api_keys]
        self.client_index = 0
        self.total_clients = len(self.clients)

    def get_next_client(self):
        self.client_index = (self.client_index + 1) % self.total_clients
        return self.clients[self.client_index]

    async def run(self, prompt):
        while True:
            agent = self.clients[self.client_index]
            try:
                result = await agent.run(prompt)

                return result.output
            except RateLimitError:
                print(f"[Rate Limit] Client {self.client_index} rate limited. Switching...")
            except Exception as e:
                raise e
            finally:
                self.get_next_client()