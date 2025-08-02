from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from app.utils import extract_text
from urllib.parse import urlparse
from typing import List
import aiofiles
import logging
import aiohttp
import uuid
import time
import os

from app.services.vector_store_service import process_and_store_document
from app.services.agent import pdf_ai_expert

hackrx_router = APIRouter()

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

DOWNLOAD_DIR = "data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@hackrx_router.post('/hackrx/run')
async def run_hackrx(
    payload: HackRxRequest,
    # token: str = Depends(verify_token)
):
    try:
        start_time = time.monotonic()

        parsed_url = urlparse(str(payload.documents))
        original_filename = f"{uuid.uuid4()}_{os.path.basename(parsed_url.path)}"
        if not original_filename:
            raise HTTPException(status_code=400, detail="Could not determine filename from URL")

        filepath = os.path.join(DOWNLOAD_DIR, original_filename)

        async with aiohttp.ClientSession() as session:
            async with session.get(str(payload.documents)) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to download file")

                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(await response.read())
     
        text = extract_text(filepath)
        await process_and_store_document(text, original_filename)

        response = {"answers": []}

        for question in payload.questions:
            result = await pdf_ai_expert.run(f"source_file is {original_filename}. user_query: {question}")

            logging.info(f"Question: {question} ----- Response: {result.output}")
            response["answers"].append(result.output)
        
        return response
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during processing.")
    
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            
        end_time = time.monotonic()
        duration = end_time - start_time
        logging.info(f"⏱️ Total response time: {duration:.2f} seconds")
