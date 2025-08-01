from fastapi import APIRouter, Depends, HTTPException
from app.utils import verify_token, extract_text
from pydantic import BaseModel, HttpUrl
from typing import List
import aiohttp
import aiofiles
import os
from urllib.parse import urlparse
from app.vector_store import process_and_store_document
import uuid
from app.agent import pdf_ai_expert

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
            response["answers"].append(result.output)
        
        return response
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during processing.")
    
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
