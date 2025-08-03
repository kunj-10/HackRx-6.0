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
import asyncio
from app.services.vector_store_service import process_and_store_document
from app.services.agent import pdf_ai_expert
from app.services.rag import answer_query
from app.utils import compute_sha256
from app.api.db.mongo import file_collection

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
        logging.info(f"Document URL: {payload.documents}")

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
        before_hash = time.monotonic()
        file_hash = await compute_sha256(filepath)
        after_hash = time.monotonic()
        logging.info(f"Time taken to hash: {(after_hash-before_hash):.2f} seconds")
        existing = await file_collection.find_one({"hash": file_hash})
        if existing:
            logging.info(f"File already processed: {existing['filename']}")
            filename = existing['filename']
        else:
            text = extract_text(filepath)
            await process_and_store_document(text, original_filename)
            await file_collection.insert_one({"hash": file_hash, "filename": original_filename})
            filename=original_filename
            logging.info("File Processed")
        # filename=original_filename

        # text = extract_text(filepath)
        # await process_and_store_document(text, original_filename)

        response = {"answers": []}
        response['answers'] = await asyncio.gather(*[
            answer_query(question, filename) for question in payload.questions
        ])
        # for question in payload.questions:
            # result = await pdf_ai_expert.run(f"source_file is {original_filename}. user_query: {question}")

            # logging.info(f"Question: {question} ----- Response: {result.output}")
            # response["answers"].append(result.output)

            # result = await answer_query(question, original_filename)
            # logging.info(f"Question: {question} ----- Response: {result}")
            # response["answers"].append(result)
        
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
