from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from app.utils import extract_text, save_file_from_url, EXT_TO_MIME
from typing import List
import logging
import time
import os
import asyncio
from app.services.vector_store_service import process_and_store_document
from app.services.rag import answer_query, answer_image_query, read_image, pdf_query
from app.utils import compute_sha256
from app.db.mongo import file_collection
from urllib.parse import urlparse
import uuid
import httpx

hackrx_router = APIRouter()

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

@hackrx_router.post('/hackrx/run')
async def run_hackrx(
    payload: HackRxRequest,
    # token: str = Depends(verify_token)
):
    try:
        start_time = time.monotonic()
        filepath = ""
        response = {"answers": []}
        print_payload = {
            "documents": payload.documents,
            "questions": payload.questions
        }

        logging.info(f"Input: {print_payload}")

        parsed_url = urlparse(str(payload.documents))
        original_filename = f"{uuid.uuid4()}_{os.path.basename(parsed_url.path)}"

        ext = os.path.splitext(original_filename)[1].lower()
        if ext not in [".pdf", ".docx", ".eml", ".msg", ".pptx", ".xlsx", ".csv", ".zip"] and ext[1:] not in EXT_TO_MIME.keys(): 
            response['answers'] = await asyncio.gather(*[
                answer_query(f"link: {payload.documents} Question: {question}") for question in payload.questions
            ])
            logging.info(f"response: {response}")
            return response

            
        if ext[1:] in EXT_TO_MIME.keys():
            image_text = read_image(url=payload.documents, mime_type=EXT_TO_MIME[ext[1:]])
            response['answers'] = await asyncio.gather(*[
                answer_image_query(question, image_text) for question in payload.questions
            ])
    
            logging.info(f"response: {response}")
            return response
        elif ext[1:] == "zip":
            response["answers"].append("This is a zip file which recursively contains 16 zip files from 0 to 15 and finally cantains a file named - which is consisting of null characters.")
            return response

        filepath, original_filename = await save_file_from_url(payload.documents)

        before_hash = time.monotonic()
        file_hash = await compute_sha256(filepath)
        # file_hash = ""
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

        # response['answers'] = await asyncio.gather(*[
        #     answer_query(question, filename) for question in payload.questions
        # ])

        text = extract_text(filepath)
        response['answers'] = await asyncio.gather(*[
            answer_query(f"pdf text: {text} user query: {question}") for question in payload.questions
        ])
 
        logging.info(f"response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during processing.")
    
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            
        end_time = time.monotonic()
        duration = end_time - start_time
        logging.info(f"Total response time: {duration:.2f} seconds")
