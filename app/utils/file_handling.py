import os
import aiofiles
import aiohttp
from fastapi import HTTPException
import uuid
from urllib.parse import urlparse

DOWNLOAD_DIR = "data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


async def save_file_from_url(file_url: str) -> str: 
    parsed_url = urlparse(str(file_url))
    original_filename = f"{uuid.uuid4()}_{os.path.basename(parsed_url.path)}"
    if not original_filename:
        raise HTTPException(status_code=400, detail="Could not determine filename from URL")

    filepath = os.path.join(DOWNLOAD_DIR, original_filename)

    async with aiohttp.ClientSession() as session:
        async with session.get(str(file_url)) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to download file")

            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
    
    return filepath, original_filename