from fastapi import Header, HTTPException, status, Depends
import os
from dotenv import load_dotenv

load_dotenv()

EXPECTED_TOKEN = os.getenv("AUTHORIZATION_TOKEN")

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")

    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

    return token 
