from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import get_settings
import logging

settings = get_settings()

app = FastAPI(
    title="Backend-API",
    description="Backend for Bajaj HackRx 6.0",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from app.api import hackrx_router
app.include_router(hackrx_router, prefix='/api/v1' )

@app.get("/", tags=["health"])
async def root():
    """Root endpoint for API health check."""
    logging.info("checking..")
    return {
        "status": "online",
        "app_name": settings.app_name,
        "version": "0.1.0"
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 