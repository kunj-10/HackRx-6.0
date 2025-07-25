"""
This script provides a convenient way to start the Backend application.
It configures environment variables and starts the uvicorn server.
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Check for required environment variables
required_vars = [
    # "GEMINI_API_KEY",
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these variables in your .env file or environment.")
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting Mahanayak Backend...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info"
    ) 