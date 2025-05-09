import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

import click

# Load environment variables from .env file
load_dotenv()

# --- Unified Logging Configuration ---
# Determine the project's base directory (assuming main.py is in 'api' subdirectory)
# Adjust if your structure is different, e.g., if main.py is at the root.
# This assumes 'api/main.py', so logs will be in 'api/logs/application.log'
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "application.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(lineno)d %(filename)s:%(funcName)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()  # Also keep logging to console
    ],
    force=True  # Ensure this configuration takes precedence and clears any existing handlers
)

# Get a logger for this main module (optional, but good practice)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required environment variables
required_env_vars = ['GOOGLE_API_KEY', 'OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    logger.warning("Some functionality may not work correctly without these variables.")

@click.command()
@click.option("--repo-url", default=None, help="Repository URL for SCM provider detection")
@click.option("--scm", default="auto", help="SCM provider (default: auto)")
def main(repo_url, scm):
    """
    CLI entrypoint for deepwiki-open.
    """
    provider = None
    if repo_url:
        if "dev.azure.com" in repo_url or repo_url.rstrip("/").endswith(".visualstudio.com"):
            provider = "azure_devops"
    if provider == "azure_devops":
        # Stub for Azure DevOps provider
        class AzureDevOpsStub:
            pass
        _ = AzureDevOpsStub()
        print("ADO selected")
        sys.exit(0)

    # Fallback: start FastAPI server as before
    port = int(os.environ.get("PORT", 8001))
    from api.api import app
    logger.info(f"Starting Streaming API on port {port}")
    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main()
