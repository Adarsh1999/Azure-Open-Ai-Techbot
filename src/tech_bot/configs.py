# Use a GPT-4 turbo model
import os

from dotenv import load_dotenv

load_dotenv()

OAI_MODEL = "gpt-4o"
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "default_deployment")
API_KEY = os.getenv("API_KEY", "default_api_key")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "default_endpoint")
API_VERSION = os.getenv("API_VERSION", "default_api_version")
