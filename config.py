import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEEPSEEK_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
