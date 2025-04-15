import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
TEMPERATURE = float(os.getenv("TEMPERATURE"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS"))
TOP_K = int(os.getenv("TOP_K"))
TOP_P = float(os.getenv("TOP_P"))