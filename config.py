import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
TEMPERATURE = float(os.getenv("TEMPERATURE"))
TOP_P = float(os.getenv("TOP_P"))
TOP_K = int(os.getenv("TOP_K"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS"))

CURRENT_DIR = Path(__file__).parent.absolute()
DATA_DIR = CURRENT_DIR / os.getenv("DATA_DIR", "data")
JSON_FILE = os.getenv("JSON_FILE")
TFIDF_MATRIX_FILE = os.getenv("TFIDF_MATRIX_FILE")
VECTORIZER_FILE = os.getenv("VECTORIZER_FILE")
STOPWORDS_FILE = os.getenv("STOPWORDS_FILE")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

CHAT_URL = os.getenv("CHAT_URL")
