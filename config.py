import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ==========================================
# Gemini Configuration
# ==========================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-2.5-flash"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)

# ==========================================
# ChromaDB Configuration
# ==========================================

CHROMA_DB_DIR = os.getenv(
    "CHROMA_DB_DIR",
    "chroma_db"
)

# ==========================================
# Text Splitter Configuration
# ==========================================

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", 1000)
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", 200)
)

# ==========================================
# Retriever Configuration
# ==========================================

TOP_K = int(
    os.getenv("TOP_K", 3)
)

# ==========================================
# Validation
# ==========================================

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing! Please add it to your .env file."
    )