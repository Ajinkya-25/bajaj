# config/settings.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class settings:
    # API Configuration
    API_TITLE: str = os.getenv("API_TITLE", "RAG API")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")
    API_DESCRIPTION: str = os.getenv("API_DESCRIPTION", "RAG System API")
    API_KEY: str = os.getenv("HACKRX_API_KEY")
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

    # Database - Cloud PostgreSQL
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")

    # Vector DB - Local FAISS
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

    # Embeddings - Google Gemini
    EMBEDDING_DIMENSION: int = 768  # Gemini embedding dimension

    # LLM - Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    # Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))