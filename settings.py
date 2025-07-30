# config/enhanced_settings.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class EnhancedSettings:
    # Database - Cloud PostgreSQL
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")

    # Vector DB - Enhanced FAISS with higher capacity
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/enhanced_faiss_index")
    MAX_VECTORS: int = int(os.getenv("MAX_VECTORS", "1000000"))  # 1M vectors capacity
    VECTOR_INDEX_TYPE: str = os.getenv("VECTOR_INDEX_TYPE", "IVF")  # IVF for scalability

    # Embeddings - Google Gemini
    EMBEDDING_DIMENSION: int = 768  # Gemini embedding dimension
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))

    # LLM - Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    # Enhanced Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2000"))  # Increased for large documents
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "400"))  # Increased overlap
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))

    # Category-based Processing
    ENABLE_SMART_CATEGORIZATION: bool = os.getenv("ENABLE_SMART_CATEGORIZATION", "true").lower() == "true"
    MIN_IMPORTANCE_SCORE: float = float(os.getenv("MIN_IMPORTANCE_SCORE", "0.1"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "15000"))  # tokens

    # Performance Optimization
    PARALLEL_PROCESSING: bool = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))  # 5 minutes

    # Document Processing Limits
    MAX_DOCUMENT_SIZE_MB: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50"))
    MAX_QUESTIONS_PER_REQUEST: int = int(os.getenv("MAX_QUESTIONS_PER_REQUEST", "20"))

    # Caching and Storage
    ENABLE_RESULT_CACHING: bool = os.getenv("ENABLE_RESULT_CACHING", "true").lower() == "true"
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour

    # Analytics and Monitoring
    ENABLE_DETAILED_ANALYTICS: bool = os.getenv("ENABLE_DETAILED_ANALYTICS", "true").lower() == "true"
    TRACK_CATEGORY_PERFORMANCE: bool = os.getenv("TRACK_CATEGORY_PERFORMANCE", "true").lower() == "true"

    # API Configuration
    API_TITLE: str = "Enhanced RAG System API"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "High-performance RAG system with category-based document processing and single-endpoint workflow"

    def validate_settings(self) -> dict:
        """Validate all required settings and return validation results"""
        validation_results = {
            "valid": True,
            "missing_required": [],
            "warnings": [],
            "config_summary": {}
        }

        # Check required settings
        required_settings = [
            ("POSTGRES_HOST", self.POSTGRES_HOST),
            ("POSTGRES_DB", self.POSTGRES_DB),
            ("POSTGRES_USER", self.POSTGRES_USER),
            ("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD),
            ("GEMINI_API_KEY", self.GEMINI_API_KEY)
        ]

        for setting_name, setting_value in required_settings:
            if not setting_value:
                validation_results["missing_required"].append(setting_name)
                validation_results["valid"] = False

        # Check configuration warnings
        if self.CHUNK_SIZE < 500:
            validation_results["warnings"].append("CHUNK_SIZE is very small, may affect performance")

        if self.MAX_VECTORS < 100000:
            validation_results["warnings"].append("MAX_VECTORS is low for large documents")

        if self.MAX_CONTEXT_LENGTH > 20000:
            validation_results["warnings"].append("MAX_CONTEXT_LENGTH is very high, may hit token limits")

        # Configuration summary
        validation_results["config_summary"] = {
            "chunk_size": self.CHUNK_SIZE,
            "max_vectors": self.MAX_VECTORS,
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "llm_model": self.LLM_MODEL,
            "smart_categorization": self.ENABLE_SMART_CATEGORIZATION,
            "parallel_processing": self.PARALLEL_PROCESSING,
            "max_questions": self.MAX_QUESTIONS_PER_REQUEST
        }

        return validation_results


# Create settings instance
settings = EnhancedSettings()

# Validate settings on import
validation_result = settings.validate_settings()
if not validation_result["valid"]:
    import logging

    logger = logging.getLogger(__name__)
    logger.error(f"Invalid configuration: Missing required settings: {validation_result['missing_required']}")

if validation_result["warnings"]:
    import logging

    logger = logging.getLogger(__name__)
    for warning in validation_result["warnings"]:
        logger.warning(f"Configuration warning: {warning}")