"""
Configuration management for Confluence Chatbot
"""

import os
from typing import List, Optional
from pathlib import Path


class Config:
    """Configuration class for the chatbot"""

    # Application settings
    APP_NAME: str = "Confluence Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    API_DATA_PATH: Path = Path(os.getenv("API_DATA_PATH", "../api_data"))

    # RAG Pipeline settings
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_DOCUMENTS: int = int(os.getenv("MAX_DOCUMENTS", "1000"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.1"))

    # Model settings (for quantized LLM integration)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-2-7b-chat.Q4_K_M.gguf")  # Quantized model file name
    LLM_MODEL_PATH: str = os.getenv("LLM_MODEL_PATH", "./models")  # Directory containing the model file
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))  # Response length limit
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # Balanced temperature for factual responses
    LLM_CONTEXT_WINDOW: int = int(os.getenv("LLM_CONTEXT_WINDOW", "2048"))  # Model's context window size
    LLM_THREADS: int = int(os.getenv("LLM_THREADS", "4"))  # Number of CPU threads to use

    # AWS settings
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: Optional[str] = os.getenv("S3_BUCKET")
    DYNAMODB_TABLE: Optional[str] = os.getenv("DYNAMODB_TABLE")

    # Security settings
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://yourdomain.com").split(",")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

    # Performance settings
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "300"))
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB

    # Cache settings
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))

    # Monitoring
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        issues = []

        # Check required paths
        if not cls.API_DATA_PATH.exists():
            issues.append(f"API data path does not exist: {cls.API_DATA_PATH}")

        # Check port range
        if not (1 <= cls.PORT <= 65535):
            issues.append(f"Invalid port number: {cls.PORT}")

        # Check chunk settings
        if cls.CHUNK_SIZE <= 0:
            issues.append("Chunk size must be positive")
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            issues.append("Chunk overlap must be less than chunk size")

        return issues


# Global config instance
config = Config()
