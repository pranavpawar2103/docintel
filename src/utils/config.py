"""
Configuration management for DocIntel.
Loads environment variables and provides app settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Pydantic V2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    
    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval Configuration
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    # Paths
    upload_dir: str = "data/uploads"
    vectordb_dir: str = "data/vectordb"


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
        
        # Create directories if they don't exist
        os.makedirs(_settings.upload_dir, exist_ok=True)
        os.makedirs(_settings.vectordb_dir, exist_ok=True)
        
    return _settings


# Convenience export
settings = get_settings()