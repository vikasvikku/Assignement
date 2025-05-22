from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Document Research & Theme Identifier"
    DEBUG: bool = True  # Set to True for development
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: Optional[str] = None
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
        
    # Model settings
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # File settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "png", "jpg", "jpeg"]
    
    # Vector store settings
    VECTOR_STORE_PATH: str = "data/vector_store"
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME: str = "documents"
    VECTOR_SIZE: int = 1536  # Dimension for text embeddings
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 