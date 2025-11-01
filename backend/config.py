"""
Configuration settings for the Movie Recommender System
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    # Model hyperparameters
    N_FACTORS: int = 20
    N_ITERATIONS: int = 60
    REG_LAMBDA: float = 0.25
    LEARNING_RATE: float = 0.005
    CONTENT_WEIGHT: float = 0.15
    SEMANTIC_WEIGHT: float = 0.20
    EARLY_STOPPING_PATIENCE: int = 10

    # Semantic search settings
    USE_SEMANTIC_SEARCH: bool = True
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_CACHE_PATH: str = str(MODELS_DIR / "embeddings.pkl")
    EMBEDDING_DIM: int = 384  # Depends on model

    # Data settings
    MIN_USER_RATINGS: int = 5
    MIN_MOVIE_RATINGS: int = 5

    # Paths - using relative paths from project root
    DATA_PATH: str = str(DATA_DIR / "ml-latest-small")
    MODEL_PATH: str = str(MODELS_DIR / "hybrid_recommender.pkl")
    PROCESSED_DATA_PATH: str = str(MODELS_DIR / "processed_data.pkl")
    
    # User data paths
    USERS_FILE: str = str(MODELS_DIR / "users.json")
    USER_RATINGS_FILE: str = str(MODELS_DIR / "user_ratings.json")
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # CORS settings - can be overridden via environment variable
    # Set CORS_ORIGINS in .env as comma-separated list:
    # CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://your-app.netlify.app
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:8000"

    @property
    def cors_origins_list(self) -> list:
        """Convert CORS_ORIGINS string to list"""
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        return self.CORS_ORIGINS

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
