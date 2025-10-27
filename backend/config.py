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
    EARLY_STOPPING_PATIENCE: int = 10

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

    # CORS settings
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8000"]

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
