"""
FastAPI Backend for Movie Recommender System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from config import settings
from services.recommender import RecommenderService
from services.user_service import UserService


# Pydantic models for API
class RecommendRequest(BaseModel):
    user_id: int
    n_items: int = 10

class TrainRequest(BaseModel):
    force_retrain: bool = False

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict]
    count: int

class UserHistoryResponse(BaseModel):
    user_id: int
    history: List[Dict]
    count: int

class TrainResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[Dict] = None

class UserRegistrationRequest(BaseModel):
    username: str
    email: str
    password: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class RatingRequest(BaseModel):
    movie_id: int
    rating: float

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    created_at: str
    last_login: Optional[str] = None

class RatingResponse(BaseModel):
    user_id: int
    movie_id: int
    rating: float
    timestamp: str

class UserStatsResponse(BaseModel):
    total_users: int
    total_ratings: int
    avg_ratings_per_user: float


# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="Hybrid collaborative filtering + content-based movie recommender",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
recommender_service = RecommenderService()
user_service = UserService(settings.USERS_FILE, settings.USER_RATINGS_FILE)

# Connect services
recommender_service.set_user_service(user_service)

# Create API router
api_router = APIRouter()

@api_router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    model_loaded = recommender_service.model is not None
    model_info = None

    if model_loaded:
        model_info = recommender_service.get_model_info()

    return HealthResponse(
        status="healthy" if model_loaded else "training_required",
        model_loaded=model_loaded,
        model_info=model_info
    )

@api_router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train the recommendation model"""
    try:
        # Run training (this might take time)
        result = recommender_service.train_model(force_retrain=request.force_retrain)

        # If training was successful, update the service
        if result["status"] == "success":
            # Reload the model in the service
            recommender_service._load_model()

        return TrainResponse(
            status=result["status"],
            message=result["message"],
            details=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@api_router.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, n: int = 10):
    """Get movie recommendations for a user"""
    try:
        recommendations = recommender_service.get_recommendations(user_id, n)

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            count=len(recommendations)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@api_router.post("/recommend", response_model=RecommendationResponse)
def recommend_with_body(request: RecommendRequest):
    """Get movie recommendations using request body"""
    return get_recommendations(request.user_id, request.n_items)

@api_router.get("/history/{user_id}", response_model=UserHistoryResponse)
def get_user_history(user_id: int):
    """Get user's rating history"""
    try:
        if recommender_service.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first"
            )

        history = recommender_service.get_user_history(user_id)

        return UserHistoryResponse(
            user_id=user_id,
            history=history,
            count=len(history)
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user history: {str(e)}")

@api_router.get("/stats")
def get_dataset_stats():
    """Get basic dataset statistics"""
    try:
        if recommender_service.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )

        data = recommender_service.data
        if data is None:
            raise HTTPException(status_code=503, detail="Data not available")

        stats = {
            "total_ratings": len(data),
            "unique_users": data['user_idx'].nunique(),
            "unique_movies": data['item_idx'].nunique(),
            "unique_genres": data['primary_genre'].nunique(),
            "avg_rating": round(data['rating'].mean(), 2),
            "rating_range": {
                "min": float(data['rating'].min()),
                "max": float(data['rating'].max())
            }
        }

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@api_router.post("/users/register", response_model=UserResponse)
def register_user(request: UserRegistrationRequest):
    """Register a new user"""
    success, message, user_id = user_service.create_user(
        request.username, request.email, request.password
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    user_data = user_service.get_user(user_id)
    return UserResponse(**user_data)

@api_router.post("/users/login")
def login_user(request: UserLoginRequest):
    """Login user"""
    success, message, user_data = user_service.authenticate_user(
        request.username, request.password
    )

    if not success:
        raise HTTPException(status_code=401, detail=message)

    return {
        "message": message,
        "user": UserResponse(**user_data)
    }

@api_router.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    """Get user information"""
    user_data = user_service.get_user(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(**user_data)

@api_router.post("/users/{user_id}/ratings", response_model=RatingResponse)
def add_user_rating(user_id: int, request: RatingRequest):
    """Add or update a rating for a user"""
    # Validate rating range
    if not (0.5 <= request.rating <= 5.0):
        raise HTTPException(status_code=400, detail="Rating must be between 0.5 and 5.0")

    # Check if user exists
    if not user_service.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")

    rating_data = user_service.add_rating(user_id, request.movie_id, request.rating)
    return RatingResponse(**rating_data)

@api_router.get("/users/{user_id}/ratings")
def get_user_ratings(user_id: int):
    """Get all ratings for a user"""
    # Check if user exists
    if not user_service.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")

    ratings = user_service.get_user_ratings(user_id)
    return {"user_id": user_id, "ratings": ratings, "count": len(ratings)}

@api_router.get("/users/{user_id}/ratings/{movie_id}")
def get_user_rating_for_movie(user_id: int, movie_id: int):
    """Get user's rating for a specific movie"""
    # Check if user exists
    if not user_service.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")

    rating = user_service.get_user_rating_for_movie(user_id, movie_id)
    if rating is None:
        raise HTTPException(status_code=404, detail="Rating not found")

    return {"user_id": user_id, "movie_id": movie_id, "rating": rating}

@api_router.get("/users/stats", response_model=UserStatsResponse)
def get_user_stats():
    """Get user statistics"""
    return UserStatsResponse(**user_service.get_stats())

# Include the API router with /api prefix
app.include_router(api_router, prefix="/api")

# Root endpoint (outside /api)
@app.get("/", response_model=Dict)
def root():
    """Root endpoint"""
    return {
        "message": "Movie Recommender API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Keep the old /health endpoint for backward compatibility
@app.get("/health", response_model=HealthResponse)
def old_health_check():
    """Health check endpoint (backward compatibility)"""
    model_loaded = recommender_service.model is not None
    model_info = None

    if model_loaded:
        model_info = recommender_service.get_model_info()

    return HealthResponse(
        status="healthy" if model_loaded else "training_required",
        model_loaded=model_loaded,
        model_info=model_info
    )