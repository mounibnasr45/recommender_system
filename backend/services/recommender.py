"""
Recommender Service
Business logic layer for movie recommendations
"""

import os
import pickle
from typing import List, Dict, Optional
import pandas as pd
from models.hybrid_mf import HybridMatrixFactorization, MovieRecommender
from config import settings


class RecommenderService:
    """Service for handling movie recommendations"""

    def __init__(self):
        self.model: Optional[HybridMatrixFactorization] = None
        self.recommender: Optional[MovieRecommender] = None
        self.data: Optional[pd.DataFrame] = None
        self.user_service = None  # Will be set later
        self._load_model()

    def _load_model(self):
        """Load the trained model and data with better error handling"""
        import sys
        import __main__
        
        try:
            if os.path.exists(settings.MODEL_PATH):
                # Fix pickle import issue by adding the class to __main__
                __main__.HybridMatrixFactorization = HybridMatrixFactorization
                
                # Also ensure the module path is correct
                sys.modules['__main__'].HybridMatrixFactorization = HybridMatrixFactorization
                
                self.model = HybridMatrixFactorization.load_model(settings.MODEL_PATH)
                print(f"✓ Model loaded from {settings.MODEL_PATH}")
    
                # Try to load the data as well if it exists
                data_path = os.path.join(os.path.dirname(settings.MODEL_PATH), "processed_data.pkl")
                if os.path.exists(data_path):
                    with open(data_path, 'rb') as f:
                        self.data = pickle.load(f)

                    # Merge with user ratings if user service is available
                    if self.user_service is not None:
                        self.data = self._merge_user_ratings(self.data)

                    self.recommender = MovieRecommender(self.model, self.data)
                    print("✓ Data and recommender initialized")
                else:
                    print("⚠️  Data file not found, recommender not initialized")
            else:
                print(f"⚠️  Model not found at {settings.MODEL_PATH}")
                print("   Run training first to create the model")
    
        except AttributeError as e:
            if "'HybridMatrixFactorization'" in str(e):
                print(f"❌ Pickle import error: The model was saved with a different module structure")
                print("   Solution: Run the fix_model_pickle.py script to fix the model file")
                print(f"   Or retrain the model with: POST /train")
            else:
                print(f"❌ Error loading model: {e}")
            self.model = None
            self.recommender = None
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   The saved model may be incompatible. Consider retraining.")
            self.model = None
            self.recommender = None

    def train_model(self, force_retrain: bool = False) -> Dict:
        """
        Train a new model

        Args:
            force_retrain: If True, retrain even if model exists

        Returns:
            Training results dictionary
        """
        if self.model is not None and not force_retrain:
            return {"status": "already_trained", "message": "Model already exists"}

        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path

            # Add project root to path if not already there
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from data.dataset_setup import DatasetSetup
            from data.data_loader import MovieLensLoader

            # Download and load data
            setup = DatasetSetup()
            if not setup.download_and_extract():
                return {"status": "error", "message": "Failed to download dataset"}

            loader = MovieLensLoader(setup.data_dir)
            data = loader.load_and_enrich(
                min_user_ratings=settings.MIN_USER_RATINGS,
                min_movie_ratings=settings.MIN_MOVIE_RATINGS
            )

            # Train/validation/test split
            from sklearn.model_selection import train_test_split
            train_df, temp_df = train_test_split(data, test_size=0.4, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

            # Initialize and train model
            model = HybridMatrixFactorization(
                n_factors=settings.N_FACTORS,
                n_iterations=settings.N_ITERATIONS,
                reg_lambda=settings.REG_LAMBDA,
                learning_rate=settings.LEARNING_RATE,
                content_weight=settings.CONTENT_WEIGHT,
                early_stopping=True,
                patience=settings.EARLY_STOPPING_PATIENCE
            )

            model.fit(train_df, val_df)

            # Save model
            os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
            model.save_model(settings.MODEL_PATH)

            # Create recommender with merged data
            combined_data = self._merge_user_ratings(data)
            self.model = model
            self.data = combined_data
            self.recommender = MovieRecommender(model, combined_data)

            return {
                "status": "success",
                "message": "Model trained successfully",
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "best_val_rmse": model.best_val_rmse if hasattr(model, 'best_val_rmse') else None
            }

        except Exception as e:
            return {"status": "error", "message": f"Training failed: {str(e)}"}

    def get_recommendations(self, user_id: int, n: int = 10) -> List[Dict]:
        """
        Get movie recommendations for a user

        Args:
            user_id: User ID
            n: Number of recommendations

        Returns:
            List of recommendation dictionaries
        """
        if self.model is None:
            print(f"❌ Model not loaded for user {user_id}")
            # Fallback: return popular movies when model is not available
            return self._get_popular_movies(n)

        try:
            # Always merge latest user ratings before generating recommendations
            if self.user_service is not None:
                current_data = self._merge_user_ratings(self.data.copy() if self.data is not None else None)
                if current_data is not None:
                    data_changed = self.data is None or not current_data.equals(self.data)
                    if data_changed:
                        print(f"✓ Data updated for user {user_id}, creating new recommender")
                        # Data has changed, create new recommender with updated data
                        self.data = current_data
                        self.recommender = MovieRecommender(self.model, self.data)
                        print(f"✓ New recommender created with {len(self.data)} total ratings")
                    else:
                        print(f"✓ Data unchanged for user {user_id}")
                else:
                    print(f"❌ Failed to merge data for user {user_id}")

            if self.recommender is None:
                print(f"❌ Recommender is None for user {user_id}")
                return self._get_popular_movies(n)

            print(f"✓ Getting recommendations for user {user_id}")
            return self.recommender.recommend(user_id, n)
        except Exception as e:
            print(f"❌ Error getting recommendations for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to popular movies
            return self._get_popular_movies(n)

    def get_user_history(self, user_id: int) -> List[Dict]:
        """
        Get user's rating history

        Args:
            user_id: User ID

        Returns:
            List of user's ratings
        """
        if self.recommender is None:
            raise ValueError("Model not loaded. Train the model first.")

        return self.recommender.get_user_history(user_id)

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "no_model", "message": "No model loaded"}

        return {
            "status": "loaded",
            "n_factors": self.model.n_factors,
            "reg_lambda": self.model.reg_lambda,
            "content_weight": self.model.content_weight,
            "cf_weight": self.model.cf_weight,
            "global_mean": float(self.model.global_mean),
            "n_users": len(self.model.user_bias) if self.model.user_bias is not None else 0,
            "n_items": len(self.model.item_bias) if self.model.item_bias is not None else 0
        }

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model is not None

    def _get_popular_movies(self, n: int = 10) -> List[Dict]:
        """Get popular movies as fallback when model is not available"""
        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path

            # Add project root to path if not already there
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from data.dataset_setup import DatasetSetup
            from data.data_loader import MovieLensLoader

            # Load basic data if not already loaded
            if self.data is None:
                setup = DatasetSetup()
                if not setup.download_and_extract():
                    return self._get_default_popular_movies(n)

                loader = MovieLensLoader(setup.data_dir)
                self.data = loader.load_and_enrich(
                    min_user_ratings=settings.MIN_USER_RATINGS,
                    min_movie_ratings=settings.MIN_MOVIE_RATINGS
                )

            # Get most rated movies
            popular_movies = self.data.groupby('movie_id').agg({
                'rating': ['count', 'mean'],
                'movie_title': 'first',
                'genres': 'first'
            }).reset_index()

            popular_movies.columns = ['movie_id', 'rating_count', 'avg_rating', 'movie_title', 'genres']
            popular_movies = popular_movies.sort_values(['rating_count', 'avg_rating'], ascending=False)

            recommendations = []
            for _, row in popular_movies.head(n).iterrows():
                recommendations.append({
                    'movie_id': int(row['movie_id']),
                    'movie_title': row['movie_title'],
                    'genres': row['genres'],
                    'predicted_rating': float(row['avg_rating']),
                    'reason': 'Popular movie (model not available)'
                })

            return recommendations

        except Exception as e:
            print(f"❌ Error getting popular movies: {e}")
            return self._get_default_popular_movies(n)

    def _get_default_popular_movies(self, n: int = 10) -> List[Dict]:
        """Return some default popular movies when data loading fails"""
        default_movies = [
            {'movie_id': 50, 'movie_title': 'Star Wars (1977)', 'genres': 'Action|Adventure|Romance|Sci-Fi|War', 'predicted_rating': 4.36, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 258, 'movie_title': 'Contact (1997)', 'genres': 'Drama|Sci-Fi', 'predicted_rating': 3.80, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 100, 'movie_title': 'Fargo (1996)', 'genres': 'Crime|Drama|Thriller', 'predicted_rating': 4.15, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 181, 'movie_title': 'Return of the Jedi (1983)', 'genres': 'Action|Adventure|Romance|Sci-Fi|War', 'predicted_rating': 4.01, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 294, 'movie_title': 'Liar Liar (1997)', 'genres': 'Comedy', 'predicted_rating': 3.16, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 286, 'movie_title': 'English Patient, The (1996)', 'genres': 'Drama|Romance|War', 'predicted_rating': 3.66, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 288, 'movie_title': 'Scream (1996)', 'genres': 'Horror|Thriller', 'predicted_rating': 3.44, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 1, 'movie_title': 'Toy Story (1995)', 'genres': 'Animation|Children\'s|Comedy', 'predicted_rating': 3.88, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 300, 'movie_title': 'Air Force One (1997)', 'genres': 'Action|Thriller', 'predicted_rating': 3.63, 'reason': 'Popular movie (fallback)'},
            {'movie_id': 121, 'movie_title': 'Independence Day (ID4) (1996)', 'genres': 'Action|Sci-Fi|War', 'predicted_rating': 3.74, 'reason': 'Popular movie (fallback)'}
        ]
        return default_movies[:n]

    def set_user_service(self, user_service):
        """Set the user service for accessing user ratings"""
        self.user_service = user_service

    def _merge_user_ratings(self, base_data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Merge user ratings with base dataset

        Args:
            base_data: Original MovieLens data (if None, loads from disk)

        Returns:
            DataFrame with user ratings merged
        """
        if self.user_service is None:
            return base_data

        # Load base data if not provided
        if base_data is None:
            try:
                import sys
                from pathlib import Path

                # Add project root to path if not already there
                project_root = Path(__file__).parent.parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))

                from data.dataset_setup import DatasetSetup
                from data.data_loader import MovieLensLoader

                setup = DatasetSetup()
                if not setup.download_and_extract():
                    return None

                loader = MovieLensLoader(setup.data_dir)
                base_data = loader.load_and_enrich(
                    min_user_ratings=settings.MIN_USER_RATINGS,
                    min_movie_ratings=settings.MIN_MOVIE_RATINGS
                )
            except Exception as e:
                print(f"❌ Error loading base data: {e}")
                return None

        # Get user ratings as DataFrame
        user_ratings_df = self.user_service.get_all_ratings_df()

        if user_ratings_df.empty:
            return base_data

        # Prepare user ratings to match base data format
        # We need to add movie metadata to user ratings
        movies_df = base_data[['movieId', 'title', 'genres', 'primary_genre']].drop_duplicates()

        # Merge user ratings with movie data
        user_ratings_enriched = user_ratings_df.merge(
            movies_df,
            left_on='movie_id',
            right_on='movieId',
            how='left'
        )

        # Filter out ratings for movies not in the base dataset
        user_ratings_enriched = user_ratings_enriched.dropna(subset=['title'])

        if user_ratings_enriched.empty:
            return base_data

        # Add required columns to match base data format
        user_ratings_enriched['userId'] = user_ratings_enriched['user_id']
        user_ratings_enriched['movie_title'] = user_ratings_enriched['title']
        user_ratings_enriched['rating'] = user_ratings_enriched['rating'].astype(float)

        # Create indices for new users (starting from max existing + 1)
        max_user_idx = base_data['user_idx'].max()
        user_ratings_enriched['user_idx'] = pd.Categorical(user_ratings_enriched['user_id']).codes + max_user_idx + 1

        # Create indices for items (reuse existing item indices where possible)
        item_mapping = base_data[['movieId', 'item_idx']].drop_duplicates().set_index('movieId')['item_idx'].to_dict()
        user_ratings_enriched['item_idx'] = user_ratings_enriched['movieId'].map(item_mapping)

        # Filter out movies not in the base dataset
        user_ratings_enriched = user_ratings_enriched.dropna(subset=['item_idx'])
        user_ratings_enriched['item_idx'] = user_ratings_enriched['item_idx'].astype(int)

        # Select only the columns we need
        columns_to_keep = ['userId', 'movieId', 'rating', 'timestamp', 'movie_title', 'genres',
                          'primary_genre', 'user_idx', 'item_idx']

        # Add missing columns if they don't exist
        for col in ['genres_list', 'num_genres']:
            if col in base_data.columns:
                if col == 'genres_list':
                    user_ratings_enriched[col] = user_ratings_enriched['genres'].str.split('|')
                elif col == 'num_genres':
                    user_ratings_enriched[col] = user_ratings_enriched['genres_list'].str.len()

        user_ratings_final = user_ratings_enriched[columns_to_keep]

        # Combine with base data
        combined_data = pd.concat([base_data, user_ratings_final], ignore_index=True)

        print(f"✓ Merged {len(user_ratings_final)} user ratings with base dataset")

        return combined_data