"""
Hybrid Matrix Factorization Model
Combines collaborative filtering with content-based features
"""

import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Set, Optional


class HybridMatrixFactorization:
    """
    Hybrid Recommender: Collaborative Filtering + Content-Based

    Features:
    - Matrix factorization for collaborative signals
    - Genre similarity for content-based signals
    - Weighted hybrid predictions
    - Early stopping to prevent overfitting
    """

    def __init__(self, n_factors: int = 20, n_iterations: int = 50,
                 reg_lambda: float = 0.1, learning_rate: float = 0.01,
                 content_weight: float = 0.15, early_stopping: bool = True,
                 patience: int = 10):

        # Hyperparameters
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.content_weight = content_weight
        self.cf_weight = 1 - content_weight
        self.early_stopping = early_stopping
        self.patience = patience

        # Model parameters
        self.P = None  # User latent factors
        self.Q = None  # Item latent factors
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0

        # Content features
        self.item_to_genre = {}
        self.item_to_genres = {}

        # Training history
        self.train_errors = []
        self.val_errors = []

        # Best model (for early stopping)
        self.best_state = None
        self.best_val_rmse = float('inf')
        self.best_epoch = 0

    def build_content_features(self, data):
        """Build genre-based content similarity"""
        print("\n--- Building Content Features ---")

        # Map item to primary genre
        self.item_to_genre = data.groupby('item_idx')['primary_genre'].first().to_dict()

        # Map item to all genres (for richer similarity)
        self.item_to_genres = data.groupby('item_idx')['genres_list'].first().to_dict()

        n_genres = data['primary_genre'].nunique()
        n_items_with_genre = len(self.item_to_genre)

        print(f"✓ Unique genres: {n_genres}")
        print(f"✓ Items with genre info: {n_items_with_genre:,}")

    def calculate_genre_similarity(self, item_idx1: int, item_idx2: int) -> float:
        """
        Calculate genre similarity between two items

        Uses Jaccard similarity on genre sets:
        similarity = |intersection| / |union|
        """
        genres1 = set(self.item_to_genres.get(item_idx1, []))
        genres2 = set(self.item_to_genres.get(item_idx2, []))

        if not genres1 or not genres2:
            return 0.0

        intersection = len(genres1 & genres2)
        union = len(genres1 | genres2)

        return intersection / union if union > 0 else 0.0

    def initialize_parameters(self, n_users: int, n_items: int):
        """Initialize model parameters with small random values"""
        print("\n--- Initializing Model Parameters ---")

        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        self.user_bias = np.zeros(n_users, dtype=np.float32)
        self.item_bias = np.zeros(n_items, dtype=np.float32)

        print(f"✓ User factors (P): {self.P.shape}")
        print(f"✓ Item factors (Q): {self.Q.shape}")
        total_params = int(self.P.size) + int(self.Q.size) + int(n_users) + int(n_items)
        print(f"✓ Parameters: {total_params:,}")

    def predict_cf(self, user_idx: int, item_idx: int) -> float:
        """Collaborative filtering prediction only"""
        if user_idx >= len(self.user_bias) or item_idx >= len(self.item_bias):
            return self.global_mean

        prediction = (
            self.global_mean +
            self.user_bias[user_idx] +
            self.item_bias[item_idx] +
            np.dot(self.P[user_idx], self.Q[item_idx])
        )

        return prediction

    def predict_content(self, user_idx: int, item_idx: int,
                       user_items: np.ndarray, user_ratings: np.ndarray) -> float:
        """Content-based prediction using genre similarity"""
        if len(user_items) == 0:
            return self.global_mean

        # Calculate similarity with all items user has rated
        similarities = np.array([
            self.calculate_genre_similarity(item_idx, int(rated_item))
            for rated_item in user_items
        ])

        # Weighted average of ratings by similarity
        if similarities.sum() > 0:
            weights = similarities / similarities.sum()
            prediction = np.dot(weights, user_ratings)
            return prediction

        return self.global_mean

    def predict_hybrid(self, user_idx: int, item_idx: int,
                      user_items: np.ndarray = None,
                      user_ratings: np.ndarray = None) -> float:
        """
        Hybrid prediction combining CF and content-based

        Final prediction = cf_weight * CF + content_weight * CB
        """
        # Collaborative filtering component
        cf_prediction = self.predict_cf(user_idx, item_idx)

        # Content-based component (if user history available)
        if user_items is not None and len(user_items) > 0:
            cb_prediction = self.predict_content(
                user_idx, item_idx, user_items, user_ratings
            )
            # Weighted combination
            prediction = (
                self.cf_weight * cf_prediction +
                self.content_weight * cb_prediction
            )
        else:
            prediction = cf_prediction

        # Clip to valid rating range
        return np.clip(prediction, 0.5, 5.0)

    def save_checkpoint(self):
        """Save current model state"""
        self.best_state = {
            'P': self.P.copy(),
            'Q': self.Q.copy(),
            'user_bias': self.user_bias.copy(),
            'item_bias': self.item_bias.copy()
        }

    def load_checkpoint(self):
        """Restore best model state"""
        if self.best_state is not None:
            self.P = self.best_state['P']
            self.Q = self.best_state['Q']
            self.user_bias = self.best_state['user_bias']
            self.item_bias = self.best_state['item_bias']

    def fit(self, train_data, val_data=None):
        """
        Train the hybrid model using SGD

        Args:
            train_data: Training ratings
            val_data: Validation ratings (for early stopping)
        """

        # Build content features
        self.build_content_features(train_data)

        # Extract training data
        users = train_data['user_idx'].values.astype(np.int32)
        items = train_data['item_idx'].values.astype(np.int32)
        ratings = train_data['rating'].values.astype(np.float32)

        # Determine matrix dimensions
        n_users = int(max(
            train_data['user_idx'].max() + 1,
            val_data['user_idx'].max() + 1 if val_data is not None else 0
        ))
        n_items = int(max(
            train_data['item_idx'].max() + 1,
            val_data['item_idx'].max() + 1 if val_data is not None else 0
        ))

        # Initialize
        self.global_mean = ratings.mean()
        self.initialize_parameters(n_users, n_items)

        # Build user profiles for predictions
        print("Building user profiles...")
        user_profiles = {}
        for user_idx in np.unique(users):
            mask = users == user_idx
            user_profiles[int(user_idx)] = (
                items[mask].astype(np.int32),
                ratings[mask].astype(np.float32)
            )
        print(f"✓ Built profiles for {len(user_profiles):,} users")

        # Training loop
        print("\n" + "="*70)
        print("TRAINING HYBRID MODEL")
        print("="*70)
        print(f"Architecture: {n_users:,} users × {n_items:,} items × {self.n_factors} factors")
        print(f"Training samples: {len(train_data):,}")
        if val_data is not None:
            print(f"Validation samples: {len(val_data):,}")
        print(f"Hybrid weights: CF={self.cf_weight:.0%}, Content={self.content_weight:.0%}")
        print(f"Regularization: λ={self.reg_lambda}")
        print(f"Learning rate: α={self.learning_rate}")
        print("="*70)

        patience_counter = 0

        for epoch in range(self.n_iterations):
            # Shuffle training data
            indices = np.random.permutation(len(users))

            # SGD updates
            for idx in indices:
                u, i, r = users[idx], items[idx], ratings[idx]

                # Predict and calculate error
                user_items, user_ratings_hist = user_profiles.get(u, (np.array([]), np.array([])))
                prediction = self.predict_hybrid(u, i, user_items, user_ratings_hist)
                error = r - prediction

                # Update biases
                self.user_bias[u] += self.learning_rate * (error - self.reg_lambda * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (error - self.reg_lambda * self.item_bias[i])

                # Update latent factors
                self.P[u, :] += self.learning_rate * (error * self.Q[i, :] - self.reg_lambda * self.P[u, :])
                self.Q[i, :] += self.learning_rate * (error * self.P[u, :] - self.reg_lambda * self.Q[i, :])

            # Evaluate training performance
            train_predictions = []
            for u, i in zip(users, items):
                user_items, user_ratings = user_profiles.get(u, (np.array([]), np.array([])))
                pred = self.predict_hybrid(u, i, user_items, user_ratings)
                train_predictions.append(pred)

            train_rmse = np.sqrt(np.mean((ratings - train_predictions) ** 2))
            self.train_errors.append(train_rmse)

            # Evaluate validation performance
            if val_data is not None:
                val_users = val_data['user_idx'].values.astype(np.int32)
                val_items = val_data['item_idx'].values.astype(np.int32)
                val_ratings = val_data['rating'].values.astype(np.float32)

                val_predictions = []
                for u, i in zip(val_users, val_items):
                    user_items, user_ratings = user_profiles.get(u, (np.array([]), np.array([])))
                    pred = self.predict_hybrid(u, i, user_items, user_ratings)
                    val_predictions.append(pred)

                val_rmse = np.sqrt(np.mean((val_ratings - val_predictions) ** 2))
                self.val_errors.append(val_rmse)

                # Print progress
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    gap = val_rmse - train_rmse
                    status = "✓" if val_rmse < self.best_val_rmse else "✗"
                    print(f"Epoch {epoch+1:3d}/{self.n_iterations} | "
                          f"Train: {train_rmse:.4f} | Val: {val_rmse:.4f} | "
                          f"Gap: {gap:+.4f} {status}")

                # Early stopping
                if self.early_stopping:
                    if val_rmse < self.best_val_rmse:
                        self.best_val_rmse = val_rmse
                        self.best_epoch = epoch
                        patience_counter = 0
                        self.save_checkpoint()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print("\n" + "─"*70)
                            print(f"⏹  Early stopping at epoch {epoch+1}")
                            print(f"   Best validation RMSE: {self.best_val_rmse:.4f} (epoch {self.best_epoch+1})")
                            print("─"*70)
                            self.load_checkpoint()
                            break
            else:
                # No validation set - just print training progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_iterations} | Train RMSE: {train_rmse:.4f}")


        print("\n✓ Training completed!")
        if val_data is not None and self.early_stopping:
            print(f"✓ Best model from epoch {self.best_epoch+1} (Val RMSE: {self.best_val_rmse:.4f})")


    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        model_data = {
            'P': self.P,
            'Q': self.Q,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'global_mean': self.global_mean,
            'item_to_genre': self.item_to_genre,
            'item_to_genres': self.item_to_genres,
            'n_factors': self.n_factors,
            'reg_lambda': self.reg_lambda,
            'content_weight': self.content_weight,
            'cf_weight': self.cf_weight
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'HybridMatrixFactorization':
        """Load a trained model from disk"""
        import sys
        # Temporarily make the class available in __main__ for pickle compatibility
        if not hasattr(sys.modules['__main__'], 'HybridMatrixFactorization'):
            sys.modules['__main__'].HybridMatrixFactorization = cls

        try:
            # Try joblib first (for newer models), then pickle
            try:
                import joblib
                loaded = joblib.load(filepath)
            except:
                with open(filepath, 'rb') as f:
                    loaded = pickle.load(f)

            # Check if it's a dict (new format) or an instance (old format)
            if isinstance(loaded, dict):
                # New format: dict of parameters
                model_data = loaded
            else:
                # Old format: pickled instance
                instance = loaded
                # Extract parameters from the instance
                model_data = {
                    'P': instance.P,
                    'Q': instance.Q,
                    'user_bias': instance.user_bias,
                    'item_bias': instance.item_bias,
                    'global_mean': instance.global_mean,
                    'item_to_genre': instance.item_to_genre,
                    'item_to_genres': instance.item_to_genres,
                    'n_factors': instance.n_factors,
                    'reg_lambda': instance.reg_lambda,
                    'content_weight': instance.content_weight,
                    'cf_weight': instance.cf_weight
                }

            # Create instance with available parameters
            instance = cls(
                n_factors=model_data.get('n_factors', 20),
                reg_lambda=model_data.get('reg_lambda', 0.1),
                content_weight=model_data.get('content_weight', 0.15)
            )

            # Restore parameters
            instance.P = model_data.get('P')
            instance.Q = model_data.get('Q')
            instance.user_bias = model_data.get('user_bias')
            instance.item_bias = model_data.get('item_bias')
            instance.global_mean = model_data.get('global_mean', 0)
            instance.item_to_genre = model_data.get('item_to_genre', {})
            instance.item_to_genres = model_data.get('item_to_genres', {})
            instance.content_weight = model_data.get('content_weight', 0.15)
            instance.cf_weight = model_data.get('cf_weight', 0.85)

            print(f"✓ Model loaded from {filepath}")
            return instance

        finally:
            # Clean up
            if hasattr(sys.modules['__main__'], 'HybridMatrixFactorization'):
                delattr(sys.modules['__main__'], 'HybridMatrixFactorization')

    def get_top_recommendations(self, user_idx: int, n: int,
                               user_items: np.ndarray, user_ratings: np.ndarray,
                               rated_items: Set[int]) -> List[Tuple[int, float]]:
        """
        Get top N movie recommendations for a user

        Args:
            user_idx: User index
            n: Number of recommendations
            user_items: Items user has rated
            user_ratings: Ratings user has given
            rated_items: Set of already rated items (to exclude)

        Returns:
            List of (item_idx, predicted_rating) tuples
        """
        n_items = self.Q.shape[0]
        predictions = []

        for item_idx in range(n_items):
            if item_idx in rated_items:
                continue

            pred = self.predict_hybrid(user_idx, item_idx, user_items, user_ratings)
            predictions.append((item_idx, pred))

        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]


class MovieRecommender:
    """User-friendly interface for getting recommendations"""

    def __init__(self, model: HybridMatrixFactorization, data: pd.DataFrame):
        self.model = model
        self.data = data

        # Create mappings
        self.user_id_to_idx = dict(zip(data['userId'], data['user_idx']))
        self.idx_to_movie_id = dict(zip(data['item_idx'], data['movieId']))

        # Movie information
        self.movie_info = data.groupby('item_idx').agg({
            'movieId': 'first',
            'title': 'first',
            'genres': 'first',
            'primary_genre': 'first',
            'rating': 'mean'
        }).to_dict('index')

        # Build user profiles
        self.user_profiles = {}
        for user_idx in data['user_idx'].unique():
            user_data = data[data['user_idx'] == user_idx]
            self.user_profiles[user_idx] = (
                user_data['item_idx'].values,
                user_data['rating'].values
            )

    def recommend(self, user_id: int, n: int = 10) -> List[Dict]:
        """
        Get top N recommendations for a user

        Args:
            user_id: User ID (from original dataset)
            n: Number of recommendations

        Returns:
            List of recommendation dictionaries
        """
        if user_id not in self.user_id_to_idx:
            return self._popular_recommendations(n)

        user_idx = self.user_id_to_idx[user_id]
        rated_items = set(self.data[self.data['userId'] == user_id]['item_idx'])
        user_items, user_ratings = self.user_profiles.get(
            user_idx,
            (np.array([]), np.array([]))
        )

        recommendations = self.model.get_top_recommendations(
            user_idx, n, user_items, user_ratings, rated_items
        )

        results = []
        for item_idx, predicted_rating in recommendations:
            info = self.movie_info[item_idx]
            results.append({
                'movieId': int(info['movieId']),
                'title': info['title'],
                'genres': info['genres'],
                'predicted_rating': round(float(predicted_rating), 2),
                'avg_rating': round(float(info['rating']), 2)
            })

        return results

    def _popular_recommendations(self, n: int) -> List[Dict]:
        """Fallback: return most popular movies (cold start)"""
        popular = self.data.groupby('item_idx').agg({
            'rating': ['mean', 'count'],
            'movieId': 'first',
            'title': 'first',
            'genres': 'first'
        })

        # Weight by rating and popularity
        popular['score'] = popular[('rating', 'mean')] * np.log1p(popular[('rating', 'count')])
        popular = popular.sort_values('score', ascending=False).head(n)

        results = []
        for _, row in popular.iterrows():
            results.append({
                'movieId': int(row[('movieId', 'first')]),
                'title': row[('title', 'first')],
                'genres': row[('genres', 'first')],
                'predicted_rating': round(float(row[('rating', 'mean')]), 2),
                'avg_rating': round(float(row[('rating', 'mean')]), 2)
            })

        return results

    def get_user_history(self, user_id: int) -> List[Dict]:
        """Get user's rating history"""
        user_data = self.data[self.data['userId'] == user_id]
        return user_data[['title', 'genres', 'rating', 'primary_genre']].to_dict('records')