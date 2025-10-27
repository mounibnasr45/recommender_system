"""
Movie Recommender Interface
User-friendly interface for getting movie recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict
from models.hybrid_mf import HybridMatrixFactorization


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

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Get top N recommendations for a user

        Args:
            user_id: User ID (from original dataset)
            n: Number of recommendations

        Returns:
            DataFrame with recommendations
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
                'movieId': info['movieId'],
                'title': info['title'],
                'genres': info['genres'],
                'predicted_rating': round(predicted_rating, 2),
                'avg_rating': round(info['rating'], 2)
            })

        return pd.DataFrame(results)

    def _popular_recommendations(self, n: int) -> pd.DataFrame:
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
                'movieId': row[('movieId', 'first')],
                'title': row[('title', 'first')],
                'genres': row[('genres', 'first')],
                'predicted_rating': round(row[('rating', 'mean')], 2),
                'avg_rating': round(row[('rating', 'mean')], 2)
            })

        return pd.DataFrame(results)

    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """Get user's rating history"""
        return self.data[self.data['userId'] == user_id][
            ['title', 'genres', 'rating', 'primary_genre']
        ].sort_values('rating', ascending=False)