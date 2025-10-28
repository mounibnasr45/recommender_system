"""
Semantic Search Service
Handles semantic search queries using movie description embeddings
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearchService:
    """
    Service for performing semantic search on movie descriptions

    Uses sentence transformers to encode queries and find semantically
    similar movies based on their descriptions (overview + tagline).
    """

    def __init__(self, encoder: SentenceTransformer, movie_embeddings: np.ndarray,
                 movie_ids: List[int], descriptions: List[str]):
        """
        Initialize the semantic search service

        Args:
            encoder: Pre-trained sentence transformer model
            movie_embeddings: Pre-computed embeddings for all movies
            movie_ids: List of movie IDs corresponding to embeddings
            descriptions: List of movie descriptions
        """
        self.encoder = encoder
        if movie_embeddings is None:
            raise ValueError("movie_embeddings is None. Ensure embeddings.pkl contains the embeddings array.")

        self.embeddings = movie_embeddings
        self.movie_ids = movie_ids
        self.descriptions = descriptions

        print(f"✓ SemanticSearchService initialized")
        print(f"  → {len(movie_ids):,} movies indexed")
        print(f"  → Embedding dimension: {movie_embeddings.shape[1]}")

    def search(self, query_text: str, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for movies semantically similar to the query

        Args:
            query_text: Natural language description of desired movie
            n: Number of results to return

        Returns:
            Tuple of (movie_indices, similarity_scores)
        """
        # Encode the query
        print(f"[SemanticSearch] Encoding query: '{query_text}'")
        query_embedding = self.encoder.encode([query_text])

        # Calculate similarities with all movies
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()

        # Get top N most similar movies
        top_indices = np.argsort(similarities)[::-1][:n]
        top_similarities = similarities[top_indices]

        # Log top matches for debugging (show up to 5)
        top_preview = list(zip(top_indices[:5].tolist(), top_similarities[:5].tolist()))
        print(f"[SemanticSearch] Top {min(len(top_preview),5)} matches (idx,score): {top_preview}")

        return top_indices, top_similarities

    def find_similar_movies(self, movie_id: int, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find movies similar to a given movie based on description

        Args:
            movie_id: Movie ID to find similar movies for
            n: Number of similar movies to return

        Returns:
            Tuple of (movie_indices, similarity_scores)
        """
        # Find the index of the target movie
        try:
            movie_idx = self.movie_ids.index(movie_id)
        except ValueError:
            print(f"Warning: Movie ID {movie_id} not found in semantic index")
            return np.array([]), np.array([])

        # Get the movie's embedding
        movie_embedding = self.embeddings[movie_idx].reshape(1, -1)

        # Calculate similarities with all other movies
        similarities = cosine_similarity(movie_embedding, self.embeddings).flatten()

        # Exclude the movie itself and get top N
        similarities[movie_idx] = -1  # Set self-similarity to minimum
        top_indices = np.argsort(similarities)[::-1][:n]
        top_similarities = similarities[top_indices]

        return top_indices, top_similarities

    def get_movie_description(self, movie_idx: int) -> str:
        """Get the description of a movie by its index"""
        if 0 <= movie_idx < len(self.descriptions):
            return self.descriptions[movie_idx]
        return ""

    def get_movie_id(self, movie_idx: int) -> int:
        """Get the movie ID by its index"""
        if 0 <= movie_idx < len(self.movie_ids):
            return self.movie_ids[movie_idx]
        return -1

    @classmethod
    def from_embeddings_file(cls, embeddings_path: str, descriptions: List[str]):
        """
        Create service from saved embeddings file

        Args:
            embeddings_path: Path to saved embeddings pickle file
            descriptions: List of movie descriptions

        Returns:
            SemanticSearchService instance
        """
        import pickle

        # Load embeddings
        with open(embeddings_path, 'rb') as f:
            embedding_data = pickle.load(f)

        embeddings = embedding_data.get('embeddings')
        if embeddings is None:
            raise ValueError(f"Embeddings file {embeddings_path} does not contain 'embeddings' (None). Regenerate with generate_data.py")
        movie_ids = embedding_data['movie_ids']
        model_name = embedding_data.get('model_name', 'all-MiniLM-L6-v2')

        # Log loaded embeddings info
        try:
            print(f"[SemanticSearch] Loaded embeddings from {embeddings_path} | shape: {embeddings.shape} | movies: {len(movie_ids)} | model: {model_name}")
        except Exception:
            print(f"[SemanticSearch] Loaded embeddings from {embeddings_path} | movies: {len(movie_ids)} | model: {model_name}")

        # Load encoder
        encoder = SentenceTransformer(model_name)

        return cls(encoder, embeddings, movie_ids, descriptions)
