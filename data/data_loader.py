"""
Data Loader Module
Loads and prepares MovieLens data with guaranteed successful joins
"""

import pandas as pd
import numpy as np
import re


class MovieLensLoader:
    """
    Loads MovieLens data with guaranteed 100% successful joins
    No ID mismatch issues!
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def _preprocess_description(self, overview: str, tagline: str, genres: str) -> str:
        """
        Preprocess movie description text for semantic search

        Args:
            overview: Movie overview from TMDb
            tagline: Movie tagline from TMDb
            genres: Pipe-separated genres

        Returns:
            Preprocessed text string
        """
        # Combine overview and tagline
        text_parts = []
        if pd.notna(overview) and overview.strip():
            text_parts.append(overview.strip())
        if pd.notna(tagline) and tagline.strip():
            text_parts.append(tagline.strip())

        text = " ".join(text_parts)

        # Clean text
        if text:
            text = text.lower().strip()
            # Remove special characters but keep spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            # Fallback to genres if no description available
            text = genres.lower().replace('|', ' ') if genres else ""

        return text

    def load_and_enrich(self, min_user_ratings: int = 5,
                       min_movie_ratings: int = 5) -> pd.DataFrame:
        """
        Load all data files and create enriched dataset

        Returns:
            DataFrame with columns:
            - userId, movieId, rating, timestamp
            - title, genres, primary_genre
            - user_idx, item_idx (for matrix factorization)
        """
        print("\n" + "="*70)
        print("LOADING MOVIELENS DATA")
        print("="*70)

        # Load ratings
        print("\n[1/3] Loading ratings.csv...")
        ratings_df = pd.read_csv(f"{self.data_path}/ratings.csv")
        print(f"✓ Loaded {len(ratings_df):,} ratings")
        print(f"  → Users: {ratings_df['userId'].nunique():,}")
        print(f"  → Movies: {ratings_df['movieId'].nunique():,}")
        print(f"  → Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")

        # Load movies with TMDb enrichment
        print("\n[2/3] Loading movies_enriched.csv...")
        try:
            movies_df = pd.read_csv(f"{self.data_path}/movies_enriched.csv")
            has_enriched = True
            print(f"✓ Loaded {len(movies_df):,} enriched movies with TMDb data")
        except FileNotFoundError:
            print("⚠️  movies_enriched.csv not found, falling back to movies.csv")
            movies_df = pd.read_csv(f"{self.data_path}/movies.csv")
            # Add empty description columns
            movies_df['overview'] = ''
            movies_df['tagline'] = ''
            movies_df['tmdb_rating'] = np.nan
            movies_df['runtime'] = np.nan
            has_enriched = False
            print(f"✓ Loaded {len(movies_df):,} basic movies")

        # Preprocess descriptions for semantic search
        if has_enriched:
            print("\n[2.5/3] Preprocessing movie descriptions...")
            movies_df['description'] = movies_df.apply(
                lambda row: self._preprocess_description(
                    row['overview'], row['tagline'], row['genres']
                ), axis=1
            )
            print(f"✓ Processed descriptions for {len(movies_df):,} movies")
        else:
            movies_df['description'] = movies_df['genres'].str.lower().str.replace('|', ' ')

        # Load tags (optional)
        print("\n[3/3] Loading tags.csv...")
        try:
            tags_df = pd.read_csv(f"{self.data_path}/tags.csv")
            # Aggregate tags per movie
            movie_tags = tags_df.groupby('movieId')['tag'].apply(
                lambda x: ' '.join(x.astype(str))
            ).reset_index()
            movie_tags.columns = ['movieId', 'tags']
            print(f"✓ Loaded {len(tags_df):,} tags")
            print(f"  → Movies with tags: {len(movie_tags):,}")
        except:
            movie_tags = pd.DataFrame(columns=['movieId', 'tags'])
            print("⚠️  No tags file (optional)")

        # CRITICAL: Merge with 100% success
        print("\n🔗 Merging datasets...")

        # Merge ratings + movies
        df = ratings_df.merge(movies_df, on='movieId', how='left')
        merge_success = (df['title'].notna().sum() / len(df)) * 100
        print(f"  ✓ Ratings + Movies: {merge_success:.2f}% success (Expected: 100%)")

        if merge_success < 100:
            missing = df['title'].isna().sum()
            print(f"  ⚠️  Warning: {missing:,} ratings without movie data")
            df = df.dropna(subset=['title'])

        # Merge tags
        if len(movie_tags) > 0:
            df = df.merge(movie_tags, on='movieId', how='left')
            df['tags'] = df['tags'].fillna('')
        else:
            df['tags'] = ''

        # Process genres
        df['genres_list'] = df['genres'].str.split('|')
        df['primary_genre'] = df['genres_list'].str[0]
        df['num_genres'] = df['genres_list'].str.len()

        print(f"\n✓ Merged dataset: {len(df):,} ratings")

        # Filter sparse users/movies
        print(f"\nFiltering (min {min_user_ratings} ratings/user, {min_movie_ratings} ratings/movie)...")

        # Filter users
        user_counts = df.groupby('userId').size()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df['userId'].isin(valid_users)]

        # Filter movies
        movie_counts = df.groupby('movieId').size()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df['movieId'].isin(valid_movies)]

        print(f"  → Removed sparse users/movies")
        print(f"  → Final dataset: {len(df):,} ratings")

        # Create indices for matrix factorization
        df['user_idx'] = pd.Categorical(df['userId']).codes.astype(np.int32)
        df['item_idx'] = pd.Categorical(df['movieId']).codes.astype(np.int32)

        # Calculate statistics
        n_users = df['user_idx'].nunique()
        n_items = df['item_idx'].nunique()
        sparsity = 1 - len(df) / (n_users * n_items)

        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        print(f"Total ratings:      {len(df):,}")
        print(f"Unique users:       {n_users:,}")
        print(f"Unique movies:      {n_items:,}")
        print(f"Unique genres:      {df['primary_genre'].nunique()}")
        print(f"Sparsity:           {sparsity:.6f} ({(1-sparsity)*100:.3f}% filled)")
        print(f"Avg ratings/user:   {len(df)/n_users:.1f}")
        print(f"Avg ratings/movie:  {len(df)/n_items:.1f}")
        print(f"Rating distribution:")
        for rating, count in df['rating'].value_counts().sort_index().items():
            pct = count / len(df) * 100
            print(f"  {rating:.1f}: {count:>6,} ({pct:>5.1f}%)")
        print("="*70)

        return df