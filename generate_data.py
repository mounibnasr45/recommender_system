#!/usr/bin/env python3
"""
Generate processed data file for the recommender system
"""

import os
import pickle
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.dataset_setup import DatasetSetup
from data.data_loader import MovieLensLoader
from backend.config import settings

def generate_processed_data():
    """Generate and save the processed data file"""
    print("🔄 Generating processed data file...")

    try:
        # Setup data directory
        setup = DatasetSetup()
        if not setup.download_and_extract():
            print("❌ Failed to download dataset")
            return False

        # Load and enrich data
        loader = MovieLensLoader(setup.data_dir)
        data = loader.load_and_enrich(
            min_user_ratings=settings.MIN_USER_RATINGS,
            min_movie_ratings=settings.MIN_MOVIE_RATINGS
        )

        # Extract movie metadata for semantic search
        movie_metadata = data[['movieId', 'title', 'genres', 'description']].drop_duplicates()
        movie_metadata = movie_metadata.set_index('movieId')

        # Generate embeddings for semantic search
        if settings.USE_SEMANTIC_SEARCH:
            print("\n🔄 Generating description embeddings...")
            try:
                from backend.models.hybrid_mf import SemanticEncoder

                encoder = SemanticEncoder(model_name=settings.EMBEDDING_MODEL)
                embeddings = encoder.encode_descriptions(movie_metadata['description'].tolist())

                # Save embeddings
                embeddings_path = Path(settings.EMBEDDING_CACHE_PATH)
                encoder.save_embeddings(str(embeddings_path), movie_metadata.index.tolist())

                print(f"✅ Embeddings saved to {embeddings_path}")
                print(f"   → {len(embeddings)} embeddings generated")
                print(f"   → Dimension: {embeddings.shape[1]}")

            except Exception as e:
                print(f"⚠️  Failed to generate embeddings: {e}")
                print("   Semantic search will not be available")
        else:
            print("\n⏭️  Skipping embedding generation (USE_SEMANTIC_SEARCH=False)")

        # Save processed data
        data_dir = Path(settings.MODEL_PATH).parent
        data_dir.mkdir(exist_ok=True)

        data_path = data_dir / "processed_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        # Save movie metadata
        metadata_path = data_dir / "movie_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(movie_metadata, f)

        print(f"✅ Processed data saved to {data_path}")
        print(f"   → {len(data):,} ratings")
        print(f"   → {data['user_idx'].nunique():,} users")
        print(f"   → {data['item_idx'].nunique():,} movies")
        print(f"✅ Movie metadata saved to {metadata_path}")
        print(f"   → {len(movie_metadata):,} movies with descriptions")

        return True

    except Exception as e:
        print(f"❌ Error generating processed data: {e}")
        return False

if __name__ == "__main__":
    success = generate_processed_data()
    sys.exit(0 if success else 1)