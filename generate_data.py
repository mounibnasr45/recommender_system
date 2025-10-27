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

        # Save processed data
        data_dir = Path(settings.MODEL_PATH).parent
        data_dir.mkdir(exist_ok=True)

        data_path = data_dir / "processed_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"✅ Processed data saved to {data_path}")
        print(f"   → {len(data):,} ratings")
        print(f"   → {data['user_idx'].nunique():,} users")
        print(f"   → {data['item_idx'].nunique():,} movies")

        return True

    except Exception as e:
        print(f"❌ Error generating processed data: {e}")
        return False

if __name__ == "__main__":
    success = generate_processed_data()
    sys.exit(0 if success else 1)