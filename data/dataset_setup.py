"""
Dataset Setup Module
Handles automatic downloading and extraction of MovieLens dataset
"""

import os
import urllib.request
import zipfile


class DatasetSetup:
    """Automatically downloads and prepares MovieLens dataset"""

    def __init__(self, data_dir: str = "./ml-latest-small"):
        self.data_dir = data_dir
        self.url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        self.zip_path = "ml-latest-small.zip"

    def download_and_extract(self):
        """Download and extract MovieLens dataset"""

        # Check if already downloaded
        if os.path.exists(self.data_dir):
            print(f"✓ Dataset already exists at {self.data_dir}")
            return True

        print("\n" + "="*70)
        print("DOWNLOADING MOVIELENS DATASET")
        print("="*70)
        print(f"Source: {self.url}")
        print(f"Size: ~1 MB (100K ratings)")

        try:
            # Download with progress
            print("Downloading...", end="")
            urllib.request.urlretrieve(self.url, self.zip_path)
            print(" ✓ Complete")

            # Extract
            print("Extracting...", end="")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            print(" ✓ Complete")

            # Cleanup
            os.remove(self.zip_path)
            print(f"✓ Dataset ready at {self.data_dir}")
            return True

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nManual download instructions:")
            print(f"1. Visit: {self.url}")
            print(f"2. Extract to: {self.data_dir}")
            return False