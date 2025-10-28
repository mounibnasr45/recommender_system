"""
TMDB Movie Description Fetcher
Enriches MovieLens dataset with movie descriptions from TMDB API
"""

import os
import pandas as pd
import requests
import time
import json
from typing import Dict, Optional, List
from pathlib import Path
import pickle

class TMDBDescriptionFetcher:
    """
    Fetches movie descriptions from TMDB API and caches them locally
    """
    
    def __init__(self, api_key: str = "3711aadbe3ebae11872b6c9725a83f4c", 
                 cache_dir: str = "./tmdb_cache"):
        """
        Initialize TMDB fetcher
        
        Args:
            api_key: Your TMDB API key (default provided)
            cache_dir: Directory to store cached descriptions
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.base_url = "https://api.themoviedb.org/3"
        self.cache_file = self.cache_dir / "descriptions_cache.pkl"
        self.image_base_url = "https://image.tmdb.org/t/p/"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        # Rate limiting
        self.request_delay = 60/35  # 35 requests per minute
        
    def _load_cache(self) -> Dict:
        """Load cached descriptions from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"✓ Loaded {len(cache)} cached descriptions")
            return cache
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def fetch_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """
        Fetch movie details from TMDB API
        
        Args:
            tmdb_id: TMDB movie ID
            
        Returns:
            Dictionary with movie details or None if failed
        """
        # Check cache first
        if tmdb_id in self.cache:
            return self.cache[tmdb_id]
        
        print(f"🔍 Fetching TMDB ID {tmdb_id}...")
        
        # Fetch from API
        url = f"{self.base_url}/movie/{tmdb_id}"
        params = {
            'api_key': self.api_key,
            'language': 'en-US'
        }
        
        try:
            print(f"🌐 Making API request to {url}")
            response = requests.get(url, params=params, timeout=30)  # Increased timeout
            print(f"📡 Response received: {response.status_code}")
            
            time.sleep(self.request_delay)  # Rate limiting
            print(f"⏳ Slept for {self.request_delay:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Successfully fetched data for TMDB ID {tmdb_id}")
                
                # Extract relevant fields
                movie_data = {
                    'tmdb_id': tmdb_id,
                    'overview': data.get('overview', ''),
                    'tagline': data.get('tagline', ''),
                    'runtime': data.get('runtime'),
                    'release_date': data.get('release_date'),
                    'vote_average': data.get('vote_average'),
                    'vote_count': data.get('vote_count'),
                    'popularity': data.get('popularity'),
                    'poster_path': data.get('poster_path'),
                    'backdrop_path': data.get('backdrop_path'),
                }
                
                # Cache the result
                self.cache[tmdb_id] = movie_data
                return movie_data
            
            elif response.status_code == 404:
                print(f"❌ Movie not found (404) for TMDB ID {tmdb_id}")
                # Movie not found - cache as None
                self.cache[tmdb_id] = None
                return None
            
            else:
                print(f"⚠️  Error {response.status_code} for TMDB ID {tmdb_id}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout error for TMDB ID {tmdb_id}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"🌐 Request error for TMDB ID {tmdb_id}: {e}")
            return None
        except Exception as e:
            print(f"💥 Exception for TMDB ID {tmdb_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def enrich_movielens_dataset(self, data_path: str = "./ml-latest-small") -> pd.DataFrame:
        """
        Enrich MovieLens dataset with TMDB descriptions
        
        Args:
            data_path: Path to MovieLens dataset
            
        Returns:
            DataFrame with added description columns
        """
        print("\n" + "="*70)
        print("ENRICHING MOVIELENS WITH TMDB DESCRIPTIONS")
        print("="*70)
        
        # Load MovieLens data
        print("\n[1/4] Loading MovieLens data...")
        movies_df = pd.read_csv(f"{data_path}/movies.csv")
        links_df = pd.read_csv(f"{data_path}/links.csv")
        
        print(f"✓ Loaded {len(movies_df):,} movies")
        print(f"✓ Loaded {len(links_df):,} TMDB links")
        
        # Merge to get TMDB IDs
        print("\n[2/4] Merging datasets...")
        enriched_df = movies_df.merge(links_df, on='movieId', how='left')
        
        # Filter movies with valid TMDB IDs
        valid_tmdb = enriched_df[enriched_df['tmdbId'].notna()].copy()
        print(f"✓ {len(valid_tmdb):,} movies have TMDB IDs")
        
        # Fetch descriptions
        print("\n[3/4] Fetching descriptions from TMDB...")
        print(f"Cache status: {len(self.cache)} already cached")
        
        descriptions = []
        total = len(valid_tmdb)
        
        for idx, (_, row) in enumerate(valid_tmdb.iterrows(), 1):
            tmdb_id = int(row['tmdbId'])
            
            # Progress indicator
            if idx % 50 == 0 or idx <= 10:  # More frequent updates for first 10 and every 50
                print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%) - "
                      f"Cache hits: {len([x for x in descriptions if x is not None])}")
            
            # Fetch movie details
            movie_data = self.fetch_movie_details(tmdb_id)
            descriptions.append(movie_data)
            
            # Save cache periodically
            if idx % 500 == 0:
                self._save_cache()
                print("💾 Cache saved")
        
        # Save final cache
        self._save_cache()
        print(f"\n✓ Fetched descriptions for {len(descriptions):,} movies")
        
        # Add descriptions to dataframe
        print("\n[4/4] Adding descriptions to dataset...")
        valid_tmdb['tmdb_data'] = descriptions
        
        # Extract fields
        valid_tmdb['overview'] = valid_tmdb['tmdb_data'].apply(
            lambda x: x['overview'] if x else ''
        )
        valid_tmdb['tagline'] = valid_tmdb['tmdb_data'].apply(
            lambda x: x['tagline'] if x else ''
        )
        valid_tmdb['runtime'] = valid_tmdb['tmdb_data'].apply(
            lambda x: x['runtime'] if x else None
        )
        valid_tmdb['tmdb_rating'] = valid_tmdb['tmdb_data'].apply(
            lambda x: x['vote_average'] if x else None
        )
        
        # Count success
        with_overview = valid_tmdb['overview'].str.len() > 0
        print(f"✓ {with_overview.sum():,} movies have descriptions")
        print(f"✓ Success rate: {with_overview.sum()/len(valid_tmdb)*100:.1f}%")
        
        # Save enriched dataset
        output_file = f"{data_path}/movies_enriched.csv"
        valid_tmdb.drop('tmdb_data', axis=1).to_csv(output_file, index=False)
        print(f"\n✓ Saved enriched dataset: {output_file}")
        
        print("\n" + "="*70)
        print("ENRICHMENT COMPLETE!")
        print("="*70)
        
        return valid_tmdb.drop('tmdb_data', axis=1)
    
    def get_statistics(self, enriched_df: pd.DataFrame):
        """Print statistics about enriched dataset"""
        
        print("\n📊 ENRICHMENT STATISTICS:")
        print(f"  • Total movies: {len(enriched_df):,}")
        print(f"  • With descriptions: {(enriched_df['overview'].str.len() > 0).sum():,}")
        print(f"  • With taglines: {(enriched_df['tagline'].str.len() > 0).sum():,}")
        print(f"  • Avg description length: {enriched_df['overview'].str.len().mean():.0f} chars")
        print(f"  • Avg TMDB rating: {enriched_df['tmdb_rating'].mean():.2f}/10")
        
        # Sample descriptions
        print("\n📝 SAMPLE DESCRIPTIONS:")
        samples = enriched_df[enriched_df['overview'].str.len() > 100].head(3)
        for _, row in samples.iterrows():
            print(f"\n{row['title']} ({row['movieId']})")
            print(f"Overview: {row['overview'][:150]}...")


def main():
    """
    Example usage
    """
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  TMDB MOVIE DESCRIPTION FETCHER                                  ║
    ║  Enriches MovieLens with semantic text data                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    SETUP INSTRUCTIONS:
    
    1. Get your TMDB API key:
       → Visit: https://www.themoviedb.org/settings/api
       → Sign up (free)
       → Request an API key
       → Copy your "API Read Access Token" (v4) or "API Key" (v3)
    
    2. Set your API key:
       → Option A: Environment variable
         export TMDB_API_KEY='your_key_here'
       
       → Option B: Direct in code
         api_key = 'your_key_here'
    
    3. Run this script:
       python tmdb_fetcher.py
    
    ══════════════════════════════════════════════════════════════════
    """)
    
    # Get API key
    api_key = os.environ.get('TMDB_API_KEY')
    
    if not api_key:
        print("❌ ERROR: TMDB API key not found!")
        print("\nPlease set your API key:")
        print("  export TMDB_API_KEY='your_key_here'")
        print("\nOr edit this script and add:")
        print("  api_key = 'your_key_here'")
        return
    
    print(f"✓ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Initialize fetcher
    fetcher = TMDBDescriptionFetcher(api_key)
    
    # Enrich dataset
    try:
        enriched_df = fetcher.enrich_movielens_dataset()
        fetcher.get_statistics(enriched_df)
        
        print("\n✅ SUCCESS!")
        print("\nNext steps:")
        print("  1. Run the semantic search script to build embeddings")
        print("  2. Integrate with your recommender system")
        print("  3. Add description-based search to your UI")
        
    except FileNotFoundError:
        print("\n❌ ERROR: MovieLens dataset not found!")
        print("\nPlease ensure ml-latest-small/ exists in the current directory.")
        print("If not, run the main recommender script first to download it.")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()