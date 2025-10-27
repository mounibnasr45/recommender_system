"""
Data handling module for MovieLens recommender system
"""

from .dataset_setup import DatasetSetup
from .data_loader import MovieLensLoader

__all__ = ['DatasetSetup', 'MovieLensLoader']