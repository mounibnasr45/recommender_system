"""
Utilities module for MovieLens recommender system
"""

from .visualization import (
    plot_learning_curves,
    plot_data_statistics,
    plot_predictions_vs_actual
)

__all__ = [
    'plot_learning_curves',
    'plot_data_statistics',
    'plot_predictions_vs_actual'
]