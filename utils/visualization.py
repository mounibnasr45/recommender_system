"""
Visualization Utilities
Functions for plotting data statistics and model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from models.hybrid_mf import HybridMatrixFactorization


def plot_learning_curves(train_errors: list, val_errors: list,
                        save_path: str = 'learning_curves.png'):
    """Plot training and validation learning curves"""

    plt.figure(figsize=(12, 6))

    epochs = range(1, len(train_errors) + 1)

    plt.plot(epochs, train_errors, 'o-', label='Train RMSE',
             linewidth=2, markersize=4, color='#2E86AB', alpha=0.8)

    if val_errors:
        plt.plot(epochs, val_errors, 's-', label='Validation RMSE',
                linewidth=2, markersize=4, color='#A23B72', alpha=0.8)

        # Highlight best epoch
        best_epoch = np.argmin(val_errors) + 1
        best_val = np.min(val_errors)
        plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.4,
                   label=f'Best Epoch ({best_epoch})')
        plt.scatter(best_epoch, best_val, color='red', s=150, zorder=5,
                   marker='*', edgecolors='darkred', linewidths=2)

        # Add text annotation
        plt.text(best_epoch, best_val - 0.02, f'RMSE: {best_val:.4f}',
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title('Hybrid Model Learning Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved learning curves: {save_path}")
    plt.close()


def plot_data_statistics(data: pd.DataFrame, save_path: str = 'data_statistics.png'):
    """Plot dataset statistics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rating distribution
    rating_counts = data['rating'].value_counts().sort_index()
    axes[0, 0].bar(rating_counts.index, rating_counts.values,
                   color='steelblue', edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('Rating Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Top genres
    top_genres = data['primary_genre'].value_counts().head(15)
    axes[0, 1].barh(range(len(top_genres)), top_genres.values, color='coral', edgecolor='black')
    axes[0, 1].set_yticks(range(len(top_genres)))
    axes[0, 1].set_yticklabels(top_genres.index)
    axes[0, 1].set_title('Top 15 Genres', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Ratings')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)

    # 3. Ratings per user
    user_rating_counts = data.groupby('user_idx').size()
    axes[1, 0].hist(user_rating_counts, bins=50, edgecolor='black',
                    color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('Ratings per User', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Ratings')
    axes[1, 0].set_ylabel('Number of Users (log scale)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Ratings per movie
    movie_rating_counts = data.groupby('item_idx').size()
    axes[1, 1].hist(movie_rating_counts, bins=50, edgecolor='black',
                    color='plum', alpha=0.8)
    axes[1, 1].set_title('Ratings per Movie', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Ratings')
    axes[1, 1].set_ylabel('Number of Movies (log scale)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved data statistics: {save_path}")
    plt.close()


def plot_predictions_vs_actual(train_data: pd.DataFrame, test_data: pd.DataFrame,
                                model: HybridMatrixFactorization,
                                user_profiles: dict,
                                save_path: str = 'predictions_analysis.png'):
    """Plot prediction quality analysis"""

    # Get predictions for test set
    test_users = test_data['user_idx'].values
    test_items = test_data['item_idx'].values
    test_ratings = test_data['rating'].values

    test_predictions = []
    for u, i in zip(test_users, test_items):
        user_items, user_ratings_hist = user_profiles.get(u, (np.array([]), np.array([])))
        pred = model.predict_hybrid(u, i, user_items, user_ratings_hist)
        test_predictions.append(pred)

    test_predictions = np.array(test_predictions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Scatter plot: Predicted vs Actual
    axes[0].scatter(test_ratings, test_predictions, alpha=0.3, s=10, color='steelblue')
    axes[0].plot([0.5, 5], [0.5, 5], 'r--', linewidth=2, label='Perfect predictions')
    axes[0].set_xlabel('Actual Rating', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
    axes[0].set_title('Predicted vs Actual Ratings', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.5, 5.5)
    axes[0].set_ylim(0.5, 5.5)

    # 2. Error distribution
    errors = test_ratings - test_predictions
    axes[1].hist(errors, bins=50, edgecolor='black', color='coral', alpha=0.8)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[1].set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # Add statistics text
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMean Error: {np.mean(errors):.4f}'
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved prediction analysis: {save_path}")
    plt.close()