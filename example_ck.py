import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from metrics import coverage_at_k, auc_catk


def generate_coverage_curve(counts, total_possible, max_k=None):
    """
    Generate coverage-at-k curve data points.
    
    Args:
        counts: Counter object with category counts
        total_possible: Total number of possible categories
        max_k: Maximum k value to plot (defaults to max count)
    
    Returns:
        tuple: (k_values, coverage_values)
    """
    if max_k is None:
        max_k = max(counts.values()) if counts else 0
    
    k_values = list(range(max_k + 1))
    coverage_values = [coverage_at_k(counts, k, total_possible) for k in k_values]
    
    return k_values, coverage_values


def plot_coverage_at_k(counts_extreme, counts_skewed, counts_skewed2, counts_skewed3, counts_uniform, total_possible_cats):
    """
    Create a visualization comparing coverage-at-k curves for different distributions.
    """
    # Generate curve data
    k_extreme, coverage_extreme = generate_coverage_curve(counts_extreme, total_possible_cats)
    k_skewed, coverage_skewed = generate_coverage_curve(counts_skewed, total_possible_cats)
    k_skewed2, coverage_skewed2 = generate_coverage_curve(counts_skewed2, total_possible_cats)
    k_skewed3, coverage_skewed3 = generate_coverage_curve(counts_skewed3, total_possible_cats)
    k_uniform, coverage_uniform = generate_coverage_curve(counts_uniform, total_possible_cats)
    
    # Create the plot
    plt.figure(figsize=(7, 7))
    
    # Plot all curves
    plt.plot(k_extreme, coverage_extreme, 'g-', linewidth=2, label='Extremely Skewed (100,0,0,0)', marker='o', markersize=3)
    plt.plot(k_skewed, coverage_skewed, 'r-', linewidth=2, label='Highly Skewed (90,3,3,4)', marker='o', markersize=3)
    plt.plot(k_skewed2, coverage_skewed2, 'orange', linewidth=2, label='Moderately Skewed (50,30,15,5)', marker='^', markersize=3)
    plt.plot(k_skewed3, coverage_skewed3, 'purple', linewidth=2, label='Slightly Skewed (35,30,25,10)', marker='d', markersize=3)
    plt.plot(k_uniform, coverage_uniform, 'b-', linewidth=2, label='Uniform (25,25,25,25)', marker='s', markersize=3)
    
    # Customize the plot
    plt.xlabel('Threshold k', fontsize=12)
    plt.ylabel('Coverage (proportion of categories with count > k)', fontsize=12)
    plt.title('Coverage-at-K Curves (100 items, 4 classes, uniform point k=25)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set axis limits - focus on the relevant range up to uniform point
    plt.xlim(0, 25)
    plt.ylim(0, 1.0)
    
    # Add AUC-C(K) values as text
    auc_extreme = auc_catk(counts_extreme, total_possible_cats)
    auc_skewed = auc_catk(counts_skewed, total_possible_cats)
    auc_skewed2 = auc_catk(counts_skewed2, total_possible_cats)
    auc_skewed3 = auc_catk(counts_skewed3, total_possible_cats)
    auc_uniform = auc_catk(counts_uniform, total_possible_cats)
    
    plt.text(0.02, 0.98, f'Extremely Skewed AUC: {auc_extreme:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    plt.text(0.02, 0.90, f'Highly Skewed AUC: {auc_skewed:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    plt.text(0.02, 0.82, f'Moderately Skewed AUC: {auc_skewed2:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    plt.text(0.02, 0.74, f'Slightly Skewed AUC: {auc_skewed3:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))
    plt.text(0.02, 0.66, f'Uniform AUC: {auc_uniform:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('coverage_at_k.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Example Usage:
    # Different types of skewed distributions (100 items, 4 classes)
    counts_extreme = Counter({'a': 100, 'b': 0, 'c': 0, 'd': 0})      # Highly skewed
    counts_skewed = Counter({'a': 90, 'b': 3, 'c': 3, 'd': 4})      # Highly skewed
    counts_skewed2 = Counter({'a': 50, 'b': 30, 'c': 15, 'd': 5})   # Moderately skewed
    counts_skewed3 = Counter({'a': 35, 'b': 30, 'c': 25, 'd': 10})  # Slightly skewed
    # Perfect uniform distribution (100 items, 4 classes)
    counts_uniform = Counter({'a': 25, 'b': 25, 'c': 25, 'd': 25})
    total_possible_cats = 4  # Only 4 possible classes

    print(f"--- Extremely Skewed Distribution (Total Items: {sum(counts_extreme.values())}) ---")
    print(f"C(0): {coverage_at_k(counts_extreme, 0, total_possible_cats):.3f}")
    print(f"AUC-C(K): {auc_catk(counts_extreme, total_possible_cats):.3f}\n")

    print(f"--- Highly Skewed Distribution (Total Items: {sum(counts_skewed.values())}) ---")
    print(f"C(0): {coverage_at_k(counts_skewed, 0, total_possible_cats):.3f}")
    print(f"AUC-C(K): {auc_catk(counts_skewed, total_possible_cats):.3f}\n")

    print(f"--- Moderately Skewed Distribution (Total Items: {sum(counts_skewed2.values())}) ---")
    print(f"C(0): {coverage_at_k(counts_skewed2, 0, total_possible_cats):.3f}")
    print(f"AUC-C(K): {auc_catk(counts_skewed2, total_possible_cats):.3f}\n")

    print(f"--- Slightly Skewed Distribution (Total Items: {sum(counts_skewed3.values())}) ---")
    print(f"C(0): {coverage_at_k(counts_skewed3, 0, total_possible_cats):.3f}")
    print(f"AUC-C(K): {auc_catk(counts_skewed3, total_possible_cats):.3f}\n")

    print(f"--- Uniform Distribution (Total Items: {sum(counts_uniform.values())}) ---")
    print(f"C(0): {coverage_at_k(counts_uniform, 0, total_possible_cats):.3f}")
    print(f"AUC-C(K): {auc_catk(counts_uniform, total_possible_cats):.3f}\n")
    
    # Create visualization
    print("Generating Coverage-at-K visualization...")
    plot_coverage_at_k(counts_extreme, counts_skewed, counts_skewed2, counts_skewed3, counts_uniform, total_possible_cats)
