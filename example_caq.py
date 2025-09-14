import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from metrics import coverage_at_q

def generate_coverage_curve(probs):
    """
    Generate coverage-at-q curve data points.
    
    Args:
        probs: Dictionary with category probabilities
    
    Returns:
        tuple: (q_values, coverage_values)
    """
    # sort probabilities in the ascending order
    sorted_probs = sorted(probs.values())

    # find points where coverage changes
    q_values = [0.0] + sorted_probs + [1.0]
    coverage_values = [coverage_at_q(probs, q) for q in q_values]

    all_q_values = []
    all_coverage_values = []
    for i, q in enumerate(q_values):
        cov = coverage_values[i]
        all_q_values.append(q)
        all_coverage_values.append(cov)
        # Add a small step to create a step function effect
        if q < 1.0:
            all_q_values.append(q)
            all_coverage_values.append(coverage_values[i + 1])
    
    return all_q_values, all_coverage_values


def plot_coverage_at_q(probs_skewed, probs_skewed2, probs_skewed3, probs_uniform):
    """
    Create a visualization comparing coverage-at-q curves for different distributions.
    """
    # Generate curve data
    q_skewed, coverage_skewed = generate_coverage_curve(probs_skewed)
    q_skewed2, coverage_skewed2 = generate_coverage_curve(probs_skewed2)
    q_skewed3, coverage_skewed3 = generate_coverage_curve(probs_skewed3)
    q_uniform, coverage_uniform = generate_coverage_curve(probs_uniform)
    
    # Create the plot
    plt.figure(figsize=(7, 7))
    
    # Plot all curves
    plt.plot(q_skewed, coverage_skewed, 'r-', linewidth=2, label='Highly Skewed (90,3,3,4)', marker='o', markersize=3)
    plt.plot(q_skewed2, coverage_skewed2, 'orange', linewidth=2, label='Moderately Skewed (50,30,15,5)', marker='^', markersize=3)
    plt.plot(q_skewed3, coverage_skewed3, 'purple', linewidth=2, label='Slightly Skewed (35,30,25,10)', marker='d', markersize=3)
    plt.plot(q_uniform, coverage_uniform, 'b-', linewidth=2, label='Uniform (25,25,25,25)', marker='s', markersize=3)
    
    # Customize the plot
    plt.xlabel('Threshold q', fontsize=12)
    plt.ylabel('Coverage (proportion of categories with count > q)', fontsize=12)
    plt.title('Coverage-at-Q Curves (100 items, 4 classes)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set axis limits
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    
    # make the plot square
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('coverage_at_q.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Example Usage:
    # Different types of skewed distributions (100 items, 4 classes)
    counts_skewed = Counter({'a': 90, 'b': 3, 'c': 3, 'd': 4})      # Highly skewed
    counts_skewed2 = Counter({'a': 50, 'b': 30, 'c': 15, 'd': 5})   # Moderately skewed
    counts_skewed3 = Counter({'a': 35, 'b': 30, 'c': 25, 'd': 10})  # Slightly skewed
    # Perfect uniform distribution (100 items, 4 classes)
    counts_uniform = Counter({'a': 25, 'b': 25, 'c': 25, 'd': 25})

    # Convert counts to probabilities for AUC-C@Q
    total_skewed = sum(counts_skewed.values())
    probs_skewed = {k: v / total_skewed for k, v in counts_skewed.items()}

    total_skewed2 = sum(counts_skewed2.values())
    probs_skewed2 = {k: v / total_skewed2 for k, v in counts_skewed2.items()}

    total_skewed3 = sum(counts_skewed3.values())
    probs_skewed3 = {k: v / total_skewed3 for k, v in counts_skewed3.items()}

    total_uniform = sum(counts_uniform.values())
    probs_uniform = {k: v / total_uniform for k, v in counts_uniform.items()}

    print(f"--- Highly Skewed Distribution (Total Items: {sum(counts_skewed.values())}) ---")
    print(f"C@0 (Coverage): {coverage_at_q(counts_skewed, 0):.3f}")

    print(f"--- Moderately Skewed Distribution (Total Items: {sum(counts_skewed2.values())}) ---")
    print(f"C@0 (Coverage): {coverage_at_q(counts_skewed2, 0):.3f}")

    print(f"--- Slightly Skewed Distribution (Total Items: {sum(counts_skewed3.values())}) ---")
    print(f"C@0 (Coverage): {coverage_at_q(counts_skewed3, 0):.3f}")

    print(f"--- Uniform Distribution (Total Items: {sum(counts_uniform.values())}) ---")
    print(f"C@0 (Coverage): {coverage_at_q(counts_uniform, 0):.3f}")
    
    # Create visualization
    print("Generating Coverage-at-Q visualization...")
    plot_coverage_at_q(probs_skewed, probs_skewed2, probs_skewed3, probs_uniform)
