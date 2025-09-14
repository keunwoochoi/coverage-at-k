import math
from collections import Counter


def coverage_at_k(counts: Counter, k: float, total_possible: int) -> float:
    """
    Calculates the proportion of possible categories with a count strictly greater than k.
    This matches the convention that C@0 equals standard coverage (non-empty proportion).

    Args:
        counts: Counter object with category counts
        k: Threshold value
        total_possible: Total number of possible categories

    Returns:
        float: Proportion of categories with a count strictly greater than k

    """
    if total_possible == 0:
        return 0.0
    # Count categories where the value is strictly greater than k
    count_greater_than_k = sum(1 for v in counts.values() if v > k)
    return count_greater_than_k / total_possible

def auc_catk(counts: Counter, total_possible: int) -> float:
    """
    Normalized AUC-C@K up to the Even Point (strict C@K with "> k").

    Steps:
    - Compute Even Point = floor(total_items / number_of_observed_categories)
    - Observed area: sum_{k=0}^{EvenPoint-1} C@k
    - Ideal area (uniform): EvenPoint * (number_of_observed_categories / total_possible)
    - Return observed / ideal ∈ [0,1]

    Args:
        counts: Counter object with category counts
        total_possible: Total number of possible categories

    Returns:
        float: Normalized AUC-C@K up to the Even Point

    """
    if total_possible == 0:
        return 0.0

    if not counts:
        return 0.0

    total_items = sum(counts.values())
    num_observed_categories = len(counts)

    if num_observed_categories == 0:
        return 0.0

    even_point = math.floor(total_items / num_observed_categories)

    if even_point <= 0:
        return 0.0

    observed_area = sum(coverage_at_k(counts, k, total_possible) for k in range(even_point))
    ideal_area = even_point * (num_observed_categories / total_possible)

    if ideal_area == 0:
        return 0.0

    return observed_area / ideal_area
