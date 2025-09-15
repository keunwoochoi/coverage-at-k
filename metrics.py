import math
from collections import Counter

def coverage_at_k(counts: Counter, k: float, total_possible: int) -> float:
    """
    Calculates the proportion of possible categories with a count strictly greater than k.
    This matches the convention that C(0) equals standard coverage (non-empty proportion).

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
    Normalized AUC-C(K) up to the Even Point (strict C(K) with "> k").

    Steps:
    - Compute Even Point = floor(total_items / number_of_observed_categories)
    - Observed area: sum_{k=0}^{EvenPoint-1} C(K)
    - Ideal area (uniform): EvenPoint * (number_of_observed_categories / total_possible)
    - Return observed / ideal ∈ [0,1]

    Args:
        counts: Counter object with category counts
        total_possible: Total number of possible categories

    Returns:
        float: Normalized AUC-C(K) up to the Even Point

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

def coverage_at_q(probs: dict, q: float) -> float:
    """
    Calculates the proportion of categories with a probability greater than or equal to q.
    This corresponds to the normalized coverage function C̅(q).

    Args:
        probs: Dictionary with category probabilities
        q: Probability threshold

    Returns:
        float: Proportion of categories with probability >= q

    """
    if not probs:
        return 0.0
    
    assert 0.0 <= q <= 1.0, "q must be in [0, 1]"

    count_greater_equal_q = sum(1 for v in probs.values() if v >= q)
    return count_greater_equal_q / len(probs)

def deviation_from_uniform(probs: dict) -> float:
    """
    Calculates the deviation from the uniform distribution using the coverage-at-q metric C̅(q).

    Args:
        probs: Dictionary with category probabilities

    Returns:
        float: \int_0^p (1 - C̅(q)) dq + \int_p^1 C̅(q) dq, where p = 1/number_of_categories
    """
    if not probs:
        return 0.0

    num_categories = len(probs)
    if num_categories == 0:
        return 0.0
    
    p = 1.0 / num_categories

    # sort the probabilities in ascending order
    sorted_probs = sorted(probs.values())

    # find points where coverage changes
    q_values = [0.0] +sorted_probs + [1.0]
    coverage_values = [coverage_at_q(probs, q) for q in q_values]

    area = 0.0

    for i, q in enumerate(q_values):
        if q <= p:
            area = area + (1 - coverage_values[i]) * (q - (q_values[i-1] if i > 0 else 0))
        else:
            area = area + coverage_values[i] * (q - (q_values[i-1] if i > 0 else 0))

    return num_categories * area

# Alias for new terminology (Uniform Divergence Score / UDS)
def uniform_divergence_score(probs: dict) -> float:
    """Alias of deviation_from_uniform for UDS naming consistency."""
    return deviation_from_uniform(probs)