import math
from collections import Counter

# Using the corrected, simpler version of this function
def coverage_at_k(counts: Counter, k: float, total_possible: int) -> float:
    """
    Calculates the proportion of possible categories with a count strictly greater than k.
    """
    if total_possible == 0:
        return 0.0
    count_greater_than_k = sum(1 for v in counts.values() if v > k)
    return count_greater_than_k / total_possible


def auc_catk(counts: Counter, total_possible: int) -> float:
    """
    Calculates the standard normalized AUC-C(K) up to the Even Point.

    Args:
        counts: Counter object with category counts
        total_possible: Total number of possible categories

    Returns:
        float: Normalized AUC-C(K) up to the Even Point
    """
    if total_possible == 0 or not counts:
        return 0.0

    total_items = sum(counts.values())

    # BUG FIX 1: Use total_possible to calculate the even point.
    if total_possible == 0: # Avoid division by zero
        return 0.0
    even_point = math.floor(total_items / total_possible)

    # If even_point is 0, the sum is over an empty range, so the area is 0.
    if even_point <= 0:
        return 0.0

    observed_area = sum(coverage_at_k(counts, k, total_possible) for k in range(even_point))

    # BUG FIX 2: The ideal area for the standard definition is just even_point.
    # The final value is the observed area divided by the ideal area.
    return observed_area / even_point

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

    count_greater_equal_q = sum(1 for v in probs.values() if v > q)
    return count_greater_equal_q / len(probs)

def deviation_from_uniform(probs: dict) -> float:
    """
    Calculates the deviation from the uniform distribution using the coverage-at-q metric C̅(q).
    This is equivalent to 1 - UCS.

    Args:
        probs: Dictionary with category probabilities

    Returns:
        float: The normalized area of deviation. Should be 1.0 for a one-hot distribution
               and 0.0 for a uniform distribution.
    """
    if not probs:
        return 0.0

    num_categories = len(probs)
    # The metric is trivial (0) for C=1 and undefined if C < 1.
    if num_categories <= 1:
        return 0.0

    p_uniform = 1.0 / num_categories

    # It's crucial that p_uniform is included to correctly split the integral.
    breakpoints = sorted(list(set([0.0, 1.0, p_uniform] + list(probs.values()))))

    raw_area = 0.0
    
    for i in range(1, len(breakpoints)):
        q_start = breakpoints[i-1]
        q_end = breakpoints[i]
        width = q_end - q_start

        if width == 0:
            continue

        # The value of C̅(q) is constant over the interval [q_start, q_end).
        # We must evaluate its value at the beginning of the interval.
        coverage = coverage_at_q(probs, q_start) # <-- THE FIX

        # Apply the correct rule based on which side of p_uniform the interval lies.
        # Note: We check the interval midpoint to handle cases where a breakpoint is exactly p_uniform
        interval_midpoint = q_start + width / 2
        if interval_midpoint < p_uniform:
            # This interval is in the first part of the integral: ∫(1 - C̅(q)) dq
            raw_area += (1 - coverage) * width
        else:
            # This interval is in the second part of the integral: ∫C̅(q) dq
            raw_area += coverage * width

    normalization_factor = (num_categories**2) / (2 * (num_categories - 1))

    return normalization_factor * raw_area

# Alias for new terminology (Uniform Divergence Score / UDS)
def uniform_divergence_score(probs: dict) -> float:
    """Alias of deviation_from_uniform for UDS naming consistency."""
    return deviation_from_uniform(probs)