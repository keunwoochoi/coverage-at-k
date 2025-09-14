import math
from collections import Counter


def coverage_at_k(counts: Counter, k: float, total_possible: int) -> float:
    """
    Calculates the proportion of possible categories with a count greater than or equal to k.
    This represents a single point on the coverage-vs-threshold curve.
    """
    if total_possible == 0:
        return 0.0
    # Sums categories where the value (v) is greater than or equal to the threshold k.
    count_greater_than_k = sum(1 for v in counts.values() if v >= k)
    return count_greater_than_k / total_possible

def auc_coverage(counts: Counter, total_possible: int) -> float:
    """
    Calculates coverage-at-k at the uniform point (k = total_items / total_possible).
    
    This represents the proportion of categories that have count > uniform_point.
    In a perfectly uniform distribution, this equals 1.0.
    """
    if total_possible == 0:
        return 0.0
    
    total_items = sum(counts.values())
    uniform_point = total_items / total_possible
    
    # Calculate coverage at the uniform point
    return coverage_at_k(counts, uniform_point, total_possible)

def normalized_evenness_auc(counts: Counter, total_possible: int) -> float:
    """
    Calculates evenness using coverage-at-k at the uniform point.
    
    This is simply the same as auc_coverage since we're now using the uniform point
    approach where perfect uniformity gives 1.0.
    """
    return auc_coverage(counts, total_possible)
