# Coverage-at-K/Q: Simple Metrics for Measuring Sharpness

Keunwoo Choi & Kyunghyun Cho, 2025 Sep.


In this repository, we describe two simple and intuitive metrics to measure the sharpness of a Categorical distribution (or a histogram). These metrics are robustness to noise by shrinking rare class probabilites or counts toward zero. 

## Motivation

A conventional approach to measure the sharpness of a Categorical distribution, or a count-based histogram, is Shannon entropy. It is not intuitive for many, unless they are trained on compression and information theory, to interpret. An alternative is the notion of "coverage", where the goal is to check which categories ever occured. That is, how many categories have non-zero probabilities or counts. This alternative is also unattractive, because it tends to be too strict and sensitive to noise. 

We generalize "coverage" by relaxing the definition of "occurrence", which is originally once or more. We do so in two different ways. First, we threhold actual counts, that is, unnormalized counts, at $k$. We call it coverage-at-$k(\tilde{p})$ (C@$k(\tilde{p})$). From this, we derive AUC-C$(\tilde{p})$ which we can use to quantify the sharpness of the count vector $\tilde{p}$ relative to the uniform count vector. 

Second, we threshold normalized counts, or the probabilities, at $q$. We call it coverage-at-$q(p)$ (C@$q(p)$). Analogously, we define DfU$(p)$ which can be used to quantify the sharpness of the normalized probability vector $p$.

These metrics can be used depending on the situation. If we are given a normalized probability vector, we can use the coverage-at-$q$ with DfU, to quantify its sharpness. If we are given an unnormalized count vector, we can use either the coverage-at-$k$ with AUC, or the coverage-at-$q$ with DfU. Of course, normalization would come with some loss of information.

## Coverage-at-$k$ and AUC-C

### Coverage-at-$k$ (C@$k$)

Coverage-at-$k(\tilde{p})$ (C@$k(\tilde{p})$) is the fraction of categories that have **more than** $k$ items in the count vector $\tilde{p}$. This makes C@$0$ equal to the standard coverage (the proportion of non-empty categories). In many cases, higher is better, though it may be the opposite depending on the application.

- $k = 0$: reduces to standard coverage (non-empty categories / total possible)
- As $k$ increases, C@$k$ likely decreases, showing how many categories are sufficiently populated
- Example (100 items, 4 classes): If the count vector $\tilde{p}$ is {A: 10, B: 15, C: 35, D:50}:
  - C@$0$ (Coverage) = $1.0$
  - C@$1$, C@$2$, .. , C@$9$ = $1.0$
  - C@$10$ = $0.75$
  - C@$14$ = $0.75$
  - C@$15$ = $0.5$
  - C@$35$ = $0.25$
  - C@$49$ = $0.25$
  - C@$50$ = $0.0$

### AUC-C

The method involves analyzing a coverage curve up to a specific cutoff point called an "even point" to aggregate the coverages at multiple K values.

### The Coverage Curve

As shown in the example above, we can compute C@$k$ while varying $k$ from $0$. This quantity will eventually vanish to $0$.

### Defining the Even Point

Although we can increase $k$ to the number of total items (e.g., 100 if there are 100 total items), we introduce a cutoff point for more meaningful analysis. We call it the **"even point"**, because it is the count each observed (non-empty) category would have if the items were distributed evenly among them.

$$
\text{even-point} = \left\lfloor \frac{C}{C@0} \right\rfloor
$$

This even point serves as the upper limit for our analysis. We will measure the area under the coverage curve from $k=0$ up to this point. 

The even point is designed so that AUC-C is normalized between $[0, 1]$, and the AUC-C of a uniform count vector would be $1$. AUC-C cannot be $0$, as an extremely sharp distribution will still have a non-zero C@$k$ for $k \in [0, \text{even-point}]$. 

With this definition of the even point, AUC-C is defined as

$$
\mathrm{AUC-C}(\tilde{p})
=
\frac{
\sum_{k=0}^{\text{even-point}}
C@k(\tilde{p})
}
{\text{even-point}} \in [0, 1].
$$

AUC-C is 1 when the count vector is uniform. It vanishes to $0$, as the count vector becomes increasingly sharper. 

## Coverage-at-Q and Difference-from-Uniform

### Coverage-at-Q (C@Q)

Given a Categorical distribution $p$, such that $\sum_{c \in C} p(c) = 1$ and $p(c) \geq 0~\forall c \in C$. The coverage-at-$q$ is defined as

$$
\mathrm{coverage}@q(p) = \sum_{c\in C} I(p(c) \geq q),
$$

where $I$ is an indicator function. In words, it counts the number of categories within the category set $C$, to which the probability greater than $q$ was assigned. 

### Difference-from-Uniform (DfU)

The sharpness can be measured as the difference from the sharpness of the uniform distribution. We measure by first realizing that the coverage-at-q is either 0 or 1, depending on whether $q \leq 1/C$ or not. That is,

$$
\mathrm{coverage}@q(u) = \begin{cases}
1,& \mathrm{if}~q \neq \frac{1}{C}, \\
0,& \mathrm{otherwise}, 
\end{cases}
$$

where $u(c) = 1/C$ for all $c \in C$.

The difference-from-uniform (DfU) is then defined as

$$
\mathrm{dfu}(p) = 
C \left(\int_{0}^{1/C}
1-\mathrm{coverage}@q(p) \mathrm{d}q
+
\int_{1/C}^1
\mathrm{coverage}@q(p) \mathrm{d}q
\right).
$$

The DfU of a uniform distribution is $0$, and the sharpest distribution, that is, a dirac distribution, will be $1$.

## Example


### Coverage-at-$q$ and AUC-C

#### Code example

```python
from collections import Counter
from metrics import auc_coverage

# Different types of distributions (100 items, 4 categories)
uniform = Counter({'a': 25, 'b': 25, 'c': 25, 'd': 25})
print(f"Uniform coverage: {auc_coverage(uniform, 4):.3f}")        # 1.000

slightly_skewed = Counter({'a': 35, 'b': 30, 'c': 25, 'd': 10})
print(f"Slightly skewed: {auc_coverage(slightly_skewed, 4):.3f}") # 0.750

moderately_skewed = Counter({'a': 50, 'b': 30, 'c': 15, 'd': 5})
print(f"Moderately skewed: {auc_coverage(moderately_skewed, 4):.3f}") # 0.500

highly_skewed = Counter({'a': 90, 'b': 3, 'c': 3, 'd': 4})
print(f"Highly skewed: {auc_coverage(highly_skewed, 4):.3f}")     # 0.250
```

#### Visualization

![Coverage-at-K Comparison](coverage_at_k.jpg)

The plot shows coverage curves for different distribution types. The uniform distribution maintains 100% coverage until k=25, while sharper distributions drop off at different rates based on their evenness.


### Coverage-at-$q$ and DfU

#### Code example

```python
from collections import Counter
from metrics import coverage_at_q, deviation_from_uniform

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
    print(f"DfU: {deviation_from_uniform(probs_skewed):.3f}\n")

    print(f"--- Moderately Skewed Distribution (Total Items: {sum(counts_skewed2.values())}) ---")
    print(f"C@0 (Coverage): {coverage_at_q(counts_skewed2, 0):.3f}")
    print(f"DfU: {deviation_from_uniform(probs_skewed2):.3f}\n")

    print(f"--- Slightly Skewed Distribution (Total Items: {sum(counts_skewed3.values())}) ---")
    print(f"C@0 (Coverage): {coverage_at_q(counts_skewed3, 0):.3f}")
    print(f"DfU: {deviation_from_uniform(probs_skewed3):.3f}\n")

    print(f"--- Uniform Distribution (Total Items: {sum(counts_uniform.values())}) ---")
    print(f"C@0 (Coverage): {coverage_at_q(counts_uniform, 0):.3f}")
    print(f"DfU: {deviation_from_uniform(probs_uniform):.3f}\n")
```

#### Visualization

![Coverage-at-Q Comparison](coverage_at_q.jpg)

We see that sharper distributions deviate away from the curve produced by the uniform distribution.

## 6. Usage

```bash
python example.py
python example_caq.py
```

These scripts generate both numerical results and the visualization above. See `metric.py` for the code to compute all the metrics.

## Citation

If you use this concept or code in your research, please cite:

```bibtex
@misc{choi2025coverageatk,
  author       = {Choi, Keunwoo and Cho, Kyunghyun},
  title        = {{Coverage-at-K/Q: Simple Metrics for Measuring Sharpness}},
  howpublished = {\url{https://github.com/keunwoochoi/coverage-at-k}},
  year         = {2025}
}
```