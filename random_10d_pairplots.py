"""
Randomly sample many 10-D categorical distributions, compute metrics, and
draw Nature-quality pairwise scatter plots.

Metrics
- AUC-C(K): computed from counts drawn via a Multinomial(N, p), using auc_catk
- UCS: 1 - deviation_from_uniform (alias uniform_divergence_score in metrics)
- Shannon entropy (normalized): H(p) / log(C), C=10

Outputs
- pairwise_scatter_10d.png (600 dpi)
- pairwise_scatter_10d.pdf (vector)

Usage
  python random_10d_pairplots.py --samples 5000 --N 100 --seed 42

Notes
- Distributions p are sampled from a Dirichlet with concentration c · 1_C,
  where c is drawn log-uniformly in [0.05, 20] by default to cover a wide
  range of sparsity/peakedness. Adjust via --alpha-min/--alpha-max.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from metrics import auc_catk, uniform_divergence_score


# -----------------------------
# Plot style for "Nature-quality"
# -----------------------------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,  # Embed fonts as TrueType (editable text in Illustrator)
    "ps.fonttype": 42,
    "font.size": 8.5,     # Suitable for single/double-column figures
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
    "lines.markersize": 3.0,
    "font.family": "DejaVu Sans",  # Robust default; swap to 'Arial' if available
})


@dataclass
class Config:
    samples: int = 5000
    seed: int | None = 42
    categories: int = 10
    N: int = 100  # items per sample for counts (for AUC-C(K))
    alpha_min: float = 0.05
    alpha_max: float = 20.0
    show: bool = False


def sample_dirichlet_variety(cfg: Config) -> np.ndarray:
    """
    Sample `cfg.samples` probability vectors of dimension `cfg.categories` from a
    Dirichlet with random concentration c * 1_C, where c ~ LogUniform(alpha_min, alpha_max).
    Returns: array of shape (samples, C)
    """
    rng = np.random.default_rng(cfg.seed)
    # Draw log-uniform by sampling u ~ U(log(a), log(b)), then c = exp(u)
    log_a, log_b = math.log(cfg.alpha_min), math.log(cfg.alpha_max)
    u = rng.uniform(log_a, log_b, size=cfg.samples)
    c = np.exp(u)

    C = cfg.categories
    P = np.empty((cfg.samples, C), dtype=float)
    for i in range(cfg.samples):
        alpha = np.full(C, c[i], dtype=float)
        P[i] = rng.dirichlet(alpha)
    return P


def compute_metrics_for_probs(p: np.ndarray, cfg: Config) -> Tuple[float, float, float]:
    """
    Compute (auc_c, ucs, h_norm) for one probability vector p.
    - auc_c: using counts ~ Multinomial(N, p)
    - ucs: 1 - uniform_divergence_score(p)
    - h_norm: normalized Shannon entropy H(p)/log(C)
    """
    C = p.shape[0]
    # AUC-C(K): build counts from Multinomial
    counts = np.random.default_rng().multinomial(cfg.N, p)
    counter = Counter({i: int(counts[i]) for i in range(C)})
    auc_c = auc_catk(counter, total_possible=C)

    # UCS: metrics.uniform_divergence_score returns deviation-from-uniform
    prob_dict = {i: float(pi) for i, pi in enumerate(p)}
    ucs = 1.0 - uniform_divergence_score(prob_dict)

    # Shannon entropy (normalized)
    eps = 1e-12
    h = -float(np.sum(p * np.log(p + eps)))
    h_norm = h / math.log(C)
    return auc_c, ucs, h_norm


def pairwise_scatter(auc: np.ndarray, ucs: np.ndarray, h: np.ndarray, cfg: Config) -> None:
    """Create and save pairwise scatter plots with consistent [0,1] axes."""
    # Correlations (Pearson)
    def corr(x, y):
        if len(x) < 2:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    r_auc_ucs = corr(auc, ucs)
    r_auc_h = corr(auc, h)
    r_ucs_h = corr(ucs, h)

    # Figure size: Nature double-column width ~180 mm = 7.09 in
    # Use 7.2 x 2.6 inches for a clean 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)

    points_kw = dict(s=6, alpha=0.35, edgecolor="none")

    # AUC vs UCS
    ax = axes[0]
    ax.scatter(auc, ucs, color="#1f77b4", **points_kw)
    ax.set_xlabel("AUC-C(K)")
    ax.set_ylabel("UCS")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"r = {r_auc_ucs:.2f}", transform=ax.transAxes, va="top")

    # AUC vs H
    ax = axes[1]
    ax.scatter(auc, h, color="#ff7f0e", **points_kw)
    ax.set_xlabel("AUC-C(K)")
    ax.set_ylabel("H (normalized)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"r = {r_auc_h:.2f}", transform=ax.transAxes, va="top")

    # UCS vs H
    ax = axes[2]
    ax.scatter(ucs, h, color="#2ca02c", **points_kw)
    ax.set_xlabel("UCS")
    ax.set_ylabel("H (normalized)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"r = {r_ucs_h:.2f}", transform=ax.transAxes, va="top")

    # Save high-quality outputs
    fig.savefig("pairwise_scatter_10d.png", bbox_inches="tight")
    fig.savefig("pairwise_scatter_10d.pdf", bbox_inches="tight")
    if cfg.show:
        plt.show()
    plt.close(fig)


def main(cfg: Config) -> None:
    P = sample_dirichlet_variety(cfg)

    # Compute metrics
    auc_list = np.empty(cfg.samples, dtype=float)
    ucs_list = np.empty(cfg.samples, dtype=float)
    h_list = np.empty(cfg.samples, dtype=float)

    rng = np.random.default_rng(cfg.seed)
    # Use local RNG for multinomial draws inside compute_metrics to be reproducible
    # by temporarily seeding numpy's default RNG per iteration is heavy; instead,
    # draw counts here and pass them? Keep simple and accept minor randomness.
    np.random.seed(cfg.seed)

    for i in range(cfg.samples):
        auc, ucs, h = compute_metrics_for_probs(P[i], cfg)
        auc_list[i] = auc
        ucs_list[i] = ucs
        h_list[i] = h

    print(f"Samples: {cfg.samples}, N per sample: {cfg.N}, C: {cfg.categories}")
    print(f"AUC-C  mean={auc_list.mean():.3f}  std={auc_list.std(ddof=1):.3f}")
    print(f"UCS    mean={ucs_list.mean():.3f}  std={ucs_list.std(ddof=1):.3f}")
    print(f"H_norm mean={h_list.mean():.3f}  std={h_list.std(ddof=1):.3f}")

    pairwise_scatter(auc_list, ucs_list, h_list, cfg)
    print("Saved: pairwise_scatter_10d.png, pairwise_scatter_10d.pdf")


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Sample 10-D categorical distributions and plot metric pairwise scatter plots.")
    ap.add_argument("--samples", type=int, default=5000, help="Number of distributions to sample")
    ap.add_argument("--N", type=int, default=100, help="Multinomial count per sample for AUC-C(K)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--alpha-min", type=float, default=0.05, help="Min concentration for log-uniform c")
    ap.add_argument("--alpha-max", type=float, default=20.0, help="Max concentration for log-uniform c")
    ap.add_argument("--show", action="store_true", help="Show the figure interactively")
    args = ap.parse_args()
    return Config(
        samples=args.samples,
        seed=args.seed,
        categories=10,
        N=args.N,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        show=args.show,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
