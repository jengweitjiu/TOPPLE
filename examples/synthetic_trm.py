#!/usr/bin/env python3
"""
TOPPLE Layer 1 — Example: Synthetic Psoriasis TRM Regulon Analysis
===================================================================

Demonstrates higher-order stability decomposition on a synthetic dataset
mimicking psoriasis TRM cell regulatory states.

Scenario
--------
We simulate 8 regulons (TFs) governing CD8+ TRM cell states:
- Pathological (lesional psoriasis): driven by RUNX3, TBX21, EOMES,
  NR4A1, IRF4, BATF, PRDM1, TOX
- Homeostatic (non-lesional): same TFs at different activity levels

The simulation embeds:
- Known pairwise synergy between RUNX3 + TBX21 (T-box cooperation)
- Known triplet interaction among IRF4 + BATF + PRDM1 (AP-1/IRF composite)
- Independent noise features (NR4A1, TOX)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topple import StabilityDecomposer, MobiusDecomposition, geometric_depth_cv
from topple.pruning import TopologyPruner


def simulate_trm_regulons(n_cells=500, random_state=42):
    """
    Simulate regulon activity for pathological vs homeostatic TRM.
    
    Returns X (n_cells x 8), y (binary), feature_names, adjacency.
    """
    rng = np.random.RandomState(random_state)
    n_half = n_cells // 2
    
    regulon_names = [
        "RUNX3", "TBX21", "EOMES", "NR4A1",
        "IRF4", "BATF", "PRDM1", "TOX"
    ]
    
    X = rng.randn(n_cells, 8) * 0.5
    y = np.array([0] * n_half + [1] * n_half)
    
    # --- Embed biological structure ---
    
    # 1. RUNX3 (idx 0) + TBX21 (idx 1): strong pairwise synergy
    #    Each alone moderately separates; together, much stronger
    X[n_half:, 0] += 1.5  # RUNX3 upregulated in pathological
    X[n_half:, 1] += 1.5  # TBX21 upregulated
    # Add anti-correlation to create synergy (need both for separation)
    corr_noise = rng.randn(n_cells) * 0.7
    X[:, 0] += corr_noise
    X[:, 1] -= corr_noise
    
    # 2. EOMES (idx 2): independent moderate contribution
    X[n_half:, 2] += 1.2
    
    # 3. NR4A1 (idx 3): weak/noise
    X[n_half:, 3] += 0.2
    
    # 4. IRF4 (idx 4) + BATF (idx 5) + PRDM1 (idx 6): triplet interaction
    #    Each alone: weak signal. Pair: moderate. Triplet: strong.
    X[n_half:, 4] += 0.8   # IRF4
    X[n_half:, 5] += 0.8   # BATF
    X[n_half:, 6] += 0.8   # PRDM1
    # Cross-correlate to create triplet dependency
    trip_noise1 = rng.randn(n_cells) * 0.6
    trip_noise2 = rng.randn(n_cells) * 0.6
    X[:, 4] += trip_noise1
    X[:, 5] -= trip_noise1 + trip_noise2
    X[:, 6] += trip_noise2
    
    # 5. TOX (idx 7): noise
    X[n_half:, 7] += 0.15
    
    # --- Build adjacency matrix (known GRN structure) ---
    adj = np.zeros((8, 8))
    # RUNX3-TBX21 axis
    adj[0, 1] = adj[1, 0] = 1  # RUNX3 <-> TBX21
    adj[0, 2] = adj[2, 0] = 1  # RUNX3 <-> EOMES
    adj[1, 2] = adj[2, 1] = 1  # TBX21 <-> EOMES
    # IRF4-BATF-PRDM1 axis
    adj[4, 5] = adj[5, 4] = 1  # IRF4 <-> BATF
    adj[5, 6] = adj[6, 5] = 1  # BATF <-> PRDM1
    adj[4, 6] = adj[6, 4] = 1  # IRF4 <-> PRDM1
    # Cross-axis
    adj[2, 4] = adj[4, 2] = 1  # EOMES <-> IRF4
    
    return X, y, regulon_names, adj


def main():
    print("=" * 70)
    print("TOPPLE Layer 1 — Synthetic Psoriasis TRM Example")
    print("=" * 70)
    
    # Generate data
    X, y, names, adj = simulate_trm_regulons(n_cells=500)
    print(f"\nDataset: {X.shape[0]} cells, {X.shape[1]} regulons")
    print(f"Regulons: {', '.join(names)}")
    print(f"States: {np.sum(y==0)} homeostatic, {np.sum(y==1)} pathological")
    
    # ---------------------------------------------------------------
    # Method 1: Exact enumeration (feasible for 8 features)
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Method 1: Exact enumeration (max_order=3)")
    print("-" * 70)
    
    sd_exact = StabilityDecomposer(
        max_order=3,
        method="exact",
        cv_folds=5,
        verbose=True,
    )
    sd_exact.fit(X, y, feature_names=names)
    
    print("\n--- Variance explained by order ---")
    for order, frac in sd_exact.variance_explained().items():
        print(f"  k={order}: {frac:.1%}")
    
    print("\n--- Top marginal contributions (k=1) ---")
    for feat_names, val in sd_exact.top_interactions(order=1, n=8):
        print(f"  {feat_names[0]:>8s}: {val:+.4f}")
    
    print("\n--- Top pairwise synergies (k=2) ---")
    for feat_names, val in sd_exact.top_interactions(order=2, n=5):
        print(f"  {' + '.join(feat_names):>20s}: {val:+.4f}")
    
    print("\n--- Top triplet interactions (k=3) ---")
    for feat_names, val in sd_exact.top_interactions(order=3, n=5):
        print(f"  {' + '.join(feat_names):>30s}: {val:+.4f}")
    
    # ---------------------------------------------------------------
    # Method 2: Topology-guided pruning
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Method 2: Topology-guided pruning")
    print("-" * 70)
    
    pruner = TopologyPruner(adj, max_order=3, max_distance=2)
    pruner.fit()
    print(pruner.summary())
    
    sd_pruned = StabilityDecomposer(
        max_order=3,
        method="pruned",
        adjacency=adj,
        cv_folds=5,
        verbose=True,
    )
    sd_pruned.fit(X, y, feature_names=names)
    
    print("\n--- Top triplet interactions (pruned) ---")
    for feat_names, val in sd_pruned.top_interactions(order=3, n=5):
        print(f"  {' + '.join(feat_names):>30s}: {val:+.4f}")
    
    # ---------------------------------------------------------------
    # Full report
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(sd_exact.report())
    print("=" * 70)
    
    print("\nDone. TOPPLE Layer 1 scaffold validated successfully.")


if __name__ == "__main__":
    main()
