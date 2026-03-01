#!/usr/bin/env python3
"""
TOPPLE End-to-End: Layer 1 → Layer 2 Pipeline
===============================================

Demonstrates the full stability-guided perturbation prediction workflow
on synthetic psoriasis TRM regulon data:

    1. Layer 1: Higher-order stability decomposition (Möbius inversion)
    2. Layer 2: Target selection → perturbation simulation → selectivity ranking
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topple import MobiusDecomposition, geometric_depth_cv
from topple.layer2 import PerturbationBridge, TargetSelector
from topple.layer2.perturbation_engine import MockPerturbationEngine
from topple.layer2.target_selection import IPAFilter


def simulate_trm_regulons(n_cells=500, random_state=42):
    """Simulate psoriasis TRM regulon activity with embedded structure."""
    rng = np.random.RandomState(random_state)
    n_half = n_cells // 2

    names = ["RUNX3", "TBX21", "EOMES", "NR4A1", "IRF4", "BATF", "PRDM1", "TOX"]
    X = rng.randn(n_cells, 8) * 0.5
    y = np.array([0]*n_half + [1]*n_half)

    # RUNX3 + TBX21: pairwise synergy (anti-correlated noise)
    X[n_half:, 0] += 1.5
    X[n_half:, 1] += 1.5
    corr = rng.randn(n_cells) * 0.7
    X[:, 0] += corr; X[:, 1] -= corr

    # EOMES: independent moderate
    X[n_half:, 2] += 1.2

    # NR4A1: noise
    X[n_half:, 3] += 0.2

    # IRF4 + BATF + PRDM1: triplet interaction
    X[n_half:, 4] += 0.8; X[n_half:, 5] += 0.8; X[n_half:, 6] += 0.8
    t1 = rng.randn(n_cells) * 0.6; t2 = rng.randn(n_cells) * 0.6
    X[:, 4] += t1; X[:, 5] -= t1 + t2; X[:, 6] += t2

    # TOX: noise
    X[n_half:, 7] += 0.15

    # DE scores (log2FC-like)
    de_scores = {
        "RUNX3": 4.2, "TBX21": 3.8, "EOMES": 2.5, "NR4A1": 0.8,
        "IRF4": 1.5, "BATF": 1.2, "PRDM1": 0.9, "TOX": 0.3,
    }

    # GRN adjacency
    adj = np.zeros((8, 8))
    adj[0,1] = adj[1,0] = adj[0,2] = adj[2,0] = adj[1,2] = adj[2,1] = 1
    adj[4,5] = adj[5,4] = adj[5,6] = adj[6,5] = adj[4,6] = adj[6,4] = 1
    adj[2,4] = adj[4,2] = 1

    return X, y, names, de_scores, adj


def main():
    print("=" * 70)
    print("TOPPLE End-to-End: Layer 1 → Layer 2")
    print("=" * 70)

    X, y, names, de_scores, adj = simulate_trm_regulons()
    print(f"\nData: {X.shape[0]} cells × {X.shape[1]} regulons")
    print(f"States: {(y==0).sum()} homeostatic, {(y==1).sum()} pathological\n")

    # ==================================================================
    # LAYER 1: Higher-Order Stability Decomposition
    # ==================================================================
    print("=" * 70)
    print("LAYER 1: Higher-Order Stability Decomposition")
    print("=" * 70)

    decomp = MobiusDecomposition(
        scorer=geometric_depth_cv, max_order=3, verbose=True,
    )
    decomp.fit(X, y, feature_names=names)

    print("\n--- Variance by order ---")
    for k, v in decomp.variance_explained().items():
        print(f"  k={k}: {v:.1%}")

    print("\n--- Top interactions (all orders) ---")
    for feat_names, val in decomp.top_interactions(n=10):
        print(f"  {' × '.join(feat_names):>30s}: {val:+.4f}")

    # ==================================================================
    # IPA CLASSIFICATION
    # ==================================================================
    print("\n" + "=" * 70)
    print("IPA Feature Classification")
    print("=" * 70)

    ipa = IPAFilter(decomp.interactions_, de_scores, names)
    classes = ipa.classify()
    for category, feats in classes.items():
        if feats:
            print(f"  {category}: {', '.join(feats)}")

    # ==================================================================
    # LAYER 2: Perturbation Bridge
    # ==================================================================
    print("\n" + "=" * 70)
    print("LAYER 2: Perturbation Bridge")
    print("=" * 70)

    engine = MockPerturbationEngine(
        X, y, names,
        effect_size=1.5,
        grn_adjacency=adj,
        random_state=42,
    )

    bridge = PerturbationBridge(
        engine=engine,
        X=X, y=y,
        feature_names=names,
        interactions=decomp.interactions_,
        de_scores=de_scores,
        max_candidates=15,
        perturbation_type="knockout",
        verbose=True,
    )

    results = bridge.run()

    # ==================================================================
    # RESULTS
    # ==================================================================
    print("\n" + bridge.report())

    # Export to DataFrame
    df = bridge.to_dataframe()
    if len(df) > 0:
        print("\n--- DataFrame preview ---")
        print(df.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
