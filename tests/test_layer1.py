"""
TOPPLE Layer 1 — Test Suite
============================

Validates:
1. Möbius inversion correctness (analytical verification)
2. Recovery of known interaction structure from synthetic data
3. Topology-guided pruning reduction ratios
4. Compressed sensing recovery accuracy
5. End-to-end pipeline on DGSA-like simulation regimes
"""

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from itertools import combinations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topple.mobius import (
    MobiusDecomposition, interaction_term, mobius_inversion, stability_loss,
)
from topple.pruning import TopologyPruner, HierarchicalScreener
from topple.compressed_sensing import CompressedInteractionSensing
from topple.stability import geometric_depth_cv, separability_gate, StabilityDecomposer


# ===========================================================================
# Helpers
# ===========================================================================

def make_separable_data(n_samples=200, n_features=6, separation=2.0, random_state=42):
    rng = np.random.RandomState(random_state)
    n_half = n_samples // 2
    X = np.vstack([rng.randn(n_half, n_features), rng.randn(n_half, n_features) + separation])
    y = np.array([0]*n_half + [1]*n_half)
    return X, y

def make_triplet_data(n_samples=400, random_state=42):
    rng = np.random.RandomState(random_state)
    n_half = n_samples // 2
    X = rng.randn(n_samples, 5) * 0.3
    y = np.array([0]*n_half + [1]*n_half)
    X[n_half:, 0] += 1.0; X[n_half:, 1] += 1.0; X[n_half:, 2] += 1.0
    noise_01 = rng.randn(n_samples) * 0.8
    X[:, 0] += noise_01; X[:, 1] -= noise_01
    noise_12 = rng.randn(n_samples) * 0.8
    X[:, 1] += noise_12; X[:, 2] -= noise_12
    return X, y


# ===========================================================================
# Test 1: Möbius inversion analytical correctness
# ===========================================================================

class TestMobiusInversion:
    def test_single_feature(self):
        cache = {frozenset(): 0.0, frozenset([0]): 0.5}
        assert np.isclose(interaction_term(frozenset([0]), cache), 0.5)

    def test_pairwise_synergy(self):
        cache = {frozenset(): 0.0, frozenset([0]): 0.3, frozenset([1]): 0.2, frozenset([0,1]): 0.7}
        assert np.isclose(interaction_term(frozenset([0,1]), cache), 0.2)

    def test_triplet_interaction(self):
        cache = {
            frozenset(): 0.0, frozenset([0]): 0.1, frozenset([1]): 0.1, frozenset([2]): 0.1,
            frozenset([0,1]): 0.25, frozenset([0,2]): 0.25, frozenset([1,2]): 0.25,
            frozenset([0,1,2]): 0.8,
        }
        expected = 0.8 - 0.25 - 0.25 - 0.25 + 0.1 + 0.1 + 0.1
        assert np.isclose(interaction_term(frozenset([0,1,2]), cache), expected)

    def test_no_synergy(self):
        cache = {frozenset(): 0.0, frozenset([0]): 0.3, frozenset([1]): 0.4, frozenset([0,1]): 0.7}
        assert np.isclose(interaction_term(frozenset([0,1]), cache), 0.0)

    def test_reconstruction_identity(self):
        """Verify: Delta(S) = sum_{T subseteq S} I(T) for all S."""
        cache = {
            frozenset(): 0.0, frozenset([0]): 0.2, frozenset([1]): 0.3, frozenset([2]): 0.1,
            frozenset([0,1]): 0.6, frozenset([0,2]): 0.4, frozenset([1,2]): 0.5,
            frozenset([0,1,2]): 0.9,
        }
        interactions = mobius_inversion(cache, max_order=3, feature_set={0,1,2})
        assert len(interactions) == 7
        for S in cache:
            if len(S) == 0: continue
            reconstructed = sum(interactions[T] for T in interactions if T.issubset(S))
            assert np.isclose(reconstructed, cache[S]), f"Failed for {S}"


# ===========================================================================
# Test 2: Topology-guided pruning
# ===========================================================================

class TestTopologyPruning:
    def test_chain_max_distance_1(self):
        n = 5
        adj = np.zeros((n, n))
        for i in range(n-1): adj[i, i+1] = adj[i+1, i] = 1
        pruner = TopologyPruner(adj, max_order=3, max_distance=1)
        pruner.fit()
        triplets = [s for s in pruner.allowed_subsets_ if len(s) == 3]
        assert len(triplets) == 0  # No triplet within distance 1 in a chain

    def test_chain_max_distance_2(self):
        n = 5
        adj = np.zeros((n, n))
        for i in range(n-1): adj[i, i+1] = adj[i+1, i] = 1
        pruner = TopologyPruner(adj, max_order=3, max_distance=2)
        pruner.fit()
        triplets = [s for s in pruner.allowed_subsets_ if len(s) == 3]
        assert len(triplets) > 0

    def test_disconnected_no_cross_cluster(self):
        adj = np.zeros((6, 6))
        adj[0,1] = adj[1,0] = adj[1,2] = adj[2,1] = 1
        adj[3,4] = adj[4,3] = adj[4,5] = adj[5,4] = 1
        pruner = TopologyPruner(adj, max_order=2, max_distance=2)
        pruner.fit()
        for pair in [s for s in pruner.allowed_subsets_ if len(s) == 2]:
            elems = sorted(pair)
            assert not (elems[0] < 3 and elems[1] >= 3)

    def test_pruning_reduces_subsets(self):
        rng = np.random.RandomState(42)
        adj = np.zeros((10, 10))
        for i, j in combinations(range(10), 2):
            if rng.rand() < 0.2: adj[i,j] = adj[j,i] = 1
        pruner = TopologyPruner(adj, max_order=3, max_distance=2)
        pruner.fit()
        assert pruner.pruning_ratio_ > 0.3


# ===========================================================================
# Test 3: Compressed sensing
# ===========================================================================

class TestCompressedSensing:
    def test_measurement_coverage(self):
        cs = CompressedInteractionSensing(n_features=8, max_order=3)
        subsets = cs.design_measurements()
        for i in range(8):
            assert frozenset([i]) in subsets

    def test_sparse_recovery_basic(self):
        cs = CompressedInteractionSensing(n_features=5, max_order=2, random_state=42)
        true_I = {s: 0.0 for s in cs.all_subsets_}
        true_I[frozenset([0])] = 0.5
        true_I[frozenset([1])] = 0.3
        true_I[frozenset([0,1])] = 0.2

        subsets = cs.design_measurements()
        cs.build_sensing_matrix()
        delta_values = {}
        for T in subsets:
            delta_values[T] = sum(true_I[S] for S in cs.all_subsets_ if T.issubset(S))

        recovered = cs.recover(delta_values, regularization=0.001)
        assert abs(recovered.get(frozenset([0]), 0) - 0.5) < 0.2


# ===========================================================================
# Test 4: End-to-end pipeline
# ===========================================================================

class TestEndToEnd:
    def test_separable_exact(self):
        X, y = make_separable_data(n_samples=200, n_features=4, separation=3.0)
        decomp = MobiusDecomposition(scorer=geometric_depth_cv, max_order=2, verbose=False)
        decomp.fit(X, y)
        assert len(decomp.interactions_) > 0
        for k, v in decomp.interactions_.items():
            if len(k) == 1: assert v > -0.05, f"Feature {k} unexpectedly negative: {v}"

    def test_gate_pass(self):
        X, y = make_separable_data(separation=3.0)
        passes, score = separability_gate(X, y, verbose=False)
        assert passes and score > 0.6

    def test_gate_fail(self):
        X, y = make_separable_data(separation=0.01)
        passes, _ = separability_gate(X, y, verbose=False)
        assert not passes

    def test_stability_decomposer(self):
        X, y = make_separable_data(n_samples=150, n_features=4, separation=3.0)
        sd = StabilityDecomposer(max_order=2, method="exact", cv_folds=3, verbose=False)
        sd.fit(X, y, feature_names=["TF_A", "TF_B", "TF_C", "TF_D"])
        top = sd.top_interactions(n=5)
        assert len(top) > 0

    def test_variance_explained_sums_to_one(self):
        X, y = make_separable_data(n_samples=150, n_features=4, separation=3.0)
        decomp = MobiusDecomposition(scorer=geometric_depth_cv, max_order=2, verbose=False)
        decomp.fit(X, y)
        total = sum(decomp.variance_explained().values())
        assert np.isclose(total, 1.0, atol=0.01)

    def test_interaction_matrix_symmetric(self):
        X, y = make_separable_data(n_samples=150, n_features=4, separation=3.0)
        decomp = MobiusDecomposition(scorer=geometric_depth_cv, max_order=2, verbose=False)
        decomp.fit(X, y)
        mat, names = decomp.interaction_matrix(order=2)
        assert np.allclose(mat, mat.T)

    def test_triplet_detection(self):
        X, y = make_triplet_data(n_samples=400)
        decomp = MobiusDecomposition(scorer=geometric_depth_cv, max_order=3, verbose=False)
        decomp.fit(X, y)
        assert frozenset([0,1,2]) in decomp.interactions_
        top3 = decomp.top_interactions(order=3, n=3)
        assert len(top3) > 0


if __name__ == "__main__":
    if pytest:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("pytest not installed; run tests manually")
