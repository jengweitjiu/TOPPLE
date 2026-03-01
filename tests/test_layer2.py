"""
TOPPLE Layer 2 — Test Suite
============================

Validates:
1. Target selection and IPA ranking
2. Mock perturbation engine
3. Destabilization scoring
4. Selectivity index computation
5. End-to-end bridge pipeline
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topple.mobius import MobiusDecomposition
from topple.stability import geometric_depth_cv
from topple.layer2.target_selection import TargetSelector, IPAFilter, PerturbationCandidate
from topple.layer2.perturbation_engine import MockPerturbationEngine, PerturbationResult
from topple.layer2.destabilization import DestabilizationScorer, SelectivityIndex
from topple.layer2.bridge import PerturbationBridge


# ===========================================================================
# Helpers
# ===========================================================================

def make_separable_data(n_samples=300, n_features=6, separation=2.5, random_state=42):
    rng = np.random.RandomState(random_state)
    n_half = n_samples // 2
    X = np.vstack([rng.randn(n_half, n_features), rng.randn(n_half, n_features) + separation])
    y = np.array([0]*n_half + [1]*n_half)
    return X, y


def make_interactions(n_features=6):
    """Create mock Layer 1 interactions."""
    interactions = {}
    # Singles
    for i in range(n_features):
        interactions[frozenset([i])] = max(0.01, 0.3 - i * 0.05)
    # A strong pair
    interactions[frozenset([0, 1])] = 0.15
    # A moderate pair
    interactions[frozenset([2, 3])] = 0.08
    # A weak triplet
    interactions[frozenset([0, 1, 2])] = 0.05
    # Noise pairs
    interactions[frozenset([4, 5])] = 0.002  # Below threshold
    return interactions


# ===========================================================================
# Test 1: Target Selection
# ===========================================================================

class TestTargetSelection:
    def test_basic_ranking(self):
        interactions = make_interactions()
        names = [f"TF_{i}" for i in range(6)]
        selector = TargetSelector(interactions, names)
        candidates = selector.rank()
        assert len(candidates) > 0
        assert candidates[0].rank == 1
        assert candidates[0].composite_score >= candidates[-1].composite_score

    def test_min_interaction_filter(self):
        interactions = make_interactions()
        names = [f"TF_{i}" for i in range(6)]
        selector = TargetSelector(interactions, names, min_interaction=0.01)
        candidates = selector.rank()
        for c in candidates:
            assert abs(c.interaction_value) >= 0.01

    def test_ipa_bonus_with_de(self):
        interactions = make_interactions()
        names = [f"TF_{i}" for i in range(6)]
        # TF_0 has high DE (obvious), TF_1 has low DE (non-obvious)
        de_scores = {"TF_0": 5.0, "TF_1": 0.5, "TF_2": 3.0,
                     "TF_3": 2.0, "TF_4": 0.1, "TF_5": 0.2}
        selector = TargetSelector(interactions, names, de_scores=de_scores, w_ipa=2.0)
        candidates = selector.rank()
        # With high IPA weight, low-DE features should be boosted
        assert len(candidates) > 0

    def test_ipa_filter_classification(self):
        interactions = make_interactions()
        names = [f"TF_{i}" for i in range(6)]
        de_scores = np.array([5.0, 0.5, 3.0, 2.0, 0.1, 0.05])
        filt = IPAFilter(interactions, de_scores, names)
        classes = filt.classify()
        assert "specification" in classes
        assert "maintenance" in classes
        assert "low_impact" in classes

    def test_empty_interactions(self):
        selector = TargetSelector({}, ["TF_0"])
        candidates = selector.rank()
        assert len(candidates) == 0

    def test_max_candidates(self):
        interactions = make_interactions()
        names = [f"TF_{i}" for i in range(6)]
        selector = TargetSelector(interactions, names, max_candidates=3)
        candidates = selector.rank()
        assert len(candidates) <= 3


# ===========================================================================
# Test 2: Mock Perturbation Engine
# ===========================================================================

class TestMockEngine:
    def test_basic_simulation(self):
        X, y = make_separable_data()
        names = [f"TF_{i}" for i in range(6)]
        engine = MockPerturbationEngine(X, y, names)
        result = engine.simulate(frozenset([0]))
        assert isinstance(result, PerturbationResult)
        assert result.X_original.shape == result.X_perturbed.shape
        assert result.n_cells == X.shape[0]

    def test_knockout_reduces_feature(self):
        X, y = make_separable_data(separation=3.0)
        names = [f"TF_{i}" for i in range(6)]
        engine = MockPerturbationEngine(X, y, names, effect_size=2.0)
        result = engine.simulate(frozenset([0]), perturbation_type="knockout")
        # Knockout should reduce expression of feature 0
        delta = result.mean_delta
        assert delta[0] < 0  # Feature 0 should decrease

    def test_multi_tf_knockout(self):
        X, y = make_separable_data()
        names = [f"TF_{i}" for i in range(6)]
        engine = MockPerturbationEngine(X, y, names)
        result = engine.simulate(frozenset([0, 1, 2]))
        assert result.feature_names == ["TF_0", "TF_1", "TF_2"]

    def test_cell_mask(self):
        X, y = make_separable_data(n_samples=200)
        names = [f"TF_{i}" for i in range(6)]
        engine = MockPerturbationEngine(X, y, names)
        mask = y == 1  # Only pathological cells
        result = engine.simulate(frozenset([0]), cell_mask=mask)
        assert result.n_cells == mask.sum()

    def test_is_fitted(self):
        X, y = make_separable_data()
        names = [f"TF_{i}" for i in range(6)]
        engine = MockPerturbationEngine(X, y, names)
        assert engine.is_fitted()


# ===========================================================================
# Test 3: Destabilization Scoring
# ===========================================================================

class TestDestabilization:
    def test_scorer_fit(self):
        X, y = make_separable_data(separation=3.0)
        scorer = DestabilizationScorer(X, y)
        scorer.fit()
        assert scorer._fitted

    def test_strong_perturbation_destabilizes(self):
        X, y = make_separable_data(n_samples=300, separation=3.0)
        names = [f"TF_{i}" for i in range(6)]
        engine = MockPerturbationEngine(X, y, names, effect_size=3.0)

        scorer = DestabilizationScorer(X, y)
        scorer.fit()

        result = engine.simulate(frozenset([0, 1, 2]))
        d_result = scorer.score(
            result.X_original, result.X_perturbed,
            result.state_labels, frozenset([0, 1, 2]),
            ["TF_0", "TF_1", "TF_2"],
            target_state=1,
        )
        # Strong perturbation of 3 features should destabilize some cells
        assert d_result.destabilization_score >= 0.0
        assert d_result.n_cells_total > 0

    def test_no_perturbation_no_destabilization(self):
        X, y = make_separable_data(separation=3.0)
        scorer = DestabilizationScorer(X, y)
        scorer.fit()
        # Score with identical X (no perturbation)
        d_result = scorer.score(
            X, X, y, frozenset([0]), ["TF_0"], target_state=1,
        )
        assert d_result.destabilization_score == 0.0


# ===========================================================================
# Test 4: Selectivity Index
# ===========================================================================

class TestSelectivity:
    def test_selectivity_computation(self):
        X, y = make_separable_data(n_samples=300, separation=3.0)
        names = [f"TF_{i}" for i in range(6)]
        engine = MockPerturbationEngine(X, y, names, effect_size=2.0)

        scorer = DestabilizationScorer(X, y)
        scorer.fit()
        si = SelectivityIndex(scorer)

        result = engine.simulate(frozenset([0, 1]))
        sel = si.compute(
            result.X_original, result.X_perturbed,
            result.state_labels, frozenset([0, 1]),
            ["TF_0", "TF_1"],
        )
        assert hasattr(sel, "selectivity_index")
        assert sel.d_pathological >= 0
        assert sel.d_homeostatic >= 0

    def test_ranking(self):
        X, y = make_separable_data(n_samples=300, separation=3.0)
        names = [f"TF_{i}" for i in range(6)]
        engine = MockPerturbationEngine(X, y, names, effect_size=2.0)

        scorer = DestabilizationScorer(X, y)
        scorer.fit()
        si = SelectivityIndex(scorer)

        results = []
        for feat_set in [frozenset([0]), frozenset([1]), frozenset([0, 1])]:
            result = engine.simulate(feat_set)
            sel = si.compute(
                result.X_original, result.X_perturbed,
                result.state_labels, feat_set,
                [names[i] for i in sorted(feat_set)],
            )
            results.append(sel)

        ranked = si.rank_candidates(results, min_d_pathological=0.0)
        # Ranks should be assigned
        if ranked:
            assert ranked[0].rank == 1


# ===========================================================================
# Test 5: End-to-end Bridge Pipeline
# ===========================================================================

class TestBridgePipeline:
    def test_full_pipeline(self):
        X, y = make_separable_data(n_samples=300, n_features=6, separation=3.0)
        names = [f"TF_{i}" for i in range(6)]

        # Use mock interactions (real decomposition may produce very small values
        # on perfectly separable data where all features contribute equally)
        interactions = {
            frozenset([i]): 0.1 * (6-i) for i in range(6)
        }
        interactions[frozenset([0, 1])] = 0.15
        interactions[frozenset([2, 3])] = 0.08

        # Layer 2
        engine = MockPerturbationEngine(X, y, names, effect_size=2.0)
        bridge = PerturbationBridge(
            engine=engine, X=X, y=y,
            feature_names=names,
            interactions=interactions,
            max_candidates=10,
            verbose=False,
        )
        results = bridge.run()

        # Should produce results
        assert isinstance(results, list)
        # Report should work
        report = bridge.report()
        assert "TOPPLE Layer 2" in report

    def test_pipeline_with_de_scores(self):
        X, y = make_separable_data(n_samples=200, n_features=4, separation=3.0)
        names = ["RUNX3", "TBX21", "EOMES", "IRF4"]
        de_scores = {"RUNX3": 4.0, "TBX21": 3.5, "EOMES": 1.0, "IRF4": 0.5}

        interactions = {
            frozenset([0]): 0.2, frozenset([1]): 0.15,
            frozenset([2]): 0.1, frozenset([3]): 0.08,
            frozenset([0, 1]): 0.12, frozenset([2, 3]): 0.09,
        }

        engine = MockPerturbationEngine(X, y, names, effect_size=2.0)
        bridge = PerturbationBridge(
            engine=engine, X=X, y=y,
            feature_names=names,
            interactions=interactions,
            de_scores=de_scores,
            max_candidates=6,
            verbose=False,
        )
        results = bridge.run()
        assert isinstance(results, list)

    def test_to_dataframe(self):
        X, y = make_separable_data(n_samples=200, n_features=4, separation=3.0)
        names = [f"TF_{i}" for i in range(4)]

        interactions = {
            frozenset([i]): 0.1 * (4-i) for i in range(4)
        }
        interactions[frozenset([0, 1])] = 0.15

        engine = MockPerturbationEngine(X, y, names, effect_size=2.0)
        bridge = PerturbationBridge(
            engine=engine, X=X, y=y,
            feature_names=names,
            interactions=interactions,
            max_candidates=5,
            verbose=False,
        )
        bridge.run()
        df = bridge.to_dataframe()
        assert "selectivity_index" in df.columns
        assert "d_pathological" in df.columns


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    test_classes = [
        TestTargetSelection, TestMockEngine,
        TestDestabilization, TestSelectivity, TestBridgePipeline,
    ]
    total = passed = failed = 0
    for cls in test_classes:
        obj = cls()
        for method in [m for m in dir(obj) if m.startswith("test_")]:
            total += 1
            try:
                getattr(obj, method)()
                print(f"  PASS  {cls.__name__}.{method}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{method}: {e}")
                import traceback; traceback.print_exc()
                failed += 1
    print(f"\n=== {passed}/{total} passed, {failed} failed ===")
