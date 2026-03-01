"""
Möbius Inversion on the Feature Subset Lattice
===============================================

Implements higher-order interaction decomposition via Möbius inversion.

For a feature set F = {f_1, ..., f_p}, the k-th order interaction term I(S)
for a subset S ⊆ F with |S| = k is defined as:

    I(S) = Σ_{T⊆S} (-1)^{|S|-|T|} Δ(T)

where Δ(T) is the stability loss upon removing all features in subset T.

When |S| = 2, this reduces to the DGSA pairwise synergy:
    S(A,B) = Δ({A,B}) - (Δ({A}) + Δ({B}))

Higher-order terms (k >= 3) capture regulatory interactions invisible to
pairwise analysis, such as cooperative TF binding, feed-forward loops,
and coherent multi-input motifs.

References
----------
- Rota, G.-C. (1964). On the foundations of combinatorial theory I.
  Theory of Möbius functions. Z. Wahrscheinlichkeitstheorie, 2, 340-368.
- Grabisch, M. & Roubens, M. (1999). An axiomatic approach to the concept
  of interaction among players in cooperative games. Int. J. Game Theory, 28.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.special import comb


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FeatureSet = FrozenSet[int]
DeltaCache = Dict[FeatureSet, float]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def stability_loss(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: FeatureSet,
    scorer: Callable,
    *,
    X_full_score: Optional[float] = None,
) -> float:
    """
    Compute stability loss Δ(T) when removing features in `feature_indices`.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Regulon activity matrix (e.g., AUCell scores from pySCENIC).
    y : np.ndarray, shape (n_samples,)
        Binary state labels (0 = homeostatic, 1 = pathological).
    feature_indices : frozenset of int
        Indices of features to ablate (remove).
    scorer : callable
        Function (X, y) -> float that returns a stability metric.
        Typically cross-validated geometric depth or AUC.
    X_full_score : float, optional
        Precomputed score on full feature set. If None, computed internally.

    Returns
    -------
    float
        Stability loss Δ(T) = score_full - score_ablated.
        Positive values indicate destabilization upon feature removal.
    """
    if X_full_score is None:
        X_full_score = scorer(X, y)

    if len(feature_indices) == 0:
        return 0.0

    # Ablate by removing specified feature columns
    remaining = [i for i in range(X.shape[1]) if i not in feature_indices]

    if len(remaining) == 0:
        # All features removed -> maximum instability
        return X_full_score

    X_ablated = X[:, remaining]
    ablated_score = scorer(X_ablated, y)

    return X_full_score - ablated_score


def interaction_term(
    subset: FeatureSet,
    delta_cache: DeltaCache,
) -> float:
    """
    Compute the Möbius interaction term I(S) for a feature subset S.

    I(S) = Σ_{T⊆S} (-1)^{|S|-|T|} Δ(T)

    Parameters
    ----------
    subset : frozenset of int
        The feature subset S for which to compute the interaction.
    delta_cache : dict
        Mapping from frozenset -> Δ(T) for all subsets T ⊆ S.
        Must include the empty set: frozenset() -> 0.0.

    Returns
    -------
    float
        The k-th order interaction term, where k = |S|.

    Raises
    ------
    KeyError
        If delta_cache is missing a required subset.

    Notes
    -----
    For |S| = 1: I({A}) = Δ({A}) - Δ(∅) = Δ({A})  [marginal contribution]
    For |S| = 2: I({A,B}) = Δ({A,B}) - Δ({A}) - Δ({B}) + Δ(∅) = synergy
    For |S| = 3: I({A,B,C}) = Δ({A,B,C}) - Δ({A,B}) - Δ({A,C}) - Δ({B,C})
                               + Δ({A}) + Δ({B}) + Δ({C}) - Δ(∅)
    """
    subset_list = sorted(subset)
    k = len(subset_list)
    total = 0.0

    # Iterate over all subsets T ⊆ S
    for size in range(k + 1):
        sign = (-1) ** (k - size)
        for combo in combinations(subset_list, size):
            T = frozenset(combo)
            if T not in delta_cache:
                raise KeyError(
                    f"Missing Δ value for subset {T}. "
                    f"Ensure all subsets of {subset} are in delta_cache."
                )
            total += sign * delta_cache[T]

    return total


def mobius_inversion(
    delta_cache: DeltaCache,
    max_order: Optional[int] = None,
    feature_set: Optional[Set[int]] = None,
) -> Dict[FeatureSet, float]:
    """
    Compute all Möbius interaction terms from a delta cache.

    Parameters
    ----------
    delta_cache : dict
        Mapping from frozenset -> Δ(T) for evaluated subsets.
        Must include frozenset() -> 0.0.
    max_order : int, optional
        Maximum interaction order to compute. Default: max subset size in cache.
    feature_set : set of int, optional
        The full feature set. If None, inferred from delta_cache keys.

    Returns
    -------
    dict
        Mapping from frozenset -> I(S) for all computed interaction terms.
    """
    # Ensure empty set is present
    if frozenset() not in delta_cache:
        delta_cache[frozenset()] = 0.0

    # Infer feature set
    if feature_set is None:
        feature_set = set()
        for key in delta_cache:
            feature_set.update(key)

    if max_order is None:
        max_order = max(len(k) for k in delta_cache) if delta_cache else 0

    interactions = {}

    for order in range(1, max_order + 1):
        for combo in combinations(sorted(feature_set), order):
            S = frozenset(combo)
            # Check if all subsets of S are available
            all_subsets_available = True
            for sub_size in range(order + 1):
                for sub_combo in combinations(sorted(S), sub_size):
                    if frozenset(sub_combo) not in delta_cache:
                        all_subsets_available = False
                        break
                if not all_subsets_available:
                    break

            if all_subsets_available:
                interactions[S] = interaction_term(S, delta_cache)

    return interactions


# ---------------------------------------------------------------------------
# Main decomposition class
# ---------------------------------------------------------------------------


class MobiusDecomposition:
    """
    Higher-order stability decomposition via Möbius inversion.

    Extends DGSA's pairwise synergy S(A,B) to arbitrary-order interaction
    terms I(S) on the feature subset lattice.

    Parameters
    ----------
    scorer : callable
        Function (X, y) -> float returning a stability metric.
        Should be cross-validated (e.g., geometric_depth_cv).
    max_order : int, default=3
        Maximum interaction order to evaluate.
        k=1: marginal contributions (single-feature ablation)
        k=2: pairwise synergy (DGSA-equivalent)
        k=3: triplet interactions (new in TOPPLE)
    allowed_subsets : list of frozenset, optional
        If provided, only evaluate these subsets (from TopologyPruner).
        Otherwise, evaluate all subsets up to max_order.
    n_jobs : int, default=1
        Number of parallel jobs for stability loss computation.
    verbose : bool, default=True
        Print progress information.

    Attributes
    ----------
    delta_cache_ : dict
        Cached Δ(T) values for all evaluated subsets.
    interactions_ : dict
        Computed I(S) interaction terms.
    feature_names_ : list of str
        Feature names (if provided).

    Examples
    --------
    >>> from topple import MobiusDecomposition, geometric_depth_cv
    >>> decomp = MobiusDecomposition(scorer=geometric_depth_cv, max_order=3)
    >>> decomp.fit(X_regulon, y_state)
    >>> # Get top triplet interactions
    >>> decomp.top_interactions(order=3, n=10)
    """

    def __init__(
        self,
        scorer: Callable,
        max_order: int = 3,
        allowed_subsets: Optional[List[FeatureSet]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        self.scorer = scorer
        self.max_order = max_order
        self.allowed_subsets = allowed_subsets
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MobiusDecomposition":
        """
        Compute all interaction terms up to max_order.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Regulon activity matrix.
        y : np.ndarray, shape (n_samples,)
            Binary state labels.
        feature_names : list of str, optional
            Names for features (e.g., regulon names from pySCENIC).

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.feature_names_ = (
            feature_names
            if feature_names is not None
            else [f"f{i}" for i in range(n_features)]
        )

        if self.max_order > n_features:
            self.max_order = n_features
            warnings.warn(
                f"max_order reduced to {n_features} (number of features)."
            )

        # Step 1: Compute full model score
        if self.verbose:
            print(f"[TOPPLE] Computing full model score...")
        full_score = self.scorer(X, y)
        if self.verbose:
            print(f"[TOPPLE] Full model score: {full_score:.4f}")

        # Step 2: Determine subsets to evaluate
        subsets_to_eval = self._determine_subsets(n_features)
        n_total = len(subsets_to_eval)
        if self.verbose:
            print(
                f"[TOPPLE] Evaluating {n_total} subsets "
                f"(max_order={self.max_order}, features={n_features})"
            )

        # Step 3: Compute Δ(T) for all subsets
        self.delta_cache_ = {frozenset(): 0.0}

        for i, subset in enumerate(subsets_to_eval):
            if self.verbose and (i + 1) % 50 == 0:
                print(f"[TOPPLE] Progress: {i+1}/{n_total} subsets evaluated")

            delta = stability_loss(
                X, y, subset, self.scorer, X_full_score=full_score
            )
            self.delta_cache_[subset] = delta

        # Step 4: Möbius inversion
        if self.verbose:
            print("[TOPPLE] Computing Möbius interaction terms...")

        self.interactions_ = mobius_inversion(
            self.delta_cache_,
            max_order=self.max_order,
            feature_set=set(range(n_features)),
        )

        # Step 5: Summary
        if self.verbose:
            self._print_summary()

        return self

    def _determine_subsets(self, n_features: int) -> List[FeatureSet]:
        """Determine which subsets to evaluate."""
        if self.allowed_subsets is not None:
            return [s for s in self.allowed_subsets if len(s) <= self.max_order]

        subsets = []
        for order in range(1, self.max_order + 1):
            n_combos = int(comb(n_features, order))
            if n_combos > 50000:
                warnings.warn(
                    f"Order {order} has {n_combos} subsets. "
                    f"Consider using TopologyPruner or reducing max_order."
                )
            for combo in combinations(range(n_features), order):
                subsets.append(frozenset(combo))
        return subsets

    def _print_summary(self):
        """Print decomposition summary."""
        print("\n[TOPPLE] === Decomposition Summary ===")
        for order in range(1, self.max_order + 1):
            order_terms = {
                k: v for k, v in self.interactions_.items() if len(k) == order
            }
            if order_terms:
                vals = list(order_terms.values())
                n_sig = sum(1 for v in vals if abs(v) > 0.01)
                print(
                    f"  Order {order}: {len(order_terms)} terms, "
                    f"{n_sig} significant (|I|>0.01), "
                    f"range [{min(vals):.4f}, {max(vals):.4f}]"
                )

    def top_interactions(
        self,
        order: Optional[int] = None,
        n: int = 10,
        abs_threshold: float = 0.0,
    ) -> List[Tuple[List[str], float]]:
        """
        Return top interaction terms ranked by absolute magnitude.

        Parameters
        ----------
        order : int, optional
            Filter to specific interaction order. None returns all orders.
        n : int, default=10
            Number of top interactions to return.
        abs_threshold : float, default=0.0
            Minimum |I(S)| to include.

        Returns
        -------
        list of (feature_names, interaction_value)
            Sorted by absolute interaction magnitude (descending).
        """
        items = self.interactions_.items()
        if order is not None:
            items = [(k, v) for k, v in items if len(k) == order]

        items = [(k, v) for k, v in items if abs(v) > abs_threshold]
        items = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:n]

        results = []
        for subset, value in items:
            names = [self.feature_names_[i] for i in sorted(subset)]
            results.append((names, value))

        return results

    def interaction_matrix(self, order: int = 2) -> Tuple[np.ndarray, List[str]]:
        """
        Return interaction terms as a matrix (for order=2) or tensor.

        Parameters
        ----------
        order : int, default=2
            Interaction order. Currently supports 2 (matrix).

        Returns
        -------
        matrix : np.ndarray
            For order=2: (n_features, n_features) symmetric matrix.
        names : list of str
            Feature names for axes.
        """
        if order != 2:
            raise NotImplementedError("Matrix view currently supports order=2 only.")

        p = self.n_features_
        mat = np.zeros((p, p))

        for subset, value in self.interactions_.items():
            if len(subset) == 2:
                i, j = sorted(subset)
                mat[i, j] = value
                mat[j, i] = value

        # Fill diagonal with marginal contributions
        for subset, value in self.interactions_.items():
            if len(subset) == 1:
                (i,) = subset
                mat[i, i] = value

        return mat, self.feature_names_

    def decomposition_table(self) -> "pd.DataFrame":
        """
        Return all interactions as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: subset, order, features, interaction, abs_interaction.
        """
        import pandas as pd

        rows = []
        for subset, value in self.interactions_.items():
            names = [self.feature_names_[i] for i in sorted(subset)]
            rows.append(
                {
                    "subset": str(sorted(subset)),
                    "order": len(subset),
                    "features": " + ".join(names),
                    "interaction": value,
                    "abs_interaction": abs(value),
                }
            )

        df = pd.DataFrame(rows)
        return df.sort_values("abs_interaction", ascending=False).reset_index(
            drop=True
        )

    def variance_explained(self) -> Dict[int, float]:
        """
        Compute fraction of total interaction variance at each order.

        Returns
        -------
        dict
            Mapping order -> fraction of total sum of squared interactions.
        """
        total_ss = sum(v ** 2 for v in self.interactions_.values())
        if total_ss == 0:
            return {}

        result = {}
        for order in range(1, self.max_order + 1):
            order_ss = sum(
                v ** 2
                for k, v in self.interactions_.items()
                if len(k) == order
            )
            result[order] = order_ss / total_ss

        return result
