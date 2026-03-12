"""
Stability Scoring and DGSA Integration
========================================

Provides the scoring functions used by MobiusDecomposition to quantify
cell state stability. Integrates with the DGSA framework for geometric
depth computation and separability gating.

Core metric: Cross-validated geometric depth (or AUC), which measures
how well a classifier can distinguish two cell states based on regulon
activity. High scores indicate separable (stable) states; low scores
indicate overlapping (unstable or poorly defined) states.

Separability Gate
-----------------
From DGSA: decomposition results are only reliable when the baseline
classification (full feature set) exceeds a CV AUC threshold (default
0.60). Below this threshold, the boundary geometry is too noisy for
meaningful ablation analysis.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def geometric_depth_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    classifier: str = "svm",
    random_state: int = 42,
) -> float:
    """
    Cross-validated classification score as a stability proxy.

    Uses stratified k-fold CV to compute mean AUC, which serves as a
    measure of state separability (geometric depth proxy).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Regulon activity matrix.
    y : np.ndarray, shape (n_samples,)
        Binary state labels.
    n_splits : int, default=5
        Number of CV folds.
    classifier : str, default="svm"
        Classifier to use: "svm" (RBF kernel) or "rf" (random forest).
    random_state : int, default=42
        Random seed.

    Returns
    -------
    float
        Mean CV AUC score ∈ [0, 1].
    """
    if X.shape[1] == 0:
        return 0.5  # No features -> random classifier

    if len(np.unique(y)) < 2:
        return 0.5  # Single class -> undefined

    # Build classifier pipeline
    if classifier == "svm":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ])
    elif classifier == "rf":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            )),
        ])
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    # Stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            aucs.append(auc)
        except Exception:
            aucs.append(0.5)

    return float(np.mean(aucs))


def separability_gate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.60,
    scorer: Optional[Callable] = None,
    verbose: bool = True,
) -> Tuple[bool, float]:
    """
    DGSA separability gate: check if states are sufficiently separable.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Regulon activity matrix (full feature set).
    y : np.ndarray, shape (n_samples,)
        Binary state labels.
    threshold : float, default=0.60
        Minimum CV AUC to pass the gate.
    scorer : callable, optional
        Custom scoring function. Default: geometric_depth_cv.
    verbose : bool, default=True
        Print gate result.

    Returns
    -------
    passes : bool
        True if full-model score exceeds threshold.
    score : float
        The full-model CV AUC score.
    """
    if scorer is None:
        scorer = geometric_depth_cv

    score = scorer(X, y)
    passes = score >= threshold

    if verbose:
        status = "PASS" if passes else "FAIL"
        print(
            f"[TOPPLE] Separability gate: CV AUC = {score:.4f} "
            f"(threshold = {threshold:.2f}) -> {status}"
        )
        if not passes:
            print(
                f"[TOPPLE] WARNING: States are not sufficiently separable. "
                f"Decomposition results may be unreliable. Consider:\n"
                f"  1. Expanding the dataset\n"
                f"  2. Using broader state definitions\n"
                f"  3. Including additional regulatory features"
            )

    return passes, score


class StabilityDecomposer:
    """
    High-level interface for TOPPLE Layer 1 stability decomposition.

    Orchestrates the full pipeline: separability gating -> topology pruning
    -> Möbius decomposition (exact or compressed) -> result analysis.

    Parameters
    ----------
    max_order : int, default=3
        Maximum interaction order.
    method : str, default="exact"
        Decomposition method:
        - "exact": Full enumeration (feasible for p ≤ ~15)
        - "pruned": Topology-guided pruning (requires adjacency matrix)
        - "compressed": Compressed interaction sensing (for large p)
        - "hierarchical": Greedy ascent screening
    classifier : str, default="svm"
        Classifier for stability scoring.
    cv_folds : int, default=5
        Number of CV folds.
    separability_threshold : float, default=0.60
        Minimum CV AUC to proceed with decomposition.
    adjacency : np.ndarray, optional
        GRN adjacency matrix (required for method="pruned").
    verbose : bool, default=True
        Print progress.

    Examples
    --------
    >>> from topple import StabilityDecomposer
    >>> # Basic usage with exact enumeration
    >>> sd = StabilityDecomposer(max_order=3)
    >>> sd.fit(X_regulon, y_state, feature_names=regulon_names)
    >>> sd.top_interactions(order=3)

    >>> # With topology-guided pruning
    >>> from topple import grn_to_adjacency
    >>> adj = grn_to_adjacency(pyscenic_adj, regulon_names)
    >>> sd = StabilityDecomposer(max_order=4, method="pruned", adjacency=adj)
    >>> sd.fit(X_regulon, y_state)

    >>> # With compressed sensing for large feature sets
    >>> sd = StabilityDecomposer(max_order=3, method="compressed")
    >>> sd.fit(X_regulon, y_state)
    """

    def __init__(
        self,
        max_order: int = 3,
        method: str = "exact",
        classifier: str = "svm",
        cv_folds: int = 5,
        separability_threshold: float = 0.60,
        adjacency: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Initialize the stability decomposer with method selection and parameters."""
        self.max_order = max_order
        self.method = method
        self.classifier = classifier
        self.cv_folds = cv_folds
        self.separability_threshold = separability_threshold
        self.adjacency = adjacency
        self.verbose = verbose

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "StabilityDecomposer":
        """
        Run the full TOPPLE Layer 1 pipeline.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Regulon activity matrix.
        y : np.ndarray, shape (n_samples,)
            Binary state labels.
        feature_names : list of str, optional
            Regulon/feature names.

        Returns
        -------
        self
        """
        from .mobius import MobiusDecomposition
        from .pruning import TopologyPruner, HierarchicalScreener
        from .compressed_sensing import CompressedInteractionSensing

        n_samples, n_features = X.shape
        if feature_names is None:
            feature_names = [f"regulon_{i}" for i in range(n_features)]
        self.feature_names_ = feature_names

        # Scorer function
        def scorer(X_sub, y_sub):
            """Evaluate cross-validated geometric depth AUC on a feature subset."""
            return geometric_depth_cv(
                X_sub, y_sub,
                n_splits=self.cv_folds,
                classifier=self.classifier,
            )

        # Step 1: Separability gate
        self.gate_passes_, self.gate_score_ = separability_gate(
            X, y, threshold=self.separability_threshold,
            scorer=scorer, verbose=self.verbose,
        )

        if not self.gate_passes_:
            warnings.warn(
                f"Separability gate FAILED (AUC={self.gate_score_:.3f}). "
                f"Proceeding with caution — results may be unreliable."
            )

        # Step 2: Determine allowed subsets based on method
        allowed_subsets = None

        if self.method == "pruned":
            if self.adjacency is None:
                raise ValueError(
                    "Adjacency matrix required for method='pruned'. "
                    "Use grn_to_adjacency() to convert pySCENIC output."
                )
            pruner = TopologyPruner(
                self.adjacency,
                max_order=self.max_order,
            )
            pruner.fit()
            allowed_subsets = pruner.allowed_subsets_
            self.pruner_ = pruner
            if self.verbose:
                print(pruner.summary())

        elif self.method == "compressed":
            cs = CompressedInteractionSensing(
                n_features=n_features,
                max_order=self.max_order,
            )
            subsets_to_eval = cs.design_measurements()
            cs.build_sensing_matrix()

            # Compute Δ for sampled subsets
            if self.verbose:
                print(f"[TOPPLE] Computing Δ for {len(subsets_to_eval)} "
                      f"sampled subsets (compressed sensing)...")

            full_score = scorer(X, y)
            from .mobius import stability_loss as _stability_loss
            delta_values = {}
            for s in subsets_to_eval:
                delta_values[s] = _stability_loss(
                    X, y, s, scorer, X_full_score=full_score
                )

            # Recover interactions
            recovered = cs.recover(delta_values)
            self.cs_ = cs
            self.interactions_ = recovered
            self.decomposition_ = None  # No MobiusDecomposition object

            if self.verbose:
                print(cs.summary())
            return self

        # Step 3: Möbius decomposition (exact or pruned)
        decomp = MobiusDecomposition(
            scorer=scorer,
            max_order=self.max_order,
            allowed_subsets=allowed_subsets,
            verbose=self.verbose,
        )
        decomp.fit(X, y, feature_names=feature_names)

        self.decomposition_ = decomp
        self.interactions_ = decomp.interactions_
        self.delta_cache_ = decomp.delta_cache_

        return self

    def top_interactions(
        self,
        order: Optional[int] = None,
        n: int = 10,
    ) -> List[Tuple[List[str], float]]:
        """Return top interactions (delegates to decomposition)."""
        if self.decomposition_ is not None:
            return self.decomposition_.top_interactions(order=order, n=n)

        # For compressed sensing results
        items = list(self.interactions_.items())
        if order is not None:
            items = [(k, v) for k, v in items if len(k) == order]
        items = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:n]
        return [
            ([self.feature_names_[i] for i in sorted(k)], v)
            for k, v in items
        ]

    def variance_explained(self) -> Dict[int, float]:
        """Variance explained by each interaction order."""
        total_ss = sum(v ** 2 for v in self.interactions_.values())
        if total_ss == 0:
            return {}
        result = {}
        for order in range(1, self.max_order + 1):
            order_ss = sum(
                v ** 2 for k, v in self.interactions_.items() if len(k) == order
            )
            result[order] = order_ss / total_ss
        return result

    def report(self) -> str:
        """Generate a text report of the decomposition."""
        lines = [
            "=" * 60,
            "TOPPLE Layer 1: Higher-Order Stability Decomposition",
            "=" * 60,
            f"Method: {self.method}",
            f"Max order: {self.max_order}",
            f"Separability gate: {'PASS' if self.gate_passes_ else 'FAIL'} "
            f"(AUC = {self.gate_score_:.4f})",
            "",
            "Variance explained by order:",
        ]
        for order, frac in self.variance_explained().items():
            lines.append(f"  k={order}: {frac:.1%}")

        lines.extend(["", "Top interactions (all orders):"])
        for names, val in self.top_interactions(n=15):
            sign = "+" if val > 0 else ""
            lines.append(f"  {' × '.join(names)}: {sign}{val:.4f}")

        return "\n".join(lines)
