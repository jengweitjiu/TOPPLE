"""
Destabilization Scoring and Selectivity Index
===============================================

After perturbation simulation, evaluates how effectively each perturbation
destabilizes pathological cell states while preserving homeostatic ones.

Key metrics:
- Destabilization score D(P): fraction of pathological cells predicted to
  cross the decision boundary after perturbation P.
- Selectivity Index SI(P) = D_pathological(P) / D_homeostatic(P):
  high SI means selective destabilization of disease state.

The scorer re-evaluates geometric depth (or classifier confidence) of
perturbed profiles against the original classification boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class DestabilizationResult:
    """Result of destabilization scoring for a single perturbation."""

    perturbation_set: FrozenSet[int]
    feature_names: List[str]

    # Per-cell scores
    original_confidence: np.ndarray    # P(pathological) before perturbation
    perturbed_confidence: np.ndarray   # P(pathological) after perturbation
    confidence_drop: np.ndarray        # original - perturbed

    # Aggregate scores
    destabilization_score: float       # Fraction crossing boundary
    mean_confidence_drop: float        # Average confidence change
    n_cells_destabilized: int          # Count of cells crossing boundary
    n_cells_total: int

    # State-specific
    state_labels: np.ndarray

    def __repr__(self) -> str:
        """Return a summary string with destabilization score and cell counts."""
        return (
            f"DestabilizationResult("
            f"features={' + '.join(self.feature_names)}, "
            f"D={self.destabilization_score:.3f}, "
            f"cells={self.n_cells_destabilized}/{self.n_cells_total})"
        )


@dataclass
class SelectivityResult:
    """Selectivity analysis comparing pathological vs homeostatic destabilization."""

    perturbation_set: FrozenSet[int]
    feature_names: List[str]

    d_pathological: float     # Destabilization of pathological state
    d_homeostatic: float      # Destabilization of homeostatic state
    selectivity_index: float  # SI = d_path / d_homeo (higher = better)

    # Detailed
    pathological_result: DestabilizationResult
    homeostatic_result: DestabilizationResult

    rank: int = 0

    def __repr__(self) -> str:
        """Return a summary string with selectivity index and destabilization scores."""
        return (
            f"SelectivityResult("
            f"features={' + '.join(self.feature_names)}, "
            f"SI={self.selectivity_index:.2f}, "
            f"D_path={self.d_pathological:.3f}, "
            f"D_homeo={self.d_homeostatic:.3f})"
        )


class DestabilizationScorer:
    """
    Score how effectively a perturbation destabilizes a cell state.

    Trains a classifier on the original (unperturbed) data, then evaluates
    how many perturbed cells are predicted to belong to the opposite state.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_cells, n_features)
        Original regulon activity matrix for classifier training.
    y_train : np.ndarray, shape (n_cells,)
        Binary state labels (0 = homeostatic, 1 = pathological).
    classifier : sklearn estimator, optional
        Classifier to use. Default: SVM with RBF kernel.
    boundary_threshold : float, default=0.5
        Probability threshold for state assignment.
        Cells crossing this threshold are "destabilized".
    cv_folds : int, default=5
        Cross-validation folds for classifier training.

    Examples
    --------
    >>> scorer = DestabilizationScorer(X_regulon, y_state)
    >>> scorer.fit()
    >>> result = scorer.score(perturbation_result)
    >>> print(f"Destabilization: {result.destabilization_score:.1%}")
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        classifier=None,
        boundary_threshold: float = 0.5,
        cv_folds: int = 5,
    ):
        """Initialize the destabilization scorer with training data and classifier."""
        self.X_train = X_train
        self.y_train = y_train
        self.boundary_threshold = boundary_threshold
        self.cv_folds = cv_folds

        if classifier is None:
            self.classifier = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
            ])
        else:
            self.classifier = classifier

        self._fitted = False

    def fit(self) -> "DestabilizationScorer":
        """Train classifier on original data."""
        self.classifier.fit(self.X_train, self.y_train)
        self._fitted = True
        return self

    def score(
        self,
        X_original: np.ndarray,
        X_perturbed: np.ndarray,
        state_labels: np.ndarray,
        perturbation_set: FrozenSet[int],
        feature_names: List[str],
        target_state: int = 1,
    ) -> DestabilizationResult:
        """
        Score destabilization of cells in target_state.

        Parameters
        ----------
        X_original : np.ndarray
            Original expression for cells to evaluate.
        X_perturbed : np.ndarray
            Predicted post-perturbation expression.
        state_labels : np.ndarray
            State labels for these cells.
        perturbation_set : frozenset
            Feature indices being perturbed.
        feature_names : list of str
            Names of perturbed features.
        target_state : int, default=1
            The state we want to destabilize (1 = pathological).

        Returns
        -------
        DestabilizationResult
        """
        if not self._fitted:
            self.fit()

        # Get classifier confidence for target state
        orig_proba = self.classifier.predict_proba(X_original)[:, target_state]
        pert_proba = self.classifier.predict_proba(X_perturbed)[:, target_state]

        # Confidence drop
        conf_drop = orig_proba - pert_proba

        # Cells that cross the boundary
        # Originally in target state (above threshold) → now below threshold
        target_mask = state_labels == target_state
        if target_mask.sum() == 0:
            return DestabilizationResult(
                perturbation_set=perturbation_set,
                feature_names=feature_names,
                original_confidence=orig_proba,
                perturbed_confidence=pert_proba,
                confidence_drop=conf_drop,
                destabilization_score=0.0,
                mean_confidence_drop=0.0,
                n_cells_destabilized=0,
                n_cells_total=0,
                state_labels=state_labels,
            )

        # Among target-state cells: how many are now below threshold?
        target_orig = orig_proba[target_mask]
        target_pert = pert_proba[target_mask]

        # Cells that were confidently in target state and now aren't
        was_above = target_orig >= self.boundary_threshold
        now_below = target_pert < self.boundary_threshold
        n_destabilized = int((was_above & now_below).sum())
        n_target = int(was_above.sum())

        d_score = n_destabilized / max(n_target, 1)

        return DestabilizationResult(
            perturbation_set=perturbation_set,
            feature_names=feature_names,
            original_confidence=orig_proba,
            perturbed_confidence=pert_proba,
            confidence_drop=conf_drop,
            destabilization_score=d_score,
            mean_confidence_drop=float(conf_drop[target_mask].mean()),
            n_cells_destabilized=n_destabilized,
            n_cells_total=n_target,
            state_labels=state_labels,
        )


class SelectivityIndex:
    """
    Compute selectivity index: SI(P) = D_pathological(P) / D_homeostatic(P).

    High SI means the perturbation selectively destabilizes the pathological
    state without disrupting homeostatic populations.

    Parameters
    ----------
    scorer : DestabilizationScorer
        Fitted destabilization scorer.
    pathological_label : int, default=1
        Label for pathological state.
    homeostatic_label : int, default=0
        Label for homeostatic state.
    min_homeostatic_d : float, default=0.01
        Floor for D_homeostatic to avoid division by zero.
        Also serves as a safety threshold: if homeostatic
        destabilization is below this, SI is set to a large value.

    Examples
    --------
    >>> si = SelectivityIndex(scorer)
    >>> result = si.compute(X_orig, X_pert, labels, perturb_set, names)
    >>> print(f"Selectivity: {result.selectivity_index:.1f}x")
    """

    def __init__(
        self,
        scorer: DestabilizationScorer,
        pathological_label: int = 1,
        homeostatic_label: int = 0,
        min_homeostatic_d: float = 0.01,
    ):
        """Initialize selectivity index with a fitted destabilization scorer."""
        self.scorer = scorer
        self.pathological_label = pathological_label
        self.homeostatic_label = homeostatic_label
        self.min_homeostatic_d = min_homeostatic_d

    def compute(
        self,
        X_original: np.ndarray,
        X_perturbed: np.ndarray,
        state_labels: np.ndarray,
        perturbation_set: FrozenSet[int],
        feature_names: List[str],
    ) -> SelectivityResult:
        """
        Compute selectivity index for a perturbation.

        Parameters
        ----------
        X_original, X_perturbed : np.ndarray
            Expression before and after perturbation.
        state_labels : np.ndarray
            Binary state labels for cells.
        perturbation_set : frozenset
            Perturbed feature indices.
        feature_names : list of str
            Perturbed feature names.

        Returns
        -------
        SelectivityResult
        """
        # Score destabilization of pathological state
        path_result = self.scorer.score(
            X_original, X_perturbed, state_labels,
            perturbation_set, feature_names,
            target_state=self.pathological_label,
        )

        # Score destabilization of homeostatic state
        homeo_result = self.scorer.score(
            X_original, X_perturbed, state_labels,
            perturbation_set, feature_names,
            target_state=self.homeostatic_label,
        )

        d_path = path_result.destabilization_score
        d_homeo = max(homeo_result.destabilization_score, self.min_homeostatic_d)
        si = d_path / d_homeo

        return SelectivityResult(
            perturbation_set=perturbation_set,
            feature_names=feature_names,
            d_pathological=d_path,
            d_homeostatic=homeo_result.destabilization_score,
            selectivity_index=si,
            pathological_result=path_result,
            homeostatic_result=homeo_result,
        )

    def rank_candidates(
        self,
        results: List[SelectivityResult],
        min_d_pathological: float = 0.05,
    ) -> List[SelectivityResult]:
        """
        Rank candidates by selectivity, filtering low-effect perturbations.

        Parameters
        ----------
        results : list of SelectivityResult
            Unranked selectivity results.
        min_d_pathological : float, default=0.05
            Minimum pathological destabilization to consider viable.

        Returns
        -------
        list of SelectivityResult
            Sorted by selectivity index (descending), with ranks assigned.
        """
        # Filter by minimum pathological effect
        viable = [r for r in results if r.d_pathological >= min_d_pathological]

        # Sort by selectivity index
        viable.sort(key=lambda r: r.selectivity_index, reverse=True)

        for i, r in enumerate(viable):
            r.rank = i + 1

        return viable
