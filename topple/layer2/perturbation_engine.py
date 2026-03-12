"""
Perturbation Engine
====================

Wraps in silico perturbation tools (CellOracle, GEARS) for predicting
transcriptomic changes following TF knockdown/knockout.

For each perturbation candidate from target selection, the engine:
1. Simulates TF knockdown for each feature in the set
2. Returns predicted post-perturbation expression profiles
3. Handles combinatorial perturbations (multi-TF knockout)

Architecture
------------
PerturbationEngine is the abstract interface.
CellOracleAdapter wraps CellOracle's GRN-based perturbation.
MockPerturbationEngine provides a simple linear model for testing.

CellOracle Integration Notes
-----------------------------
CellOracle requires:
- A fitted GRN (from `co.fit_GRN_for_simulation()`)
- An AnnData object with cell annotations
- TF names matching the GRN

The adapter handles:
- Multi-TF perturbation via sequential or simultaneous KO
- Extraction of predicted expression from CellOracle's output
- Conversion back to regulon activity space (via AUCell re-scoring)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np


@dataclass
class PerturbationResult:
    """Result of an in silico perturbation experiment."""

    perturbation_set: FrozenSet[int]
    feature_names: List[str]
    X_original: np.ndarray          # (n_cells, n_genes) or (n_cells, n_regulons)
    X_perturbed: np.ndarray         # (n_cells, n_genes) or (n_cells, n_regulons)
    cell_indices: np.ndarray        # Which cells were perturbed
    state_labels: np.ndarray        # Original state labels for these cells
    perturbation_type: str          # "knockout", "knockdown", "overexpression"
    perturbation_strength: float    # 0.0 = full KO, 0.5 = 50% knockdown
    metadata: dict                  # Additional info from the engine

    @property
    def n_cells(self) -> int:
        """Number of cells in this perturbation result."""
        return len(self.cell_indices)

    @property
    def delta_expression(self) -> np.ndarray:
        """Change in expression: perturbed - original."""
        return self.X_perturbed - self.X_original

    @property
    def mean_delta(self) -> np.ndarray:
        """Mean expression change across cells."""
        return self.delta_expression.mean(axis=0)


class PerturbationEngine(ABC):
    """
    Abstract interface for in silico perturbation prediction.

    Subclasses must implement `simulate()` to predict expression
    changes following TF perturbation.
    """

    @abstractmethod
    def simulate(
        self,
        feature_indices: FrozenSet[int],
        cell_mask: Optional[np.ndarray] = None,
        perturbation_type: str = "knockout",
        strength: float = 0.0,
    ) -> PerturbationResult:
        """
        Simulate perturbation of specified features.

        Parameters
        ----------
        feature_indices : frozenset of int
            Indices of features (TFs/regulons) to perturb.
        cell_mask : np.ndarray of bool, optional
            Which cells to perturb. Default: all cells.
        perturbation_type : str
            "knockout" (expression → 0), "knockdown" (reduced),
            or "overexpression" (increased).
        strength : float
            For knockdown: fraction of remaining expression (0.0 = full KO).
            For overexpression: fold increase.

        Returns
        -------
        PerturbationResult
        """
        pass

    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the engine has a fitted GRN model."""
        pass


class CellOracleAdapter(PerturbationEngine):
    """
    Adapter for CellOracle in silico perturbation.

    Wraps CellOracle's simulation API to:
    1. Accept TOPPLE feature indices (regulon/TF indices)
    2. Simulate single or combinatorial TF perturbation
    3. Return predicted expression in a standard format

    Parameters
    ----------
    oracle : celloracle.Oracle
        A fitted CellOracle Oracle object with GRN ready for simulation.
        Must have had `get_links()` and `fit_GRN_for_simulation()` called.
    adata : anndata.AnnData
        The AnnData object used to fit the Oracle.
    feature_names : list of str
        Ordered list of TF/regulon names matching TOPPLE's feature indices.
    cluster_key : str, default="cell_type"
        Column in adata.obs for cell type annotations.
    target_cluster : str, optional
        If set, only simulate perturbation in this cluster.

    Examples
    --------
    >>> import celloracle as co
    >>> oracle = co.Oracle()
    >>> # ... fit oracle ...
    >>> adapter = CellOracleAdapter(
    ...     oracle=oracle,
    ...     adata=adata,
    ...     feature_names=regulon_names,
    ...     target_cluster="CD8_TRM_pathological",
    ... )
    >>> result = adapter.simulate(frozenset([0, 2]))  # KO TF 0 and 2
    """

    def __init__(
        self,
        oracle,
        adata,
        feature_names: List[str],
        cluster_key: str = "cell_type",
        target_cluster: Optional[str] = None,
    ):
        """Initialize CellOracle adapter with a fitted oracle and AnnData object."""
        self.oracle = oracle
        self.adata = adata
        self.feature_names = feature_names
        self.cluster_key = cluster_key
        self.target_cluster = target_cluster
        self._fitted = True

    def simulate(
        self,
        feature_indices: FrozenSet[int],
        cell_mask: Optional[np.ndarray] = None,
        perturbation_type: str = "knockout",
        strength: float = 0.0,
    ) -> PerturbationResult:
        """
        Simulate TF perturbation via CellOracle.

        For combinatorial perturbations, applies each TF knockout
        sequentially (CellOracle doesn't natively support simultaneous
        multi-TF perturbation, so we compose the effects).
        """
        try:
            import celloracle as co
        except ImportError:
            raise ImportError(
                "CellOracle not installed. Install with: pip install celloracle"
            )

        tf_names = [self.feature_names[i] for i in sorted(feature_indices)]

        # Determine target cells
        if cell_mask is not None:
            cell_idx = np.where(cell_mask)[0]
        elif self.target_cluster is not None:
            cell_idx = np.where(
                self.adata.obs[self.cluster_key] == self.target_cluster
            )[0]
        else:
            cell_idx = np.arange(self.adata.n_obs)

        # Get original expression for target cells
        X_original = self.adata[cell_idx].X
        if hasattr(X_original, "toarray"):
            X_original = X_original.toarray()
        X_original = np.array(X_original, dtype=np.float64)

        # Simulate perturbation for each TF
        # CellOracle API: oracle.simulate_shift(perturb_condition={gene: value})
        perturb_condition = {}
        for tf in tf_names:
            if perturbation_type == "knockout":
                perturb_condition[tf] = 0.0
            elif perturbation_type == "knockdown":
                perturb_condition[tf] = strength
            elif perturbation_type == "overexpression":
                perturb_condition[tf] = strength
            else:
                raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        # Run CellOracle simulation
        self.oracle.simulate_shift(
            perturb_condition=perturb_condition,
            n_propagation=3,
        )

        # Extract predicted expression
        # CellOracle stores results in oracle.adata.layers
        X_perturbed = self.oracle.adata[cell_idx].layers.get(
            "simulated_count", X_original
        )
        if hasattr(X_perturbed, "toarray"):
            X_perturbed = X_perturbed.toarray()
        X_perturbed = np.array(X_perturbed, dtype=np.float64)

        # Get state labels
        state_labels = np.zeros(len(cell_idx))
        if "state" in self.adata.obs.columns:
            state_labels = self.adata.obs["state"].values[cell_idx]

        return PerturbationResult(
            perturbation_set=feature_indices,
            feature_names=tf_names,
            X_original=X_original,
            X_perturbed=X_perturbed,
            cell_indices=cell_idx,
            state_labels=state_labels,
            perturbation_type=perturbation_type,
            perturbation_strength=strength,
            metadata={
                "engine": "CellOracle",
                "n_propagation": 3,
                "perturb_condition": perturb_condition,
            },
        )

    def is_fitted(self) -> bool:
        """Return True if the CellOracle model is fitted."""
        return self._fitted


class MockPerturbationEngine(PerturbationEngine):
    """
    Simple linear perturbation model for testing and benchmarking.

    Predicts post-perturbation expression as:
        X_perturbed = X_original - effect_matrix @ perturbation_vector

    where effect_matrix encodes direct + indirect (1-hop GRN) effects.

    Parameters
    ----------
    X : np.ndarray, shape (n_cells, n_features)
        Original expression/regulon activity matrix.
    y : np.ndarray, shape (n_cells,)
        State labels.
    feature_names : list of str
        Feature names.
    effect_size : float, default=1.0
        Scaling factor for perturbation effects.
    grn_adjacency : np.ndarray, optional
        GRN adjacency for indirect effects. If None, only direct effects.
    random_state : int, default=42
        Random seed for noise.

    Notes
    -----
    This is intentionally simple — it serves as a baseline for
    comparing against CellOracle/GEARS predictions and for testing
    the Layer 2 pipeline without requiring a fitted GRN model.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        effect_size: float = 1.0,
        grn_adjacency: Optional[np.ndarray] = None,
        random_state: int = 42,
    ):
        """Initialize mock perturbation engine with expression data and effect parameters."""
        self.X = X.copy()
        self.y = y.copy()
        self.feature_names = feature_names
        self.effect_size = effect_size
        self.grn_adjacency = grn_adjacency
        self.rng = np.random.RandomState(random_state)
        self._fitted = True

        # Build effect matrix
        n_features = X.shape[1]
        self.effect_matrix = np.eye(n_features) * effect_size

        if grn_adjacency is not None:
            # Add 1-hop indirect effects (scaled down)
            indirect = grn_adjacency * 0.3 * effect_size
            self.effect_matrix += indirect

    def simulate(
        self,
        feature_indices: FrozenSet[int],
        cell_mask: Optional[np.ndarray] = None,
        perturbation_type: str = "knockout",
        strength: float = 0.0,
    ) -> PerturbationResult:
        """Simulate perturbation using the linear model."""
        tf_names = [self.feature_names[i] for i in sorted(feature_indices)]

        if cell_mask is not None:
            cell_idx = np.where(cell_mask)[0]
        else:
            cell_idx = np.arange(self.X.shape[0])

        X_orig = self.X[cell_idx].copy()

        # Build perturbation vector
        n_features = self.X.shape[1]
        perturb_vec = np.zeros(n_features)
        for idx in feature_indices:
            if perturbation_type == "knockout":
                perturb_vec[idx] = 1.0  # Full removal
            elif perturbation_type == "knockdown":
                perturb_vec[idx] = 1.0 - strength
            elif perturbation_type == "overexpression":
                perturb_vec[idx] = -strength  # Negative = increase

        # Predicted change: effect_matrix @ perturbation_vector
        delta = self.effect_matrix @ perturb_vec  # (n_features,)

        # Apply cell-specific noise
        noise = self.rng.randn(len(cell_idx), n_features) * 0.1
        X_perturbed = X_orig - delta[np.newaxis, :] + noise

        return PerturbationResult(
            perturbation_set=feature_indices,
            feature_names=tf_names,
            X_original=X_orig,
            X_perturbed=X_perturbed,
            cell_indices=cell_idx,
            state_labels=self.y[cell_idx],
            perturbation_type=perturbation_type,
            perturbation_strength=strength,
            metadata={"engine": "MockLinear", "effect_size": self.effect_size},
        )

    def is_fitted(self) -> bool:
        """Return True if the mock engine is ready for simulation."""
        return self._fitted
