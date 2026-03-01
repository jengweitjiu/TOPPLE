"""
Stromal Buffering Estimation
==============================

Estimates the stromal buffering coefficient β_spatial(i) for each cell,
quantifying the degree to which neighboring stromal cells provide paracrine
signals that reinforce the target cell's regulatory state.

Cells with high β_spatial are predicted to RESIST perturbation (buffered),
while cells with low β_spatial are predicted to be MORE SUSCEPTIBLE (exposed).

Integration with SICAI
-----------------------
SICAI (Stromal-Immune Coupled Attractor Index) quantifies coupling between
stromal and immune cell states in spatial niches. We convert SICAI coupling
scores into buffering coefficients:

    β_spatial(i) = f(coupling_score(i), neighborhood_density(i), LR_score(i))

where:
- coupling_score: SICAI-derived attractor coupling strength
- neighborhood_density: local stromal cell density around cell i
- LR_score: ligand-receptor interaction score from spatial neighbors

Spatial Neighborhoods
---------------------
Neighborhoods are defined by either:
1. k-nearest neighbors (kNN) in physical coordinates
2. Radius-based neighborhoods (r-ball)
3. Voronoi/Delaunay tessellation

For Visium data: spot neighborhoods from tissue_positions.
For MERFISH/single-cell spatial: cell centroid coordinates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


def compute_ligand_receptor_score(
    expression: np.ndarray,
    coordinates: np.ndarray,
    ligand_indices: List[int],
    receptor_indices: List[int],
    k_neighbors: int = 10,
    radius: Optional[float] = None,
) -> np.ndarray:
    """
    Compute ligand-receptor interaction scores for each cell.

    For each cell i, the LR score is the average product of receptor
    expression in cell i and ligand expression in its spatial neighbors:

        LR(i) = mean_j∈N(i) [ Σ_l R_l(i) · L_l(j) ]

    Parameters
    ----------
    expression : np.ndarray, shape (n_cells, n_genes)
        Gene expression matrix.
    coordinates : np.ndarray, shape (n_cells, 2) or (n_cells, 3)
        Spatial coordinates.
    ligand_indices : list of int
        Column indices for ligand genes.
    receptor_indices : list of int
        Column indices for receptor genes (matched with ligands).
    k_neighbors : int, default=10
        Number of spatial neighbors.
    radius : float, optional
        If set, use radius-based neighborhoods instead of kNN.

    Returns
    -------
    np.ndarray, shape (n_cells,)
        LR interaction score per cell.
    """
    n_cells = expression.shape[0]
    tree = KDTree(coordinates)

    lr_scores = np.zeros(n_cells)

    for i in range(n_cells):
        # Get neighbors
        if radius is not None:
            neighbor_idx = tree.query_ball_point(coordinates[i], radius)
            neighbor_idx = [j for j in neighbor_idx if j != i]
        else:
            _, neighbor_idx = tree.query(coordinates[i], k=k_neighbors + 1)
            neighbor_idx = [j for j in neighbor_idx if j != i][:k_neighbors]

        if len(neighbor_idx) == 0:
            continue

        # Receptor expression in cell i
        receptor_expr = expression[i, receptor_indices]

        # Ligand expression in neighbors
        ligand_expr = expression[neighbor_idx][:, ligand_indices]

        # Interaction score: mean over neighbors of dot product
        interactions = ligand_expr @ receptor_expr  # (n_neighbors,)
        lr_scores[i] = interactions.mean()

    return lr_scores


def neighborhood_coupling(
    target_expression: np.ndarray,
    neighbor_expression: np.ndarray,
    coordinates: np.ndarray,
    target_mask: np.ndarray,
    stromal_mask: np.ndarray,
    k_neighbors: int = 15,
) -> np.ndarray:
    """
    Compute stromal-immune coupling score for each target cell.

    For each target cell i, measures the correlation between its
    regulatory state and the regulatory states of surrounding stromal
    cells, weighted by spatial proximity.

    Parameters
    ----------
    target_expression : np.ndarray, shape (n_cells, n_features)
        Regulon activity for target (immune) cells.
    neighbor_expression : np.ndarray, shape (n_cells, n_features)
        Regulon activity for potential neighbor (stromal) cells.
    coordinates : np.ndarray, shape (n_cells, 2)
        Spatial coordinates for ALL cells.
    target_mask : np.ndarray of bool
        Which cells are targets (e.g., TRM cells).
    stromal_mask : np.ndarray of bool
        Which cells are stromal.
    k_neighbors : int, default=15
        Number of stromal neighbors to consider.

    Returns
    -------
    np.ndarray, shape (n_target_cells,)
        Coupling score ∈ [0, 1] for each target cell.
    """
    target_idx = np.where(target_mask)[0]
    stromal_idx = np.where(stromal_mask)[0]

    if len(stromal_idx) == 0:
        return np.zeros(len(target_idx))

    # Build KDTree on stromal cells only
    stromal_coords = coordinates[stromal_idx]
    tree = KDTree(stromal_coords)

    coupling_scores = np.zeros(len(target_idx))

    for ii, ti in enumerate(target_idx):
        target_coord = coordinates[ti]
        k = min(k_neighbors, len(stromal_idx))
        distances, nn_idx = tree.query(target_coord, k=k)

        if k == 1:
            distances = [distances]
            nn_idx = [nn_idx]

        # Distance-weighted correlation
        # Weight by inverse distance (closer neighbors matter more)
        weights = 1.0 / (np.array(distances) + 1e-6)
        weights = weights / weights.sum()

        # Coupling: weighted mean correlation between target and stromal regulons
        target_vec = target_expression[ti]
        if np.std(target_vec) < 1e-10:
            coupling_scores[ii] = 0.0
            continue

        correlations = []
        for j, sj_local in enumerate(nn_idx):
            sj_global = stromal_idx[sj_local]
            stromal_vec = neighbor_expression[sj_global]
            if np.std(stromal_vec) < 1e-10:
                correlations.append(0.0)
                continue
            corr = np.corrcoef(target_vec, stromal_vec)[0, 1]
            correlations.append(max(0, corr))  # Only positive coupling

        coupling_scores[ii] = np.average(correlations, weights=weights)

    return coupling_scores


class StromalBufferingEstimator:
    """
    Estimate stromal buffering coefficient β_spatial for each cell.

    Combines multiple spatial signals into a single buffering score:
    1. Stromal neighborhood density (more stromal neighbors = more buffered)
    2. Coupling strength (SICAI-derived regulatory correlation)
    3. Ligand-receptor signaling (paracrine support intensity)

    β_spatial(i) = w_density * ρ(i) + w_coupling * C(i) + w_lr * LR(i)

    All components are normalized to [0, 1] before combination.

    Parameters
    ----------
    coordinates : np.ndarray, shape (n_cells, 2)
        Spatial coordinates for all cells.
    cell_types : np.ndarray of str
        Cell type labels for all cells.
    regulon_activity : np.ndarray, shape (n_cells, n_regulons)
        Regulon activity matrix (AUCell scores).
    target_type : str
        Cell type to compute buffering for (e.g., "CD8_TRM").
    stromal_types : list of str
        Cell types considered stromal (e.g., ["fibroblast", "endothelial"]).
    k_neighbors : int, default=15
        Number of spatial neighbors.
    w_density : float, default=0.3
        Weight for neighborhood density component.
    w_coupling : float, default=0.5
        Weight for coupling component.
    w_lr : float, default=0.2
        Weight for ligand-receptor component.

    Attributes
    ----------
    beta_ : np.ndarray, shape (n_target_cells,)
        Estimated buffering coefficients ∈ [0, 1].
    target_indices_ : np.ndarray
        Indices of target cells in the full coordinate array.
    density_scores_ : np.ndarray
        Normalized density component.
    coupling_scores_ : np.ndarray
        Normalized coupling component.
    lr_scores_ : np.ndarray
        Normalized LR component (if computed).

    Examples
    --------
    >>> estimator = StromalBufferingEstimator(
    ...     coordinates=spatial_coords,
    ...     cell_types=cell_labels,
    ...     regulon_activity=aucell_matrix,
    ...     target_type="CD8_TRM",
    ...     stromal_types=["fibroblast", "pericyte", "endothelial"],
    ... )
    >>> estimator.fit()
    >>> beta = estimator.beta_  # Buffering coefficients
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        cell_types: np.ndarray,
        regulon_activity: np.ndarray,
        target_type: str,
        stromal_types: List[str],
        k_neighbors: int = 15,
        w_density: float = 0.3,
        w_coupling: float = 0.5,
        w_lr: float = 0.2,
    ):
        self.coordinates = coordinates
        self.cell_types = np.array(cell_types)
        self.regulon_activity = regulon_activity
        self.target_type = target_type
        self.stromal_types = stromal_types
        self.k_neighbors = k_neighbors
        self.w_density = w_density
        self.w_coupling = w_coupling
        self.w_lr = w_lr

    def fit(
        self,
        expression: Optional[np.ndarray] = None,
        ligand_indices: Optional[List[int]] = None,
        receptor_indices: Optional[List[int]] = None,
    ) -> "StromalBufferingEstimator":
        """
        Compute buffering coefficients.

        Parameters
        ----------
        expression : np.ndarray, optional
            Full gene expression matrix (for LR scoring).
            If None, LR component is skipped and weights redistributed.
        ligand_indices, receptor_indices : list of int, optional
            Gene indices for LR pairs. Required if expression is provided.

        Returns
        -------
        self
        """
        target_mask = self.cell_types == self.target_type
        stromal_mask = np.isin(self.cell_types, self.stromal_types)

        self.target_indices_ = np.where(target_mask)[0]
        n_target = len(self.target_indices_)

        if n_target == 0:
            self.beta_ = np.array([])
            return self

        # Component 1: Stromal neighborhood density
        self.density_scores_ = self._compute_density(target_mask, stromal_mask)

        # Component 2: Regulatory coupling (SICAI-style)
        self.coupling_scores_ = neighborhood_coupling(
            target_expression=self.regulon_activity,
            neighbor_expression=self.regulon_activity,
            coordinates=self.coordinates,
            target_mask=target_mask,
            stromal_mask=stromal_mask,
            k_neighbors=self.k_neighbors,
        )

        # Component 3: Ligand-receptor scoring (optional)
        compute_lr = (
            expression is not None
            and ligand_indices is not None
            and receptor_indices is not None
        )

        if compute_lr:
            lr_all = compute_ligand_receptor_score(
                expression=expression,
                coordinates=self.coordinates,
                ligand_indices=ligand_indices,
                receptor_indices=receptor_indices,
                k_neighbors=self.k_neighbors,
            )
            self.lr_scores_ = self._normalize(lr_all[self.target_indices_])
            w_d, w_c, w_l = self.w_density, self.w_coupling, self.w_lr
        else:
            self.lr_scores_ = np.zeros(n_target)
            # Redistribute LR weight to other components
            total = self.w_density + self.w_coupling
            w_d = self.w_density / total if total > 0 else 0.5
            w_c = self.w_coupling / total if total > 0 else 0.5
            w_l = 0.0

        # Combine into β_spatial
        self.beta_ = (
            w_d * self._normalize(self.density_scores_)
            + w_c * self._normalize(self.coupling_scores_)
            + w_l * self.lr_scores_
        )

        # Clip to [0, 1]
        self.beta_ = np.clip(self.beta_, 0.0, 1.0)

        return self

    def _compute_density(
        self,
        target_mask: np.ndarray,
        stromal_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute local stromal density around each target cell."""
        target_idx = np.where(target_mask)[0]
        stromal_idx = np.where(stromal_mask)[0]

        if len(stromal_idx) == 0:
            return np.zeros(len(target_idx))

        stromal_coords = self.coordinates[stromal_idx]
        tree = KDTree(stromal_coords)

        densities = np.zeros(len(target_idx))
        for ii, ti in enumerate(target_idx):
            k = min(self.k_neighbors, len(stromal_idx))
            distances, _ = tree.query(self.coordinates[ti], k=k)
            if k == 1:
                distances = [distances]
            # Density: inverse mean distance to k nearest stromal cells
            mean_dist = np.mean(distances)
            densities[ii] = 1.0 / (mean_dist + 1e-6)

        return densities

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        if len(arr) == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    def summary(self) -> str:
        """Return summary of buffering estimation."""
        if not hasattr(self, "beta_") or len(self.beta_) == 0:
            return "StromalBufferingEstimator: not fitted or no target cells."
        lines = [
            "Stromal Buffering Summary",
            f"  Target type: {self.target_type}",
            f"  Stromal types: {', '.join(self.stromal_types)}",
            f"  Target cells: {len(self.beta_)}",
            f"  β_spatial: mean={self.beta_.mean():.3f}, "
            f"std={self.beta_.std():.3f}, "
            f"range=[{self.beta_.min():.3f}, {self.beta_.max():.3f}]",
            f"  Highly buffered (β>0.7): {(self.beta_ > 0.7).sum()}",
            f"  Exposed (β<0.3): {(self.beta_ < 0.3).sum()}",
        ]
        return "\n".join(lines)
