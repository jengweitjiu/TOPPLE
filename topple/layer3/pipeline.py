"""
Spatial Vulnerability Pipeline
================================

Orchestrates the complete TOPPLE pipeline:
    Layer 1 (stability decomposition)
    → Layer 2 (perturbation bridge)
    → Layer 3 (spatial vulnerability mapping + niche ranking)

This is the main entry point for spatially-informed perturbation prediction.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .spatial_buffering import StromalBufferingEstimator
from .vulnerability import SpatialVulnerabilityScorer, VulnerabilityMap
from .niche import NicheStratifier, NichePerturbationRanker


class SpatialVulnerabilityPipeline:
    """
    Full Layer 3 pipeline: buffering → vulnerability → niche ranking.

    Requires Layer 2 results (per-cell destabilization scores) as input.

    Parameters
    ----------
    coordinates : np.ndarray, shape (n_cells, 2)
        Spatial coordinates for ALL cells.
    cell_types : np.ndarray of str
        Cell type labels for ALL cells.
    regulon_activity : np.ndarray, shape (n_cells, n_regulons)
        Regulon activity matrix.
    target_type : str
        Cell type to analyze (e.g., "CD8_TRM").
    stromal_types : list of str
        Stromal cell types for buffering estimation.
    n_niches : int, default=3
        Number of spatial niches.
    strata_zones : np.ndarray, optional
        Pre-computed STRATA zone labels.
    k_neighbors : int, default=15
        Spatial neighborhood size.
    verbose : bool, default=True
        Print progress.

    Examples
    --------
    >>> pipeline = SpatialVulnerabilityPipeline(
    ...     coordinates=spatial_coords,
    ...     cell_types=cell_labels,
    ...     regulon_activity=aucell,
    ...     target_type="CD8_TRM",
    ...     stromal_types=["fibroblast", "endothelial"],
    ... )
    >>> vmaps = pipeline.run(layer2_destabilizations)
    >>> print(pipeline.report())
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        cell_types: np.ndarray,
        regulon_activity: np.ndarray,
        target_type: str,
        stromal_types: List[str],
        n_niches: int = 3,
        strata_zones: Optional[np.ndarray] = None,
        k_neighbors: int = 15,
        verbose: bool = True,
    ):
        self.coordinates = coordinates
        self.cell_types = np.array(cell_types)
        self.regulon_activity = regulon_activity
        self.target_type = target_type
        self.stromal_types = stromal_types
        self.n_niches = n_niches
        self.strata_zones = strata_zones
        self.k_neighbors = k_neighbors
        self.verbose = verbose

    def run(
        self,
        destabilizations: List[Tuple[np.ndarray, FrozenSet[int], List[str]]],
    ) -> List[VulnerabilityMap]:
        """
        Run the full Layer 3 pipeline.

        Parameters
        ----------
        destabilizations : list of (destab_array, feature_set, feature_names)
            Per-cell destabilization scores from Layer 2 for each
            perturbation candidate. destab_array has shape (n_ALL_cells,)
            or (n_target_cells,).

        Returns
        -------
        list of VulnerabilityMap
            Sorted by mean vulnerability.
        """
        target_mask = self.cell_types == self.target_type
        target_idx = np.where(target_mask)[0]
        n_target = len(target_idx)

        if n_target == 0:
            if self.verbose:
                print(f"[TOPPLE L3] WARNING: No cells of type '{self.target_type}' found.")
            return []

        if self.verbose:
            print(f"[TOPPLE L3] Target cells: {n_target} ({self.target_type})")

        # Step 1: Stromal buffering estimation
        if self.verbose:
            print("[TOPPLE L3] Step 1: Estimating stromal buffering...")

        self.buffering_ = StromalBufferingEstimator(
            coordinates=self.coordinates,
            cell_types=self.cell_types,
            regulon_activity=self.regulon_activity,
            target_type=self.target_type,
            stromal_types=self.stromal_types,
            k_neighbors=self.k_neighbors,
        )
        self.buffering_.fit()

        if self.verbose:
            print(self.buffering_.summary())

        # Step 2: Spatial vulnerability scoring
        if self.verbose:
            print(f"\n[TOPPLE L3] Step 2: Scoring {len(destabilizations)} perturbations...")

        target_coords = self.coordinates[target_idx]
        scorer = SpatialVulnerabilityScorer(
            beta_spatial=self.buffering_.beta_,
            coordinates=target_coords,
            cell_indices=target_idx,
        )

        # Process destabilizations (handle both full-size and target-only arrays)
        processed = []
        for destab, fset, fnames in destabilizations:
            if len(destab) == len(self.cell_types):
                destab_target = destab[target_idx]
            elif len(destab) == n_target:
                destab_target = destab
            else:
                raise ValueError(
                    f"Destabilization array length ({len(destab)}) doesn't match "
                    f"total cells ({len(self.cell_types)}) or target cells ({n_target})."
                )
            processed.append((destab_target, fset, fnames))

        self.vulnerability_maps_ = scorer.score_multiple(processed)

        if self.verbose:
            for vm in self.vulnerability_maps_[:5]:
                print(f"  {' + '.join(vm.feature_names):<30s} "
                      f"V={vm.mean_vulnerability:.3f} "
                      f"(frac={vm.fraction_vulnerable:.0%})")

        # Step 3: Niche stratification
        if self.verbose:
            print(f"\n[TOPPLE L3] Step 3: Niche stratification...")

        self.stratifier_ = NicheStratifier(
            coordinates=self.coordinates,
            cell_types=self.cell_types,
            target_mask=target_mask,
            k_neighbors=self.k_neighbors,
            n_niches=self.n_niches,
            strata_zones=self.strata_zones,
        )
        self.stratifier_.fit()

        if self.verbose:
            print(self.stratifier_.summary())

        # Step 4: Niche-stratified ranking
        if self.verbose:
            print(f"\n[TOPPLE L3] Step 4: Niche-stratified perturbation ranking...")

        self.ranker_ = NichePerturbationRanker(
            niche_labels=self.stratifier_.niche_labels_,
            vulnerability_maps=self.vulnerability_maps_,
        )
        self.ranker_.rank()

        if self.verbose:
            print("[TOPPLE L3] Done.")

        return self.vulnerability_maps_

    def report(self) -> str:
        """Generate full Layer 3 report."""
        if not hasattr(self, "ranker_"):
            return "Pipeline not run. Call .run() first."

        lines = [
            "=" * 70,
            "TOPPLE Layer 3: Spatial Vulnerability Report",
            "=" * 70,
            "",
            self.buffering_.summary(),
            "",
            self.stratifier_.summary(),
            "",
            self.ranker_.report(),
        ]
        return "\n".join(lines)

    def to_dataframe(self):
        """Export per-cell vulnerability data as a DataFrame."""
        import pandas as pd

        if not hasattr(self, "vulnerability_maps_") or not self.vulnerability_maps_:
            return pd.DataFrame()

        # Use the top vulnerability map
        vmap = self.vulnerability_maps_[0]
        data = {
            "cell_index": vmap.cell_indices,
            "x": vmap.coordinates[:, 0],
            "y": vmap.coordinates[:, 1],
            "vulnerability": vmap.vulnerability,
            "destabilization": vmap.destabilization,
            "buffering": vmap.buffering,
        }

        if hasattr(self, "stratifier_"):
            data["niche"] = self.stratifier_.niche_labels_

        return pd.DataFrame(data)
