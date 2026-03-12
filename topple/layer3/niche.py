"""
Niche Stratification and Niche-Optimal Perturbation Ranking
=============================================================

Groups target cells into spatial niches based on their microenvironment
composition (cell type proportions, buffering level, tissue zone).
Then ranks perturbations separately within each niche, identifying
niche-specific optimal interventions.

Biological rationale: in psoriasis, TRM cells near fibroblast-rich
stroma may require different perturbation sets than TRM cells in
peri-vascular niches, because the stromal buffering differs.

Niche discovery:
1. Compute neighborhood composition vector for each cell
   (proportions of each cell type within k nearest neighbors)
2. Cluster composition vectors → spatial niches
3. Annotate niches by dominant stromal type and buffering level

Integration with STRATA:
STRATA defines spatial regulatory zones from TF activity gradients.
If STRATA zone labels are available, they can be used directly as
niche assignments instead of (or in addition to) composition clustering.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


class NicheStratifier:
    """
    Discover and assign spatial niches from tissue architecture.

    Parameters
    ----------
    coordinates : np.ndarray, shape (n_cells, 2)
        Spatial coordinates for ALL cells.
    cell_types : np.ndarray of str, shape (n_cells,)
        Cell type labels for ALL cells.
    target_mask : np.ndarray of bool
        Which cells are targets for niche assignment.
    k_neighbors : int, default=20
        Neighborhood size for composition profiling.
    n_niches : int, default=3
        Number of niches to discover (if not using STRATA zones).
    strata_zones : np.ndarray, optional
        Pre-computed STRATA zone assignments. If provided, used directly
        as niche labels instead of composition clustering.

    Attributes
    ----------
    niche_labels_ : np.ndarray, shape (n_target_cells,)
        Niche assignment for each target cell.
    composition_profiles_ : np.ndarray, shape (n_target_cells, n_cell_types)
        Neighborhood composition vectors.
    niche_summaries_ : dict
        Per-niche statistics (size, dominant types, mean buffering).

    Examples
    --------
    >>> stratifier = NicheStratifier(
    ...     coordinates=coords, cell_types=labels,
    ...     target_mask=(labels == "CD8_TRM"), n_niches=3,
    ... )
    >>> stratifier.fit()
    >>> stratifier.niche_labels_  # array([0, 2, 1, 0, ...])
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        cell_types: np.ndarray,
        target_mask: np.ndarray,
        k_neighbors: int = 20,
        n_niches: int = 3,
        strata_zones: Optional[np.ndarray] = None,
    ):
        """Initialize niche stratifier with spatial coordinates and cell type labels."""
        self.coordinates = coordinates
        self.cell_types = np.array(cell_types)
        self.target_mask = target_mask
        self.k_neighbors = k_neighbors
        self.n_niches = n_niches
        self.strata_zones = strata_zones

    def fit(self) -> "NicheStratifier":
        """Discover niches and assign target cells."""
        self.target_indices_ = np.where(self.target_mask)[0]
        n_target = len(self.target_indices_)

        if n_target == 0:
            self.niche_labels_ = np.array([], dtype=int)
            self.composition_profiles_ = np.array([])
            self.niche_summaries_ = {}
            return self

        # If STRATA zones provided, use directly
        if self.strata_zones is not None:
            self.niche_labels_ = self.strata_zones[self.target_indices_]
            self.composition_profiles_ = self._compute_compositions()
            self._summarize_niches()
            return self

        # Compute neighborhood composition profiles
        self.composition_profiles_ = self._compute_compositions()

        # Cluster into niches
        self.niche_labels_ = self._cluster_niches(self.composition_profiles_)

        # Summarize
        self._summarize_niches()

        return self

    def _compute_compositions(self) -> np.ndarray:
        """Compute cell type composition in each target cell's neighborhood."""
        unique_types = np.unique(self.cell_types)
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        self.cell_type_order_ = list(unique_types)

        tree = KDTree(self.coordinates)
        n_target = len(self.target_indices_)
        n_types = len(unique_types)
        compositions = np.zeros((n_target, n_types))

        for ii, ti in enumerate(self.target_indices_):
            k = min(self.k_neighbors, len(self.coordinates))
            _, nn_idx = tree.query(self.coordinates[ti], k=k)
            if k == 1:
                nn_idx = [nn_idx]
            for j in nn_idx:
                ct = self.cell_types[j]
                compositions[ii, type_to_idx[ct]] += 1
            compositions[ii] /= max(compositions[ii].sum(), 1)

        return compositions

    def _cluster_niches(self, profiles: np.ndarray) -> np.ndarray:
        """Simple k-means clustering on composition profiles."""
        from sklearn.cluster import KMeans

        n_samples = profiles.shape[0]
        n_clusters = min(self.n_niches, n_samples)

        if n_clusters <= 1:
            return np.zeros(n_samples, dtype=int)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return km.fit_predict(profiles)

    def _summarize_niches(self):
        """Compute per-niche summary statistics."""
        self.niche_summaries_ = {}
        unique_niches = np.unique(self.niche_labels_)

        for niche in unique_niches:
            mask = self.niche_labels_ == niche
            n_cells = int(mask.sum())

            if hasattr(self, "composition_profiles_") and len(self.composition_profiles_) > 0:
                mean_comp = self.composition_profiles_[mask].mean(axis=0)
                dominant_idx = np.argmax(mean_comp)
                dominant_type = self.cell_type_order_[dominant_idx]
                dominant_frac = float(mean_comp[dominant_idx])
            else:
                dominant_type = "unknown"
                dominant_frac = 0.0

            self.niche_summaries_[int(niche)] = {
                "n_cells": n_cells,
                "dominant_type": dominant_type,
                "dominant_fraction": dominant_frac,
            }

    def summary(self) -> str:
        """Return niche summary."""
        lines = [
            "Niche Stratification Summary",
            f"  Target cells: {len(self.target_indices_)}",
            f"  Niches discovered: {len(self.niche_summaries_)}",
        ]
        for niche, info in self.niche_summaries_.items():
            lines.append(
                f"  Niche {niche}: {info['n_cells']} cells, "
                f"dominant={info['dominant_type']} ({info['dominant_fraction']:.0%})"
            )
        return "\n".join(lines)


class NichePerturbationRanker:
    """
    Rank perturbations within each spatial niche.

    Identifies niche-specific optimal perturbation sets — a key translational
    insight, since different tissue zones may respond differently to the
    same intervention.

    Parameters
    ----------
    niche_labels : np.ndarray
        Niche assignment for each target cell.
    vulnerability_maps : list of VulnerabilityMap
        Vulnerability maps for each candidate perturbation.
    min_cells_per_niche : int, default=5
        Minimum cells to consider a niche for ranking.

    Attributes
    ----------
    niche_rankings_ : dict
        {niche_id: list of (feature_names, mean_V, fraction_vulnerable)}.
    global_ranking_ : list
        Overall ranking across all niches.

    Examples
    --------
    >>> ranker = NichePerturbationRanker(
    ...     niche_labels=stratifier.niche_labels_,
    ...     vulnerability_maps=vmaps,
    ... )
    >>> ranker.rank()
    >>> ranker.niche_rankings_[0]  # Best perturbations for niche 0
    """

    def __init__(
        self,
        niche_labels: np.ndarray,
        vulnerability_maps: list,
        min_cells_per_niche: int = 5,
    ):
        """Initialize niche perturbation ranker with niche labels and vulnerability maps."""
        self.niche_labels = niche_labels
        self.vulnerability_maps = vulnerability_maps
        self.min_cells_per_niche = min_cells_per_niche

    def rank(self) -> "NichePerturbationRanker":
        """Rank perturbations within each niche."""
        unique_niches = np.unique(self.niche_labels)
        self.niche_rankings_ = {}
        self.global_ranking_ = []

        for niche in unique_niches:
            niche_mask = self.niche_labels == niche
            if niche_mask.sum() < self.min_cells_per_niche:
                continue

            niche_results = []
            for vmap in self.vulnerability_maps:
                niche_v = vmap.vulnerability[niche_mask]
                mean_v = float(niche_v.mean())
                frac_v = float((niche_v > 0.3).mean())

                niche_results.append({
                    "features": vmap.feature_names,
                    "perturbation_set": vmap.perturbation_set,
                    "mean_vulnerability": mean_v,
                    "fraction_vulnerable": frac_v,
                    "max_vulnerability": float(niche_v.max()),
                    "n_cells": int(niche_mask.sum()),
                })

            niche_results.sort(key=lambda x: x["mean_vulnerability"], reverse=True)
            self.niche_rankings_[int(niche)] = niche_results

        # Global ranking: weight by niche size
        perturbation_global_scores = {}
        total_cells = len(self.niche_labels)

        for niche, results in self.niche_rankings_.items():
            niche_weight = sum(1 for n in self.niche_labels if n == niche) / max(total_cells, 1)
            for r in results:
                key = tuple(sorted(r["perturbation_set"]))
                if key not in perturbation_global_scores:
                    perturbation_global_scores[key] = {
                        "features": r["features"],
                        "weighted_V": 0.0,
                        "niche_scores": {},
                    }
                perturbation_global_scores[key]["weighted_V"] += (
                    niche_weight * r["mean_vulnerability"]
                )
                perturbation_global_scores[key]["niche_scores"][niche] = r["mean_vulnerability"]

        self.global_ranking_ = sorted(
            perturbation_global_scores.values(),
            key=lambda x: x["weighted_V"],
            reverse=True,
        )

        return self

    def report(self, top_n: int = 5) -> str:
        """Generate niche-stratified perturbation report."""
        lines = [
            "=" * 65,
            "TOPPLE Layer 3: Niche-Stratified Perturbation Ranking",
            "=" * 65,
        ]

        # Per-niche rankings
        for niche, results in self.niche_rankings_.items():
            if not results:
                lines.append(f"\n--- Niche {niche} (no candidates) ---")
                continue
            lines.append(f"\n--- Niche {niche} ({results[0]['n_cells']} cells) ---")
            for r in results[:top_n]:
                feat_str = " + ".join(r["features"])
                lines.append(
                    f"  {feat_str:<30s} V={r['mean_vulnerability']:.3f} "
                    f"frac={r['fraction_vulnerable']:.0%}"
                )

        # Global ranking
        lines.extend(["", "--- Global Ranking (niche-size-weighted) ---"])
        for r in self.global_ranking_[:top_n]:
            feat_str = " + ".join(r["features"])
            niche_str = ", ".join(
                f"N{n}={v:.3f}" for n, v in r["niche_scores"].items()
            )
            lines.append(
                f"  {feat_str:<30s} wV={r['weighted_V']:.3f} [{niche_str}]"
            )

        # Niche-discordant perturbations (work in one niche but not another)
        discordant = []
        for r in self.global_ranking_:
            scores = list(r["niche_scores"].values())
            if len(scores) >= 2:
                spread = max(scores) - min(scores)
                if spread > 0.2:
                    discordant.append((r["features"], spread, r["niche_scores"]))

        if discordant:
            lines.extend(["", "--- Niche-Discordant Perturbations (spread > 0.2) ---"])
            for feat, spread, scores in discordant[:5]:
                feat_str = " + ".join(feat)
                niche_str = ", ".join(f"N{n}={v:.3f}" for n, v in scores.items())
                lines.append(f"  {feat_str:<30s} spread={spread:.3f} [{niche_str}]")

        return "\n".join(lines)
