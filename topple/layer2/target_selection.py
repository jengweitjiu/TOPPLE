"""
Stability-Guided Target Selection
===================================

Takes Layer 1 decomposition results and produces a ranked list of
perturbation candidates: single features, pairs, and higher-order sets
ordered by their stability impact and interaction structure.

Incorporates the IPA (Inverse Prioritization Analysis) filter:
features with high stability contribution but low specification rank
(maintenance-dominant) are prioritized, as they represent non-obvious
intervention targets that standard DE analysis would miss.

Candidate scoring:
    score(S) = w_stability * |I(S)| + w_ipa * ipa_bonus(S) + w_parsimony * (1/|S|)

where:
- |I(S)| is the absolute Möbius interaction magnitude from Layer 1
- ipa_bonus(S) penalizes high-DE features (standard targets) and
  rewards maintenance-dominant features (non-obvious targets)
- parsimony term favors smaller perturbation sets (Occam)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np


@dataclass
class PerturbationCandidate:
    """A ranked perturbation candidate from target selection."""

    features: FrozenSet[int]
    feature_names: List[str]
    order: int
    interaction_value: float
    stability_score: float
    ipa_bonus: float
    parsimony_score: float
    composite_score: float
    rank: int = 0

    def __repr__(self) -> str:
        sign = "+" if self.interaction_value > 0 else ""
        return (
            f"PerturbationCandidate(rank={self.rank}, "
            f"features={' + '.join(self.feature_names)}, "
            f"I={sign}{self.interaction_value:.4f}, "
            f"score={self.composite_score:.4f})"
        )


class TargetSelector:
    """
    Rank perturbation candidates from Layer 1 decomposition.

    Parameters
    ----------
    interactions : dict
        Möbius interaction terms {frozenset -> I(S)} from Layer 1.
    feature_names : list of str
        Feature/regulon names.
    de_scores : dict or np.ndarray, optional
        Differential expression scores (e.g., log2FC or -log10 p-value)
        for each feature. Used for IPA bonus calculation.
        Higher DE = more "obvious" target = lower IPA bonus.
    w_stability : float, default=1.0
        Weight for stability magnitude.
    w_ipa : float, default=0.5
        Weight for IPA (inverse prioritization) bonus.
    w_parsimony : float, default=0.2
        Weight for parsimony (smaller sets preferred).
    min_interaction : float, default=0.001
        Minimum |I(S)| to consider a candidate.
    max_candidates : int, default=50
        Maximum number of candidates to return.

    Examples
    --------
    >>> selector = TargetSelector(
    ...     interactions=decomp.interactions_,
    ...     feature_names=regulon_names,
    ...     de_scores=de_log2fc,
    ... )
    >>> candidates = selector.rank()
    >>> candidates[0]  # Top candidate
    PerturbationCandidate(rank=1, features=RUNX3 + EOMES, ...)
    """

    def __init__(
        self,
        interactions: Dict[FrozenSet[int], float],
        feature_names: List[str],
        de_scores: Optional[dict] = None,
        w_stability: float = 1.0,
        w_ipa: float = 0.5,
        w_parsimony: float = 0.2,
        min_interaction: float = 0.001,
        max_candidates: int = 50,
    ):
        self.interactions = interactions
        self.feature_names = feature_names
        self.de_scores = de_scores
        self.w_stability = w_stability
        self.w_ipa = w_ipa
        self.w_parsimony = w_parsimony
        self.min_interaction = min_interaction
        self.max_candidates = max_candidates

    def rank(self) -> List[PerturbationCandidate]:
        """
        Produce ranked list of perturbation candidates.

        Returns
        -------
        list of PerturbationCandidate
            Sorted by composite score (descending).
        """
        # Filter by minimum interaction threshold
        eligible = {
            k: v
            for k, v in self.interactions.items()
            if abs(v) >= self.min_interaction
        }

        if not eligible:
            return []

        # Normalize stability scores to [0, 1]
        max_abs = max(abs(v) for v in eligible.values())
        if max_abs == 0:
            max_abs = 1.0

        # Compute IPA bonus for each feature
        ipa_map = self._compute_ipa_bonuses()

        candidates = []
        for subset, interaction_val in eligible.items():
            names = [self.feature_names[i] for i in sorted(subset)]
            order = len(subset)

            # Stability score (normalized absolute interaction)
            stability = abs(interaction_val) / max_abs

            # IPA bonus: average across features in set
            # Higher bonus = less obvious targets (maintenance-dominant)
            if ipa_map:
                ipa_bonus = np.mean([ipa_map.get(i, 0.5) for i in subset])
            else:
                ipa_bonus = 0.5  # Neutral if no DE data

            # Parsimony: favor smaller sets
            parsimony = 1.0 / order

            # Composite score
            composite = (
                self.w_stability * stability
                + self.w_ipa * ipa_bonus
                + self.w_parsimony * parsimony
            )

            candidates.append(
                PerturbationCandidate(
                    features=subset,
                    feature_names=names,
                    order=order,
                    interaction_value=interaction_val,
                    stability_score=stability,
                    ipa_bonus=ipa_bonus,
                    parsimony_score=parsimony,
                    composite_score=composite,
                )
            )

        # Sort by composite score
        candidates.sort(key=lambda c: c.composite_score, reverse=True)
        candidates = candidates[: self.max_candidates]

        # Assign ranks
        for i, c in enumerate(candidates):
            c.rank = i + 1

        return candidates

    def _compute_ipa_bonuses(self) -> Dict[int, float]:
        """
        Compute IPA bonus for each feature.

        IPA logic: features with HIGH stability contribution but LOW DE
        (specification) score get a bonus. These are maintenance-dominant
        features that standard analysis would miss.

        bonus(f) = 1 - normalized_de(f)
        """
        if self.de_scores is None:
            return {}

        # Convert DE scores to per-feature dict
        if isinstance(self.de_scores, np.ndarray):
            de_dict = {i: abs(self.de_scores[i]) for i in range(len(self.de_scores))}
        elif isinstance(self.de_scores, dict):
            de_dict = {}
            for k, v in self.de_scores.items():
                if isinstance(k, str):
                    try:
                        idx = self.feature_names.index(k)
                        de_dict[idx] = abs(v)
                    except ValueError:
                        continue
                else:
                    de_dict[k] = abs(v)
        else:
            return {}

        if not de_dict:
            return {}

        # Normalize to [0, 1]
        max_de = max(de_dict.values())
        if max_de == 0:
            return {k: 0.5 for k in de_dict}

        # IPA bonus = 1 - normalized_de (low DE -> high bonus)
        return {k: 1.0 - v / max_de for k, v in de_dict.items()}

    def summary(self, top_n: int = 10) -> str:
        """Return text summary of top candidates."""
        candidates = self.rank()
        lines = [
            "Target Selection Summary",
            f"  Total candidates: {len(candidates)}",
            f"  Top {min(top_n, len(candidates))} candidates:",
        ]
        for c in candidates[:top_n]:
            lines.append(
                f"    #{c.rank}: {' + '.join(c.feature_names)} "
                f"(I={c.interaction_value:+.4f}, "
                f"score={c.composite_score:.3f}, "
                f"IPA={c.ipa_bonus:.2f})"
            )
        return "\n".join(lines)


class IPAFilter:
    """
    Inverse Prioritization Analysis filter.

    Separates features into:
    - Specification-dominant: high DE, high stability -> obvious targets
    - Maintenance-dominant: low DE, high stability -> non-obvious, prioritized
    - Low-impact: low stability regardless of DE -> deprioritized

    Parameters
    ----------
    interactions : dict
        Layer 1 interaction terms.
    de_scores : dict or np.ndarray
        Differential expression scores.
    feature_names : list of str
        Feature names.
    de_threshold : float, default=0.5
        Normalized DE threshold for specification/maintenance split.
    stability_threshold : float, default=0.3
        Normalized stability threshold for low-impact exclusion.
    """

    def __init__(
        self,
        interactions: Dict[FrozenSet[int], float],
        de_scores,
        feature_names: List[str],
        de_threshold: float = 0.5,
        stability_threshold: float = 0.3,
    ):
        self.interactions = interactions
        self.de_scores = de_scores
        self.feature_names = feature_names
        self.de_threshold = de_threshold
        self.stability_threshold = stability_threshold

    def classify(self) -> Dict[str, List[str]]:
        """
        Classify features into specification/maintenance/low-impact.

        Returns
        -------
        dict with keys "specification", "maintenance", "low_impact",
        each mapping to a list of feature names.
        """
        # Get marginal stability contributions (order-1 interactions)
        marginals = {}
        for k, v in self.interactions.items():
            if len(k) == 1:
                (idx,) = k
                marginals[idx] = abs(v)

        if not marginals:
            return {"specification": [], "maintenance": [], "low_impact": list(self.feature_names)}

        max_stability = max(marginals.values()) if marginals else 1.0

        # Normalize DE scores
        if isinstance(self.de_scores, np.ndarray):
            de_vals = {i: abs(self.de_scores[i]) for i in range(len(self.de_scores))}
        elif isinstance(self.de_scores, dict):
            de_vals = {}
            for k, v in self.de_scores.items():
                if isinstance(k, str):
                    try:
                        idx = self.feature_names.index(k)
                        de_vals[idx] = abs(v)
                    except ValueError:
                        continue
                else:
                    de_vals[k] = abs(v)
        else:
            de_vals = {}

        max_de = max(de_vals.values()) if de_vals else 1.0

        result = {"specification": [], "maintenance": [], "low_impact": []}

        for idx, name in enumerate(self.feature_names):
            norm_stab = marginals.get(idx, 0) / max_stability if max_stability > 0 else 0
            norm_de = de_vals.get(idx, 0) / max_de if max_de > 0 else 0

            if norm_stab < self.stability_threshold:
                result["low_impact"].append(name)
            elif norm_de >= self.de_threshold:
                result["specification"].append(name)
            else:
                result["maintenance"].append(name)

        return result
