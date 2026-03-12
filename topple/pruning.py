"""
Topology-Guided Pruning
========================

Restricts higher-order subset evaluation to feature sets that are connected
in the gene regulatory network (GRN), reducing the search space from O(2^p)
to approximately O(p · d^{k-1}), where d is the mean network degree and k
is the maximum interaction order.

Rationale: Features that are not connected in the regulatory network are
unlikely to exhibit higher-order interactions, because their regulatory
effects are mediated through independent pathways. Connected features, by
contrast, share regulatory targets or participate in feed-forward loops
and coherent multi-input motifs that generate genuine higher-order effects.

Integration with pySCENIC
--------------------------
The GRN is inferred via pySCENIC, which produces:
1. Regulons: sets of target genes for each TF
2. Adjacencies: TF-target regulatory links with importance scores

We convert pySCENIC adjacencies into a regulon-level graph where two
regulons are connected if they share targets above a threshold or if
one TF is a target of another regulon.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


def grn_to_adjacency(
    adjacencies,
    regulon_names: List[str],
    *,
    mode: str = "shared_targets",
    min_shared: int = 5,
    min_importance: float = 0.0,
) -> np.ndarray:
    """
    Convert pySCENIC adjacencies to a regulon-level adjacency matrix.

    Parameters
    ----------
    adjacencies : pd.DataFrame
        pySCENIC adjacency table with columns: TF, target, importance.
    regulon_names : list of str
        Ordered list of regulon/TF names matching feature columns.
    mode : str, default="shared_targets"
        How to define regulon connectivity:
        - "shared_targets": Connected if regulons share >= min_shared targets.
        - "regulatory": Connected if one TF is a target of another's regulon.
        - "combined": Union of both.
    min_shared : int, default=5
        Minimum shared targets for "shared_targets" mode.
    min_importance : float, default=0.0
        Minimum importance score to include an edge.

    Returns
    -------
    np.ndarray, shape (n_regulons, n_regulons)
        Binary adjacency matrix (symmetric, zero diagonal).
    """
    import pandas as pd

    n = len(regulon_names)
    adj_matrix = np.zeros((n, n), dtype=int)

    # Filter by importance
    if min_importance > 0:
        adjacencies = adjacencies[adjacencies["importance"] >= min_importance]

    # Build TF -> target sets
    tf_targets: Dict[str, Set[str]] = defaultdict(set)
    for _, row in adjacencies.iterrows():
        tf_targets[row["TF"]].add(row["target"])

    name_to_idx = {name: i for i, name in enumerate(regulon_names)}

    if mode in ("shared_targets", "combined"):
        # Connect regulons with shared targets
        for i, j in combinations(range(n), 2):
            tf_i, tf_j = regulon_names[i], regulon_names[j]
            targets_i = tf_targets.get(tf_i, set())
            targets_j = tf_targets.get(tf_j, set())
            shared = len(targets_i & targets_j)
            if shared >= min_shared:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    if mode in ("regulatory", "combined"):
        # Connect if one TF is a target of another
        for i in range(n):
            tf_i = regulon_names[i]
            targets_i = tf_targets.get(tf_i, set())
            for j in range(n):
                if i == j:
                    continue
                tf_j = regulon_names[j]
                if tf_j in targets_i:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

    return adj_matrix


class TopologyPruner:
    """
    Prune the feature subset lattice using GRN topology.

    Only generates subsets where all features are connected (within a
    specified graph distance) in the regulatory network. This dramatically
    reduces the number of subsets to evaluate while preserving biologically
    relevant interactions.

    Parameters
    ----------
    adjacency : np.ndarray, shape (n_features, n_features)
        Binary adjacency matrix from grn_to_adjacency().
    max_order : int, default=3
        Maximum subset size (interaction order).
    max_distance : int, default=2
        Maximum graph distance between any pair of features in a subset.
        1 = direct neighbors only.
        2 = neighbors of neighbors (captures feed-forward loops).
    min_degree : int, default=1
        Minimum node degree to include a feature. Features with fewer
        connections are excluded from higher-order subsets (but still
        evaluated for marginal contributions).

    Attributes
    ----------
    allowed_subsets_ : list of frozenset
        Pruned list of subsets to evaluate.
    pruning_ratio_ : float
        Fraction of subsets pruned (1.0 = all pruned, 0.0 = none).
    feature_degrees_ : dict
        Mapping from feature index to network degree.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        max_order: int = 3,
        max_distance: int = 2,
        min_degree: int = 1,
    ):
        """Initialize topology pruner with a GRN adjacency matrix."""
        self.adjacency = adjacency
        self.max_order = max_order
        self.max_distance = max_distance
        self.min_degree = min_degree

    def fit(self) -> "TopologyPruner":
        """
        Generate pruned subset list based on network topology.

        Returns
        -------
        self
        """
        n = self.adjacency.shape[0]

        # Compute distance matrix (BFS shortest paths)
        self.distance_matrix_ = self._compute_distances(self.adjacency)

        # Compute degrees
        self.feature_degrees_ = {
            i: int(self.adjacency[i].sum()) for i in range(n)
        }

        # Features eligible for higher-order interactions
        eligible = {
            i for i, d in self.feature_degrees_.items() if d >= self.min_degree
        }

        # Generate subsets
        self.allowed_subsets_ = []
        total_possible = 0

        for order in range(1, self.max_order + 1):
            if order == 1:
                # All single features (marginal contributions always evaluated)
                for i in range(n):
                    self.allowed_subsets_.append(frozenset([i]))
                total_possible += n
                continue

            # Higher-order: only connected subsets among eligible features
            candidates = list(combinations(sorted(eligible), order))
            total_possible += len(candidates)

            for combo in candidates:
                if self._is_connected_subset(combo):
                    self.allowed_subsets_.append(frozenset(combo))

        # Pruning statistics
        self.pruning_ratio_ = (
            1.0 - len(self.allowed_subsets_) / max(total_possible, 1)
        )

        return self

    def _compute_distances(self, adj: np.ndarray) -> np.ndarray:
        """Compute all-pairs shortest path distances via BFS."""
        n = adj.shape[0]
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)

        for source in range(n):
            # BFS from source
            visited = {source}
            queue = [(source, 0)]
            while queue:
                node, d = queue.pop(0)
                for neighbor in range(n):
                    if adj[node, neighbor] > 0 and neighbor not in visited:
                        dist[source, neighbor] = d + 1
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        return dist

    def _is_connected_subset(self, combo: Tuple[int, ...]) -> bool:
        """
        Check if all pairs in the subset are within max_distance.

        A subset is "connected" if the induced subgraph has diameter
        ≤ max_distance (i.e., every pair of features can reach each
        other within max_distance hops).
        """
        for i, j in combinations(combo, 2):
            if self.distance_matrix_[i, j] > self.max_distance:
                return False
        return True

    def summary(self) -> str:
        """Return a human-readable pruning summary."""
        n = self.adjacency.shape[0]
        lines = [
            f"TopologyPruner Summary",
            f"  Features: {n}",
            f"  Max order: {self.max_order}",
            f"  Max distance: {self.max_distance}",
            f"  Min degree: {self.min_degree}",
            f"  Allowed subsets: {len(self.allowed_subsets_)}",
            f"  Pruning ratio: {self.pruning_ratio_:.1%}",
            f"  Subsets by order:",
        ]
        for order in range(1, self.max_order + 1):
            n_order = sum(
                1 for s in self.allowed_subsets_ if len(s) == order
            )
            lines.append(f"    k={order}: {n_order}")

        mean_degree = np.mean(list(self.feature_degrees_.values()))
        lines.append(f"  Mean network degree: {mean_degree:.1f}")

        return "\n".join(lines)


class HierarchicalScreener:
    """
    Hierarchical (greedy ascent) screening on the interaction lattice.

    Evaluates interactions bottom-up: only promotes features to higher
    orders if they show significant contributions at lower orders.

    1. Evaluate all single-feature ablations.
    2. Screen pairwise only among features with significant marginal Δ.
    3. Screen triplets only among pairs with significant pairwise I(S).
    ... and so on.

    Parameters
    ----------
    significance_threshold : float, default=0.01
        Minimum |I(S)| or |Δ(f)| to promote to the next order.
    max_order : int, default=4
        Maximum interaction order.
    top_k_per_order : int, optional
        If set, only promote the top-k features/subsets at each order.
    """

    def __init__(
        self,
        significance_threshold: float = 0.01,
        max_order: int = 4,
        top_k_per_order: Optional[int] = None,
    ):
        """Initialize hierarchical screener with significance and order parameters."""
        self.significance_threshold = significance_threshold
        self.max_order = max_order
        self.top_k_per_order = top_k_per_order

    def generate_subsets(
        self,
        delta_cache: Dict[FrozenSet[int], float],
        interactions: Dict[FrozenSet[int], float],
        n_features: int,
    ) -> List[FrozenSet[int]]:
        """
        Generate next-order subsets based on current results.

        Parameters
        ----------
        delta_cache : dict
            Current Δ(T) values.
        interactions : dict
            Current I(S) values.
        n_features : int
            Total number of features.

        Returns
        -------
        list of frozenset
            New subsets to evaluate at the next order.
        """
        if not interactions:
            # Bootstrap: return all single features
            return [frozenset([i]) for i in range(n_features)]

        # Find current maximum order
        current_max_order = max(len(k) for k in interactions)
        next_order = current_max_order + 1

        if next_order > self.max_order:
            return []

        # Identify significant features at current order
        significant = [
            k
            for k, v in interactions.items()
            if len(k) == current_max_order
            and abs(v) >= self.significance_threshold
        ]

        if self.top_k_per_order is not None:
            significant = sorted(
                significant, key=lambda k: abs(interactions[k]), reverse=True
            )[: self.top_k_per_order]

        if not significant:
            return []

        # Generate next-order subsets by extending significant sets
        # Union all features that appear in significant subsets
        active_features = set()
        for s in significant:
            active_features.update(s)

        # Also include features significant at order 1
        for k, v in interactions.items():
            if len(k) == 1 and abs(v) >= self.significance_threshold:
                active_features.update(k)

        # Generate (next_order)-subsets from active features
        new_subsets = []
        for combo in combinations(sorted(active_features), next_order):
            fs = frozenset(combo)
            # Only include if all sub-subsets are already evaluated
            if fs not in delta_cache:
                new_subsets.append(fs)

        return new_subsets
