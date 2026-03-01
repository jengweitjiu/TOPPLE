"""
Perturbation Bridge Pipeline
==============================

Orchestrates the complete Layer 1 -> Layer 2 workflow:

    Layer 1 decomposition
    -> Target selection (stability + IPA ranking)
    -> Perturbation simulation (CellOracle / Mock)
    -> Destabilization scoring
    -> Selectivity ranking

This is the main entry point for TOPPLE Layer 2 analysis.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional

import numpy as np

from .target_selection import TargetSelector, PerturbationCandidate
from .perturbation_engine import PerturbationEngine, PerturbationResult
from .destabilization import (
    DestabilizationScorer,
    SelectivityIndex,
    SelectivityResult,
)


class PerturbationBridge:
    """
    Full Layer 1 -> Layer 2 bridge pipeline.

    Connects stability decomposition to perturbation prediction with
    selectivity constraints.

    Parameters
    ----------
    engine : PerturbationEngine
        Perturbation simulation engine (CellOracleAdapter or MockPerturbationEngine).
    X : np.ndarray, shape (n_cells, n_features)
        Original regulon activity matrix.
    y : np.ndarray, shape (n_cells,)
        Binary state labels (0 = homeostatic, 1 = pathological).
    feature_names : list of str
        Regulon/TF names.
    interactions : dict
        Möbius interaction terms from Layer 1.
    de_scores : dict, optional
        Differential expression scores for IPA weighting.
    max_candidates : int, default=20
        Maximum perturbation candidates to simulate.
    perturbation_type : str, default="knockout"
        Type of perturbation to simulate.
    verbose : bool, default=True
        Print progress.

    Examples
    --------
    >>> from topple.layer2 import PerturbationBridge
    >>> from topple.layer2.perturbation_engine import MockPerturbationEngine
    >>>
    >>> engine = MockPerturbationEngine(X, y, feature_names)
    >>> bridge = PerturbationBridge(
    ...     engine=engine, X=X, y=y,
    ...     feature_names=feature_names,
    ...     interactions=decomp.interactions_,
    ... )
    >>> results = bridge.run()
    >>> bridge.report()
    """

    def __init__(
        self,
        engine: PerturbationEngine,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        interactions: Dict[FrozenSet[int], float],
        de_scores: Optional[dict] = None,
        max_candidates: int = 20,
        perturbation_type: str = "knockout",
        verbose: bool = True,
    ):
        self.engine = engine
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.interactions = interactions
        self.de_scores = de_scores
        self.max_candidates = max_candidates
        self.perturbation_type = perturbation_type
        self.verbose = verbose

    def run(self) -> List[SelectivityResult]:
        """
        Execute the full bridge pipeline.

        Returns
        -------
        list of SelectivityResult
            Ranked by selectivity index.
        """
        # Step 1: Target selection
        if self.verbose:
            print("[TOPPLE L2] Step 1: Target selection...")

        selector = TargetSelector(
            interactions=self.interactions,
            feature_names=self.feature_names,
            de_scores=self.de_scores,
            max_candidates=self.max_candidates,
        )
        candidates = selector.rank()

        if self.verbose:
            print(f"[TOPPLE L2]   {len(candidates)} candidates selected")

        if not candidates:
            print("[TOPPLE L2] WARNING: No candidates passed threshold.")
            return []

        # Step 2: Train destabilization scorer
        if self.verbose:
            print("[TOPPLE L2] Step 2: Training boundary classifier...")

        scorer = DestabilizationScorer(self.X, self.y)
        scorer.fit()

        selectivity = SelectivityIndex(scorer)

        # Step 3: Simulate and score each candidate
        if self.verbose:
            print(f"[TOPPLE L2] Step 3: Simulating {len(candidates)} perturbations...")

        selectivity_results = []

        for i, candidate in enumerate(candidates):
            if self.verbose and (i + 1) % 5 == 0:
                print(f"[TOPPLE L2]   Progress: {i+1}/{len(candidates)}")

            try:
                # Simulate perturbation
                pert_result = self.engine.simulate(
                    feature_indices=candidate.features,
                    perturbation_type=self.perturbation_type,
                )

                # Compute selectivity
                sel_result = selectivity.compute(
                    X_original=pert_result.X_original,
                    X_perturbed=pert_result.X_perturbed,
                    state_labels=pert_result.state_labels,
                    perturbation_set=candidate.features,
                    feature_names=candidate.feature_names,
                )
                selectivity_results.append(sel_result)

            except Exception as e:
                if self.verbose:
                    print(
                        f"[TOPPLE L2]   WARN: Failed for "
                        f"{' + '.join(candidate.feature_names)}: {e}"
                    )
                continue

        # Step 4: Rank by selectivity
        if self.verbose:
            print("[TOPPLE L2] Step 4: Ranking by selectivity...")

        ranked = selectivity.rank_candidates(
            selectivity_results, min_d_pathological=0.01
        )

        self.results_ = ranked
        self.all_results_ = selectivity_results
        self.candidates_ = candidates

        if self.verbose:
            print(f"[TOPPLE L2] Done. {len(ranked)} viable candidates ranked.")

        return ranked

    def report(self, top_n: int = 15) -> str:
        """
        Generate a text report of the bridge results.

        Parameters
        ----------
        top_n : int, default=15
            Number of top results to show.

        Returns
        -------
        str
            Formatted report.
        """
        if not hasattr(self, "results_"):
            return "No results. Call .run() first."

        lines = [
            "=" * 65,
            "TOPPLE Layer 2: Perturbation Bridge Report",
            "=" * 65,
            f"Candidates evaluated: {len(self.all_results_)}",
            f"Viable (D_path >= 0.01): {len(self.results_)}",
            f"Perturbation type: {self.perturbation_type}",
            "",
        ]

        if self.results_:
            lines.append(
                f"{'Rank':<5} {'Perturbation':<30} {'D_path':<8} "
                f"{'D_homeo':<8} {'SI':<8}"
            )
            lines.append("-" * 65)

            for r in self.results_[:top_n]:
                feat_str = " + ".join(r.feature_names)
                if len(feat_str) > 28:
                    feat_str = feat_str[:25] + "..."
                lines.append(
                    f"#{r.rank:<4} {feat_str:<30} {r.d_pathological:<8.3f} "
                    f"{r.d_homeostatic:<8.3f} {r.selectivity_index:<8.2f}"
                )

            # Summary stats
            top5 = self.results_[:5]
            if top5:
                lines.extend([
                    "",
                    "--- Top 5 Summary ---",
                    f"  Best SI: {top5[0].selectivity_index:.2f}x "
                    f"({' + '.join(top5[0].feature_names)})",
                    f"  Mean D_pathological: {np.mean([r.d_pathological for r in top5]):.3f}",
                    f"  Mean D_homeostatic: {np.mean([r.d_homeostatic for r in top5]):.3f}",
                ])

                # Check for any candidates with D_homeo > D_path (bad)
                risky = [r for r in self.results_ if r.selectivity_index < 1.0]
                if risky:
                    lines.append(
                        f"\n  WARNING: {len(risky)} candidates have SI < 1.0 "
                        f"(destabilize homeostatic MORE than pathological)"
                    )
        else:
            lines.append("No viable candidates found.")

        return "\n".join(lines)

    def to_dataframe(self):
        """Export results as pandas DataFrame."""
        import pandas as pd

        if not hasattr(self, "results_"):
            return pd.DataFrame()

        rows = []
        for r in self.all_results_:
            rows.append({
                "rank": r.rank,
                "features": " + ".join(r.feature_names),
                "order": len(r.perturbation_set),
                "d_pathological": r.d_pathological,
                "d_homeostatic": r.d_homeostatic,
                "selectivity_index": r.selectivity_index,
                "viable": r.rank > 0,
            })

        df = pd.DataFrame(rows)
        return df.sort_values("selectivity_index", ascending=False).reset_index(drop=True)
