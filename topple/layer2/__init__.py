"""
TOPPLE Layer 2: Causal Perturbation Bridge
===========================================

Bridges the gap between computational stability decomposition (Layer 1)
and biological perturbation outcomes. Integrates with CellOracle for
in silico TF perturbation prediction, with selectivity constraints
ensuring pathological state destabilization without homeostatic disruption.

Pipeline:
    Layer 1 interactions → Target Selection → CellOracle Simulation
    → Destabilization Scoring → Selectivity Ranking
"""

from .target_selection import TargetSelector, PerturbationCandidate
from .perturbation_engine import PerturbationEngine, CellOracleAdapter
from .destabilization import DestabilizationScorer, SelectivityIndex
from .bridge import PerturbationBridge

__all__ = [
    "TargetSelector",
    "PerturbationCandidate",
    "PerturbationEngine",
    "CellOracleAdapter",
    "DestabilizationScorer",
    "SelectivityIndex",
    "PerturbationBridge",
]
