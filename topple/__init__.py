"""
TOPPLE: Topological Perturbation Prediction from Landscape Estimation
=====================================================================

Layer 1 — Higher-Order Stability Decomposition

Extends DGSA (Decomposable Geometric Stability Analysis) from pairwise synergy
to arbitrary-order interaction terms via Möbius inversion on the feature subset
lattice, with topology-guided pruning and compressed interaction sensing for
computational efficiency.

Author: Jeng-Wei Tjiu
Affiliation: National Taiwan University Hospital / NTU College of Medicine
"""

__version__ = "0.1.0"

from .mobius import (
    MobiusDecomposition,
    mobius_inversion,
    interaction_term,
    stability_loss,
)
from .pruning import TopologyPruner, grn_to_adjacency
from .compressed_sensing import CompressedInteractionSensing
from .stability import (
    StabilityDecomposer,
    geometric_depth_cv,
    separability_gate,
)
from .layer2 import (
    TargetSelector,
    PerturbationCandidate,
    PerturbationEngine,
    CellOracleAdapter,
    DestabilizationScorer,
    SelectivityIndex,
    PerturbationBridge,
)

__all__ = [
    # Layer 1
    "MobiusDecomposition",
    "mobius_inversion",
    "interaction_term",
    "stability_loss",
    "TopologyPruner",
    "grn_to_adjacency",
    "CompressedInteractionSensing",
    "StabilityDecomposer",
    "geometric_depth_cv",
    "separability_gate",
    # Layer 2
    "TargetSelector",
    "PerturbationCandidate",
    "PerturbationEngine",
    "CellOracleAdapter",
    "DestabilizationScorer",
    "SelectivityIndex",
    "PerturbationBridge",
]
