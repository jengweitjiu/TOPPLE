"""
TOPPLE Layer 3: Spatial Vulnerability Mapping
===============================================

Contextualizes perturbation predictions within tissue architecture.
Integrates STRATA (spatial regulatory zones) and SICAI (stromal-immune
coupling) to produce spatially-resolved vulnerability maps.

Core equation:
    V(i, P) = D(i, P) · (1 - β_spatial(i))

where D(i,P) is cell-level destabilization from Layer 2, and
β_spatial(i) ∈ [0,1] is the stromal buffering coefficient from SICAI.

Pipeline:
    Layer 2 destabilization → Spatial buffering → Vulnerability scoring
    → Niche stratification → Niche-optimal perturbation ranking
"""

from .spatial_buffering import (
    StromalBufferingEstimator,
    compute_ligand_receptor_score,
    neighborhood_coupling,
)
from .vulnerability import (
    SpatialVulnerabilityScorer,
    VulnerabilityMap,
)
from .niche import (
    NicheStratifier,
    NichePerturbationRanker,
)
from .pipeline import SpatialVulnerabilityPipeline

__all__ = [
    "StromalBufferingEstimator",
    "compute_ligand_receptor_score",
    "neighborhood_coupling",
    "SpatialVulnerabilityScorer",
    "VulnerabilityMap",
    "NicheStratifier",
    "NichePerturbationRanker",
    "SpatialVulnerabilityPipeline",
]
