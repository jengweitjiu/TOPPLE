# TOPPLE — Topological Perturbation Prediction from Landscape Estimation

**Layer 1: Higher-Order Stability Decomposition**

TOPPLE extends [DGSA](https://github.com/jengweitjiu/DGSA-stability) from pairwise synergy to arbitrary-order interaction terms via Möbius inversion on the feature subset lattice.

## Overview

Cell state stability depends on complex regulatory interactions that extend beyond pairwise relationships. TOPPLE Layer 1 decomposes stability into higher-order interaction terms *I(S)* for arbitrary feature subsets *S*, revealing cooperative regulatory logic invisible to pairwise analysis.

### Key components

| Module | Description |
|--------|-------------|
| `mobius.py` | Möbius inversion on the feature subset lattice |
| `pruning.py` | Topology-guided pruning via GRN structure (pySCENIC) |
| `compressed_sensing.py` | L1-regularized recovery of sparse interaction coefficients |
| `stability.py` | CV geometric depth scoring + DGSA separability gate |

## Installation

```bash
pip install -e .          # Core dependencies
pip install -e ".[all]"   # All optional dependencies
```

## Quick start

```python
from topple import StabilityDecomposer

# X: regulon activity matrix (n_cells x n_regulons)
# y: binary state labels (0 = homeostatic, 1 = pathological)
sd = StabilityDecomposer(max_order=3, method="exact")
sd.fit(X, y, feature_names=regulon_names)
sd.top_interactions(order=3, n=10)
sd.variance_explained()
print(sd.report())
```

### With topology-guided pruning (p > 15)

```python
from topple import StabilityDecomposer, grn_to_adjacency
adj = grn_to_adjacency(pyscenic_adjacencies, regulon_names, mode="combined")
sd = StabilityDecomposer(max_order=4, method="pruned", adjacency=adj)
sd.fit(X, y)
```

### With compressed sensing (large feature sets)

```python
sd = StabilityDecomposer(max_order=3, method="compressed")
sd.fit(X, y)
```

## Mathematical framework

For subset S, the k-th order interaction term:

    I(S) = sum_{T in S} (-1)^{|S|-|T|} Delta(T)

| Order | Interpretation | DGSA equivalent |
|-------|---------------|-----------------|
| k=1 | Marginal contribution | Single-feature ablation |
| k=2 | Pairwise synergy | S(A,B) |
| k=3+ | Higher-order cooperation | **New in TOPPLE** |

## Efficiency strategies

| Strategy | Use when |
|----------|----------|
| Exact | p <= 15 |
| Topology-guided pruning | p <= 30, GRN available |
| Compressed sensing | p > 30 |
| Hierarchical screening | Exploratory |

## Layer 3: Spatial Vulnerability Mapping

Contextualizes perturbation predictions within tissue architecture.

```python
from topple.layer3 import SpatialVulnerabilityPipeline

pipeline = SpatialVulnerabilityPipeline(
    coordinates=spatial_coords,
    cell_types=cell_labels,
    regulon_activity=aucell_matrix,
    target_type="CD8_TRM",
    stromal_types=["fibroblast", "endothelial"],
    n_niches=3,
)
vmaps = pipeline.run(layer2_destabilizations)
print(pipeline.report())
```

### Layer 3 modules

| Module | Description |
|--------|-------------|
| `spatial_buffering.py` | Stromal buffering β(i) from neighborhood density + SICAI coupling + LR signaling |
| `vulnerability.py` | Core equation: V(i,P) = D(i,P) · (1 - β(i)) |
| `niche.py` | Niche discovery + niche-stratified perturbation ranking |
| `pipeline.py` | Full L3 pipeline orchestrator |

### Spatial Vulnerability Equation

    V(i, P) = D_pathological(i, P) · (1 - β_spatial(i))

High V = strong destabilization AND weak stromal protection. Niche-discordant perturbations (high spread across niches) are flagged automatically.

## Integration

Layer 1 → Layer 2 → Layer 3 (complete)

Related: [DGSA](https://github.com/jengweitjiu/DGSA-stability) | [IPA](https://github.com/jengweitjiu/IPA-kill-experiment) | STRATA | SICAI

## Tests

```bash
pytest tests/ -v
# 55 tests: 18 Layer 1 + 19 Layer 2 + 18 Layer 3
```

## License

MIT
