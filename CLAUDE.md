# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TOPPLE (Topological Perturbation Prediction from Landscape Estimation) is a computational biology Python package for analyzing cell state stability and predicting perturbation outcomes in single-cell and spatial transcriptomics. It decomposes stability via Mobius inversion on feature lattices and bridges to causal perturbation prediction.

## Common Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_layer1.py -v

# Run a single test by name
pytest tests/test_layer1.py -k "test_name" -v

# Run with coverage
pytest tests/ --cov=topple --cov-report=xml

# Lint
ruff check topple/ tests/ --ignore E501,F401,E402

# Format
black topple/ tests/
```

## Architecture

The package is organized into three computational layers plus a data module:

**Layer 1 (`topple/`)** - Higher-order stability decomposition. Core math: for subset S, interaction `I(S) = sum_{T subset S} (-1)^{|S|-|T|} * Delta(T)` where Delta(T) is the stability loss from ablating features in T. Four methods scale to different feature counts:
- Exact enumeration (p <= 15) via `MobiusDecomposition` in `mobius.py`
- Topology-guided pruning (p <= 30 with GRN) via `TopologyPruner` in `pruning.py`
- Compressed sensing (p > 30) via `CompressedInteractionSensing` in `compressed_sensing.py`
- Hierarchical screening (exploratory) via `HierarchicalScreener` in `pruning.py`

The main API entry point is `StabilityDecomposer` in `stability.py`, which orchestrates method selection. Scoring uses `geometric_depth_cv()` (cross-validated AUC) and `separability_gate()` as a quality threshold.

**Layer 2 (`topple/layer2/`)** - Causal perturbation bridge. Connects Layer 1 interactions to biological perturbation outcomes:
- `bridge.py`: `PerturbationBridge` orchestrator
- `target_selection.py`: `TargetSelector` ranks candidate TFs using IPA (Interaction Participation Analysis)
- `perturbation_engine.py`: `PerturbationEngine` ABC with `CellOracleAdapter` (real) and `MockPerturbationEngine` (synthetic)
- `destabilization.py`: `DestabilizationScorer` for cell-level impact, `SelectivityIndex` for pathological vs homeostatic selectivity

**Layer 3 (`topple/layer3/`)** - Spatial vulnerability mapping. Contextualizes perturbations within tissue architecture:
- `pipeline.py`: `SpatialVulnerabilityPipeline` orchestrator
- `spatial_buffering.py`: `StromalBufferingEstimator` computes SICAI stromal buffering coefficient (beta)
- `vulnerability.py`: Core equation `V(i,P) = D(i,P) * (1 - beta_spatial(i))`
- `niche.py`: `NicheStratifier` (K-means tissue niche discovery) and `NichePerturbationRanker`

**Data (`topple/data/`)** - `TOPPLEData` in `loader.py` provides unified loading for AnnData (.h5ad), pySCENIC outputs (adjacencies/regulons/aucell CSVs), Seurat conversions, and Visium/MERFISH spatial coordinates.

## Key Conventions

- Line length: 100 characters (ruff and black)
- Target Python version: 3.9+ (tested on 3.9-3.12)
- Build system: setuptools via pyproject.toml (PEP 517, no setup.py)
- Core dependencies are minimal (numpy, scipy, scikit-learn, pandas); heavy bio packages are optional extras
- Tests use synthetic data fixtures; no external data files needed
- Windows compatibility: avoid Unicode characters in source (cp950 encoding constraint, see commit 77d388e)
