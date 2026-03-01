"""
TOPPLE Data Connectors
========================

Load and prepare real single-cell / spatial data for the TOPPLE pipeline.

Supported inputs:
- AnnData (.h5ad) with pySCENIC AUCell scores
- pySCENIC outputs (adjacencies.csv, regulons.csv, aucell.csv)
- Seurat objects (via .h5ad conversion or .rds → anndata)
- Visium / MERFISH spatial coordinates

The main entry point is `TOPPLEData.from_anndata()` or `TOPPLEData.from_files()`.
"""

from .loader import (
    TOPPLEData,
    load_pyscenic_adjacencies,
    load_pyscenic_regulons,
    load_aucell_matrix,
)

__all__ = [
    "TOPPLEData",
    "load_pyscenic_adjacencies",
    "load_pyscenic_regulons",
    "load_aucell_matrix",
]
