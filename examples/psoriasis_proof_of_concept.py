#!/usr/bin/env python3
"""
TOPPLE Psoriasis Proof-of-Concept
====================================

Complete workflow for running TOPPLE on psoriatic skin spatial
transcriptomics with pySCENIC regulon data.

Prerequisites
--------------
1. pySCENIC completed on scRNA-seq data:
   - GRNBoost2 adjacencies → adjacencies.csv
   - cisTarget regulons → regulons.csv or regulons.json
   - AUCell scores → stored in AnnData obsm['X_aucell']

2. Spatial transcriptomics (Visium or MERFISH):
   - Coordinates in adata.obsm['spatial']

3. Cell type annotation + Harmony integration (if multi-sample):
   - adata.obs['cell_type'] with labels including 'CD8_TRM'
   - adata.obs['condition'] with 'psoriasis' vs 'healthy'

Data structure expected
------------------------
adata.h5ad:
    adata.obsm['X_aucell']     # (n_cells, n_regulons) AUCell matrix
    adata.obsm['spatial']      # (n_cells, 2) spatial coordinates
    adata.obs['cell_type']     # Cell type annotations
    adata.obs['condition']     # 'psoriasis' or 'healthy'
    adata.obs['sample_id']     # Sample batch (post-Harmony)

adjacencies.csv:
    TF,target,importance
    RUNX3,ITGAE,12.5
    RUNX3,CD103,8.2
    ...

Usage
------
    python psoriasis_proof_of_concept.py \\
        --h5ad psoriasis_scenic.h5ad \\
        --adjacencies adjacencies.csv \\
        --regulons regulons.csv \\
        --output results/
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="TOPPLE psoriasis proof-of-concept"
    )
    parser.add_argument(
        "--h5ad", type=str, required=True,
        help="Path to AnnData .h5ad with pySCENIC results"
    )
    parser.add_argument(
        "--adjacencies", type=str, default=None,
        help="Path to GRNBoost2 adjacencies.csv"
    )
    parser.add_argument(
        "--regulons", type=str, default=None,
        help="Path to regulon definitions (csv/json/gmt)"
    )
    parser.add_argument(
        "--aucell-key", type=str, default="X_aucell",
        help="Key in adata.obsm for AUCell matrix"
    )
    parser.add_argument(
        "--cell-type-key", type=str, default="cell_type",
        help="Key in adata.obs for cell type annotations"
    )
    parser.add_argument(
        "--condition-key", type=str, default="condition",
        help="Key in adata.obs for disease condition"
    )
    parser.add_argument(
        "--pathological", type=str, default="psoriasis",
        help="Value in condition_key marking pathological state"
    )
    parser.add_argument(
        "--target-type", type=str, default="CD8_TRM",
        help="Target cell type"
    )
    parser.add_argument(
        "--stromal-types", type=str, nargs="+",
        default=["fibroblast", "pericyte", "endothelial"],
        help="Stromal cell types for buffering estimation"
    )
    parser.add_argument(
        "--regulon-list", type=str, nargs="+", default=None,
        help="Specific regulons to analyze (e.g., RUNX3 TBX21 EOMES)"
    )
    parser.add_argument(
        "--max-order", type=int, default=3,
        help="Maximum interaction order for L1 decomposition"
    )
    parser.add_argument(
        "--n-niches", type=int, default=3,
        help="Number of spatial niches"
    )
    parser.add_argument(
        "--output", type=str, default="topple_results",
        help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("TOPPLE: Psoriasis Proof-of-Concept")
    print("=" * 70)

    # ================================================================
    # Step 1: Load data
    # ================================================================
    print("\n[Step 1] Loading data...")

    from topple.data import TOPPLEData

    data = TOPPLEData.from_anndata(
        args.h5ad,
        aucell_key=args.aucell_key,
        cell_type_key=args.cell_type_key,
        condition_key=args.condition_key,
        pathological_value=args.pathological,
        adjacencies_path=args.adjacencies,
        regulons_path=args.regulons,
    )

    print(data.summary())

    # ================================================================
    # Step 2: Select regulons
    # ================================================================
    print("\n[Step 2] Selecting regulons...")

    if args.regulon_list:
        data.select_regulons(regulon_list=args.regulon_list)
    else:
        # Auto-select: top 15 most variable regulons with >= 20 active cells
        data.select_regulons(
            min_cells_active=20,
            min_activity_threshold=0.01,
            top_n_variable=15,
        )

    print(f"  Selected {data.n_regulons} regulons: {', '.join(data.regulon_names)}")

    # ================================================================
    # Step 3: Define states
    # ================================================================
    print("\n[Step 3] Defining pathological vs homeostatic states...")

    if data.state_labels is not None:
        n_path = (data.state_labels == 1).sum()
        n_homeo = (data.state_labels == 0).sum()
        print(f"  Pathological: {n_path}, Homeostatic: {n_homeo}")
    else:
        print("  WARNING: No condition column found. You may need to set labels manually.")
        print("  Example: data.define_states(labels=your_binary_labels)")
        return

    # ================================================================
    # Step 4: Choose decomposition method
    # ================================================================
    n_reg = data.n_regulons
    if n_reg <= 15:
        method = "exact"
    elif n_reg <= 30 and data.adjacencies is not None:
        method = "pruned"
    else:
        method = "compressed"

    print(f"\n[Step 4] Decomposition method: {method} "
          f"(p={n_reg}, adjacencies={'yes' if data.adjacencies is not None else 'no'})")

    # ================================================================
    # Step 5: Run full pipeline
    # ================================================================
    print(f"\n[Step 5] Running TOPPLE L1→L2→L3...")

    results = data.run_topple(
        target_type=args.target_type,
        stromal_types=args.stromal_types,
        max_order=args.max_order,
        method=method,
        n_niches=args.n_niches,
        verbose=True,
    )

    # ================================================================
    # Step 6: Export results
    # ================================================================
    print(f"\n[Step 6] Saving results to {args.output}/")

    # L1: Interactions
    sd = results["decomposition"]
    with open(os.path.join(args.output, "L1_interactions.tsv"), "w") as f:
        f.write("features\torder\tinteraction_value\n")
        for feat, val in sd.top_interactions(n=50):
            f.write(f"{' × '.join(feat)}\t{len(feat)}\t{val:.6f}\n")
    print(f"  L1_interactions.tsv")

    # L1: Variance explained
    with open(os.path.join(args.output, "L1_variance_explained.tsv"), "w") as f:
        f.write("order\tvariance_fraction\tcumulative\n")
        ve = sd.variance_explained()
        cum = 0
        for k, v in sorted(ve.items()):
            cum += v
            f.write(f"{k}\t{v:.6f}\t{cum:.6f}\n")
    print(f"  L1_variance_explained.tsv")

    # L2: Bridge report
    with open(os.path.join(args.output, "L2_bridge_report.txt"), "w") as f:
        f.write(results["bridge_report"])
    print(f"  L2_bridge_report.txt")

    # L2: Candidates table
    if results["bridge_results"]:
        with open(os.path.join(args.output, "L2_candidates.tsv"), "w") as f:
            f.write("rank\tfeatures\tSI\tD_pathological\tD_homeostatic\n")
            for i, r in enumerate(results["bridge_results"]):
                feat_str = " + ".join(r["features"])
                si = r.get("selectivity_index", 0)
                dp = r.get("d_pathological", 0)
                dh = r.get("d_homeostatic", 0)
                f.write(f"{i+1}\t{feat_str}\t{si:.3f}\t{dp:.3f}\t{dh:.3f}\n")
        print(f"  L2_candidates.tsv")

    # L3: Spatial report
    if results["spatial_report"]:
        with open(os.path.join(args.output, "L3_spatial_report.txt"), "w") as f:
            f.write(results["spatial_report"])
        print(f"  L3_spatial_report.txt")

    # L3: Per-cell DataFrame
    if results["dataframe"] is not None:
        results["dataframe"].to_csv(
            os.path.join(args.output, "L3_vulnerability_per_cell.csv"), index=False,
        )
        print(f"  L3_vulnerability_per_cell.csv")

    # Summary
    with open(os.path.join(args.output, "TOPPLE_summary.txt"), "w") as f:
        f.write("TOPPLE Psoriasis Proof-of-Concept Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target: {args.target_type}\n")
        f.write(f"Regulons: {', '.join(data.regulon_names)}\n")
        f.write(f"Method: {method}, max_order={args.max_order}\n\n")
        f.write(sd.report() + "\n\n")
        if results["spatial_report"]:
            f.write(results["spatial_report"])
    print(f"  TOPPLE_summary.txt")

    print(f"\n{'='*70}")
    print("TOPPLE analysis complete.")
    print(f"{'='*70}")


# =====================================================================
# Alternative: interactive / notebook workflow
# =====================================================================

NOTEBOOK_TEMPLATE = '''
# TOPPLE Psoriasis Analysis — Interactive Notebook Workflow
# =========================================================

import scanpy as sc
import numpy as np
from topple.data import TOPPLEData

# ---- 1. Load your AnnData ----
adata = sc.read_h5ad("psoriasis_scenic.h5ad")
print(adata)

# ---- 2. Create TOPPLEData ----
data = TOPPLEData.from_anndata(
    adata,
    aucell_key="X_aucell",          # Where AUCell scores live
    cell_type_key="cell_type",       # Cell type column
    spatial_key="spatial",           # Spatial coordinates
    condition_key="condition",       # Disease condition
    pathological_value="psoriasis",  # Which value = pathological
    adjacencies_path="adjacencies.csv",  # GRNBoost2 output
)
print(data.summary())

# ---- 3. Select TRM-relevant regulons ----
# Option A: Manually specify TRM regulons
data.select_regulons(regulon_list=[
    "RUNX3", "TBX21", "EOMES", "NR4A1",
    "IRF4", "BATF", "PRDM1", "TOX",
    "BHLHE40", "HOBIT", "IKZF2",
])

# Option B: Auto-select top variable regulons
# data.select_regulons(top_n_variable=12, min_cells_active=20)

# ---- 4. Run full pipeline ----
results = data.run_topple(
    target_type="CD8_TRM",
    stromal_types=["fibroblast", "pericyte", "endothelial"],
    max_order=3,
    n_niches=3,
)

# ---- 5. Examine results ----

# Top interactions (L1)
sd = results["decomposition"]
for feat, val in sd.top_interactions(n=10):
    print(f"  {' × '.join(feat):>30s}: {val:+.4f}")

# Variance by order
print(sd.variance_explained())

# Perturbation candidates (L2)
print(results["bridge_report"])

# Spatial vulnerability (L3)
if results["spatial_report"]:
    print(results["spatial_report"])

# Per-cell vulnerability DataFrame
df = results["dataframe"]
if df is not None:
    # Plot vulnerability map with scanpy
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(["vulnerability", "destabilization", "buffering"]):
        sc_ = axes[i].scatter(df["x"], df["y"], c=df[col], s=3, cmap="RdYlBu_r")
        axes[i].set_title(col)
        plt.colorbar(sc_, ax=axes[i])
    plt.tight_layout()
    plt.savefig("topple_vulnerability_map.pdf", dpi=300)
    plt.show()

# ---- 6. Seurat integration (R → Python) ----
# In R:
#   library(SeuratDisk)
#   SaveH5Seurat(seurat_obj, "seurat_export.h5seurat")
#   Convert("seurat_export.h5seurat", dest="h5ad")
#
# Then in Python:
#   data = TOPPLEData.from_seurat("seurat_export.h5ad", aucell_key="X_aucell")
'''


if __name__ == "__main__":
    main()
