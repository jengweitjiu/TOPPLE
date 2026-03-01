#!/usr/bin/env python3
"""
TOPPLE Full Pipeline: Layer 1 → Layer 2 → Layer 3
====================================================

Complete stability-guided, spatially-informed perturbation prediction
on synthetic psoriasis tissue with TRM cells, fibroblasts, and
endothelial cells in distinct spatial niches.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topple import MobiusDecomposition, geometric_depth_cv
from topple.layer2.perturbation_engine import MockPerturbationEngine
from topple.layer2.destabilization import DestabilizationScorer
from topple.layer3.pipeline import SpatialVulnerabilityPipeline


def simulate_spatial_psoriasis(n_cells=600, random_state=42):
    """Simulate spatial tissue with embedded regulatory structure."""
    rng = np.random.RandomState(random_state)

    n_trm = 150
    n_fibro = 200
    n_endo = 100
    n_kerat = n_cells - n_trm - n_fibro - n_endo

    # Spatial layout
    coords_trm = rng.rand(n_trm, 2) * 10
    coords_fibro = np.column_stack([rng.rand(n_fibro)*4, rng.rand(n_fibro)*10])
    coords_endo = np.column_stack([rng.rand(n_endo)*3+7, rng.rand(n_endo)*10])
    coords_kerat = np.column_stack([rng.rand(n_kerat)*10, rng.rand(n_kerat)*10])
    coordinates = np.vstack([coords_trm, coords_fibro, coords_endo, coords_kerat])

    cell_types = np.array(
        ["CD8_TRM"]*n_trm + ["fibroblast"]*n_fibro
        + ["endothelial"]*n_endo + ["keratinocyte"]*n_kerat
    )

    names = ["RUNX3", "TBX21", "EOMES", "NR4A1", "IRF4", "BATF", "PRDM1", "TOX"]

    # Regulon activity
    regulon_activity = rng.randn(n_cells, 8) * 0.5

    # TRM state: pathological in center (x: 3.5-6.5, sparse stroma)
    trm_x = coordinates[:n_trm, 0]
    y_trm = np.zeros(n_trm)
    y_trm[(trm_x > 3.5) & (trm_x < 6.5)] = 1

    # Embed regulatory signal in pathological TRM
    path_mask_trm = y_trm == 1
    regulon_activity[:n_trm][path_mask_trm, 0] += 1.5   # RUNX3
    regulon_activity[:n_trm][path_mask_trm, 1] += 1.5   # TBX21
    regulon_activity[:n_trm][path_mask_trm, 2] += 1.0   # EOMES
    regulon_activity[:n_trm][path_mask_trm, 4] += 0.8   # IRF4
    regulon_activity[:n_trm][path_mask_trm, 5] += 0.8   # BATF
    regulon_activity[:n_trm][path_mask_trm, 6] += 0.8   # PRDM1

    # Add synergy noise
    corr = rng.randn(n_trm) * 0.7
    regulon_activity[:n_trm, 0] += corr
    regulon_activity[:n_trm, 1] -= corr

    # Full state labels (non-TRM = 0)
    y_full = np.zeros(n_cells)
    y_full[:n_trm] = y_trm

    de_scores = {
        "RUNX3": 4.2, "TBX21": 3.8, "EOMES": 2.5, "NR4A1": 0.8,
        "IRF4": 1.5, "BATF": 1.2, "PRDM1": 0.9, "TOX": 0.3,
    }

    return coordinates, cell_types, regulon_activity, y_trm, y_full, names, de_scores


def main():
    print("=" * 70)
    print("TOPPLE Full Pipeline: Layer 1 → Layer 2 → Layer 3")
    print("=" * 70)

    coords, ctypes, regulons, y_trm, y_full, names, de_scores = simulate_spatial_psoriasis()
    n_trm = (ctypes == "CD8_TRM").sum()
    print(f"\nTissue: {len(ctypes)} cells")
    print(f"  CD8_TRM: {n_trm} ({(y_trm==1).sum()} pathological, {(y_trm==0).sum()} homeostatic)")
    print(f"  fibroblast: {(ctypes=='fibroblast').sum()}")
    print(f"  endothelial: {(ctypes=='endothelial').sum()}")
    print(f"  keratinocyte: {(ctypes=='keratinocyte').sum()}")

    # ==================================================================
    # LAYER 1: Stability decomposition on TRM regulons
    # ==================================================================
    print("\n" + "=" * 70)
    print("LAYER 1: Higher-Order Stability Decomposition")
    print("=" * 70)

    X_trm = regulons[:n_trm]
    decomp = MobiusDecomposition(scorer=geometric_depth_cv, max_order=2, verbose=True)
    decomp.fit(X_trm, y_trm, feature_names=names)

    print("\n--- Top interactions ---")
    for feat, val in decomp.top_interactions(n=8):
        print(f"  {' × '.join(feat):>25s}: {val:+.4f}")

    # ==================================================================
    # LAYER 2: Perturbation simulation + selectivity
    # ==================================================================
    print("\n" + "=" * 70)
    print("LAYER 2: Perturbation Bridge")
    print("=" * 70)

    engine = MockPerturbationEngine(X_trm, y_trm, names, effect_size=1.5)
    scorer = DestabilizationScorer(X_trm, y_trm)
    scorer.fit()

    # Select top perturbation candidates from L1
    candidates = [
        (frozenset([0, 1]), ["RUNX3", "TBX21"]),
        (frozenset([2]), ["EOMES"]),
        (frozenset([4, 5, 6]), ["IRF4", "BATF", "PRDM1"]),
        (frozenset([0, 2]), ["RUNX3", "EOMES"]),
        (frozenset([1, 5]), ["TBX21", "BATF"]),
    ]

    # Simulate and get per-cell destabilizations for Layer 3
    destabilizations = []
    for fset, fnames in candidates:
        result = engine.simulate(fset)
        # Get per-cell confidence drop for pathological TRM
        orig_conf = scorer.classifier.predict_proba(result.X_original)[:, 1]
        pert_conf = scorer.classifier.predict_proba(result.X_perturbed)[:, 1]
        conf_drop = np.clip(orig_conf - pert_conf, 0, 1)

        mean_d = conf_drop[y_trm == 1].mean() if (y_trm == 1).sum() > 0 else 0
        print(f"  {' + '.join(fnames):<30s} mean_D={mean_d:.3f}")

        destabilizations.append((conf_drop, fset, fnames))

    # ==================================================================
    # LAYER 3: Spatial vulnerability mapping
    # ==================================================================
    print("\n" + "=" * 70)
    print("LAYER 3: Spatial Vulnerability Mapping")
    print("=" * 70)

    pipeline = SpatialVulnerabilityPipeline(
        coordinates=coords,
        cell_types=ctypes,
        regulon_activity=regulons,
        target_type="CD8_TRM",
        stromal_types=["fibroblast", "endothelial"],
        n_niches=3,
        verbose=True,
    )
    vmaps = pipeline.run(destabilizations)

    # ==================================================================
    # FULL REPORT
    # ==================================================================
    print("\n" + pipeline.report())

    # Export DataFrame
    df = pipeline.to_dataframe()
    print(f"\n--- Per-cell DataFrame: {len(df)} rows ---")
    print(df.describe().round(3).to_string())

    print("\n" + "=" * 70)
    print("Full TOPPLE pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
