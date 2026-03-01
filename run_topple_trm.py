# encoding: utf-8
"""
TOPPLE on GSE173706: CD8+ TRM in Psoriatic Lesional Skin
==========================================================

Correct configuration based on diagnostic:
  - Target: CD8+Trm (380 cells)
  - Condition: status2 -> Pso_LS vs HC (lesional vs healthy)
  - Also run on CD8+Tex (503) and CD8+Tem (596) for comparison
  - Stromal: Smooth muscle, CXCL12+/CCN5+/COL23A1+/COL11A1+/SFRP1+ Fibroblasts
  - 12 TFs available

Run: python run_topple_trm.py
"""

import os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H5AD = os.path.join(os.path.expanduser("~"), "Downloads", "GSE173706_cellxgene_data.h5ad")

TFS = ["RUNX3", "TBX21", "EOMES", "NR4A1", "IRF4", "BATF",
       "PRDM1", "TOX", "BHLHE40", "ZNF683", "IKZF2", "ID2"]

# =====================================================================
print("=" * 65)
print("TOPPLE: CD8+ TRM in Psoriatic Lesional Skin")
print("=" * 65)

import anndata
import scipy.sparse as sp
from scipy.stats import zscore

print("\nLoading data...")
adata = anndata.read_h5ad(H5AD)
print("  %d cells x %d genes" % (adata.n_obs, adata.n_vars))

# Extract TF expression
available = [g for g in TFS if g in adata.var_names]
print("  TFs (%d): %s" % (len(available), ", ".join(available)))

X_tf = adata[:, available].X
if sp.issparse(X_tf):
    X_tf = X_tf.toarray()
X_tf = np.array(X_tf, dtype=np.float64)
if X_tf.max() > 50:
    X_tf = np.log1p(X_tf)
X_tf = zscore(X_tf, axis=0, nan_policy="omit")
X_tf = np.nan_to_num(X_tf, 0.0)

cell_types = adata.obs["fine_celltype"].values.astype(str)
status2 = adata.obs["status2"].values.astype(str)

# State: Pso_LS = 1 (lesional), HC = 0 (healthy)
# Exclude Pso_NL (non-lesional) for cleaner contrast
keep_mask = np.isin(status2, ["Pso_LS", "HC"])
state_labels = np.zeros(len(adata), dtype=int)
state_labels[status2 == "Pso_LS"] = 1

del adata  # Free memory

# =====================================================================
# Analyze multiple CD8 populations
# =====================================================================

targets = [
    ("CD8+Trm", "CD8+Trm"),     # Primary target: tissue-resident memory
    ("CD8+Tex", "CD8+Tex"),     # Exhausted CD8
    ("CD8+Tem", "CD8+Tem"),     # Effector memory CD8
]

output_dir = "topple_results_GSE173706"
os.makedirs(output_dir, exist_ok=True)

from topple import StabilityDecomposer
from topple.layer2.perturbation_engine import MockPerturbationEngine
from topple.layer2.bridge import PerturbationBridge
from topple.data.loader import TOPPLEData

all_reports = []

for label, target_name in targets:
    print("\n" + "=" * 65)
    print("  Analyzing: %s" % label)
    print("=" * 65)

    target_mask = (cell_types == target_name) & keep_mask
    target_idx = np.where(target_mask)[0]
    n_target = len(target_idx)

    if n_target < 15:
        print("  Skipping: only %d cells (need >= 15)" % n_target)
        continue

    X_t = X_tf[target_idx]
    y_t = state_labels[target_idx]
    n_pso = (y_t == 1).sum()
    n_hc = (y_t == 0).sum()

    print("  Cells: %d (Pso_LS: %d, HC: %d)" % (n_target, n_pso, n_hc))

    if n_pso < 5 or n_hc < 5:
        print("  Skipping: need >= 5 in each condition")
        continue

    # --- Layer 1 ---
    print("\n  --- LAYER 1: Stability Decomposition ---")
    max_order = 3 if len(available) <= 12 else 2
    sd = StabilityDecomposer(max_order=max_order, method="exact", verbose=True)
    sd.fit(X_t, y_t, feature_names=available)

    print("\n  Top interactions:")
    top_ints = sd.top_interactions(n=15)
    for feat, val in top_ints:
        print("    %s: %+.4f" % (" x ".join(feat), val))

    ve = sd.variance_explained()
    print("\n  Variance by order:")
    for k, v in sorted(ve.items()):
        print("    k=%d: %.1f%%" % (k, v * 100))

    # --- Layer 2 ---
    print("\n  --- LAYER 2: Perturbation Bridge ---")

    # DE scores on this subset
    tmp = TOPPLEData(
        aucell=X_t, regulon_names=available,
        cell_types=np.array(["target"] * n_target),
        state_labels=y_t,
    )
    de_scores = {}
    for i, name in enumerate(available):
        v1 = X_t[y_t == 1, i]
        v0 = X_t[y_t == 0, i]
        if len(v1) >= 2 and len(v0) >= 2:
            from scipy.stats import ttest_ind
            t, _ = ttest_ind(v1, v0)
            de_scores[name] = abs(float(t))
        else:
            de_scores[name] = 0.0

    print("  DE scores (|t-stat|, Pso_LS vs HC):")
    for tf, score in sorted(de_scores.items(), key=lambda x: -x[1]):
        print("    %s: %.2f" % (tf, score))

    engine = MockPerturbationEngine(X_t, y_t, available, effect_size=1.5)
    bridge = PerturbationBridge(
        engine=engine, X=X_t, y=y_t,
        feature_names=available,
        interactions=sd.interactions_,
        de_scores=de_scores,
    )
    l2_results = bridge.run()
    report = bridge.report()
    print(report)

    # Save per-target results
    prefix = os.path.join(output_dir, label)

    with open(prefix + "_L1_interactions.tsv", "w") as f:
        f.write("features\torder\tvalue\n")
        for feat, val in sd.top_interactions(n=50):
            f.write("%s\t%d\t%.6f\n" % (" x ".join(feat), len(feat), val))

    with open(prefix + "_L1_variance.tsv", "w") as f:
        f.write("order\tfraction\n")
        for k, v in sorted(ve.items()):
            f.write("%d\t%.6f\n" % (k, v))

    with open(prefix + "_L2_report.txt", "w") as f:
        f.write(report)

    with open(prefix + "_DE_scores.tsv", "w") as f:
        f.write("TF\tDE_score\n")
        for tf, score in sorted(de_scores.items(), key=lambda x: -x[1]):
            f.write("%s\t%.4f\n" % (tf, score))

    summary = (
        "%s: %d cells (Pso_LS=%d, HC=%d)\n"
        "  AUC=%.3f, Top: %s\n"
        "  Variance: %s\n"
    ) % (
        label, n_target, n_pso, n_hc,
        sd.metadata_.get("cv_auc", 0),
        ", ".join(["%s(%+.3f)" % (" x ".join(f), v) for f, v in top_ints[:3]]),
        ", ".join(["k%d=%.0f%%" % (k, v*100) for k, v in sorted(ve.items())]),
    )
    all_reports.append(summary)

# Combined report
with open(os.path.join(output_dir, "TOPPLE_combined_report.txt"), "w") as f:
    f.write("TOPPLE: GSE173706 Psoriasis - CD8+ T Cell Populations\n")
    f.write("=" * 55 + "\n\n")
    f.write("Dataset: GSE173706_cellxgene_data.h5ad (83,352 cells)\n")
    f.write("Condition: Pso_LS (lesional) vs HC (healthy)\n")
    f.write("TFs: %s\n\n" % ", ".join(available))
    for r in all_reports:
        f.write(r + "\n")

print("\n" + "=" * 65)
print("All results saved to %s/" % output_dir)
print("\nFiles per cell type: *_L1_interactions.tsv, *_L1_variance.tsv,")
print("  *_L2_report.txt, *_DE_scores.tsv")
print("Combined: TOPPLE_combined_report.txt")
print("=" * 65)
