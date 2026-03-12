# encoding: utf-8
"""
TOPPLE on GSE173706: CD8+ TRM — FIXED
=======================================

Fixes:
1. Unicode error: use utf-8 encoding for file writes
2. Class imbalance: CD8+Trm has 320 Pso_LS vs 11 HC
   -> Also compare Pso_LS vs Pso_NL (lesional vs non-lesional)
   -> Use balanced subsampling for L2

Run: python run_topple_trm_v2.py
"""

import os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H5AD = os.path.join(os.path.expanduser("~"), "Downloads", "GSE173706_cellxgene_data.h5ad")

TFS = ["RUNX3", "TBX21", "EOMES", "NR4A1", "IRF4", "BATF",
       "PRDM1", "TOX", "BHLHE40", "ZNF683", "IKZF2", "ID2"]

print("=" * 65)
print("TOPPLE: CD8+ TRM in Psoriatic Skin (v2 - fixed)")
print("=" * 65)

import anndata
import scipy.sparse as sp
from scipy.stats import zscore, ttest_ind

print("\nLoading data...")
adata = anndata.read_h5ad(H5AD)
print("  %d cells x %d genes" % (adata.n_obs, adata.n_vars))

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
del adata

# =====================================================================
# Analysis configurations
# =====================================================================

configs = [
    # (label, target, state1_name, state1_value, state0_name, state0_value)
    ("CD8+Trm_LSvsHC", "CD8+Trm", "Pso_LS", "Pso_LS", "HC", "HC"),
    ("CD8+Trm_LSvsNL", "CD8+Trm", "Pso_LS", "Pso_LS", "Pso_NL", "Pso_NL"),
    ("CD8+Tex_LSvsHC", "CD8+Tex", "Pso_LS", "Pso_LS", "HC", "HC"),
    ("CD8+Tex_LSvsNL", "CD8+Tex", "Pso_LS", "Pso_LS", "Pso_NL", "Pso_NL"),
    ("CD8+Tem_LSvsHC", "CD8+Tem", "Pso_LS", "Pso_LS", "HC", "HC"),
    ("CD8+Tem_LSvsNL", "CD8+Tem", "Pso_LS", "Pso_LS", "Pso_NL", "Pso_NL"),
]

output_dir = "topple_results_v2"
os.makedirs(output_dir, exist_ok=True)

from topple import StabilityDecomposer
from topple.layer2.perturbation_engine import MockPerturbationEngine
from topple.layer2.bridge import PerturbationBridge

all_summaries = []

for label, target_name, s1_name, s1_val, s0_name, s0_val in configs:
    print("\n" + "=" * 65)
    print("  %s" % label)
    print("=" * 65)

    # Select cells: target type AND (state1 OR state0)
    mask = (cell_types == target_name) & np.isin(status2, [s1_val, s0_val])
    idx = np.where(mask)[0]
    n = len(idx)

    X_t = X_tf[idx]
    y_t = (status2[idx] == s1_val).astype(int)
    n1 = (y_t == 1).sum()
    n0 = (y_t == 0).sum()

    print("  %d cells (%s: %d, %s: %d)" % (n, s1_name, n1, s0_name, n0))

    if n < 15 or n1 < 5 or n0 < 5:
        print("  SKIP: insufficient cells")
        continue

    # --- Layer 1 ---
    print("\n  --- LAYER 1 ---")
    max_order = 3 if len(available) <= 12 else 2
    sd = StabilityDecomposer(max_order=max_order, method="exact", verbose=True)
    sd.fit(X_t, y_t, feature_names=available)

    top_ints = sd.top_interactions(n=15)
    print("\n  Top interactions:")
    for feat, val in top_ints:
        print("    %s: %+.4f" % (" x ".join(feat), val))

    ve = sd.variance_explained()
    print("\n  Variance: %s" % ", ".join(
        ["k%d=%.1f%%" % (k, v*100) for k, v in sorted(ve.items())]
    ))

    cv_auc = getattr(sd, "gate_score_", 0)

    # --- Layer 2 ---
    print("\n  --- LAYER 2 ---")

    # DE scores
    de_scores = {}
    for i, name in enumerate(available):
        v1 = X_t[y_t == 1, i]
        v0 = X_t[y_t == 0, i]
        if len(v1) >= 2 and len(v0) >= 2:
            t, _ = ttest_ind(v1, v0)
            de_scores[name] = abs(float(t))
        else:
            de_scores[name] = 0.0

    print("  DE (%s vs %s):" % (s1_name, s0_name))
    for tf, score in sorted(de_scores.items(), key=lambda x: -x[1])[:8]:
        print("    %s: %.2f" % (tf, score))

    # For imbalanced data, balance by subsampling majority class
    if n1 > 3 * n0:
        print("  Balancing: subsampling %s from %d to %d" % (s1_name, n1, min(n1, 3*n0)))
        rng = np.random.RandomState(42)
        idx1 = np.where(y_t == 1)[0]
        idx0 = np.where(y_t == 0)[0]
        subsample = rng.choice(idx1, size=min(len(idx1), 3*len(idx0)), replace=False)
        bal_idx = np.concatenate([subsample, idx0])
        X_bal = X_t[bal_idx]
        y_bal = y_t[bal_idx]
    elif n0 > 3 * n1:
        print("  Balancing: subsampling %s from %d to %d" % (s0_name, n0, min(n0, 3*n1)))
        rng = np.random.RandomState(42)
        idx1 = np.where(y_t == 1)[0]
        idx0 = np.where(y_t == 0)[0]
        subsample = rng.choice(idx0, size=min(len(idx0), 3*len(idx1)), replace=False)
        bal_idx = np.concatenate([idx1, subsample])
        X_bal = X_t[bal_idx]
        y_bal = y_t[bal_idx]
    else:
        X_bal = X_t
        y_bal = y_t

    engine = MockPerturbationEngine(X_bal, y_bal, available, effect_size=1.5)
    bridge = PerturbationBridge(
        engine=engine, X=X_bal, y=y_bal,
        feature_names=available,
        interactions=sd.interactions_,
        de_scores=de_scores,
    )
    l2_results = bridge.run()
    report = bridge.report()
    print(report)

    # --- Save (UTF-8!) ---
    prefix = os.path.join(output_dir, label)

    with open(prefix + "_L1.tsv", "w", encoding="utf-8") as f:
        f.write("features\torder\tvalue\n")
        for feat, val in sd.top_interactions(n=50):
            f.write("%s\t%d\t%.6f\n" % (" x ".join(feat), len(feat), val))

    with open(prefix + "_variance.tsv", "w", encoding="utf-8") as f:
        f.write("order\tfraction\n")
        for k, v in sorted(ve.items()):
            f.write("%d\t%.6f\n" % (k, v))

    with open(prefix + "_L2.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(prefix + "_DE.tsv", "w", encoding="utf-8") as f:
        f.write("TF\tDE_score\n")
        for tf, score in sorted(de_scores.items(), key=lambda x: -x[1]):
            f.write("%s\t%.4f\n" % (tf, score))

    summary = "%s: n=%d (%s=%d, %s=%d), AUC=%.3f\n" % (
        label, n, s1_name, n1, s0_name, n0, cv_auc
    )
    summary += "  Top: %s\n" % ", ".join(
        ["%s(%+.3f)" % ("x".join(f), v) for f, v in top_ints[:3]]
    )
    summary += "  Var: %s\n" % ", ".join(
        ["k%d=%.0f%%" % (k, v*100) for k, v in sorted(ve.items())]
    )
    summary += "  L2 viable: %d\n" % len(l2_results)
    all_summaries.append(summary)

# Combined report
with open(os.path.join(output_dir, "TOPPLE_combined.txt"), "w", encoding="utf-8") as f:
    f.write("TOPPLE: GSE173706 CD8+ T Cell Populations\n")
    f.write("=" * 55 + "\n\n")
    f.write("Dataset: GSE173706 (83,352 cells, Harmony)\n")
    f.write("TFs: %s\n" % ", ".join(available))
    f.write("Note: TF expression proxy (no pySCENIC AUCell)\n\n")
    for s in all_summaries:
        f.write(s + "\n")

print("\n" + "=" * 65)
print("Results saved to %s/" % output_dir)
print("=" * 65)
