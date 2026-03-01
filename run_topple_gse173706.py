# encoding: utf-8
"""
TOPPLE on GSE173706 Psoriasis Skin Data
=========================================

Your data: 83,352 cells, Harmony-integrated, Pso vs HC
No AUCell yet -> uses TF gene expression as regulon proxy

Run: python run_topple_gse173706.py
"""

import os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Add TOPPLE to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# CONFIGURATION
# =====================================================================

H5AD_PATH = os.path.join(
    os.path.expanduser("~"), "Downloads",
    "GSE173706_cellxgene_data.h5ad"
)

# TRM-relevant transcription factors
TF_GENES = [
    "RUNX3", "TBX21", "EOMES", "NR4A1",
    "IRF4", "BATF", "PRDM1", "TOX",
    "BHLHE40", "ZNF683",  # HOBIT
    "IKZF2",  # Helios
    "ID2",
]

# Data keys (from diagnostic)
CELLTYPE_KEY = "fine_celltype"       # or "major_celltype"
CONDITION_KEY = "status"             # Pso vs HC
PATHOLOGICAL = "Pso"
TARGET_TYPE = "cytotoxic T cell"     # closest to CD8_TRM in this dataset
STROMAL_TYPES = [
    "pericyte",
    "endothelial cell",
    "fibroblast of papillary layer of dermis",
    "skin fibroblast",
    "fibroblast",
]

# =====================================================================
# LOAD
# =====================================================================

print("=" * 65)
print("TOPPLE: GSE173706 Psoriasis Skin Analysis")
print("=" * 65)

import anndata
import pandas as pd

print("\n[1] Loading %s ..." % os.path.basename(H5AD_PATH))
adata = anndata.read_h5ad(H5AD_PATH)
print("    %d cells x %d genes" % (adata.n_obs, adata.n_vars))

# Show cell types
print("\n[2] Cell types (%s):" % CELLTYPE_KEY)
ct = adata.obs[CELLTYPE_KEY].value_counts()
for name, count in ct.head(15).items():
    marker = " <-- TARGET" if name == TARGET_TYPE else ""
    marker2 = " <-- STROMAL" if name in STROMAL_TYPES else ""
    print("    %s: %d%s%s" % (name, count, marker, marker2))

print("\n[3] Condition (%s):" % CONDITION_KEY)
print("    %s" % dict(adata.obs[CONDITION_KEY].value_counts()))

# =====================================================================
# EXTRACT TF EXPRESSION AS REGULON PROXY
# =====================================================================

print("\n[4] Extracting TF expression as regulon activity proxy...")
available_tfs = [g for g in TF_GENES if g in adata.var_names]
missing_tfs = [g for g in TF_GENES if g not in adata.var_names]
if missing_tfs:
    print("    Missing TFs (skipped): %s" % missing_tfs)
print("    Using %d TFs: %s" % (len(available_tfs), available_tfs))

# Extract and normalize TF expression
import scipy.sparse as sp
X_full = adata[:, available_tfs].X
if sp.issparse(X_full):
    X_full = X_full.toarray()
tf_expression = np.array(X_full, dtype=np.float64)

# Log-normalize if not already
if tf_expression.max() > 50:
    print("    Log-normalizing TF expression...")
    tf_expression = np.log1p(tf_expression)

# Z-score per TF
from scipy.stats import zscore
tf_expression = zscore(tf_expression, axis=0, nan_policy="omit")
tf_expression = np.nan_to_num(tf_expression, 0.0)
print("    TF matrix: %s" % str(tf_expression.shape))

# =====================================================================
# BUILD TOPPLEData
# =====================================================================

print("\n[5] Building TOPPLEData...")
from topple.data.loader import TOPPLEData

# State labels: Pso = 1, HC = 0
state_labels = (adata.obs[CONDITION_KEY].values == PATHOLOGICAL).astype(int)
cell_types = adata.obs[CELLTYPE_KEY].values.astype(str)

data = TOPPLEData(
    aucell=tf_expression,
    regulon_names=available_tfs,
    cell_types=cell_types,
    coordinates=None,  # No spatial in this scRNA-seq dataset
    state_labels=state_labels,
    metadata={"source": os.path.basename(H5AD_PATH), "note": "TF expression proxy"},
)

print(data.summary())

# =====================================================================
# RUN TOPPLE (L1 + L2, no L3 since no spatial)
# =====================================================================

print("\n[6] Running TOPPLE L1 -> L2 on %s cells..." % TARGET_TYPE)

target_mask = cell_types == TARGET_TYPE
target_idx = np.where(target_mask)[0]
n_target = len(target_idx)
print("    Target cells: %d" % n_target)
print("    Pathological: %d, Homeostatic: %d" % (
    (state_labels[target_idx] == 1).sum(),
    (state_labels[target_idx] == 0).sum(),
))

if n_target < 20:
    print("\n    WARNING: Very few target cells. Trying major_celltype...")
    # Fallback to major_celltype
    if "major_celltype" in adata.obs.columns:
        cell_types2 = adata.obs["major_celltype"].values.astype(str)
        print("    major_celltype values: %s" % dict(
            pd.Series(cell_types2).value_counts().head(10)
        ))
        # Look for T cell types
        for ct_name in cell_types2:
            if "T" in ct_name or "cytotoxic" in ct_name.lower():
                print("    Found: %s" % ct_name)
                break

X_target = tf_expression[target_idx]
y_target = state_labels[target_idx]

# Check separability
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
if len(np.unique(y_target)) >= 2:
    auc = cross_val_score(
        LogisticRegression(max_iter=1000), X_target, y_target,
        cv=min(5, min((y_target==0).sum(), (y_target==1).sum())),
        scoring="roc_auc",
    ).mean()
    print("    Cross-validated AUC: %.3f" % auc)

# Layer 1: Stability decomposition
print("\n--- LAYER 1: Stability Decomposition ---")
from topple import StabilityDecomposer

n_tfs = len(available_tfs)
if n_tfs <= 12:
    method = "exact"
    max_order = 3
elif n_tfs <= 20:
    method = "exact"
    max_order = 2
else:
    method = "compressed"
    max_order = 2

print("    Method: %s, max_order: %d, features: %d" % (method, max_order, n_tfs))

sd = StabilityDecomposer(max_order=max_order, method=method, verbose=True)
sd.fit(X_target, y_target, feature_names=available_tfs)

print("\n    Top interactions:")
for feat, val in sd.top_interactions(n=15):
    print("      %s: %+.4f" % (" x ".join(feat), val))

print("\n    Variance by order:")
ve = sd.variance_explained()
for k, v in sorted(ve.items()):
    print("      k=%d: %.1f%%" % (k, v * 100))

# Layer 2: Perturbation bridge
print("\n--- LAYER 2: Perturbation Bridge ---")
from topple.layer2.perturbation_engine import MockPerturbationEngine
from topple.layer2.bridge import PerturbationBridge

de_scores = data.get_de_scores()
print("    DE scores:")
for tf, score in sorted(de_scores.items(), key=lambda x: -x[1]):
    print("      %s: %.2f" % (tf, score))

engine = MockPerturbationEngine(X_target, y_target, available_tfs, effect_size=1.5)
bridge = PerturbationBridge(
    engine=engine, X=X_target, y=y_target,
    feature_names=available_tfs,
    interactions=sd.interactions_,
    de_scores=de_scores,
)
l2_results = bridge.run()
print(bridge.report())

# =====================================================================
# SAVE RESULTS
# =====================================================================

output_dir = "topple_results_GSE173706"
os.makedirs(output_dir, exist_ok=True)

# L1
with open(os.path.join(output_dir, "L1_interactions.tsv"), "w") as f:
    f.write("features\torder\tvalue\n")
    for feat, val in sd.top_interactions(n=50):
        f.write("%s\t%d\t%.6f\n" % (" x ".join(feat), len(feat), val))

with open(os.path.join(output_dir, "L1_variance.tsv"), "w") as f:
    f.write("order\tfraction\n")
    for k, v in sorted(ve.items()):
        f.write("%d\t%.6f\n" % (k, v))

# L2
with open(os.path.join(output_dir, "L2_report.txt"), "w") as f:
    f.write(bridge.report())

# DE scores
with open(os.path.join(output_dir, "DE_scores.tsv"), "w") as f:
    f.write("TF\tDE_score\n")
    for tf, score in sorted(de_scores.items(), key=lambda x: -x[1]):
        f.write("%s\t%.4f\n" % (tf, score))

# Full report
with open(os.path.join(output_dir, "TOPPLE_report.txt"), "w") as f:
    f.write("TOPPLE Analysis: GSE173706 Psoriasis\n")
    f.write("=" * 50 + "\n\n")
    f.write("Data: %s\n" % os.path.basename(H5AD_PATH))
    f.write("Target: %s (%d cells)\n" % (TARGET_TYPE, n_target))
    f.write("TFs: %s\n" % ", ".join(available_tfs))
    f.write("Method: %s, max_order=%d\n\n" % (method, max_order))
    f.write("NOTE: Using TF gene expression as regulon proxy.\n")
    f.write("For publication, run pySCENIC first for AUCell scores.\n\n")
    f.write(sd.report() + "\n\n")
    f.write(bridge.report())

print("\n" + "=" * 65)
print("Results saved to %s/" % output_dir)
print("")
print("NOTE: This uses TF expression as regulon proxy.")
print("For the manuscript, run pySCENIC on this dataset to get")
print("proper AUCell scores, then re-run with TOPPLEData.from_anndata().")
print("=" * 65)
