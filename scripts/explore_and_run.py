# encoding: utf-8
"""
Step 1: Show actual cell types in GSE173706
Step 2: Run TOPPLE with the correct names

Run: python explore_and_run.py
"""

import os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H5AD = os.path.join(os.path.expanduser("~"), "Downloads", "GSE173706_cellxgene_data.h5ad")

print("=" * 65)
print("Loading GSE173706...")
print("=" * 65)

import anndata
adata = anndata.read_h5ad(H5AD)
print("%d cells x %d genes" % (adata.n_obs, adata.n_vars))

# Show ALL cell type columns
for key in ["fine_celltype", "major_celltype", "celltype_classifiy"]:
    if key in adata.obs.columns:
        vc = adata.obs[key].value_counts()
        print("\n--- %s (%d unique) ---" % (key, len(vc)))
        for name, count in vc.items():
            print("  %s: %d" % (name, count))

# Show condition
for key in ["status", "status2", "status3"]:
    if key in adata.obs.columns:
        print("\n--- %s ---" % key)
        print("  %s" % dict(adata.obs[key].value_counts()))

# Check TF availability
TFS = ["RUNX3", "TBX21", "EOMES", "NR4A1", "IRF4", "BATF", "PRDM1",
       "TOX", "BHLHE40", "ZNF683", "IKZF2", "ID2"]
available = [g for g in TFS if g in adata.var_names]
missing = [g for g in TFS if g not in adata.var_names]
print("\n--- TF availability ---")
print("  Available (%d): %s" % (len(available), available))
if missing:
    print("  Missing: %s" % missing)

print("\n" + "=" * 65)
print("Copy the T cell type name from above and edit TARGET below,")
print("then re-run this script.")
print("=" * 65)

# =====================================================================
# EDIT THIS: pick the T cell type from the list above
# =====================================================================
# Common options in skin datasets:
#   "T cell", "CD8+ T cell", "Tc", "CTL", etc.
# Set to None to auto-detect (picks largest T-cell-like type)
TARGET = None

# Auto-detect T cell type
if TARGET is None:
    for key in ["fine_celltype", "major_celltype", "celltype_classifiy"]:
        if key not in adata.obs.columns:
            continue
        vc = adata.obs[key].value_counts()
        # Look for T cell types (case-insensitive)
        t_types = [(name, count) for name, count in vc.items()
                   if any(x in str(name).lower() for x in ["t cell", "cd8", "cytotoxic", "trm", " tc", "ctl"])]
        if t_types:
            # Pick the largest
            TARGET = max(t_types, key=lambda x: x[1])[0]
            CELLTYPE_KEY = key
            print("\nAuto-detected target: '%s' from %s (%d cells)" % (TARGET, key, dict(vc)[TARGET]))
            # Also show all T types found
            print("All T-cell types found:")
            for name, count in sorted(t_types, key=lambda x: -x[1]):
                print("  %s: %d" % (name, count))
            break

if TARGET is None:
    print("\nCould not auto-detect T cell type.")
    print("Please set TARGET manually in the script and re-run.")
    sys.exit(0)

# Identify stromal types
STROMAL = []
for key in ["fine_celltype", "major_celltype"]:
    if key not in adata.obs.columns:
        continue
    vc = adata.obs[key].value_counts()
    for name in vc.index:
        nl = str(name).lower()
        if any(x in nl for x in ["fibroblast", "pericyte", "endothelial", "stromal", "smooth muscle"]):
            STROMAL.append(str(name))
    if STROMAL:
        break

print("Stromal types: %s" % STROMAL)

# =====================================================================
# RUN TOPPLE
# =====================================================================

import scipy.sparse as sp
from scipy.stats import zscore

cell_types = adata.obs[CELLTYPE_KEY].values.astype(str)
state_labels = (adata.obs["status"].values == "Pso").astype(int)

# Extract TF expression
X_tf = adata[:, available].X
if sp.issparse(X_tf):
    X_tf = X_tf.toarray()
X_tf = np.array(X_tf, dtype=np.float64)
if X_tf.max() > 50:
    X_tf = np.log1p(X_tf)
X_tf = zscore(X_tf, axis=0, nan_policy="omit")
X_tf = np.nan_to_num(X_tf, 0.0)

# Target subset
target_mask = cell_types == TARGET
target_idx = np.where(target_mask)[0]
n_target = len(target_idx)
X_target = X_tf[target_idx]
y_target = state_labels[target_idx]
n_pso = (y_target == 1).sum()
n_hc = (y_target == 0).sum()

print("\n" + "=" * 65)
print("TOPPLE on %s (%d cells: %d Pso, %d HC)" % (TARGET, n_target, n_pso, n_hc))
print("TFs: %s" % ", ".join(available))
print("=" * 65)

if n_target < 10 or n_pso < 5 or n_hc < 5:
    print("ERROR: Not enough cells in both conditions.")
    print("Need >= 5 Pso and >= 5 HC for meaningful analysis.")
    sys.exit(1)

# L1
print("\n--- LAYER 1: Stability Decomposition ---")
from topple import StabilityDecomposer

n_tf = len(available)
max_order = 3 if n_tf <= 12 else 2
sd = StabilityDecomposer(max_order=max_order, method="exact", verbose=True)
sd.fit(X_target, y_target, feature_names=available)

print("\nTop interactions:")
for feat, val in sd.top_interactions(n=15):
    print("  %s: %+.4f" % (" x ".join(feat), val))

print("\nVariance by order:")
ve = sd.variance_explained()
for k, v in sorted(ve.items()):
    print("  k=%d: %.1f%%" % (k, v * 100))

# L2
print("\n--- LAYER 2: Perturbation Bridge ---")
from topple.layer2.perturbation_engine import MockPerturbationEngine
from topple.layer2.bridge import PerturbationBridge
from topple.data.loader import TOPPLEData

tmp_data = TOPPLEData(
    aucell=X_tf, regulon_names=available,
    cell_types=cell_types, state_labels=state_labels,
)
de_scores = tmp_data.get_de_scores()
print("DE scores (|t-stat|):")
for tf, score in sorted(de_scores.items(), key=lambda x: -x[1]):
    print("  %s: %.2f" % (tf, score))

engine = MockPerturbationEngine(X_target, y_target, available, effect_size=1.5)
bridge = PerturbationBridge(
    engine=engine, X=X_target, y=y_target,
    feature_names=available,
    interactions=sd.interactions_,
    de_scores=de_scores,
)
l2_results = bridge.run()
print(bridge.report())

# Save
output_dir = "topple_results_GSE173706"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "L1_interactions.tsv"), "w") as f:
    f.write("features\torder\tvalue\n")
    for feat, val in sd.top_interactions(n=50):
        f.write("%s\t%d\t%.6f\n" % (" x ".join(feat), len(feat), val))

with open(os.path.join(output_dir, "L1_variance.tsv"), "w") as f:
    f.write("order\tfraction\n")
    for k, v in sorted(ve.items()):
        f.write("%d\t%.6f\n" % (k, v))

with open(os.path.join(output_dir, "L2_report.txt"), "w") as f:
    f.write(bridge.report())

with open(os.path.join(output_dir, "DE_scores.tsv"), "w") as f:
    f.write("TF\tDE_score\n")
    for tf, score in sorted(de_scores.items(), key=lambda x: -x[1]):
        f.write("%s\t%.4f\n" % (tf, score))

with open(os.path.join(output_dir, "TOPPLE_report.txt"), "w") as f:
    f.write("TOPPLE: GSE173706 Psoriasis (%s)\n" % TARGET)
    f.write("=" * 50 + "\n\n")
    f.write("Target: %s (%d cells: %d Pso, %d HC)\n" % (TARGET, n_target, n_pso, n_hc))
    f.write("TFs: %s\n" % ", ".join(available))
    f.write("Method: exact, max_order=%d\n\n" % max_order)
    f.write(sd.report() + "\n\n")
    f.write(bridge.report())

print("\n" + "=" * 65)
print("Results saved to %s/" % output_dir)
print("=" * 65)
