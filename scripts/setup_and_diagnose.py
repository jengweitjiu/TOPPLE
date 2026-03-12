"""
TOPPLE Setup & Diagnostic Script
==================================

Run in Anaconda Prompt:
    cd C:\Users\user\Downloads\TOPPLE_github_ready
    python setup_and_diagnose.py

This script:
1. Fixes the pip install issue
2. Finds your .h5ad files
3. Shows data structure for TOPPLE configuration
"""

import os
import sys
import glob

print("=" * 60)
print("TOPPLE Setup & Diagnostic")
print("=" * 60)

# ---- Step 1: Fix pyproject.toml build backend ----
print("\n[Step 1] Fixing pyproject.toml...")
pyproject_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
if os.path.exists(pyproject_path):
    with open(pyproject_path, "r") as f:
        content = f.read()
    # Fix the build backend
    old = 'build-backend = "setuptools.backends._legacy:_Backend"'
    new = 'build-backend = "setuptools.build_meta"'
    if old in content:
        content = content.replace(old, new)
        with open(pyproject_path, "w") as f:
            f.write(content)
        print("  Fixed build-backend in pyproject.toml")
    elif new in content:
        print("  Already fixed")
    else:
        print("  WARNING: Could not find build-backend line")

# ---- Step 2: Add TOPPLE to sys.path ----
topple_dir = os.path.dirname(os.path.abspath(__file__))
if topple_dir not in sys.path:
    sys.path.insert(0, topple_dir)
print(f"\n[Step 2] TOPPLE path: {topple_dir}")

# Test import
try:
    import topple
    print("  import topple: OK")
except Exception as e:
    print(f"  import topple: FAILED - {e}")

# ---- Step 3: Find .h5ad files ----
print("\n[Step 3] Searching for .h5ad files...")
search_paths = [
    os.path.expanduser("~"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Downloads"),
    "D:\\",
    "E:\\",
]

h5ad_files = []
for base in search_paths:
    if not os.path.exists(base):
        continue
    for root, dirs, files in os.walk(base):
        # Skip deep directories
        depth = root.replace(base, "").count(os.sep)
        if depth > 4:
            dirs.clear()
            continue
        # Skip system/hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in (
            "AppData", "node_modules", ".git", "__pycache__",
            "miniconda3", "anaconda3", "Program Files",
        )]
        for f in files:
            if f.endswith(".h5ad"):
                full_path = os.path.join(root, f)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                h5ad_files.append((full_path, size_mb))
                print(f"  Found: {full_path} ({size_mb:.1f} MB)")

if not h5ad_files:
    print("  No .h5ad files found in common locations.")
    print("  Please tell me the full path to your psoriasis .h5ad file.")
else:
    print(f"\n  Total: {len(h5ad_files)} .h5ad files found")

# ---- Step 4: Inspect the first/largest h5ad ----
if h5ad_files:
    try:
        import anndata
        print("\n[Step 4] Inspecting .h5ad files...")
        for path, size in h5ad_files[:5]:  # Check top 5
            print(f"\n  --- {os.path.basename(path)} ({size:.1f} MB) ---")
            try:
                adata = anndata.read_h5ad(path)
                print(f"  Shape: {adata.n_obs} cells x {adata.n_vars} genes")
                print(f"  obs columns: {list(adata.obs.columns[:15])}")
                if len(adata.obs.columns) > 15:
                    print(f"    ... +{len(adata.obs.columns)-15} more")
                print(f"  obsm keys: {list(adata.obsm.keys())}")
                if "X_aucell" in adata.obsm:
                    print(f"  AUCell: {adata.obsm['X_aucell'].shape}")
                    if hasattr(adata.obsm["X_aucell"], "columns"):
                        cols = list(adata.obsm["X_aucell"].columns[:10])
                        print(f"  Regulons: {cols}...")
                # Check for regulon columns in obs
                reg_cols = [c for c in adata.obs.columns if "(+)" in c]
                if reg_cols:
                    print(f"  Regulon obs columns: {len(reg_cols)} (e.g., {reg_cols[:5]})")
                # Cell types
                for key in ["cell_type", "celltype", "cluster", "seurat_clusters", "CellType"]:
                    if key in adata.obs.columns:
                        vals = adata.obs[key].value_counts()
                        print(f"  Cell types ({key}): {dict(vals.head(10))}")
                        break
                # Spatial
                for key in ["spatial", "X_spatial"]:
                    if key in adata.obsm:
                        print(f"  Spatial ({key}): {adata.obsm[key].shape}")
                        break
                del adata
            except Exception as e:
                print(f"  Error reading: {e}")
    except ImportError:
        print("\n[Step 4] anndata not installed. Installing...")
        os.system(f"{sys.executable} -m pip install anndata")
        print("  Please re-run this script.")

# ---- Step 5: Try pip install ----
print("\n[Step 5] Installing TOPPLE...")
print(f"  Run: pip install -e {topple_dir}")
os.system(f"{sys.executable} -m pip install -e {topple_dir}")

print("\n" + "=" * 60)
print("Next: copy the file path of your .h5ad above and share it with me.")
print("I will configure run_topple.py with the correct keys.")
print("=" * 60)
