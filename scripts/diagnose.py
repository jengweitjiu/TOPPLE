# encoding: utf-8
import os, sys, glob

print("TOPPLE Diagnostic")
print("=" * 50)

# Find h5ad files
home = os.path.expanduser("~")
search = [
    os.path.join(home, "Desktop"),
    os.path.join(home, "Documents"),
    os.path.join(home, "Downloads"),
    home,
    "D:" + os.sep,
    "E:" + os.sep,
]

found = []
for base in search:
    if not os.path.isdir(base):
        continue
    for root, dirs, files in os.walk(base):
        depth = root.replace(base, "").count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        skip = {"AppData", ".git", "node_modules", "__pycache__", "miniconda3", "anaconda3"}
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for f in files:
            if f.endswith(".h5ad"):
                p = os.path.join(root, f)
                mb = os.path.getsize(p) / 1048576
                found.append(p)
                print("  FOUND: %s (%.1f MB)" % (p, mb))

if not found:
    print("  No .h5ad files found.")
    print("  Please type the full path to your h5ad file:")
    sys.exit(0)

# Inspect
try:
    import anndata
    print("\nInspecting files...")
    for p in found[:5]:
        print("\n--- %s ---" % os.path.basename(p))
        try:
            ad = anndata.read_h5ad(p)
            print("  Cells: %d, Genes: %d" % (ad.n_obs, ad.n_vars))
            print("  obs cols: %s" % list(ad.obs.columns[:20]))
            print("  obsm keys: %s" % list(ad.obsm.keys()))
            for k in ad.obsm:
                print("    %s: shape %s" % (k, str(ad.obsm[k].shape)))
                if hasattr(ad.obsm[k], "columns"):
                    cols = list(ad.obsm[k].columns[:8])
                    print("    columns: %s ..." % cols)
            reg = [c for c in ad.obs.columns if "(+)" in c]
            if reg:
                print("  Regulon obs cols: %d (e.g. %s)" % (len(reg), reg[:5]))
            for ck in ["cell_type", "celltype", "CellType", "cluster", "seurat_clusters", "cell_type_fine"]:
                if ck in ad.obs.columns:
                    vc = ad.obs[ck].value_counts()
                    print("  Cell types [%s]: %s" % (ck, dict(vc.head(10))))
            for ck in ["condition", "disease", "group", "status", "sample"]:
                if ck in ad.obs.columns:
                    vc = ad.obs[ck].value_counts()
                    print("  Condition [%s]: %s" % (ck, dict(vc.head(5))))
            del ad
        except Exception as e:
            print("  Error: %s" % e)
except ImportError:
    print("\nanndata not installed. Run: pip install anndata")

print("\nDone. Share this output screenshot.")
