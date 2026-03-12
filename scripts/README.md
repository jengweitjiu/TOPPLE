# Scripts

Personal workflow scripts for running TOPPLE on specific datasets.
These are **not** part of the TOPPLE package and contain hardcoded paths
specific to the original development environment.

For reusable examples, see the `examples/` directory.

## Contents

| Script | Purpose |
|---|---|
| `diagnose.py` | Scan local filesystem for .h5ad files and inspect their structure |
| `setup_and_diagnose.py` | Install TOPPLE, find data files, and inspect them |
| `run_topple_gse173706.py` | Run TOPPLE L1+L2 on GSE173706 psoriasis skin data |
| `run_topple_trm_v2.py` | Run TOPPLE on CD8+ TRM subsets with class-imbalance handling |
| `explore_and_run.py` | Explore GSE173706 cell types and auto-run TOPPLE |
