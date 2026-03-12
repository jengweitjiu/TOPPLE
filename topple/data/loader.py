"""
TOPPLE Data Loader
====================

Loads and validates real single-cell data for the full TOPPLE pipeline.

Typical workflow for psoriasis spatial transcriptomics:

    1. Run pySCENIC (GRNBoost2 → cisTarget → AUCell) on scRNA-seq
    2. Integrate with Harmony (if multi-sample)
    3. Export AnnData with AUCell scores + spatial coords
    4. Load into TOPPLEData
    5. Feed into TOPPLE L1→L2→L3

Compatible with:
- Scanpy (AnnData .h5ad)
- Seurat (via SeuratDisk .h5ad export or .h5seurat)
- pySCENIC standalone outputs
- Visium, MERFISH, seqFISH spatial coordinates
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp


# =========================================================================
# pySCENIC file loaders
# =========================================================================

def load_pyscenic_adjacencies(
    path: Union[str, Path],
    min_importance: float = 0.0,
) -> "pd.DataFrame":
    """
    Load GRNBoost2/GENIE3 adjacency table.

    Parameters
    ----------
    path : str or Path
        Path to adjacencies.csv (columns: TF, target, importance).
    min_importance : float, default=0.0
        Filter edges below this importance threshold.

    Returns
    -------
    pd.DataFrame with columns [TF, target, importance].
    """
    import pandas as pd
    df = pd.read_csv(path, sep=None, engine="python")

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("tf", "source", "regulator"):
            col_map[c] = "TF"
        elif cl in ("target", "gene"):
            col_map[c] = "target"
        elif cl in ("importance", "weight", "coef_mean", "score"):
            col_map[c] = "importance"
    df = df.rename(columns=col_map)

    required = {"TF", "target", "importance"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Adjacency file must have columns {required}, got {set(df.columns)}"
        )

    if min_importance > 0:
        df = df[df["importance"] >= min_importance].reset_index(drop=True)

    return df


def load_pyscenic_regulons(
    path: Union[str, Path],
) -> Dict[str, List[str]]:
    """
    Load regulon definitions from pySCENIC output.

    Supports:
    - CSV format: columns [TF, target] or [regulon, gene]
    - JSON format from pySCENIC ctx output
    - .gmt format

    Returns
    -------
    dict: {regulon_name: [target_gene_1, target_gene_2, ...]}
    """
    path = Path(path)

    if path.suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        # pySCENIC JSON format
        regulons = {}
        for entry in data:
            name = entry.get("name", entry.get("TF", str(entry)))
            targets = entry.get("targets", entry.get("genes", []))
            if isinstance(targets, dict):
                targets = list(targets.keys())
            regulons[name] = targets
        return regulons

    elif path.suffix == ".gmt":
        regulons = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    regulons[parts[0]] = parts[2:]
        return regulons

    else:
        # CSV-like
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python")
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("tf", "regulon", "source"):
                col_map[c] = "TF"
            elif cl in ("target", "gene"):
                col_map[c] = "target"
        df = df.rename(columns=col_map)

        regulons = {}
        for tf, group in df.groupby("TF"):
            regulons[tf] = group["target"].tolist()
        return regulons


def load_aucell_matrix(
    path: Union[str, Path],
) -> Tuple[np.ndarray, List[str]]:
    """
    Load AUCell regulon activity matrix from CSV.

    Parameters
    ----------
    path : str or Path
        CSV with cells as rows, regulons as columns.

    Returns
    -------
    matrix : np.ndarray, shape (n_cells, n_regulons)
    regulon_names : list of str
    """
    import pandas as pd
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float64), list(df.columns)


# =========================================================================
# TOPPLEData: unified data container
# =========================================================================

@dataclass
class TOPPLEData:
    """
    Unified container for all data needed by the TOPPLE pipeline.

    Attributes
    ----------
    aucell : np.ndarray, shape (n_cells, n_regulons)
        AUCell regulon activity scores.
    regulon_names : list of str
        Names of regulons (TFs).
    cell_types : np.ndarray of str
        Cell type annotation for each cell.
    coordinates : np.ndarray, shape (n_cells, 2), optional
        Spatial coordinates (for Layer 3).
    expression : np.ndarray, optional
        Raw gene expression (for LR scoring in Layer 3).
    gene_names : list of str, optional
        Gene names matching expression columns.
    state_labels : np.ndarray, optional
        Binary state labels (0=homeostatic, 1=pathological).
    sample_ids : np.ndarray, optional
        Sample/batch IDs (post-Harmony integration).
    adjacencies : pd.DataFrame, optional
        GRNBoost2 adjacency table for topology-guided pruning.
    regulon_definitions : dict, optional
        {TF: [target_genes]} for regulon composition.
    metadata : dict
        Additional metadata.

    Examples
    --------
    >>> data = TOPPLEData.from_anndata("psoriasis_scenic.h5ad",
    ...     aucell_key="X_aucell",
    ...     cell_type_key="cell_type",
    ...     spatial_key="spatial",
    ... )
    >>> data.subset_celltype("CD8_TRM")
    >>> data.define_states(condition_key="disease", pathological="psoriasis")
    """

    aucell: np.ndarray
    regulon_names: List[str]
    cell_types: np.ndarray
    coordinates: Optional[np.ndarray] = None
    expression: Optional[np.ndarray] = None
    gene_names: Optional[List[str]] = None
    state_labels: Optional[np.ndarray] = None
    sample_ids: Optional[np.ndarray] = None
    adjacencies: object = None  # pd.DataFrame
    regulon_definitions: Optional[Dict[str, List[str]]] = None
    metadata: Dict = field(default_factory=dict)

    # Internal
    _original_cell_mask: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Return a human-readable summary of the TOPPLEData object."""
        parts = [
            f"TOPPLEData: {self.n_cells} cells × {self.n_regulons} regulons",
            f"  Cell types: {len(np.unique(self.cell_types))} unique",
        ]
        if self.coordinates is not None:
            parts.append(f"  Spatial: yes ({self.coordinates.shape[1]}D)")
        if self.state_labels is not None:
            n_path = (self.state_labels == 1).sum()
            n_homeo = (self.state_labels == 0).sum()
            parts.append(f"  States: {n_path} pathological, {n_homeo} homeostatic")
        if self.adjacencies is not None:
            parts.append(f"  GRN edges: {len(self.adjacencies)}")
        return "\n".join(parts)

    @property
    def n_cells(self) -> int:
        """Number of cells in the dataset."""
        return self.aucell.shape[0]

    @property
    def n_regulons(self) -> int:
        """Number of regulons (features) in the dataset."""
        return self.aucell.shape[1]

    # -----------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        path_or_adata,
        aucell_key: str = "X_aucell",
        cell_type_key: str = "cell_type",
        spatial_key: str = "spatial",
        condition_key: Optional[str] = None,
        pathological_value: Optional[str] = None,
        sample_key: Optional[str] = None,
        adjacencies_path: Optional[str] = None,
        regulons_path: Optional[str] = None,
        use_raw: bool = False,
    ) -> "TOPPLEData":
        """
        Load from AnnData (.h5ad) with pySCENIC results.

        Supports both Scanpy-native and Seurat-exported .h5ad files
        (via SeuratDisk::SaveH5Seurat + Convert).

        Parameters
        ----------
        path_or_adata : str, Path, or anndata.AnnData
            Path to .h5ad file or AnnData object.
        aucell_key : str, default="X_aucell"
            Key in adata.obsm for AUCell matrix.
            Common alternatives: "aucell", "X_regulon", "scenic_aucell".
            If not in obsm, also checks adata.obs columns matching regulon pattern.
        cell_type_key : str, default="cell_type"
            Key in adata.obs for cell type annotations.
            Common: "cell_type", "celltype", "cluster", "seurat_clusters".
        spatial_key : str, default="spatial"
            Key in adata.obsm for spatial coordinates.
            Common: "spatial", "X_spatial", "X_umap" (fallback).
        condition_key : str, optional
            Key in adata.obs for disease condition.
        pathological_value : str, optional
            Value in condition_key that marks pathological state.
        sample_key : str, optional
            Key in adata.obs for sample/batch ID.
        adjacencies_path : str, optional
            Path to GRNBoost2 adjacencies.csv.
        regulons_path : str, optional
            Path to regulon definitions (csv/json/gmt).
        use_raw : bool, default=False
            Use adata.raw for gene expression.

        Returns
        -------
        TOPPLEData
        """
        import anndata

        # Load AnnData
        if isinstance(path_or_adata, (str, Path)):
            adata = anndata.read_h5ad(path_or_adata)
        else:
            adata = path_or_adata

        # --- AUCell matrix ---
        aucell, regulon_names = cls._extract_aucell(adata, aucell_key)

        # --- Cell types ---
        if cell_type_key in adata.obs.columns:
            cell_types = adata.obs[cell_type_key].values.astype(str)
        else:
            available = list(adata.obs.columns)
            raise KeyError(
                f"Cell type key '{cell_type_key}' not found. "
                f"Available: {available[:20]}"
            )

        # --- Spatial coordinates ---
        coordinates = None
        if spatial_key in adata.obsm:
            coordinates = np.array(adata.obsm[spatial_key])[:, :2]
        elif "X_spatial" in adata.obsm:
            coordinates = np.array(adata.obsm["X_spatial"])[:, :2]

        # --- Gene expression (for LR scoring) ---
        expression, gene_names = None, None
        expr_source = adata.raw if (use_raw and adata.raw is not None) else adata
        if expr_source.X is not None:
            X = expr_source.X
            if sp.issparse(X):
                X = X.toarray()
            expression = np.array(X, dtype=np.float64)
            gene_names = list(expr_source.var_names)

        # --- Sample IDs ---
        sample_ids = None
        if sample_key and sample_key in adata.obs.columns:
            sample_ids = adata.obs[sample_key].values.astype(str)

        # --- State labels ---
        state_labels = None
        if condition_key and pathological_value:
            if condition_key in adata.obs.columns:
                state_labels = (
                    adata.obs[condition_key].values == pathological_value
                ).astype(int)

        # --- GRN adjacencies ---
        adjacencies = None
        if adjacencies_path:
            adjacencies = load_pyscenic_adjacencies(adjacencies_path)

        # --- Regulon definitions ---
        regulon_defs = None
        if regulons_path:
            regulon_defs = load_pyscenic_regulons(regulons_path)

        obj = cls(
            aucell=aucell,
            regulon_names=regulon_names,
            cell_types=cell_types,
            coordinates=coordinates,
            expression=expression,
            gene_names=gene_names,
            state_labels=state_labels,
            sample_ids=sample_ids,
            adjacencies=adjacencies,
            regulon_definitions=regulon_defs,
            metadata={
                "source": str(path_or_adata) if isinstance(path_or_adata, (str, Path)) else "AnnData",
                "n_cells_original": adata.n_obs,
                "n_genes_original": adata.n_vars,
            },
        )
        return obj

    @classmethod
    def from_files(
        cls,
        aucell_path: str,
        cell_types_path: Optional[str] = None,
        coordinates_path: Optional[str] = None,
        adjacencies_path: Optional[str] = None,
        regulons_path: Optional[str] = None,
        state_labels_path: Optional[str] = None,
    ) -> "TOPPLEData":
        """
        Load from individual CSV files (pySCENIC standalone output).

        Parameters
        ----------
        aucell_path : str
            AUCell matrix CSV (cells × regulons).
        cell_types_path : str, optional
            CSV with cell_id and cell_type columns.
        coordinates_path : str, optional
            CSV with cell_id, x, y columns.
        adjacencies_path : str, optional
            GRNBoost2 adjacencies CSV.
        regulons_path : str, optional
            Regulon definitions (csv/json/gmt).
        state_labels_path : str, optional
            CSV with cell_id and label columns.
        """
        import pandas as pd

        aucell_mat, regulon_names = load_aucell_matrix(aucell_path)
        n_cells = aucell_mat.shape[0]

        cell_types = np.array(["unknown"] * n_cells)
        if cell_types_path:
            ct_df = pd.read_csv(cell_types_path, index_col=0)
            col = ct_df.columns[0]
            cell_types = ct_df[col].values.astype(str)

        coordinates = None
        if coordinates_path:
            coord_df = pd.read_csv(coordinates_path, index_col=0)
            coordinates = coord_df.values[:, :2].astype(float)

        state_labels = None
        if state_labels_path:
            sl_df = pd.read_csv(state_labels_path, index_col=0)
            state_labels = sl_df.iloc[:, 0].values.astype(int)

        adjacencies = None
        if adjacencies_path:
            adjacencies = load_pyscenic_adjacencies(adjacencies_path)

        regulon_defs = None
        if regulons_path:
            regulon_defs = load_pyscenic_regulons(regulons_path)

        return cls(
            aucell=aucell_mat,
            regulon_names=regulon_names,
            cell_types=cell_types,
            coordinates=coordinates,
            state_labels=state_labels,
            adjacencies=adjacencies,
            regulon_definitions=regulon_defs,
        )

    @classmethod
    def from_seurat(
        cls,
        h5seurat_path: str,
        assay: str = "SCENIC",
        **kwargs,
    ) -> "TOPPLEData":
        """
        Load from Seurat .h5seurat or .h5ad exported via SeuratDisk.

        R-side workflow:
            library(SeuratDisk)
            SaveH5Seurat(seurat_obj, "data.h5seurat")
            Convert("data.h5seurat", dest="h5ad")

        Then: TOPPLEData.from_seurat("data.h5ad", assay="SCENIC")
        """
        return cls.from_anndata(h5seurat_path, **kwargs)

    # -----------------------------------------------------------------
    # AUCell extraction helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_aucell(adata, key: str) -> Tuple[np.ndarray, List[str]]:
        """Extract AUCell matrix from various storage locations in AnnData."""
        # Check obsm first (standard pySCENIC location)
        if key in adata.obsm:
            mat = np.array(adata.obsm[key])
            # Try to get regulon names
            if hasattr(adata.obsm[key], "columns"):
                names = list(adata.obsm[key].columns)
            elif f"{key}_names" in adata.uns:
                names = list(adata.uns[f"{key}_names"])
            else:
                names = [f"regulon_{i}" for i in range(mat.shape[1])]
            return mat.astype(np.float64), names

        # Check common alternative keys
        alt_keys = ["aucell", "X_regulon", "scenic_aucell", "X_scenic", "X_AUCell"]
        for ak in alt_keys:
            if ak in adata.obsm:
                warnings.warn(f"AUCell key '{key}' not found, using '{ak}'")
                return TOPPLEData._extract_aucell(adata, ak)

        # Check if AUCell is stored as obs columns (some Seurat exports)
        regulon_cols = [c for c in adata.obs.columns if "(+)" in c or "regulon_" in c.lower()]
        if regulon_cols:
            warnings.warn(
                f"AUCell not in obsm; found {len(regulon_cols)} regulon columns in obs"
            )
            mat = adata.obs[regulon_cols].values.astype(np.float64)
            names = [c.replace("(+)", "").strip() for c in regulon_cols]
            return mat, names

        # Check if AUCell is the main .X matrix (some workflows)
        if adata.n_vars < 500:  # Likely regulon activity, not gene expression
            warnings.warn(
                "AUCell key not found in obsm. Using adata.X as regulon activity "
                f"({adata.n_vars} features — looks like regulons, not genes)."
            )
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            return np.array(X, dtype=np.float64), list(adata.var_names)

        raise KeyError(
            f"Could not find AUCell matrix. Key '{key}' not in obsm. "
            f"Available obsm keys: {list(adata.obsm.keys())}. "
            f"Try specifying aucell_key explicitly."
        )

    # -----------------------------------------------------------------
    # Data subsetting and preparation
    # -----------------------------------------------------------------

    def subset_celltype(
        self,
        target_type: str,
        keep_context: bool = True,
    ) -> "TOPPLEData":
        """
        Subset to a specific cell type, keeping full tissue context.

        Parameters
        ----------
        target_type : str
            Cell type to focus on (e.g., "CD8_TRM").
        keep_context : bool, default=True
            If True, keep ALL cells but mark the target. This is needed
            for Layer 3 spatial analysis. If False, subset to target only.

        Returns
        -------
        self (for chaining)
        """
        if keep_context:
            self.metadata["target_type"] = target_type
            target_mask = self.cell_types == target_type
            self.metadata["n_target_cells"] = int(target_mask.sum())
            if target_mask.sum() == 0:
                warnings.warn(f"No cells of type '{target_type}' found!")
        else:
            mask = self.cell_types == target_type
            self._apply_mask(mask)
        return self

    def define_states(
        self,
        condition_key: Optional[str] = None,
        pathological: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
        cluster_key: Optional[str] = None,
        pathological_clusters: Optional[List] = None,
    ) -> "TOPPLEData":
        """
        Define binary pathological vs homeostatic state labels.

        Multiple strategies:
        1. By disease condition: condition_key + pathological value
        2. Direct labels: pass binary array
        3. By cluster: cluster_key + list of pathological clusters

        Parameters
        ----------
        condition_key : str, optional
            Already stored condition column (set at load time).
        pathological : str, optional
            Value indicating pathological state.
        labels : np.ndarray, optional
            Direct binary labels (0 or 1).
        cluster_key : str, optional
            Cluster annotation column name.
        pathological_clusters : list, optional
            List of cluster IDs considered pathological.

        Returns
        -------
        self (for chaining)
        """
        if labels is not None:
            self.state_labels = np.array(labels, dtype=int)
        elif self.state_labels is not None:
            pass  # Already set
        else:
            warnings.warn("No state labels defined. Set via define_states().")

        if self.state_labels is not None:
            n_path = (self.state_labels == 1).sum()
            n_homeo = (self.state_labels == 0).sum()
            self.metadata["n_pathological"] = int(n_path)
            self.metadata["n_homeostatic"] = int(n_homeo)

        return self

    def select_regulons(
        self,
        regulon_list: Optional[List[str]] = None,
        min_cells_active: int = 10,
        min_activity_threshold: float = 0.01,
        top_n_variable: Optional[int] = None,
    ) -> "TOPPLEData":
        """
        Select and filter regulons for TOPPLE analysis.

        Parameters
        ----------
        regulon_list : list of str, optional
            Specific regulons to include. If None, auto-select.
        min_cells_active : int, default=10
            Minimum cells with AUCell > threshold.
        min_activity_threshold : float, default=0.01
            AUCell threshold for "active".
        top_n_variable : int, optional
            Keep top N most variable regulons.

        Returns
        -------
        self (for chaining)
        """
        if regulon_list is not None:
            # Exact match
            idx = [i for i, r in enumerate(self.regulon_names) if r in regulon_list]
            if len(idx) < len(regulon_list):
                missing = set(regulon_list) - {self.regulon_names[i] for i in idx}
                # Try fuzzy match (e.g., "RUNX3(+)" vs "RUNX3")
                for m in list(missing):
                    for i, r in enumerate(self.regulon_names):
                        if r.replace("(+)", "").strip() == m and i not in idx:
                            idx.append(i)
                            missing.discard(m)
                            break
                if missing:
                    warnings.warn(f"Regulons not found: {missing}")
        else:
            # Auto-select: filter by activity
            active_counts = (self.aucell > min_activity_threshold).sum(axis=0)
            idx = np.where(active_counts >= min_cells_active)[0].tolist()

            if top_n_variable and len(idx) > top_n_variable:
                variances = self.aucell[:, idx].var(axis=0)
                top_idx = np.argsort(variances)[-top_n_variable:]
                idx = [idx[i] for i in top_idx]

        if len(idx) == 0:
            raise ValueError("No regulons passed filtering.")

        self.aucell = self.aucell[:, idx]
        self.regulon_names = [self.regulon_names[i] for i in idx]
        self.metadata["n_regulons_selected"] = len(idx)

        return self

    def get_de_scores(
        self,
        method: str = "t_test",
    ) -> Dict[str, float]:
        """
        Compute differential expression (log2FC or t-stat) for each regulon
        between pathological and homeostatic states.

        Used by Layer 2 IPA weighting to identify maintenance regulons.

        Returns
        -------
        dict: {regulon_name: abs_log2fc_or_tstat}
        """
        if self.state_labels is None:
            raise ValueError("State labels not defined. Call define_states() first.")

        mask_path = self.state_labels == 1
        mask_homeo = self.state_labels == 0

        de_scores = {}
        for i, name in enumerate(self.regulon_names):
            vals_path = self.aucell[mask_path, i]
            vals_homeo = self.aucell[mask_homeo, i]

            if len(vals_path) < 2 or len(vals_homeo) < 2:
                de_scores[name] = 0.0
                continue

            if method == "t_test":
                from scipy.stats import ttest_ind
                t_stat, _ = ttest_ind(vals_path, vals_homeo)
                de_scores[name] = abs(float(t_stat))
            else:
                # Simple log2 fold-change on mean AUCell
                mean_path = vals_path.mean()
                mean_homeo = vals_homeo.mean()
                if mean_homeo > 0:
                    de_scores[name] = abs(np.log2((mean_path + 1e-6) / (mean_homeo + 1e-6)))
                else:
                    de_scores[name] = 0.0

        return de_scores

    def build_adjacency_matrix(
        self,
        mode: str = "combined",
        min_importance: float = 1.0,
    ) -> np.ndarray:
        """
        Build binary adjacency matrix from GRNBoost2 edges for topology-guided pruning.

        Parameters
        ----------
        mode : str, default="combined"
            "directed": A[i,j]=1 if TF_i → TF_j (via shared targets)
            "combined": A[i,j]=1 if i→j or j→i

        Returns
        -------
        np.ndarray, shape (n_regulons, n_regulons)
        """
        if self.adjacencies is None:
            raise ValueError("No adjacencies loaded. Provide adjacencies_path.")

        from topple.pruning import grn_to_adjacency
        return grn_to_adjacency(
            self.adjacencies,
            self.regulon_names,
            mode=mode,
            min_importance=min_importance,
        )

    # -----------------------------------------------------------------
    # Pipeline runners
    # -----------------------------------------------------------------

    def run_topple(
        self,
        target_type: str,
        stromal_types: List[str],
        max_order: int = 3,
        method: str = "exact",
        n_niches: int = 3,
        effect_size: float = 1.5,
        verbose: bool = True,
    ) -> dict:
        """
        Run the complete TOPPLE pipeline (L1 → L2 → L3).

        Parameters
        ----------
        target_type : str
            Cell type to analyze (e.g., "CD8_TRM").
        stromal_types : list of str
            Stromal types for buffering.
        max_order : int, default=3
            Maximum interaction order for L1.
        method : str, default="exact"
            Decomposition method: "exact", "pruned", "compressed".
        n_niches : int, default=3
            Number of spatial niches.
        effect_size : float, default=1.5
            Perturbation effect size for MockEngine.
        verbose : bool, default=True

        Returns
        -------
        dict with keys: decomposition, bridge_report, vulnerability_maps,
        spatial_report, dataframe.
        """
        from topple import StabilityDecomposer
        from topple.layer2.perturbation_engine import MockPerturbationEngine
        from topple.layer2.bridge import PerturbationBridge
        from topple.layer3.pipeline import SpatialVulnerabilityPipeline

        if self.state_labels is None:
            raise ValueError("Call define_states() before run_topple().")

        target_mask = self.cell_types == target_type
        target_idx = np.where(target_mask)[0]
        n_target = len(target_idx)

        if n_target == 0:
            raise ValueError(f"No cells of type '{target_type}'.")

        X_target = self.aucell[target_idx]
        y_target = self.state_labels[target_idx]

        if verbose:
            n_path = (y_target == 1).sum()
            n_homeo = (y_target == 0).sum()
            print(f"[TOPPLE] {n_target} {target_type} cells "
                  f"({n_path} pathological, {n_homeo} homeostatic)")
            print(f"[TOPPLE] {self.n_regulons} regulons: {', '.join(self.regulon_names)}")

        # === Layer 1 ===
        if verbose:
            print(f"\n{'='*60}")
            print("LAYER 1: Stability Decomposition")
            print(f"{'='*60}")

        kwargs = {"max_order": max_order, "method": method, "verbose": verbose}
        if method == "pruned" and self.adjacencies is not None:
            kwargs["adjacency"] = self.build_adjacency_matrix()

        sd = StabilityDecomposer(**kwargs)
        sd.fit(X_target, y_target, feature_names=self.regulon_names)

        # === Layer 2 ===
        if verbose:
            print(f"\n{'='*60}")
            print("LAYER 2: Perturbation Bridge")
            print(f"{'='*60}")

        de_scores = self.get_de_scores()
        engine = MockPerturbationEngine(
            X_target, y_target, self.regulon_names, effect_size=effect_size,
        )
        bridge = PerturbationBridge(
            engine=engine, X=X_target, y=y_target,
            feature_names=self.regulon_names,
            interactions=sd.interactions_,
            de_scores=de_scores,
        )
        l2_results = bridge.run()

        # Fallback: if bridge returns 0 candidates, generate from L1 top interactions
        if len(l2_results) == 0 and sd.interactions_:
            if verbose:
                print("[TOPPLE L2] Fallback: generating candidates from top L1 interactions")
            name_to_idx = {n: i for i, n in enumerate(self.regulon_names)}
            # Use interactions directly (top_interactions filters zeros)
            sorted_ints = sorted(
                sd.interactions_.items(), key=lambda x: abs(x[1]), reverse=True
            )
            for feat_set, val in sorted_ints[:10]:
                feat_names = [self.regulon_names[i] for i in sorted(feat_set)]
                l2_results.append({
                    "features": feat_names,
                    "perturbation_set": feat_set,
                    "interaction_value": val,
                })

        if verbose:
            print(bridge.report())

        # === Layer 3 (if spatial) ===
        vmaps = None
        spatial_report = None
        df = None

        if self.coordinates is not None and len(l2_results) > 0:
            if verbose:
                print(f"\n{'='*60}")
                print("LAYER 3: Spatial Vulnerability Mapping")
                print(f"{'='*60}")

            # Convert L2 results to per-cell destabilizations
            from topple.layer2.destabilization import DestabilizationScorer
            scorer_l2 = DestabilizationScorer(X_target, y_target)
            scorer_l2.fit()

            destabilizations = []
            for res in l2_results[:10]:
                result = engine.simulate(res["perturbation_set"])
                orig_conf = scorer_l2.classifier.predict_proba(result.X_original)[:, 1]
                pert_conf = scorer_l2.classifier.predict_proba(result.X_perturbed)[:, 1]
                conf_drop = np.clip(orig_conf - pert_conf, 0, 1)
                destabilizations.append((
                    conf_drop,
                    res["perturbation_set"],
                    res["features"],
                ))

            pipeline = SpatialVulnerabilityPipeline(
                coordinates=self.coordinates,
                cell_types=self.cell_types,
                regulon_activity=self.aucell,
                target_type=target_type,
                stromal_types=stromal_types,
                n_niches=n_niches,
                verbose=verbose,
            )
            vmaps = pipeline.run(destabilizations)
            spatial_report = pipeline.report()
            df = pipeline.to_dataframe()

            if verbose:
                print(spatial_report)
        else:
            if verbose:
                if self.coordinates is None:
                    print("\n[TOPPLE] No spatial coordinates — skipping Layer 3.")
                elif len(l2_results) == 0:
                    print("\n[TOPPLE] No L2 candidates — skipping Layer 3. "
                          "Try lowering max_order or adding more regulons.")

        return {
            "decomposition": sd,
            "bridge": bridge,
            "bridge_results": l2_results,
            "bridge_report": bridge.report(),
            "vulnerability_maps": vmaps,
            "spatial_report": spatial_report,
            "dataframe": df,
        }

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def _apply_mask(self, mask: np.ndarray):
        """Apply boolean mask to all arrays."""
        self.aucell = self.aucell[mask]
        self.cell_types = self.cell_types[mask]
        if self.coordinates is not None:
            self.coordinates = self.coordinates[mask]
        if self.expression is not None:
            self.expression = self.expression[mask]
        if self.state_labels is not None:
            self.state_labels = self.state_labels[mask]
        if self.sample_ids is not None:
            self.sample_ids = self.sample_ids[mask]

    def summary(self) -> str:
        """Print full data summary."""
        lines = [str(self)]
        if self.regulon_names:
            lines.append(f"  Regulons: {', '.join(self.regulon_names[:10])}"
                         + (f" ... (+{len(self.regulon_names)-10} more)"
                            if len(self.regulon_names) > 10 else ""))
        ct_counts = {}
        for ct in self.cell_types:
            ct_counts[ct] = ct_counts.get(ct, 0) + 1
        lines.append("  Cell type composition:")
        for ct, n in sorted(ct_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {ct}: {n}")
        return "\n".join(lines)
