"""
TOPPLE Data Module — Test Suite
==================================

Tests are split into:
- Core tests (no anndata dependency): CSV loading, TOPPLEData operations
- AnnData tests (require `pip install anndata`): .h5ad loading, full pipeline

Run: python tests/test_data.py
"""

import numpy as np
import pandas as pd
import os, sys, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topple.data.loader import (
    TOPPLEData,
    load_pyscenic_adjacencies,
    load_aucell_matrix,
)

try:
    import anndata
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


# ===========================================================================
# Helpers
# ===========================================================================

def make_adjacencies_csv(tmp_dir, regulon_names):
    """Create a minimal adjacencies.csv."""
    import pandas as pd
    rows = []
    for i, tf in enumerate(regulon_names):
        for j in range(5):
            rows.append({"TF": tf, "target": f"gene_{i*5+j}", "importance": np.random.rand() * 10})
    for i in range(len(regulon_names) - 1):
        rows.append({
            "TF": regulon_names[i],
            "target": regulon_names[i+1],
            "importance": 5.0,
        })
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_dir, "adjacencies.csv")
    df.to_csv(path, index=False)
    return path


def make_aucell_csv(tmp_dir, n_cells=100, n_regulons=6):
    """Create standalone AUCell CSV."""
    import pandas as pd
    rng = np.random.RandomState(42)
    names = [f"TF_{i}" for i in range(n_regulons)]
    df = pd.DataFrame(
        rng.rand(n_cells, n_regulons),
        columns=names,
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    path = os.path.join(tmp_dir, "aucell.csv")
    df.to_csv(path)
    return path, names


def make_topple_data(n_cells=300, n_regulons=8, with_spatial=True, rng_seed=42):
    """Create TOPPLEData directly (no anndata needed)."""
    rng = np.random.RandomState(rng_seed)
    names = ["RUNX3", "TBX21", "EOMES", "NR4A1", "IRF4", "BATF", "PRDM1", "TOX"][:n_regulons]
    aucell = rng.rand(n_cells, n_regulons)
    ct_choices = ["CD8_TRM", "fibroblast", "endothelial", "keratinocyte"]
    cell_types = np.array(rng.choice(ct_choices, size=n_cells))
    coordinates = rng.rand(n_cells, 2) * 10 if with_spatial else None
    state_labels = rng.choice([0, 1], size=n_cells)

    # Inject signal: pathological TRM have higher regulon activity
    trm_mask = cell_types == "CD8_TRM"
    path_trm = trm_mask & (state_labels == 1)
    aucell[path_trm, :3] += 2.5   # Strong signal for RUNX3, TBX21, EOMES
    aucell[path_trm, 4:7] += 1.5  # Moderate for IRF4, BATF, PRDM1
    # Add pairwise synergy
    synergy = rng.randn(int(path_trm.sum())) * 0.5
    aucell[path_trm, 0] += synergy
    aucell[path_trm, 1] -= synergy

    return TOPPLEData(
        aucell=aucell,
        regulon_names=names,
        cell_types=cell_types,
        coordinates=coordinates,
        state_labels=state_labels,
    )


def make_synthetic_h5ad(tmp_dir, n_cells=300, n_regulons=8, with_spatial=True):
    """Create a minimal AnnData .h5ad for testing (requires anndata)."""
    import pandas as pd
    rng = np.random.RandomState(42)
    n_genes = 200
    X = rng.rand(n_cells, n_genes).astype(np.float32)
    aucell = rng.rand(n_cells, n_regulons).astype(np.float64)
    regulon_names = ["RUNX3", "TBX21", "EOMES", "NR4A1", "IRF4", "BATF", "PRDM1", "TOX"][:n_regulons]
    ct_choices = ["CD8_TRM", "fibroblast", "endothelial", "keratinocyte"]
    cell_types = rng.choice(ct_choices, size=n_cells)
    conditions = rng.choice(["psoriasis", "healthy"], size=n_cells)
    obs = pd.DataFrame({
        "cell_type": cell_types, "condition": conditions,
        "sample_id": rng.choice(["S1", "S2", "S3"], size=n_cells),
    }, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    aucell_df = pd.DataFrame(aucell, columns=regulon_names, index=obs.index)
    adata.obsm["X_aucell"] = aucell_df
    if with_spatial:
        adata.obsm["spatial"] = rng.rand(n_cells, 2) * 10
    path = os.path.join(tmp_dir, "test.h5ad")
    adata.write(path)
    return path, regulon_names


# ===========================================================================
# Core tests (no anndata required)
# ===========================================================================

class TestLoadPySCENIC:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_load_adjacencies(self):
        path = make_adjacencies_csv(self.tmp, ["RUNX3", "TBX21", "EOMES"])
        df = load_pyscenic_adjacencies(path)
        assert "TF" in df.columns
        assert "target" in df.columns
        assert "importance" in df.columns
        assert len(df) > 0

    def test_load_adjacencies_with_filter(self):
        path = make_adjacencies_csv(self.tmp, ["RUNX3", "TBX21"])
        df = load_pyscenic_adjacencies(path, min_importance=5.0)
        assert all(df["importance"] >= 5.0)

    def test_load_aucell_csv(self):
        path, names = make_aucell_csv(self.tmp)
        mat, loaded_names = load_aucell_matrix(path)
        assert mat.shape == (100, 6)
        assert loaded_names == names


class TestTOPPLEDataDirect:
    """Tests using directly-constructed TOPPLEData (no anndata)."""

    def setup_method(self):
        pass
    def teardown_method(self):
        pass

    def test_basic_properties(self):
        data = make_topple_data()
        assert data.n_cells == 300
        assert data.n_regulons == 8

    def test_repr(self):
        data = make_topple_data()
        r = repr(data)
        assert "300 cells" in r
        assert "8 regulons" in r

    def test_summary(self):
        data = make_topple_data()
        s = data.summary()
        assert "CD8_TRM" in s
        assert "fibroblast" in s

    def test_select_regulons_by_list(self):
        data = make_topple_data()
        data.select_regulons(regulon_list=["RUNX3", "TBX21", "EOMES"])
        assert data.n_regulons == 3
        assert "RUNX3" in data.regulon_names

    def test_select_regulons_auto(self):
        data = make_topple_data()
        data.select_regulons(top_n_variable=5)
        assert data.n_regulons == 5

    def test_de_scores(self):
        data = make_topple_data()
        de = data.get_de_scores()
        assert len(de) == 8
        assert all(v >= 0 for v in de.values())
        # RUNX3 should have high DE (we injected signal)
        assert de["RUNX3"] > de["TOX"]

    def test_run_topple_full(self):
        data = make_topple_data(n_cells=400)
        data.select_regulons(top_n_variable=6)
        results = data.run_topple(
            target_type="CD8_TRM",
            stromal_types=["fibroblast", "endothelial"],
            max_order=2, n_niches=2, verbose=False,
        )
        assert results["decomposition"] is not None
        assert results["bridge_report"] is not None
        assert results["vulnerability_maps"] is not None
        assert "vulnerability" in results["dataframe"].columns

    def test_run_topple_no_spatial(self):
        data = make_topple_data(with_spatial=False)
        data.select_regulons(top_n_variable=5)
        results = data.run_topple(
            target_type="CD8_TRM",
            stromal_types=["fibroblast"],
            max_order=2, verbose=False,
        )
        assert results["vulnerability_maps"] is None
        assert results["spatial_report"] is None


# ===========================================================================
# AnnData tests (skipped if anndata not installed)
# ===========================================================================

class TestTOPPLEDataFromAnnData:
    """Tests requiring anndata. Skipped if not installed."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_basic_load(self):
        if not HAS_ANNDATA: return  # SKIP
        h5ad_path, _ = make_synthetic_h5ad(self.tmp)
        data = TOPPLEData.from_anndata(
            h5ad_path, aucell_key="X_aucell",
            cell_type_key="cell_type",
            condition_key="condition",
            pathological_value="psoriasis",
        )
        assert data.n_cells == 300
        assert data.n_regulons == 8
        assert data.coordinates is not None
        assert data.state_labels is not None

    def test_with_adjacencies(self):
        if not HAS_ANNDATA: return
        h5ad_path, reg_names = make_synthetic_h5ad(self.tmp)
        adj_path = make_adjacencies_csv(self.tmp, reg_names)
        data = TOPPLEData.from_anndata(h5ad_path, adjacencies_path=adj_path)
        assert data.adjacencies is not None

    def test_no_spatial(self):
        if not HAS_ANNDATA: return
        h5ad_path, _ = make_synthetic_h5ad(self.tmp, with_spatial=False)
        data = TOPPLEData.from_anndata(h5ad_path)
        assert data.coordinates is None

    def test_full_pipeline_h5ad(self):
        if not HAS_ANNDATA: return
        h5ad_path, _ = make_synthetic_h5ad(self.tmp, n_cells=400)
        data = TOPPLEData.from_anndata(
            h5ad_path, condition_key="condition",
            pathological_value="psoriasis",
        )
        data.select_regulons(top_n_variable=6)
        results = data.run_topple(
            target_type="CD8_TRM",
            stromal_types=["fibroblast", "endothelial"],
            max_order=2, n_niches=2, verbose=False,
        )
        assert results["dataframe"] is not None


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    test_classes = [
        TestLoadPySCENIC, TestTOPPLEDataDirect,
        TestTOPPLEDataFromAnnData,
    ]
    total = passed = failed = 0
    for cls in test_classes:
        obj = cls()
        for method in sorted([m for m in dir(obj) if m.startswith("test_")]):
            total += 1
            try:
                obj.setup_method()
                getattr(obj, method)()
                obj.teardown_method()
                print(f"  PASS  {cls.__name__}.{method}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{method}: {e}")
                import traceback; traceback.print_exc()
                failed += 1
                try:
                    obj.teardown_method()
                except:
                    pass
    print(f"\n=== {passed}/{total} passed, {failed} failed ===")
