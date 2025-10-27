import os
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List
import yaml

import numpy as np
import pandas as pd
import joblib
import networkx as nx
from sklearn.linear_model import LinearRegression
from models import LinearAddSCM, CausalBayesianNetwork, Intervention  # noqa: F401
# pgmpy imports left for compatibility, not used directly
from pgmpy.models import BayesianNetwork as BN  # noqa: F401
from pgmpy.factors.discrete import TabularCPD as cpd  # noqa: F401

warnings.filterwarnings("ignore")


# -------------------------------
# CONFIG
# -------------------------------
def _load_config(config_path: str = "configs/battery_config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------
# UTILITIES
# -------------------------------
def _bootstrap_augment(data: np.ndarray, min_samples: int = 10, seed: int = 42) -> np.ndarray:
    """
    Conservative bootstrap: only upsample when N < min_samples; otherwise return data unchanged.
    Uses a deterministic seed so results are reproducible per bucket.
    """
    rng = np.random.default_rng(seed)
    current_size = len(data)
    if current_size >= min_samples:
        return data
    idx = rng.choice(current_size, size=min_samples, replace=True)
    return data[idx]


def _parents_map(graph: nx.DiGraph, node_order: List[str]) -> Dict[str, List[str]]:
    p = {n: [] for n in node_order}
    for u, v in graph.edges():
        p[v].append(u)
    return p


# -------------------------------
# IO + FRAMING
# -------------------------------
def _load_graphs_and_dfs(config: Dict[str, Any]) -> Tuple[nx.DiGraph, nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """Load LL/HL DAGs and raw dataframes, then normalize headers and rename columns from config."""
    data_paths = config["data_paths"]
    P_M_BASE = Path(data_paths["ll_scm"])
    P_M_ABST = Path(data_paths["hl_scm"])
    P_DF_BASE = Path(data_paths["ll_data"])
    P_DF_ABST = Path(data_paths["hl_data"])

    # Load SCM graphs
    M_base = joblib.load(P_M_BASE)
    M_abst = joblib.load(P_M_ABST)

    Gll = nx.DiGraph(); Gll.add_nodes_from(M_base.nodes()); Gll.add_edges_from(M_base.edges())
    Ghl = nx.DiGraph(); Ghl.add_nodes_from(M_abst.nodes()); Ghl.add_edges_from(M_abst.edges())

    prefer_parquet = config.get("advanced", {}).get("prefer_parquet", True)
    if prefer_parquet:
        P_DF_BASE_PARQ = P_DF_BASE.with_suffix(".parquet")
        P_DF_ABST_PARQ = P_DF_ABST.with_suffix(".parquet")
        if P_DF_BASE_PARQ.exists():
            df_base_raw = pd.read_parquet(P_DF_BASE_PARQ)
        else:
            df_base_raw = joblib.load(P_DF_BASE)
        if P_DF_ABST_PARQ.exists():
            df_abst_raw = pd.read_parquet(P_DF_ABST_PARQ)
        else:
            df_abst_raw = joblib.load(P_DF_ABST)
    else:
        df_base_raw = joblib.load(P_DF_BASE)
        df_abst_raw = joblib.load(P_DF_ABST)

    # Normalize headers (trim whitespace)
    df_base_raw.columns = df_base_raw.columns.str.strip()
    df_abst_raw.columns = df_abst_raw.columns.str.strip()

    # Apply column renaming from config robustly
    ll_rename = config["preprocessing"]["ll_rename"]  # {'ML_avg0': 'ML0', 'ML_avg1': 'ML1'}
    hl_rename = config["preprocessing"]["hl_rename"]  # {'Comma gap (µm)': 'CG', 'Mass Loading (mg cm-2)': 'ML'}

    # Build raw column lists from rename maps + CG
    ll_raw_cols = ["CG"] + list(ll_rename.keys())
    hl_raw_cols = list(hl_rename.keys())

    # Select by raw names (after strip) then rename → canonical names
    try:
        df_base = df_base_raw[ll_raw_cols].rename(columns=ll_rename)
    except KeyError as e:
        missing = [c for c in ll_raw_cols if c not in df_base_raw.columns]
        raise KeyError(f"LL columns missing in raw dataframe: {missing}") from e

    try:
        df_abst = df_abst_raw[hl_raw_cols].rename(columns=hl_rename)
    except KeyError as e:
        missing = [c for c in hl_raw_cols if c not in df_abst_raw.columns]
        raise KeyError(f"HL columns missing in raw dataframe: {missing}") from e

    # Ensure numeric types early (fail fast if not numeric)
    for c in df_base.columns:
        df_base[c] = pd.to_numeric(df_base[c], errors="raise")
    for c in df_abst.columns:
        df_abst[c] = pd.to_numeric(df_abst[c], errors="raise")

    return Gll, Ghl, df_base, df_abst


# -------------------------------
# ABDUCTION
# -------------------------------
def _fit_coeffs_intercepts(
    X: np.ndarray, graph: nx.DiGraph, node_order: List[str], use_intercept: bool
):
    """Fit linear mechanisms per node; return coeffs, intercepts, residuals, R2."""
    from sklearn.metrics import r2_score
    coeffs: Dict[tuple, float] = {}
    intercepts: Dict[str, float] = {}
    residuals: Dict[str, np.ndarray] = {}
    r2_train: Dict[str, float] = {}

    for node in nx.topological_sort(graph):
        j = node_order.index(node)
        parents = list(graph.predecessors(node))
        if not parents:
            # root: model X = c + U; with use_intercept True, c is the global mean
            intercepts[node] = float(X[:, j].mean()) if use_intercept else 0.0
            residuals[node] = X[:, j] - intercepts[node]
            r2_train[node] = np.nan
            continue
        X_pa = X[:, [node_order.index(p) for p in parents]]
        y = X[:, j]
        lr = LinearRegression(fit_intercept=use_intercept)
        lr.fit(X_pa, y)
        yhat = lr.predict(X_pa)
        for p, b in zip(parents, lr.coef_):
            coeffs[(p, node)] = float(b)
        intercepts[node] = float(lr.intercept_) if use_intercept else 0.0
        residuals[node] = y - yhat
        r2_train[node] = float(r2_score(y, yhat))
    return coeffs, intercepts, residuals, r2_train


def _abduce_noise(
    X: np.ndarray, graph: nx.DiGraph, node_order: List[str], coeffs: dict, intercepts: dict
) -> np.ndarray:
    """Compute U = X - (c + BX_pa); for roots use U = X - c."""
    N, d = X.shape
    U = np.zeros((N, d), dtype=float)
    for node in nx.topological_sort(graph):
        j = node_order.index(node)
        parents = list(graph.predecessors(node))
        if not parents:
            b0 = float(intercepts.get(node, 0.0))
            U[:, j] = X[:, j] - b0
        else:
            pred = intercepts.get(node, 0.0) * np.ones(N)
            for p in parents:
                pred += X[:, node_order.index(p)] * coeffs[(p, node)]
            U[:, j] = X[:, j] - pred
    return U


# -------------------------------
# ALIGNMENT
# -------------------------------
def _align_by_cg(
    X_ll: np.ndarray, X_hl: np.ndarray, seed: int = 23
) -> Tuple[np.ndarray, np.ndarray, Dict[float, float], np.ndarray, np.ndarray]:
    """
    Align LL/HL rows by CG via ω: {75→75, 110→100, 180→200, 200→200}.
    Shuffle within buckets to avoid positional bias.
    Returns aligned matrices and origin indices into the original arrays.
    """
    df_ll = pd.DataFrame(X_ll, columns=["CG", "ML0", "ML1"])
    df_hl = pd.DataFrame(X_hl, columns=["CG", "ML"])
    omega_cg_map = {75.0: 75.0, 110.0: 100.0, 180.0: 200.0, 200.0: 200.0}

    rng = np.random.default_rng(seed)

    hl_to_ll: Dict[float, List[float]] = {}
    for cg_ll, cg_hl in omega_cg_map.items():
        hl_to_ll.setdefault(cg_hl, []).append(cg_ll)

    Xll_chunks, Xhl_chunks = [], []
    ll_origin_idx, hl_origin_idx = [], []

    for cg_hl, ll_cgs in hl_to_ll.items():
        hl_idx = df_hl.index[df_hl["CG"] == cg_hl].to_numpy()
        if hl_idx.size == 0:
            continue
        ll_idx = np.concatenate([df_ll.index[df_ll["CG"] == cg].to_numpy() for cg in ll_cgs])  # pool
        if ll_idx.size == 0:
            continue

        rng.shuffle(hl_idx)
        rng.shuffle(ll_idx)

        n = min(len(ll_idx), len(hl_idx))
        Xll_chunks.append(X_ll[ll_idx[:n]])
        Xhl_chunks.append(X_hl[hl_idx[:n]])
        ll_origin_idx.append(ll_idx[:n])
        hl_origin_idx.append(hl_idx[:n])

    X_ll_aligned = np.vstack(Xll_chunks)
    X_hl_aligned = np.vstack(Xhl_chunks)
    ll_origin_idx = np.concatenate(ll_origin_idx)
    hl_origin_idx = np.concatenate(hl_origin_idx)

    return X_ll_aligned, X_hl_aligned, omega_cg_map, ll_origin_idx, hl_origin_idx


def _select_by_cg(mat: np.ndarray, cg_value: float, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rows with CG≈cg_value, boolean mask) from an aligned matrix."""
    m = np.isclose(mat[:, 0], float(cg_value), atol=tol)
    return mat[m], m


# -------------------------------
# DETERMINISTIC PART PER INTERVENTION
# -------------------------------
def _compute_D_per_iv(model_dict: dict, coeffs: dict, intercepts: dict) -> Dict[Any, np.ndarray]:
    """
    For each intervention iv and node j:
      - if j is in do(iv): D_j = do_value
      - else if j is root: D_j = intercept
      - else: D_j = intercept + sum β X_pa
    """
    data_by_iv = model_dict["data"]
    node_order = model_dict["node_order"]
    parents = _parents_map(model_dict["graph"], node_order)

    def _do_value(iv, node):
        if iv is None:
            return None
        vv = iv.vv()
        return vv.get(node, None)

    D_dict: Dict[Any, np.ndarray] = {}
    for iv, X in data_by_iv.items():
        if X is None:
            continue
        X = np.asarray(X)
        N, d = X.shape
        D = np.zeros((N, d), dtype=float)

        for j, child in enumerate(node_order):
            do_val = _do_value(iv, child)
            pa = parents.get(child, [])

            if do_val is not None:
                D[:, j] = float(do_val)
                continue

            if len(pa) == 0:
                D[:, j] = float(intercepts.get(child, 0.0))
                continue

            X_pa = X[:, [node_order.index(p) for p in pa]]
            if (child in intercepts) and all(((p, child) in coeffs) for p in pa):
                b0 = float(intercepts[child])
                betas = np.array([coeffs[(p, child)] for p in pa], dtype=float)
                D[:, j] = b0 + X_pa @ betas
            else:
                # fallback (rare)
                lr = LinearRegression(fit_intercept=True).fit(X_pa, X[:, j])
                D[:, j] = lr.intercept_ + X_pa @ lr.coef_.astype(float)

        D_dict[iv] = D
    return D_dict


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def load_and_prepare_battery(save: bool = True, config_path: str = "configs/battery_config.yaml") -> Dict[str, Any]:
    config = _load_config(config_path)

    # Seed
    seed = int(config.get("seed", 23))
    np.random.seed(seed)

    # Load graphs + dataframes
    Gll, Ghl, df_base, df_abst = _load_graphs_and_dfs(config)

    # Node orders (canonical)
    LL_NODES = config["variables"]["low_level"]["nodes"]   # ['CG','ML0','ML1']
    HL_NODES = config["variables"]["high_level"]["nodes"]  # ['CG','ML']

    # Sanity: DAG nodes should match node orders (or be supersets with same names)
    assert set(Gll.nodes()).issuperset(set(LL_NODES)), f"LL DAG missing nodes in {LL_NODES}"
    assert set(Ghl.nodes()).issuperset(set(HL_NODES)), f"HL DAG missing nodes in {HL_NODES}"

    # Arrays
    X_ll = df_base[LL_NODES].to_numpy().astype(float)
    X_hl = df_abst[HL_NODES].to_numpy().astype(float)

    # Abduction (fit + residuals)
    use_intercept = bool(config["abduction"]["use_intercept"])
    ll_coeffs, ll_intercepts, ll_resid, ll_r2 = _fit_coeffs_intercepts(X_ll, Gll, LL_NODES, use_intercept)
    hl_coeffs, hl_intercepts, hl_resid, hl_r2 = _fit_coeffs_intercepts(X_hl, Ghl, HL_NODES, use_intercept)
    U_ll = _abduce_noise(X_ll, Gll, LL_NODES, ll_coeffs, ll_intercepts)
    U_hl = _abduce_noise(X_hl, Ghl, HL_NODES, hl_coeffs, hl_intercepts)

    # Optional QC prints
    qc = config.get("quality_control", {})
    if qc.get("print_r2_scores", False):
        print("[QC] LL R² (non-roots):", {k: v for k, v in ll_r2.items() if not np.isnan(v)})
        print("[QC] HL R² (non-roots):", {k: v for k, v in hl_r2.items() if not np.isnan(v)})

    # Align LL/HL by CG with shuffle + origin indices (for noise alignment & future folds)
    X_ll_aligned, X_hl_aligned, omega_cg_map, ll_origin_idx, hl_origin_idx = _align_by_cg(X_ll, X_hl, seed=seed)

    # Tie noise to the exact aligned rows
    U_ll_aligned = U_ll[ll_origin_idx]
    U_hl_aligned = U_hl[hl_origin_idx]

    # Buckets (observational-aligned as "None"; then per CG do-buckets)
    def _cg_values(side: str) -> List[float]:
        return list(config["interventions"][side]["cg_values"])

    # Create intervention objects first to ensure consistent references
    ll_interventions = {cg: Intervention({"CG": cg}) for cg in _cg_values("low_level")}
    hl_interventions = {cg: Intervention({"CG": cg}) for cg in _cg_values("high_level")}

    # Initialize with aligned data and build row indices during bucket creation
    K = X_ll_aligned.shape[0]
    aligned_pos = np.arange(K)
    CG_ll_aligned = X_ll_aligned[:, 0].astype(float)
    CG_hl_aligned = X_hl_aligned[:, 0].astype(float)

    Dll_samples: Dict[Any, np.ndarray] = {None: X_ll_aligned}
    Dhl_samples: Dict[Any, np.ndarray] = {None: X_hl_aligned}
    Dll_noise: Dict[Any, np.ndarray] = {None: U_ll_aligned}
    Dhl_noise: Dict[Any, np.ndarray] = {None: U_hl_aligned}

    row_idx_ll: Dict[Any, np.ndarray] = {None: aligned_pos}
    row_idx_hl: Dict[Any, np.ndarray] = {None: aligned_pos}

    # Bootstrap configuration
    bootstrap_cfg = config["bootstrap"]
    min_samples = int(bootstrap_cfg["min_samples"]) if bootstrap_cfg["enabled"] else 0
    rng = np.random.default_rng(seed)
    tol = float(config.get("alignment", {}).get("tolerance", 1e-9))

    # LL per-intervention (build indices first, then sample with replacement if needed)
    for cg_ll in _cg_values("low_level"):
        iv = ll_interventions[cg_ll]
        base_idx = np.where(np.isclose(CG_ll_aligned, float(cg_ll), atol=tol))[0]
        if base_idx.size == 0:
            continue
        if bootstrap_cfg["enabled"] and base_idx.size < min_samples:
            sel = rng.choice(base_idx, size=min_samples, replace=True)
        else:
            sel = base_idx
        Dll_samples[iv] = X_ll_aligned[sel]
        Dll_noise[iv] = U_ll_aligned[sel]
        row_idx_ll[iv] = sel

    # HL per-intervention
    for cg_hl in _cg_values("high_level"):
        iv = hl_interventions[cg_hl]
        base_idx = np.where(np.isclose(CG_hl_aligned, float(cg_hl), atol=tol))[0]
        if base_idx.size == 0:
            continue
        if bootstrap_cfg["enabled"] and base_idx.size < min_samples:
            sel = rng.choice(base_idx, size=min_samples, replace=True)
        else:
            sel = base_idx
        Dhl_samples[iv] = X_hl_aligned[sel]
        Dhl_noise[iv] = U_hl_aligned[sel]
        row_idx_hl[iv] = sel

    # Define intervention sets using the same objects
    Ill_relevant = [None] + list(ll_interventions.values())
    Ihl_relevant = [None] + list(hl_interventions.values())

    # Center noise per intervention (mean-zero), then zero CG noise under do(CG=·)
    ll_cg_idx = LL_NODES.index("CG")
    hl_cg_idx = HL_NODES.index("CG")

    for iv, U in list(Dll_noise.items()):
        if U is None:
            continue
        Uc = U - U.mean(axis=0, keepdims=True)
        if iv is not None and "CG" in iv.vv():
            Uc[:, ll_cg_idx] = 0.0  # do(CG=·): no stochasticity on CG
        Dll_noise[iv] = Uc

    for iv, U in list(Dhl_noise.items()):
        if U is None:
            continue
        Uc = U - U.mean(axis=0, keepdims=True)
        if iv is not None and "CG" in iv.vv():
            Uc[:, hl_cg_idx] = 0.0
        Dhl_noise[iv] = Uc

    # Package causal graphs (edges from fitted coeffs)
    ll_causal_graph = CausalBayesianNetwork(list(ll_coeffs.keys()))
    hl_causal_graph = CausalBayesianNetwork(list(hl_coeffs.keys()))

    # Build model dicts
    LLmodel = {
        "graph": ll_causal_graph,
        "intervention_set": Ill_relevant,
        "coeffs": ll_coeffs,
        "intercepts": ll_intercepts,
        "data": Dll_samples,
        "noise": Dll_noise,
        "node_order": LL_NODES,
    }
    HLmodel = {
        "graph": hl_causal_graph,
        "intervention_set": Ihl_relevant,
        "coeffs": hl_coeffs,
        "intercepts": hl_intercepts,
        "data": Dhl_samples,
        "noise": Dhl_noise,
        "node_order": HL_NODES,
    }

    # Deterministic parts per intervention (with do-semantics + roots handled)
    det_ll_dict = _compute_D_per_iv(LLmodel, ll_coeffs, ll_intercepts)
    det_hl_dict = _compute_D_per_iv(HLmodel, hl_coeffs, hl_intercepts)

    LLmodel["deterministic"] = det_ll_dict
    HLmodel["deterministic"] = det_hl_dict

    # Add aligned global noise keys for optimization compatibility (use aligned!)
    LLmodel["noise"]["U_ll_hat"] = U_ll_aligned
    HLmodel["noise"]["U_hl_hat"] = U_hl_aligned

    # Add row indices to models (already computed during bucket building)
    LLmodel["row_idx"] = row_idx_ll
    HLmodel["row_idx"] = row_idx_hl

    # ω mapping (reuse same Intervention objects for key identity)
    omega = {
        None: None,
        ll_interventions[75.0]:  hl_interventions[75.0],
        ll_interventions[110.0]: hl_interventions[100.0],
        ll_interventions[180.0]: hl_interventions[200.0],
        ll_interventions[200.0]: hl_interventions[200.0],
    }
    abstraction_data = {"T": None, "omega": omega}

    # QC: sample sizes per intervention (optional)
    if config.get("quality_control", {}).get("print_sample_sizes", False):
        def _count(buckets: Dict[Any, np.ndarray]) -> Dict[str, int]:
            out = {}
            for k, v in buckets.items():
                key = "None" if k is None else f"do(CG={list(k.vv().values())[0]})"
                out[key] = 0 if v is None else len(v)
            return out

        print("[QC] LL sample sizes:", _count(Dll_samples))
        print("[QC] HL sample sizes:", _count(Dhl_samples))

    out = {
        "LLmodel": LLmodel,
        "HLmodel": HLmodel,
        "abstraction_data": abstraction_data,
    }

    # Save
    if save:
        output_config = config["output"]
        save_dir = Path(output_config["save_directory"])
        save_dir.mkdir(parents=True, exist_ok=True)

        if output_config.get("save_models", True):
            joblib.dump(LLmodel, save_dir / output_config["ll_model_file"])
            joblib.dump(HLmodel, save_dir / output_config["hl_model_file"])

        if output_config.get("save_abstraction", True):
            joblib.dump(abstraction_data, save_dir / output_config["abstraction_file"])

    return out


if __name__ == "__main__":
    bundles = load_and_prepare_battery(save=True)
    print("Battery data generation completed successfully!")
