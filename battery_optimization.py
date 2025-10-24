#!/usr/bin/env python3
"""
Battery Empirical Optimization (Repeated 90/10 Splits)

- Uses your existing utilities/optimizers.
- Replaces K-folds with repeated stratified 90/10 splits over multiple seeds.
- Keeps the same 'saved_folds' structure expected by your run_* functions.
"""

import argparse
import os
import sys
import logging
import joblib
import numpy as np

import utilities as ut
import opt_tools as optools

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("battery_optimization")


# ---------- small helpers ----------
def get_config_path(base_name: str, exp_name: str) -> str:
    """Load specialized config if present, else default."""
    specialized_path = f"configs/{base_name}_{exp_name}.yaml"
    default_path = f"configs/{base_name}.yaml"
    return specialized_path if os.path.exists(specialized_path) else default_path


def make_stratified_9010_splits(Dhl_obs: np.ndarray, test_size: float, seeds: list[int]) -> list[dict]:
    """
    Build a list of 'folds' dicts (one per seed) with 'train' and 'test' indices.
    Stratify by HL CG (assumed to be column 0 of Dhl_obs).
    """
    assert 0.0 < test_size < 1.0, "test_size must be in (0,1)"
    N = Dhl_obs.shape[0]
    y = Dhl_obs[:, 0]  # CG values

    # group indices by label (exact match on CG values)
    label_to_indices = {}
    for idx, lab in enumerate(y):
        label_to_indices.setdefault(lab, []).append(idx)

    folds = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test_idx = []
        for lab, idxs in label_to_indices.items():
            idxs = np.array(idxs, dtype=int)
            n = len(idxs)
            n_test = max(1, int(round(test_size * n)))
            chosen = rng.choice(idxs, size=n_test, replace=False)
            test_idx.append(chosen)
        test_idx = np.unique(np.concatenate(test_idx))
        train_mask = np.ones(N, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        folds.append({"train": train_idx, "test": test_idx})
    return folds


# ---------- Affine ANM helpers (deterministic + noise) implemented locally ----------
def _get_parents(graph_edges: list[tuple], node_order: list[str]) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {n: [] for n in node_order}
    for u, v in graph_edges:
        parents[v].append(u)
    return parents


def compute_deterministic_parts_per_intervention(model_dict: dict) -> dict:
    """
    Build deterministic part D^{(iota)} for each intervention in model_dict['data'] using
    per-intervention linear regressions with intercept on child ~ parents.

    Returns: { iota: D (N_iota x d) } with columns in model_dict['node_order'] order.
    """
    from sklearn.linear_model import LinearRegression

    data_by_iv: dict = model_dict["data"]  # {iota -> X (N x d)}
    node_order: list[str] = model_dict["node_order"]
    graph_edges = list(model_dict["graph"].edges())
    parents_map = _get_parents(graph_edges, node_order)

    det_dict: dict = {}
    for iv, X in data_by_iv.items():
        if X is None:
            continue
        X = np.asarray(X)
        N, d = X.shape
        D = np.zeros((N, d), dtype=float)
        # Roots: deterministic = 0 by convention (all signal in residual/offset handled via intercepts on children)
        for j, child in enumerate(node_order):
            pa = parents_map.get(child, [])
            if len(pa) == 0:
                D[:, j] = 0.0
                continue
            X_pa = X[:, [node_order.index(p) for p in pa]]
            y = X[:, j]
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_pa, y)
            D[:, j] = lr.intercept_ + X_pa @ lr.coef_.astype(float)
        det_dict[iv] = D
    return det_dict


def _build_intervention_indices_by_cg(X_obs: np.ndarray, omega_cg_map: dict) -> dict:
    """
    Build boolean index arrays for each intervention by matching CG (first column).
    Returns: { iota: idx_array }
    """
    idx_map: dict = {}
    # observational None present in model data as well, but here we only need per-CG masks
    unique_cg = np.unique(X_obs[:, 0])
    for cg in unique_cg:
        mask = np.isclose(X_obs[:, 0], float(cg))
        idx_map[cg] = np.where(mask)[0]
    return idx_map


def optimize_affine_map_erica(det_ll: dict, det_hl: dict,
                              U_ll_hat: np.ndarray, U_hl_hat: np.ndarray,
                              omega: dict, saved_folds: list[dict],
                              epsilon: float = 1.0, delta: float = 1.0,
                              eta_min: float = 1e-3, eta_max: float = 1e-2,
                              num_steps_min: int = 5, num_steps_max: int = 5,
                              max_iter: int = 200, tol: float = 1e-6,
                              seed: int = 42, initialization: str = 'zeros',
                              gain: float = 0.0, optimizers: str = 'adam') -> dict:
    """
    ERICA-style min-max optimization for affine ANM objective.
    
    Minimize T: E_iota || T (D_l(iota) + U_l(iota) + Theta_l) - (D_h(omega) + U_h(omega) + Phi_h) ||_F^2
    Maximize Theta, Phi: Adversarial perturbations within Frobenius balls
    
    Args:
        det_ll: {intervention: deterministic_part} for LL
        det_hl: {intervention: deterministic_part} for HL  
        U_ll_hat: LL noise (N_ll, d_l)
        U_hl_hat: HL noise (N_hl, d_h)
        omega: intervention mapping
        saved_folds: list of train/test splits
        epsilon: radius for Theta perturbations
        delta: radius for Phi perturbations
        eta_min: learning rate for T (minimization)
        eta_max: learning rate for Theta, Phi (maximization)
        num_steps_min: steps for T minimization
        num_steps_max: steps for Theta, Phi maximization
        max_iter: max iterations
        tol: convergence tolerance
        seed: random seed
        initialization: 'zeros' or 'random' for perturbations
        gain: Xavier initialization gain for T
        optimizers: 'adam' or 'adam_betas'
    """
    rng = np.random.default_rng(seed)

    # Map interventions to CG values (keys are Intervention or None). We'll infer CG from data content.
    # We build masks from observed Dll_obs/Dhl_obs using first column (CG) to index U slices consistent with det_* dicts.
    def _iv_key_to_cg(iv):
        if iv is None:
            return None
        # Intervention is likely an object with vv() or dict-like repr; we try to extract CG value
        try:
            vv = iv.vv()  # type: ignore
            return vv.get('CG', None)
        except Exception:
            return None

    # Build per-CG index arrays for LL and HL obs matrices
    ll_idx_by_cg = _build_intervention_indices_by_cg(Dll_obs, {})
    hl_idx_by_cg = _build_intervention_indices_by_cg(Dhl_obs, {})

    results: dict = {}
    d_l = Dll_obs.shape[1]
    d_h = Dhl_obs.shape[1]

    for fold_id, fold in enumerate(saved_folds):
        train_idx_h = set(fold["train"].tolist())

        # Build concatenated X (LL) and Y (HL) over all interventions, restricted to train indices
        X_chunks, Y_chunks = [], []
        # Also keep per-bucket shapes for perturbation constraints
        bucket_slices = []  # list of (slice_ll, slice_hl)

        for iv, D_l in det_ll.items():
            if iv is None:
                continue
            cg_ll = _iv_key_to_cg(iv)
            if cg_ll is None:
                continue
            # map to omega(iv)
            iv_h = omega.get(iv, None)
            cg_h = _iv_key_to_cg(iv_h)
            if cg_h is None:
                continue

            idx_ll = ll_idx_by_cg.get(cg_ll, np.array([], dtype=int))
            idx_hl = hl_idx_by_cg.get(cg_h, np.array([], dtype=int))
            # restrict HL indices to train fold
            idx_hl = np.array([i for i in idx_hl if i in train_idx_h], dtype=int)
            if len(idx_ll) == 0 or len(idx_hl) == 0:
                continue

            # Align counts by min length to allow concatenation
            n = min(len(idx_ll), len(idx_hl))
            idx_ll = idx_ll[:n]
            idx_hl = idx_hl[:n]

            X_block = D_l[:n, :] + U_ll_hat[idx_ll, :]
            D_h = det_hl.get(iv_h, None)
            if D_h is None:
                continue
            Y_block = D_h[:n, :] + U_hl_hat[idx_hl, :]

            bucket_slices.append((n, n))
            X_chunks.append(X_block)
            Y_chunks.append(Y_block)

        if len(X_chunks) == 0:
            logging.warning("No data buckets found for fold %s; skipping.", fold_id)
            continue

        X_all = np.vstack(X_chunks)  # (N_total, d_l)
        Y_all = np.vstack(Y_chunks)  # (N_total, d_h)

        # Initialize T by ridge least squares
        # T solves min ||T X - Y||_F^2 + ridge ||T||_F^2 → closed form: T = Y X^T (X X^T + ridge I)^{-1}
        Xt = X_all.T
        G = Xt @ X_all + ridge * np.eye(d_l)
        T = (Y_all.T @ X_all) @ np.linalg.pinv(G)

        # Initialize perturbations
        Theta = np.zeros_like(X_all)
        Phi = np.zeros_like(Y_all)

        # Projector onto Frobenius ball per bucket
        def proj_bucket(Z: np.ndarray, start: int, length: int, radius: float):
            seg = Z[start:start+length]
            nrm = np.linalg.norm(seg, ord='fro')
            cap = radius * (length ** 0.5)
            if nrm > cap and nrm > 0:
                Z[start:start+length] = seg * (cap / nrm)

        # Gradient steps
        offset = 0
        bucket_offsets = []
        for n_l, _ in bucket_slices:
            bucket_offsets.append(offset)
            offset += n_l

        for _ in range(max_iters):
            # Update T by ridge LS for current perturbations
            Xp = X_all + Theta
            Yp = Y_all + Phi
            Xt = Xp.T
            G = Xt @ Xp + ridge * np.eye(d_l)
            T = (Yp.T @ Xp) @ np.linalg.pinv(G)

            # Residuals
            R = (T @ Xp.T).T - Yp  # (N_total, d_h)

            # Gradient w.r.t Theta and Phi (least-squares gradients)
            # d/dTheta: 2 X^T T^T (T X - Y) in sample form → we use matrix form below
            grad_Theta = (R @ T).astype(float)  # (N_total, d_l)
            grad_Phi = (-R).astype(float)       # (N_total, d_h)

            Theta -= lr * grad_Theta
            Phi   -= lr * grad_Phi

            # Project per bucket
            for b, (n_l, _) in enumerate(bucket_slices):
                s = bucket_offsets[b]
                proj_bucket(Theta, s, n_l, rho_ll)
                proj_bucket(Phi,   s, n_l, rho_hl)

        results[f"seed_{fold_id}"] = {"T": T, "train_N": X_all.shape[0]}

    return results


# ---------- per-method runners (same signatures as your original) ----------
def run_diroca_empirical_optimization(all_data, saved_folds, U_ll_hat, U_hl_hat, hyperparams_diroca, output_path):
    logger.info("Starting DiRoCA empirical optimization (affine ANM objective)...")

    # Load precomputed deterministic parts from model bundles (preferred)
    det_ll = all_data["LLmodel"].get("deterministic") or compute_deterministic_parts_per_intervention(all_data["LLmodel"])  # {iota: D_ll}
    det_hl = all_data["HLmodel"].get("deterministic") or compute_deterministic_parts_per_intervention(all_data["HLmodel"])  # {eta:  D_hl}

    # Radii from config or defaults
    rho_ll = float(hyperparams_diroca.get("rho_ll", 1.0))
    rho_hl = float(hyperparams_diroca.get("rho_hl", 1.0))
    max_iters = int(hyperparams_diroca.get("max_iters", 200))
    lr = float(hyperparams_diroca.get("lr", 1e-2))
    ridge = float(hyperparams_diroca.get("ridge", 1e-6))
    seed = int(hyperparams_diroca.get("seed", 42))

    omega = all_data["abstraction_data"]["omega"] if "abstraction_data" in all_data else all_data["omega"]

    # Endogenous aligned data (observational) to build per-CG masks
    Dll_obs = all_data["LLmodel"]["data"][None]
    Dhl_obs = all_data["HLmodel"]["data"][None]

    # Optimize T per split (returns a T per seed)
    results = optimize_affine_map(det_ll, det_hl,
                                  Dll_obs, Dhl_obs,
                                  U_ll_hat, U_hl_hat,
                                  omega, saved_folds,
                                  rho_ll=rho_ll, rho_hl=rho_hl,
                                  max_iters=max_iters, lr=lr, ridge=ridge, seed=seed)

    joblib.dump(results, output_path)
    logger.info(f"✓ DiRoCA affine empirical results saved to {output_path}")
    return results


def run_gradca_empirical_optimization(all_data, saved_folds, U_ll_hat, U_hl_hat, hyperparams_gradca, output_path):
    logger.info("Starting GRADCA empirical optimization (affine ANM objective)...")

    # Use precomputed deterministic parts if present
    det_ll = all_data["LLmodel"].get("deterministic") or compute_deterministic_parts_per_intervention(all_data["LLmodel"])  # {iota: D_ll}
    det_hl = all_data["HLmodel"].get("deterministic") or compute_deterministic_parts_per_intervention(all_data["HLmodel"])  # {eta:  D_hl}

    rho_ll = float(hyperparams_gradca.get("rho_ll", 1.0))
    rho_hl = float(hyperparams_gradca.get("rho_hl", 1.0))
    max_iters = int(hyperparams_gradca.get("max_iters", 200))
    lr = float(hyperparams_gradca.get("lr", 1e-2))
    ridge = float(hyperparams_gradca.get("ridge", 1e-6))
    seed = int(hyperparams_gradca.get("seed", 42))

    omega = all_data["abstraction_data"]["omega"] if "abstraction_data" in all_data else all_data["omega"]

    Dll_obs = all_data["LLmodel"]["data"][None]
    Dhl_obs = all_data["HLmodel"]["data"][None]

    results = optimize_affine_map(det_ll, det_hl,
                                  Dll_obs, Dhl_obs,
                                  U_ll_hat, U_hl_hat,
                                  omega, saved_folds,
                                  rho_ll=rho_ll, rho_hl=rho_hl,
                                  max_iters=max_iters, lr=lr, ridge=ridge, seed=seed)

    joblib.dump(results, output_path)
    logger.info(f"✓ GRADCA affine empirical results saved to {output_path}")
    return results


def run_baryca_empirical_optimization(all_data, saved_folds, U_ll_hat, U_hl_hat, hyperparams_baryca, output_path):
    logger.warning("BARYCA path disabled here: structure-matrix baseline assumes X=MU. Skipping.")
    joblib.dump({"skipped": True}, output_path)
    return {"skipped": True}


def run_abslingam_optimization(all_data, saved_folds, Dll_obs, Dhl_obs, output_path):
    logger.info("Starting Abs-LiNGAM optimization...")
    abs_results = {}

    for i, fold in enumerate(saved_folds):
        fold_key = f"seed_{i}"
        tr = fold["train"]
        Dll_tr = Dll_obs[tr]
        Dhl_tr = Dhl_obs[tr]

        res = optools.run_abs_lingam_complete(Dll_tr, Dhl_tr)
        abs_results[fold_key] = {
            "Perfect": {"T_matrix": res["Perfect"]["T"].T, "test_indices": fold["test"]},
            "Noisy": {"T_matrix": res["Noisy"]["T"].T, "test_indices": fold["test"]},
        }

    joblib.dump(abs_results, output_path)
    logger.info(f"✓ Abs-LiNGAM results saved to {output_path}")
    return abs_results


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Battery 90/10 Empirical Optimization (multi-seed)")
    parser.add_argument("--experiment", type=str, default="battery", help="Experiment name (default: battery)")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test fraction (default: 0.1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="Seeds for repeated 90/10 splits (default: 42 43 44 45 46)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/{experiment}/results_empirical_9010)")
    parser.add_argument("--skip-diroca", action="store_true")
    parser.add_argument("--skip-gradca", action="store_true")
    parser.add_argument("--skip-baryca", action="store_true")
    parser.add_argument("--skip-abslingam", action="store_true")
    args = parser.parse_args()

    exp = args.experiment
    outdir = args.output_dir or f"data/{exp}/results_empirical_9010"
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"Output directory: {outdir}")

    # 1) Load configs (unchanged pattern)
    cfg_files = {
        "hyperparams_diroca": get_config_path("diroca_opt_config_empirical", exp),
        "hyperparams_gradca": get_config_path("gradca_opt_config_empirical", exp),
        "hyperparams_baryca": get_config_path("baryca_opt_config_empirical", exp),
    }
    configs = ut.load_configs(cfg_files)

    # 2) Load data
    all_data = ut.load_all_data(exp)
    all_data["experiment_name"] = exp

    Dll_obs = all_data["LLmodel"]["data"][None]  # endogenous obs (aligned)
    Dhl_obs = all_data["HLmodel"]["data"][None]
    U_ll_hat = all_data["LLmodel"]["noise"][None]  # exogenous obs (aligned)
    U_hl_hat = all_data["HLmodel"]["noise"][None]

    # 3) Build repeated stratified 90/10 splits (one "fold" per seed)
    saved_folds = make_stratified_9010_splits(Dhl_obs, args.test_size, args.seeds)
    logger.info(f"Built {len(saved_folds)} stratified 90/10 splits over seeds: {args.seeds}")

    # 4) Run optimizations (same interfaces)
    results = {}

    if not args.skip_diroca:
        diroca_path = os.path.join(outdir, "diroca_empirical_splits.pkl")
        results["diroca"] = run_diroca_empirical_optimization(
            all_data, saved_folds, U_ll_hat, U_hl_hat, configs["hyperparams_diroca"], diroca_path
        )

    if not args.skip_gradca:
        gradca_path = os.path.join(outdir, "gradca_empirical_splits.pkl")
        results["gradca"] = run_gradca_empirical_optimization(
            all_data, saved_folds, U_ll_hat, U_hl_hat, configs["hyperparams_gradca"], gradca_path
        )

    if not args.skip_baryca:
        baryca_path = os.path.join(outdir, "baryca_empirical_splits.pkl")
        results["baryca"] = run_baryca_empirical_optimization(
            all_data, saved_folds, U_ll_hat, U_hl_hat, configs["hyperparams_baryca"], baryca_path
        )

    if not args.skip_abslingam:
        abs_path = os.path.join(outdir, "abslingam_empirical_splits.pkl")
        results["abslingam"] = run_abslingam_optimization(
            all_data, saved_folds, Dll_obs, Dhl_obs, abs_path
        )

    logger.info("All 90/10 empirical optimizations completed successfully!")
    print("\n" + "=" * 60)
    print("BATTERY 90/10 EMPIRICAL OPTIMIZATION — COMPLETED")
    print("=" * 60)
    print(f"Experiment: {exp}")
    print(f"Output dir: {outdir}")
    for k in ["diroca", "gradca", "baryca", "abslingam"]:
        if k in results:
            print(f"  - {k}: saved")
    print("=" * 60)


if __name__ == "__main__":
    main()
