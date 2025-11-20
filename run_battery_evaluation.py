#!/usr/bin/env python3
"""
Battery empirical evaluation script.

Usage example:
    python run_battery_evaluation.py \
        --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
        --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
        --trials 20 --distribution gaussian
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm


# ----------------------------------------------------------------------
# Helpers: fold reconstruction + T orientation + contamination + error
# ----------------------------------------------------------------------

def make_stratified_9010_splits(Dhl_obs: np.ndarray,
                                test_size: float,
                                seeds: List[int]) -> List[dict]:
    """Rebuild the same stratified 90/10 splits used in optimization."""
    assert Dhl_obs.ndim == 2 and 0.0 < test_size < 1.0
    N = Dhl_obs.shape[0]
    y = Dhl_obs[:, 0]  # CG value for stratification

    label_to_indices: Dict[Any, List[int]] = {}
    for idx, lab in enumerate(y):
        label_to_indices.setdefault(lab, []).append(idx)

    folds = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test_idx = []
        for _, idxs in label_to_indices.items():
            n = len(idxs)
            n_test = max(1, int(round(test_size * n)))
            chosen = rng.choice(idxs, size=n_test, replace=False)
            test_idx.append(chosen)
        test_idx = np.unique(np.concatenate(test_idx))
        train_idx = np.setdiff1d(np.arange(N), test_idx)
        folds.append({"train": train_idx, "test": test_idx, "seed": seed})
    return folds


def coerce_T(T, d_h: int, d_l: int) -> np.ndarray:
    """Ensure T matrix has correct orientation (d_h, d_l)."""
    T = np.asarray(T)
    if T.shape == (d_h, d_l):
        return T
    if T.shape == (d_l, d_h):
        return T.T
    raise ValueError(f"Unexpected T shape {T.shape}, expected {(d_h, d_l)} or {(d_l, d_h)}")


def apply_huber_contamination_battery(
    clean_data,
    alpha: float,
    noise_scale: float,
    noise_dims,
    distribution: str,
    seed: int = None,
) -> np.ndarray:
    """
    Huber contamination on selected dimensions with *additive* noise.

    Args:
        clean_data: array-like (n_samples, n_features)
        alpha: fraction of rows to replace with noisy version (0..1)
        noise_scale: scale parameter for the noise
        noise_dims: indices/slice of dimensions to contaminate (e.g. slice(1, d))
        distribution: 'gaussian' | 'student-t' | 'exponential'
        seed: RNG seed

    Returns:
        contaminated_data: np.ndarray, same shape as input
    """
    data = np.asarray(clean_data, dtype=np.float32).copy()
    if alpha <= 0 or noise_scale == 0.0:
        return data

    rng = np.random.default_rng(seed)
    sub = data[:, noise_dims]  # view on selected dims
    shape = sub.shape

    # Sample noise
    if distribution == "gaussian":
        noise = rng.normal(loc=0.0, scale=noise_scale, size=shape).astype(np.float32)
    elif distribution == "student-t":
        df = 3.0
        # standard t, scaled by noise_scale
        noise = (rng.standard_t(df=df, size=shape) * noise_scale).astype(np.float32)
    elif distribution == "exponential":
        # strictly positive shift; mimics the old "exponential" contamination
        noise = rng.exponential(scale=noise_scale, size=shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    noisy_sub = sub + noise

    if alpha >= 1.0:
        data[:, noise_dims] = noisy_sub
        return data

    n_samples = data.shape[0]
    n_contaminate = int(alpha * n_samples)
    if n_contaminate <= 0:
        return data

    idx_to_contaminate = rng.choice(n_samples, size=n_contaminate, replace=False)
    data[idx_to_contaminate, noise_dims] = noisy_sub[idx_to_contaminate]
    return data


def calculate_empirical_error_battery(T_matrix, Dll_test, Dhl_test) -> float:
    """
    Frobenius error of abstraction on battery data:
        || Dll_test * T^T - Dhl_test ||_F^2 / (n * d)
    """
    try:
        T_matrix = T_matrix if isinstance(T_matrix, torch.Tensor) else torch.tensor(T_matrix, dtype=torch.float32)
        Dll_test = Dll_test if isinstance(Dll_test, torch.Tensor) else torch.tensor(Dll_test, dtype=torch.float32)
        Dhl_test = Dhl_test if isinstance(Dhl_test, torch.Tensor) else torch.tensor(Dhl_test, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T_matrix = T_matrix.to(device)
        Dll_test = Dll_test.to(device)
        Dhl_test = Dhl_test.to(device)

        if T_matrix.shape[1] != Dll_test.shape[1]:
            print(f"Dimension mismatch T vs LL: T={T_matrix.shape}, LL={Dll_test.shape}")
            return float("inf")
        if T_matrix.shape[0] != Dhl_test.shape[1]:
            print(f"Dimension mismatch T vs HL: T={T_matrix.shape}, HL={Dhl_test.shape}")
            return float("inf")

        with torch.no_grad():
            Dhl_pred = Dll_test @ T_matrix.T
            diff = Dhl_pred - Dhl_test
            err = torch.norm(diff, p="fro") ** 2 / (diff.shape[0] * diff.shape[1])
        return float(err.item())
    except Exception as e:
        print(f"Error in calculate_empirical_error_battery: {e}")
        return float("inf")


# ----------------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------------

def run_battery_evaluation(
    alpha_min: float,
    alpha_max: float,
    alpha_steps: int,
    noise_min: float,
    noise_max: float,
    noise_steps: int,
    trials: int,
    distribution: str,
    output: str = None,
):
    base_dir = Path("data/battery")
    battery_data_path = base_dir / "LLmodel.pkl"
    hl_data_path = base_dir / "HLmodel.pkl"
    abstraction_data_path = base_dir / "abstraction_data.pkl"

    # ----- Load data -----
    print("Loading battery data...")
    ll_data = joblib.load(battery_data_path)
    hl_data = joblib.load(hl_data_path)
    abstraction_data = joblib.load(abstraction_data_path)

    Dll_samples = ll_data["data"]          # Dict[intervention, np.ndarray]
    Dhl_samples = hl_data["data"]          # Dict[intervention, np.ndarray]
    row_idx_ll  = ll_data["row_idx"]       # Dict[intervention, np.ndarray of global ids]
    row_idx_hl  = hl_data["row_idx"]
    omega       = abstraction_data["omega"]

    print("✅ Battery data loaded.")
    print(f"LL interventions: {list(Dll_samples.keys())}")
    print(f"HL interventions: {list(Dhl_samples.keys())}")
    print(f"Omega mapping: {omega}")

    # ----- Rebuild 90/10 folds (same as optimization) -----
    Dhl_obs_full = hl_data["data"][None]  # observational HL (N, d_h)
    folds = make_stratified_9010_splits(
        Dhl_obs_full, test_size=0.1, seeds=[42, 43, 44, 45, 46]
    )
    print(f"✅ Rebuilt {len(folds)} folds.")
    print(f"Test sizes per fold: {[len(f['test']) for f in folds]}")

    # ----- Load optimization results -----
    results_dir = base_dir / "results_empirical_9010"
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pkl")]
    print(f"\nFound {len(result_files)} result files in {results_dir}:")
    for f in result_files:
        print("  -", f)

    all_results: Dict[str, Any] = {}

    # DiRoCA: keys epsilon_..._delta_...
    if "diroca_cv_results_empirical.pkl" in result_files:
        diroca_results = joblib.load(results_dir / "diroca_cv_results_empirical.pkl")
        print(f"✅ DiRoCA results loaded: {len(diroca_results)} epsilon configs")
        all_results.update(diroca_results)

    # GradCA
    if "gradca_cv_results_empirical.pkl" in result_files:
        gradca_results = joblib.load(results_dir / "gradca_cv_results_empirical.pkl")
        print(f"✅ GradCA results loaded: {len(gradca_results)} folds")
        all_results["gradca"] = gradca_results

    # BaryCA
    if "baryca_cv_results_empirical.pkl" in result_files:
        baryca_results = joblib.load(results_dir / "baryca_cv_results_empirical.pkl")
        print(f"✅ BaryCA results loaded: {len(baryca_results)} folds")
        all_results["baryca"] = baryca_results

    # Abs-LiNGAM
    if "abslingam_cv_results_empirical.pkl" in result_files:
        abslingam_results = joblib.load(results_dir / "abslingam_cv_results_empirical.pkl")
        print(f"✅ Abs-LiNGAM results loaded: {len(abslingam_results)} folds")
        all_results["abslingam"] = abslingam_results

    print("\nCombined result keys:", list(all_results.keys()))

    if not all_results:
        raise RuntimeError("No optimization results found; cannot run evaluation.")

    # ----- Dimensions and noise dims -----
    d_h = hl_data["data"][None].shape[1]
    d_l = ll_data["data"][None].shape[1]
    print(f"\nExpected T shape: ({d_h}, {d_l})")

    # Choose noise dims: all except first (CG)
    sample_ll = next(iter(Dll_samples.values()))
    sample_hl = next(iter(Dhl_samples.values()))
    n_features_ll = sample_ll.shape[1]
    n_features_hl = sample_hl.shape[1]
    LL_NOISE_DIMS = slice(1, n_features_ll)
    HL_NOISE_DIMS = slice(1, n_features_hl)
    print(f"LL noise dims: {LL_NOISE_DIMS}, HL noise dims: {HL_NOISE_DIMS}")

    # ----- Grids -----
    alpha_values = np.linspace(alpha_min, alpha_max, alpha_steps)
    noise_values = np.linspace(noise_min, noise_max, noise_steps)

    print("\nEvaluation configuration:")
    print(f"  - distribution: {distribution}")
    print(f"  - alpha: {alpha_steps} points from {alpha_values[0]:.2f} to {alpha_values[-1]:.2f}")
    print(f"  - noise: {noise_steps} points from {noise_values[0]:.2f} to {noise_values[-1]:.2f}")
    print(f"  - trials: {trials}")

    # ----- Count total configs for progress bar -----
    base_runs = 0
    for method_group_key, method_results_inner in all_results.items():
        if not isinstance(method_results_inner, dict):
            continue
        for fold_key, fold_data in method_results_inner.items():
            if not str(fold_key).startswith("fold_"):
                continue
            if method_group_key.startswith("epsilon_"):  # DiRoCA
                base_runs += 1
            elif method_group_key in ["gradca", "baryca"]:
                base_runs += 1
            elif method_group_key == "abslingam":
                base_runs += len(fold_data)  # Perfect + Noisy

    total_configs = (
        base_runs * len(alpha_values) * len(noise_values) * trials
    )
    print(f"\nApprox. total evaluation configs: {total_configs}")

    # ----- Main evaluation loop -----
    evaluation_records = []

    if total_configs == 0:
        print("❌ No valid training results found. Aborting.")
        return pd.DataFrame()

    pbar = tqdm(total=total_configs, desc="Evaluating Battery Methods")

    for alpha in alpha_values:
        for noise_scale in noise_values:
            for trial in range(trials):

                for method_group_key, method_results_inner in all_results.items():
                    if not isinstance(method_results_inner, dict):
                        continue

                    for fold_key, fold_data in method_results_inner.items():
                        if not str(fold_key).startswith("fold_"):
                            continue

                        # Fold index & global test indices
                        fold_idx = int(str(fold_key).split("_")[-1])
                        global_test_idx = set(folds[fold_idx]["test"])

                        # ---- DiRoCA (epsilon_* dicts at top level) ----
                        if method_group_key.startswith("epsilon_"):
                            run_result = fold_data
                            run_key = method_group_key
                            eval_method_name = f"DiRoCA ({run_key})"

                            if "T" not in run_result:
                                pbar.update(1)
                                continue

                            T_matrix = coerce_T(run_result["T"], d_h, d_l)
                            trial_errors = []

                            for iota, eta in list(omega.items()):
                                try:
                                    if iota not in Dll_samples or eta not in Dhl_samples:
                                        continue

                                    ll_bucket_idx = row_idx_ll.get(iota, None)
                                    hl_bucket_idx = row_idx_hl.get(eta, None)
                                    if ll_bucket_idx is None or hl_bucket_idx is None:
                                        continue

                                    ll_test_mask = np.array(
                                        [g in global_test_idx for g in ll_bucket_idx],
                                        dtype=bool,
                                    )
                                    hl_test_mask = np.array(
                                        [g in global_test_idx for g in hl_bucket_idx],
                                        dtype=bool,
                                    )
                                    if not ll_test_mask.any() or not hl_test_mask.any():
                                        continue

                                    Dll_test_clean = Dll_samples[iota][ll_test_mask]
                                    Dhl_test_clean = Dhl_samples[eta][hl_test_mask]

                                    ll_ids = ll_bucket_idx[ll_test_mask]
                                    hl_ids = hl_bucket_idx[hl_test_mask]
                                    common = np.intersect1d(ll_ids, hl_ids)
                                    if common.size == 0:
                                        continue

                                    pos_l = {g: p for p, g in enumerate(ll_ids)}
                                    pos_h = {g: p for p, g in enumerate(hl_ids)}
                                    sel_l = np.array([pos_l[g] for g in common], dtype=int)
                                    sel_h = np.array([pos_h[g] for g in common], dtype=int)

                                    Dll_test_clean = Dll_test_clean[sel_l]
                                    Dhl_test_clean = Dhl_test_clean[sel_h]

                                    seed = hash(
                                        (
                                            fold_idx,
                                            run_key,
                                            float(alpha),
                                            float(noise_scale),
                                            trial,
                                            str(iota),
                                        )
                                    ) % (2**32)

                                    Dll_test_cont = apply_huber_contamination_battery(
                                        Dll_test_clean,
                                        alpha=float(alpha),
                                        noise_scale=float(noise_scale),
                                        noise_dims=LL_NOISE_DIMS,
                                        distribution=distribution,
                                        seed=seed,
                                    )
                                    Dhl_test_cont = apply_huber_contamination_battery(
                                        Dhl_test_clean,
                                        alpha=float(alpha),
                                        noise_scale=float(noise_scale),
                                        noise_dims=HL_NOISE_DIMS,
                                        distribution=distribution,
                                        seed=seed,
                                    )

                                    error = calculate_empirical_error_battery(
                                        T_matrix, Dll_test_cont, Dhl_test_cont
                                    )
                                    if not np.isnan(error) and error != float("inf"):
                                        trial_errors.append(error)

                                except Exception as e:
                                    print(
                                        f"ERROR: {e} | Context: M={eval_method_name}, "
                                        f"F={fold_idx}, R={run_key}, A={alpha}, "
                                        f"S={noise_scale}, T={trial}, Iota={iota}"
                                    )
                                    trial_errors.append(np.nan)

                            record = {
                                "method": eval_method_name,
                                "fold": fold_idx,
                                "alpha": float(alpha),
                                "noise_scale": float(noise_scale),
                                "trial": trial,
                                "error": float(np.nanmean(trial_errors))
                                if trial_errors
                                else np.nan,
                            }
                            if "epsilon" in run_result:
                                record["train_epsilon"] = run_result.get("epsilon", np.nan)
                                record["train_delta"] = run_result.get("delta", np.nan)

                            evaluation_records.append(record)
                            pbar.update(1)

                        # ---- GradCA / BaryCA ----
                        elif method_group_key in ["gradca", "baryca"]:
                            run_result = fold_data
                            run_key = f"{method_group_key}_run"
                            eval_method_name = method_group_key.title()

                            if "T" not in run_result:
                                pbar.update(1)
                                continue

                            T_matrix = coerce_T(run_result["T"], d_h, d_l)
                            trial_errors = []

                            for iota, eta in list(omega.items()):
                                try:
                                    if iota not in Dll_samples or eta not in Dhl_samples:
                                        continue

                                    ll_bucket_idx = row_idx_ll.get(iota, None)
                                    hl_bucket_idx = row_idx_hl.get(eta, None)
                                    if ll_bucket_idx is None or hl_bucket_idx is None:
                                        continue

                                    ll_test_mask = np.array(
                                        [g in global_test_idx for g in ll_bucket_idx],
                                        dtype=bool,
                                    )
                                    hl_test_mask = np.array(
                                        [g in global_test_idx for g in hl_bucket_idx],
                                        dtype=bool,
                                    )
                                    if not ll_test_mask.any() or not hl_test_mask.any():
                                        continue

                                    Dll_test_clean = Dll_samples[iota][ll_test_mask]
                                    Dhl_test_clean = Dhl_samples[eta][hl_test_mask]

                                    ll_ids = ll_bucket_idx[ll_test_mask]
                                    hl_ids = hl_bucket_idx[hl_test_mask]
                                    common = np.intersect1d(ll_ids, hl_ids)
                                    if common.size == 0:
                                        continue

                                    pos_l = {g: p for p, g in enumerate(ll_ids)}
                                    pos_h = {g: p for p, g in enumerate(hl_ids)}
                                    sel_l = np.array([pos_l[g] for g in common], dtype=int)
                                    sel_h = np.array([pos_h[g] for g in common], dtype=int)

                                    Dll_test_clean = Dll_test_clean[sel_l]
                                    Dhl_test_clean = Dhl_test_clean[sel_h]

                                    seed = hash(
                                        (
                                            fold_idx,
                                            run_key,
                                            float(alpha),
                                            float(noise_scale),
                                            trial,
                                            str(iota),
                                        )
                                    ) % (2**32)

                                    Dll_test_cont = apply_huber_contamination_battery(
                                        Dll_test_clean,
                                        alpha=float(alpha),
                                        noise_scale=float(noise_scale),
                                        noise_dims=LL_NOISE_DIMS,
                                        distribution=distribution,
                                        seed=seed,
                                    )
                                    Dhl_test_cont = apply_huber_contamination_battery(
                                        Dhl_test_clean,
                                        alpha=float(alpha),
                                        noise_scale=float(noise_scale),
                                        noise_dims=HL_NOISE_DIMS,
                                        distribution=distribution,
                                        seed=seed,
                                    )

                                    error = calculate_empirical_error_battery(
                                        T_matrix, Dll_test_cont, Dhl_test_cont
                                    )
                                    if not np.isnan(error) and error != float("inf"):
                                        trial_errors.append(error)

                                except Exception as e:
                                    print(
                                        f"ERROR: {e} | Context: M={eval_method_name}, "
                                        f"F={fold_idx}, R={run_key}, A={alpha}, "
                                        f"S={noise_scale}, T={trial}, Iota={iota}"
                                    )
                                    trial_errors.append(np.nan)

                            record = {
                                "method": eval_method_name,
                                "fold": fold_idx,
                                "alpha": float(alpha),
                                "noise_scale": float(noise_scale),
                                "trial": trial,
                                "error": float(np.nanmean(trial_errors))
                                if trial_errors
                                else np.nan,
                            }
                            evaluation_records.append(record)
                            pbar.update(1)

                        # ---- Abs-LiNGAM ----
                        elif method_group_key == "abslingam":
                            # fold_data: dict[style -> run_result]
                            for run_key, run_result in fold_data.items():
                                if "T" not in run_result:
                                    pbar.update(1)
                                    continue

                                eval_method_name = f"Abs-LiNGAM ({run_key})"
                                T_matrix = coerce_T(run_result["T"], d_h, d_l)
                                trial_errors = []

                                for iota, eta in list(omega.items()):
                                    try:
                                        if iota not in Dll_samples or eta not in Dhl_samples:
                                            continue

                                        ll_bucket_idx = row_idx_ll.get(iota, None)
                                        hl_bucket_idx = row_idx_hl.get(eta, None)
                                        if ll_bucket_idx is None or hl_bucket_idx is None:
                                            continue

                                        ll_test_mask = np.array(
                                            [g in global_test_idx for g in ll_bucket_idx],
                                            dtype=bool,
                                        )
                                        hl_test_mask = np.array(
                                            [g in global_test_idx for g in hl_bucket_idx],
                                            dtype=bool,
                                        )
                                        if not ll_test_mask.any() or not hl_test_mask.any():
                                            continue

                                        Dll_test_clean = Dll_samples[iota][ll_test_mask]
                                        Dhl_test_clean = Dhl_samples[eta][hl_test_mask]

                                        ll_ids = ll_bucket_idx[ll_test_mask]
                                        hl_ids = hl_bucket_idx[hl_test_mask]
                                        common = np.intersect1d(ll_ids, hl_ids)
                                        if common.size == 0:
                                            continue

                                        pos_l = {g: p for p, g in enumerate(ll_ids)}
                                        pos_h = {g: p for p, g in enumerate(hl_ids)}
                                        sel_l = np.array(
                                            [pos_l[g] for g in common], dtype=int
                                        )
                                        sel_h = np.array(
                                            [pos_h[g] for g in common], dtype=int
                                        )

                                        Dll_test_clean = Dll_test_clean[sel_l]
                                        Dhl_test_clean = Dhl_test_clean[sel_h]

                                        seed = hash(
                                            (
                                                fold_idx,
                                                run_key,
                                                float(alpha),
                                                float(noise_scale),
                                                trial,
                                                str(iota),
                                            )
                                        ) % (2**32)

                                        Dll_test_cont = apply_huber_contamination_battery(
                                            Dll_test_clean,
                                            alpha=float(alpha),
                                            noise_scale=float(noise_scale),
                                            noise_dims=LL_NOISE_DIMS,
                                            distribution=distribution,
                                            seed=seed,
                                        )
                                        Dhl_test_cont = apply_huber_contamination_battery(
                                            Dhl_test_clean,
                                            alpha=float(alpha),
                                            noise_scale=float(noise_scale),
                                            noise_dims=HL_NOISE_DIMS,
                                            distribution=distribution,
                                            seed=seed,
                                        )

                                        error = calculate_empirical_error_battery(
                                            T_matrix, Dll_test_cont, Dhl_test_cont
                                        )
                                        if not np.isnan(error) and error != float("inf"):
                                            trial_errors.append(error)

                                    except Exception as e:
                                        print(
                                            f"ERROR: {e} | Context: M={eval_method_name}, "
                                            f"F={fold_idx}, R={run_key}, A={alpha}, "
                                            f"S={noise_scale}, T={trial}, Iota={iota}"
                                        )
                                        trial_errors.append(np.nan)

                                record = {
                                    "method": eval_method_name,
                                    "fold": fold_idx,
                                    "alpha": float(alpha),
                                    "noise_scale": float(noise_scale),
                                    "trial": trial,
                                    "error": float(np.nanmean(trial_errors))
                                    if trial_errors
                                    else np.nan,
                                }
                                evaluation_records.append(record)
                                pbar.update(1)

    pbar.close()

    full_results_df = pd.DataFrame(evaluation_records)

    # ----- Save -----
    output_dir = base_dir / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            f"battery_eval_additive_{distribution}_"
            f"alpha{alpha_steps}-{alpha_min:.2f}-{alpha_max:.2f}_"
            f"noise{noise_steps}-{noise_min:.2f}-{noise_max:.2f}_"
            f"trials{trials}_{timestamp}.pkl"
        )
        output_path = output_dir / fname
    else:
        out_path = Path(output)
        # force into evaluation_results dir
        output_path = output_dir / out_path.name

    full_results_df.to_pickle(output_path)
    print(f"\n✅ Evaluation results saved to: {output_path}")
    print(f"Total records: {len(full_results_df)}")
    print("Methods:", sorted(full_results_df["method"].unique()))
    return full_results_df


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Battery empirical evaluation (additive noise, various distributions)."
    )

    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_steps", type=int, default=10)

    parser.add_argument("--noise_min", type=float, default=0.0)
    parser.add_argument("--noise_max", type=float, default=10.0)
    parser.add_argument("--noise_steps", type=int, default=20)

    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--distribution",
        type=str,
        default="gaussian",
        choices=["gaussian", "student-t", "exponential"],
        help="Contamination distribution (additive Huber-style).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output filename (will be placed in data/battery/evaluation_results).",
    )

    args = parser.parse_args()

    run_battery_evaluation(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_steps=args.alpha_steps,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        noise_steps=args.noise_steps,
        trials=args.trials,
        distribution=args.distribution,
        output=args.output,
    )


if __name__ == "__main__":
    main()
