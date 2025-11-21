#!/usr/bin/env python3
"""
General empirical evaluation (nonlinear-ready).

Default target: LUCAS nonlinear pack + results_nonlinear outputs.
But structured so you can reuse it for future datasets that follow the same
"pack + results folder" convention.

Usage:
  python run_general_evaluation.py --experiment lucas
  python run_general_evaluation.py --experiment lucas --alpha_steps 6 --noise_steps 10 --trials 5
  python run_general_evaluation.py --experiment lucas --distribution student-t --shift_type multiplicative

Outputs:
  data/{experiment}/evaluation_results/
    empirical_evaluation_{...}.csv
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys
from tqdm import tqdm
import warnings
import scipy.stats as stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Contamination helpers (same logic as your old script)
# ---------------------------------------------------------------------

def apply_shift(clean_data, shift_config, all_var_names, model_level, seed=None):
    """
    Applies a specified contamination, using a dedicated RNG for reproducibility.
    """
    rng = np.random.default_rng(seed)

    shift_type = shift_config.get('type')
    dist_type = shift_config.get('distribution', 'gaussian')
    n_samples, n_dims = clean_data.shape

    level_key = 'll_params' if model_level == 'L' else 'hl_params'
    params = shift_config.get(level_key, {})

    noise_matrix = np.zeros_like(clean_data)

    if dist_type == 'gaussian':
        mu = np.array(params.get('mu', np.zeros(n_dims)))
        sigma_def = params.get('sigma', np.eye(n_dims))
        sigma = np.diag(np.array(sigma_def)) if np.array(sigma_def).ndim == 1 else np.array(sigma_def)
        noise_matrix = rng.multivariate_normal(mean=mu, cov=sigma, size=n_samples)

    elif dist_type == 'student-t':
        df = params.get('df', 3)
        loc = np.array(params.get('loc', np.zeros(n_dims)))
        shape_def = params.get('shape', np.eye(n_dims))
        shape = np.diag(np.array(shape_def)) if np.array(shape_def).ndim == 1 else np.array(shape_def)
        noise_matrix = stats.multivariate_t.rvs(loc=loc, shape=shape, df=df, size=n_samples, random_state=rng)

    elif dist_type == 'exponential':
        scale = params.get('scale', 1.0)
        noise_matrix = rng.exponential(scale=scale, size=(n_samples, n_dims))

    elif dist_type == 'translation':
        c = params.get('c', 0.5)
        noise_matrix = np.ones((n_samples, n_dims)) * c

    elif dist_type == 'scaling':
        c = params.get('c', 1.5)
        noise_matrix = np.ones((n_samples, n_dims)) * c

    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

    final_noise = np.zeros_like(clean_data)
    vars_to_affect = params.get('apply_to_vars')

    if vars_to_affect is None:
        final_noise = noise_matrix
    else:
        indices_to_affect = [
            all_var_names.index(var) for var in vars_to_affect if var in all_var_names
        ]
        final_noise[:, indices_to_affect] = noise_matrix[:, indices_to_affect]

    if shift_type == 'additive':
        return clean_data + final_noise
    elif shift_type == 'multiplicative':
        return clean_data * final_noise
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")


def apply_huber_contamination(clean_data, alpha, shift_config, all_var_names, model_level, seed=None):
    """
    Contaminates a dataset using a Huber mixture with a specific RNG seed.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if alpha == 0:
        return clean_data

    noisy_data = apply_shift(clean_data, shift_config, all_var_names, model_level, seed=seed)

    if alpha == 1:
        return noisy_data

    n_samples = clean_data.shape[0]
    n_to_contaminate = int(alpha * n_samples)

    rng = np.random.default_rng(seed)
    indices_to_replace = rng.choice(n_samples, n_to_contaminate, replace=False)

    contaminated = clean_data.copy()
    contaminated[indices_to_replace] = noisy_data[indices_to_replace]
    return contaminated


# ---------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------

def calculate_empirical_error(T_matrix, Xll_test, Xhl_test):
    """
    Frobenius abstraction error on endo data (X = D + U):
        || Xll_test @ T^T - Xhl_test ||_F^2 / (n * d_h)
    """
    if Xll_test.shape[0] == 0 or Xhl_test.shape[0] == 0:
        return np.nan
    try:
        Xhl_pred = Xll_test @ T_matrix.T
        diff = Xhl_pred - Xhl_test
        err = np.linalg.norm(diff, ord="fro")**2 / (diff.shape[0] * diff.shape[1])
        return float(err)
    except Exception as e:
        print(f"[WARN] empirical error failed: {e}")
        return np.nan


# ---------------------------------------------------------------------
# Loading packs + results (nonlinear LUCAS format)
# ---------------------------------------------------------------------

def load_pack(experiment):
    """
    Expected location for nonlinear datasets:
      data/{experiment}/{experiment}_pack.pkl
    For LUCAS: data/lucas/lucas_pack.pkl
    """
    pack_path = os.path.join("data", experiment, f"{experiment}_pack.pkl")
    if not os.path.exists(pack_path):
        # LUCAS special-case filename
        alt = os.path.join("data", experiment, "lucas_pack.pkl")
        if os.path.exists(alt):
            pack_path = alt
        else:
            raise FileNotFoundError(f"Could not find pack file at {pack_path} or {alt}")

    pack = joblib.load(pack_path)

    if "ll" not in pack or "hl" not in pack or "omega" not in pack:
        raise KeyError("Pack missing required keys among {'ll','hl','omega'}.")

    omega = pack["omega"]
    ll_data = pack["ll"]
    hl_data = pack["hl"]

    # observational sizes
    N = ll_data["iota0"]["X"].shape[0]
    d_l = ll_data["iota0"]["X"].shape[1]
    d_h = hl_data[omega.get("iota0", "eta0")]["X"].shape[1]

    return pack, omega, N, d_l, d_h, pack_path


def load_results(experiment):
    """
    Default nonlinear results directory:
      data/{experiment}/results_nonlinear/
    Fallbacks:
      results_empirical/
      results_empirical_9010/
    """
    base = os.path.join("data", experiment)
    candidates = [
        os.path.join(base, "results_nonlinear"),
        os.path.join(base, "results_empirical"),
        os.path.join(base, "results_empirical_9010"),
    ]
    results_dir = None
    for c in candidates:
        if os.path.isdir(c):
            results_dir = c
            break
    if results_dir is None:
        raise FileNotFoundError(f"No results dir found under {candidates}")

    def pkl(name):
        return os.path.join(results_dir, name)

    results_to_eval = {}

    # ---------- DiRoCA ----------
    diroca_path = pkl("diroca_cv_results_empirical.pkl")
    if os.path.exists(diroca_path):
        diroca_results = joblib.load(diroca_path)
        # explode into separate method entries per radius key
        if diroca_results:
            first_fold = next(iter(diroca_results.keys()))
            run_ids = list(diroca_results[first_fold].keys())
            for run_id in run_ids:
                mname = f"DiRoCA ({run_id})"
                sliced = {}
                for fold_key, fold_res in diroca_results.items():
                    if run_id in fold_res:
                        sliced[fold_key] = {run_id: fold_res[run_id]}
                results_to_eval[mname] = sliced

    # ---------- GradCA ----------
    gradca_path = pkl("gradca_cv_results_empirical.pkl")
    if os.path.exists(gradca_path):
        results_to_eval["GradCA"] = joblib.load(gradca_path)

    # ---------- BaryCA ----------
    baryca_path = pkl("baryca_cv_results_empirical.pkl")
    if os.path.exists(baryca_path):
        results_to_eval["BaryCA"] = joblib.load(baryca_path)

    # ---------- Abs-LiNGAM ----------
    abslingam_path = pkl("abslingam_cv_results_empirical.pkl")
    if os.path.exists(abslingam_path):
        abslingam_results = joblib.load(abslingam_path)
        if abslingam_results:
            first_fold = next(iter(abslingam_results.keys()))
            styles = list(abslingam_results[first_fold].keys())
            for style in styles:
                mname = f"Abs-LiNGAM ({style})"
                sliced = {}
                for fold_key, fold_res in abslingam_results.items():
                    if style in fold_res:
                        sliced[fold_key] = {style: fold_res[style]}
                results_to_eval[mname] = sliced

    if not results_to_eval:
        raise RuntimeError(f"No result files found in {results_dir}")

    return results_to_eval, results_dir


def rebuild_kfolds(N, k_folds=5, seed=23):
    """Rebuild the same KFold splits used in lucas optimization."""
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = []

    # manual KFold to avoid sklearn dependency in evaluation if you want lightweight
    fold_sizes = np.full(k_folds, N // k_folds, dtype=int)
    fold_sizes[: N % k_folds] += 1

    cur = 0
    for fs in fold_sizes:
        te = indices[cur : cur + fs]
        tr = np.setdiff1d(indices, te)
        folds.append({"train": tr, "test": te})
        cur += fs
    return folds


# ---------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------

def run_evaluation(
    experiment="lucas",
    alpha_values=None,
    noise_levels=None,
    num_trials=2,
    zero_mean=True,
    shift_type="additive",
    distribution="gaussian",
    k_folds=5,
    seed=23,
    output_file=None,
):

    if alpha_values is None:
        alpha_values = np.linspace(0, 1.0, 10)
    if noise_levels is None:
        noise_levels = np.linspace(0, 5.0, 20)

    print(f"[SETUP] Experiment: {experiment}")
    pack, omega, N, d_l, d_h, pack_path = load_pack(experiment)
    results_to_eval, results_dir = load_results(experiment)
    folds = rebuild_kfolds(N, k_folds=k_folds, seed=seed)

    print(f"[DATA] pack: {pack_path}")
    print(f"[DATA] results_dir: {results_dir}")
    print(f"[CV] k_folds={k_folds}, seed={seed}, N={N}")
    print(f"[CV] {len(folds)} folds, ~{len(folds[0]['train'])} train / fold")
    print(f"[METHODS] {list(results_to_eval.keys())}")

    # placeholder var names: allow "apply_to_vars" later if you want
    ll_var_names = [f"ll_{i}" for i in range(d_l)]
    hl_var_names = [f"hl_{i}" for i in range(d_h)]

    base_sigma_L = np.eye(d_l)
    base_sigma_H = np.eye(d_h)

    total_configs = (
        len(alpha_values) * len(noise_levels) * num_trials * len(folds) * len(results_to_eval)
    )
    print(f"[EVAL] Total configs: {total_configs:,}")

    records = []

    for alpha in tqdm(alpha_values, desc="Alpha Levels"):
        for scale in noise_levels:
            for trial in range(num_trials):
                for fi, fold in enumerate(folds):
                    te = fold["test"]

                    for method_name, res_dict in results_to_eval.items():
                        fold_key = f"fold_{fi}"
                        if fold_key not in res_dict:
                            continue

                        fold_results = res_dict[fold_key]
                        for run_key, run_data in fold_results.items():

                            T_learned = run_data["T_matrix"]
                            # normalize to numpy float
                            T_learned = np.asarray(T_learned, dtype=np.float32)

                            errors_per_iota = []
                            for iota, eta in omega.items():
                                if iota not in pack["ll"] or eta not in pack["hl"]:
                                    continue

                                Xll_clean = pack["ll"][iota]["X"][te]  # (n_test, d_l)
                                Xhl_clean = pack["hl"][eta]["X"][te]  # (n_test, d_h)

                                # shift config
                                if zero_mean:
                                    mu_L = np.zeros(d_l)
                                    mu_H = np.zeros(d_h)
                                else:
                                    mu_L = np.ones(d_l) * scale
                                    mu_H = np.ones(d_h) * scale

                                sigma_L = base_sigma_L * (scale**2)
                                sigma_H = base_sigma_H * (scale**2)

                                if distribution == "gaussian":
                                    shift_config = {
                                        "type": shift_type,
                                        "distribution": distribution,
                                        "ll_params": {"mu": mu_L, "sigma": sigma_L},
                                        "hl_params": {"mu": mu_H, "sigma": sigma_H},
                                    }
                                elif distribution == "exponential":
                                    shift_config = {
                                        "type": shift_type,
                                        "distribution": distribution,
                                        "ll_params": {"scale": scale},
                                        "hl_params": {"scale": scale},
                                    }
                                elif distribution == "student-t":
                                    shift_config = {
                                        "type": shift_type,
                                        "distribution": distribution,
                                        "ll_params": {
                                            "df": 3,
                                            "loc": np.zeros(d_l),
                                            "shape": base_sigma_L * (scale**2),
                                        },
                                        "hl_params": {
                                            "df": 3,
                                            "loc": np.zeros(d_h),
                                            "shape": base_sigma_H * (scale**2),
                                        },
                                    }
                                elif distribution == "translation":
                                    shift_config = {
                                        "type": "additive",
                                        "distribution": distribution,
                                        "ll_params": {"c": scale},
                                        "hl_params": {"c": scale},
                                    }
                                elif distribution == "scaling":
                                    shift_config = {
                                        "type": "multiplicative",
                                        "distribution": distribution,
                                        "ll_params": {"c": scale},
                                        "hl_params": {"c": scale},
                                    }
                                else:
                                    raise ValueError(f"Unknown distribution {distribution}")

                                # unique deterministic seed per config
                                contam_seed = hash((fi, method_name, run_key, alpha, scale, trial, iota)) % (2**32)

                                Xll_cont = apply_huber_contamination(
                                    Xll_clean, alpha, shift_config, ll_var_names, "L", seed=contam_seed
                                )
                                Xhl_cont = apply_huber_contamination(
                                    Xhl_clean, alpha, shift_config, hl_var_names, "H", seed=contam_seed + 19
                                )

                                err = calculate_empirical_error(T_learned, Xll_cont, Xhl_cont)
                                if not np.isnan(err):
                                    errors_per_iota.append(err)

                            avg_err = np.mean(errors_per_iota) if errors_per_iota else np.nan

                            records.append(
                                {
                                    "method": method_name,
                                    "run_key": run_key,
                                    "alpha": float(alpha),
                                    "noise_scale": float(scale),
                                    "trial": int(trial),
                                    "fold": int(fi),
                                    "error": float(avg_err),
                                }
                            )

    df = pd.DataFrame(records)
    print(f"[DONE] Generated {len(df)} records.")

    out_dir = os.path.join("data", experiment, "evaluation_results")
    os.makedirs(out_dir, exist_ok=True)

    if output_file is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            f"empirical_evaluation_{shift_type}_{distribution}_"
            f"alpha{len(alpha_values)}-{alpha_values[0]:.2f}-{alpha_values[-1]:.2f}_"
            f"noise{len(noise_levels)}-{noise_levels[0]:.2f}-{noise_levels[-1]:.2f}_"
            f"trials{num_trials}_zero_mean{zero_mean}_"
            f"kfolds{k_folds}_seed{seed}_{timestamp}.csv"
        )
        output_file = os.path.join(out_dir, fname)
    else:
        # force into evaluation_results
        output_file = os.path.join(out_dir, os.path.basename(output_file))

    df.to_csv(output_file, index=False)
    print(f"[SAVE] Results -> {output_file}")

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="General empirical evaluation (nonlinear-ready).")

    parser.add_argument("--experiment", type=str, default="lucas")
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_steps", type=int, default=10)

    parser.add_argument("--noise_min", type=float, default=0.0)
    parser.add_argument("--noise_max", type=float, default=5.0)
    parser.add_argument("--noise_steps", type=int, default=20)

    parser.add_argument("--trials", type=int, default=2)

    parser.add_argument("--zero_mean", type=str, default="True")
    parser.add_argument(
        "--shift_type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="gaussian",
        choices=["gaussian", "exponential", "student-t", "translation", "scaling"],
    )

    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)

    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    zero_mean = args.zero_mean.lower() in ["true", "1", "yes", "y"]

    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    noise_levels = np.linspace(args.noise_min, args.noise_max, args.noise_steps)

    try:
        run_evaluation(
            experiment=args.experiment,
            alpha_values=alpha_values,
            noise_levels=noise_levels,
            num_trials=args.trials,
            zero_mean=zero_mean,
            shift_type=args.shift_type,
            distribution=args.distribution,
            k_folds=args.k_folds,
            seed=args.seed,
            output_file=args.output,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
