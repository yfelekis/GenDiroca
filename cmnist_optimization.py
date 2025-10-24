#!/usr/bin/env python3
"""
ColorMNIST Optimization Pipeline

This script runs empirical optimization for causal abstraction methods
on the ColorMNIST dataset with configurable hyperparameters.
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import joblib
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from operations import Intervention

# Import the model classes
from train_cmnist_models import ImageColorizerUNet


def load_cmnist_data(data_dir='data/cmnist'):
    """Load the generated ColorMNIST data and models."""
    print(f"Loading ColorMNIST data from {data_dir}...")
    
    # Load data
    Dll_samples = torch.load(f'{data_dir}/dll_samples.pkl')
    Dhl_samples = torch.load(f'{data_dir}/dhl_samples.pkl')
    omega = torch.load(f'{data_dir}/intervention_mapping.pkl')
    
    # Load models
    ll_model_state = torch.load(f'{data_dir}/ll_model_unet.pth')
    ll_model = ImageColorizerUNet()
    ll_model.load_state_dict(ll_model_state)
    ll_model.eval()
    
    hl_model = torch.load(f'{data_dir}/hl_model.pkl')
    
    # Load noise
    U_ll_hat = torch.load(f'{data_dir}/U_ll_hat.pkl')
    U_hl_hat = torch.load(f'{data_dir}/U_hl_hat.pkl')
    
    print("Data loaded successfully!")
    return Dll_samples, Dhl_samples, omega, ll_model, hl_model, U_ll_hat, U_hl_hat


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_cv_folds(data_tuple, n_splits=5, seed=42):
    """Create k-fold cross-validation splits."""
    print(f"Setting up {n_splits}-fold cross-validation...")
    num_samples = data_tuple[0].shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(np.arange(num_samples)):
        folds.append({'train': train_idx, 'test': test_idx})
    print(f"✓ CV folds created. Each fold will have ~{len(folds[0]['train'])} training samples.")
    return folds


def det_hl_func(hl_model, parent_info_hl, intervention):
    """High-level deterministic function."""
    features = parent_info_hl.copy()
    if intervention is not None:
        iv_dict = intervention.vv()
        if 'D_HL' in globals() and D_HL in iv_dict:
            new_digits = np.zeros_like(features[:, :10])
            new_digits[:, iv_dict[D_HL]] = 1
            features[:, :10] = new_digits
        if 'C_HL' in globals() and C_HL in iv_dict:
            new_colors = np.zeros_like(features[:, 10:])
            new_colors[:, iv_dict[C_HL]] = 1
            features[:, 10:] = new_colors
    
    image_features = hl_model.predict(features)
    if image_features.ndim == 1:
        image_features = image_features.reshape(-1, 1)
    
    full_vector = np.concatenate([features, image_features], axis=1)
    return torch.tensor(full_vector, dtype=torch.float32)


def det_ll_func(ll_model, parent_info_ll_tuple, intervention):
    """Low-level deterministic function."""
    _, img_shapes, digits, colors = parent_info_ll_tuple
    n_samples = digits.shape[0]

    if intervention is not None:
        iv_dict = intervention.vv()
        if 'D_LL' in globals() and D_LL in iv_dict:
            digits = torch.full_like(digits, iv_dict[D_LL])
        if 'C_LL' in globals() and C_LL in iv_dict:
            colors = torch.full_like(colors, iv_dict[C_LL])

    ll_model.eval()
    device = next(ll_model.parameters()).device
    img_shapes = img_shapes.to(device)
    digits = digits.to(device).long()
    colors = colors.to(device).long()

    with torch.no_grad():
        predicted_images = ll_model(img_shapes, digits, colors)

    flattened_images = predicted_images.cpu().view(n_samples, -1)
    digits_onehot = F.one_hot(digits.cpu(), num_classes=10).float()
    colors_onehot = F.one_hot(colors.cpu(), num_classes=10).float()
    
    full_vector = torch.cat([flattened_images, digits_onehot, colors_onehot], dim=1)
    return full_vector


def empirical_objective_cmnist_full(T, U_ll, U_hl, Theta_ll, Phi_hl, det_ll_dict, det_hl_dict, parent_info, omega):
    """Empirical objective function for full vectors."""
    total_loss = 0.0

    for iota, eta in omega.items():
        det_ll_full = det_ll_dict[iota]
        det_hl_full = det_hl_dict[eta]
        
        # Add noise to appropriate parts
        image_part = det_ll_full[:, :3072]
        label_part_ll = det_ll_full[:, 3072:]
        image_part_noisy = image_part + U_ll + Theta_ll
        endo_ll_full = torch.cat([image_part_noisy, label_part_ll], dim=1)
        
        label_part_hl = det_hl_full[:, :20]
        feature_part = det_hl_full[:, 20:]
        feature_part_noisy = feature_part + U_hl + Phi_hl
        endo_hl_full = torch.cat([label_part_hl, feature_part_noisy], dim=1)

        # Apply linear abstraction map
        mapped_ll = (T @ endo_ll_full.T).T
        diff = mapped_ll - endo_hl_full
        total_loss += torch.norm(diff, p='fro') ** 2

    return total_loss / len(omega)


def project_onto_frobenius_ball(tensor, radius):
    """Project tensor onto Frobenius norm ball."""
    with torch.no_grad():
        current_norm = torch.norm(tensor, p='fro')
        if current_norm > radius:
            tensor.mul_(radius / (current_norm + 1e-12))
    return tensor


def compute_empirical_radius(N, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=1):
    """Compute empirical radius for adversarial noise bounds."""
    return c1 * (np.log(N) / N) ** (1/alpha) + c2 * np.sqrt(m * np.log(1/eta) / N)


def run_empirical_erica_optimization(U_L, U_H, L_models, H_models, omega, epsilon, delta, 
                                   eta_min, eta_max, num_steps_min, num_steps_max, 
                                   max_iter, tol, seed, robust_L, robust_H, 
                                   initialization, experiment, gain, optimizers):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Ill = list(L_models.keys())
    method = 'erica' if robust_L or robust_H else 'enrico'        

    U_L = torch.as_tensor(U_L, dtype=torch.float32)
    U_H = torch.as_tensor(U_H, dtype=torch.float32)
    
    N, l = U_L.shape
    _, h = U_H.shape
    
    T = torch.randn(h+20, l+20, requires_grad=True)
    
    if gain > 0:
        torch.nn.init.xavier_normal_(T, gain=gain)

    if initialization == 'zeros':  
        Theta = torch.zeros(N, l, requires_grad=False)  
        Phi   = torch.zeros(N, h, requires_grad=False) 
    elif initialization == 'random':
        Theta = torch.randn(N, l, requires_grad=True)
        Phi   = torch.randn(N, h, requires_grad=True)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")

    if optimizers == 'adam':
        optimizer_T = torch.optim.Adam([T], lr=eta_min)
        optimizer_max = torch.optim.Adam([Theta, Phi], lr=eta_max)
    elif optimizers == 'adam_betas':
        optimizer_T  = torch.optim.Adam([T],   lr=eta_min, betas=(0.9,0.999), eps=1e-8, amsgrad=True)
        optimizer_max = torch.optim.Adam([Theta, Phi], lr=eta_max, betas=(0.9,0.999), eps=1e-8, amsgrad=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizers}")

    prev_T_objective = float('inf')
    
    for iteration in tqdm(range(max_iter)):
        objs_T, objs_max = [], []
        
        # Step 1: Minimize with respect to T
        for _ in range(num_steps_min):
            optimizer_T.zero_grad()
            T_objective = empirical_objective_cmnist_full(T, U_L, U_H, Theta, Phi, L_models, H_models, None, omega)
            objs_T.append(T_objective.item())
            T_objective.backward()
            optimizer_T.step()
            
        # Step 2: Maximize with respect to Theta and Phi
        if method == 'erica':
            for _ in range(num_steps_max):
                optimizer_max.zero_grad()
                max_objective = -empirical_objective_cmnist_full(T, U_L, U_H, Theta, Phi, L_models, H_models, None, omega)
                max_objective.backward()
                optimizer_max.step()
                
                with torch.no_grad():
                    Theta.data = project_onto_frobenius_ball(Theta, epsilon)
                    Phi.data   = project_onto_frobenius_ball(Phi, delta)

                mobj = empirical_objective_cmnist_full(T, U_L, U_H, Theta, Phi, L_models, H_models, None, omega)
                objs_max.append(mobj.item())

        with torch.no_grad():
            current_T_objective = T_objective.item()
            if abs(prev_T_objective - current_T_objective) < tol:
                print(f"Converged at iteration {iteration + 1}")
                break
            prev_T_objective = current_T_objective
            
    T = T.detach().numpy()
    
    paramsL = {'pert_U': Theta.detach().numpy(), 'radius_worst': epsilon,
               'pert_hat': U_L, 'radius': epsilon}
    paramsH = {'pert_U': Phi.detach().numpy(), 'radius_worst': delta,
               'pert_hat': U_H, 'radius': delta}
    
    opt_params = {'L': paramsL, 'H': paramsH}

    return opt_params, T


def run_diroca_optimization(config, cv_folds, U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega, output_dir):
    """Run DiRoCA optimization."""
    print("Starting DiRoCA optimization...")
    
    diroca_results = {}
    
    for i, fold_info in enumerate(cv_folds):
        print(f"DiRoCA Fold {i+1}/{len(cv_folds)}")
        fold_key = f'fold_{i}'
        diroca_results[fold_key] = {}
        
        # Calculate bounds for this fold
        train_n = len(fold_info['train'])
        l = U_ll_hat.shape[1]
        h = U_hl_hat.shape[1]
        
        ll_bound = round(compute_empirical_radius(N=train_n, **config['empirical_radius']), 3)
        hl_bound = round(compute_empirical_radius(N=train_n, **config['empirical_radius']), 3)
        
        # Get training data
        U_ll_train = U_ll_hat[fold_info['train']]
        U_hl_train = U_hl_hat[fold_info['train']]
        det_ll_train = {k: v[fold_info['train']] for k, v in det_ll_dict.items()}
        det_hl_train = {k: v[fold_info['train']] for k, v in det_hl_dict.items()}

        # Test different hyperparameters
        radius_pairs = config['radius_pairs']
        for epsilon, delta in radius_pairs:
            print(f"  Testing ε={epsilon}, δ={delta}")
            
            opt_params, trained_T = run_empirical_erica_optimization(
                U_ll_train, U_hl_train, det_ll_train, det_hl_train, omega, 
                epsilon, delta, **config['optimization']
            )
            
            hyperparam_key = f'eps_{epsilon}_delta_{delta}'
            diroca_results[fold_key][hyperparam_key] = {
                'T_matrix': torch.tensor(trained_T),
                'optimization_params': opt_params,
                'test_indices': fold_info['test'],
                'epsilon': epsilon,
                'delta': delta
            }
    
    # Save results
    output_path = os.path.join(output_dir, "diroca_cv_results_empirical.pkl")
    joblib.dump(diroca_results, output_path)
    print(f"DiRoCA results saved to {output_path}")
    
    return diroca_results


def run_gradca_optimization(config, cv_folds, U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega, output_dir):
    """Run GradCA optimization."""
    print("Starting GradCA optimization...")
    
    gradca_results = {}
    
    for i, fold_info in enumerate(cv_folds):
        print(f"GradCA Fold {i+1}/{len(cv_folds)}")
        fold_key = f'fold_{i}'
        
        # Get training data
        U_ll_train = U_ll_hat[fold_info['train']]
        U_hl_train = U_hl_hat[fold_info['train']]
        det_ll_train = {k: v[fold_info['train']] for k, v in det_ll_dict.items()}
        det_hl_train = {k: v[fold_info['train']] for k, v in det_hl_dict.items()}

        # Run optimization (non-robust)
        opt_params, trained_T = run_empirical_erica_optimization(
            U_ll_train, U_hl_train, det_ll_train, det_hl_train, omega, 
            0.0, 0.0, **config['optimization']
        )
        
        gradca_results[fold_key] = {
            'gradca_run': {
                'T_matrix': torch.tensor(trained_T),
                'optimization_params': opt_params,
                'test_indices': fold_info['test'] 
            }
        }
    
    # Save results
    output_path = os.path.join(output_dir, "gradca_cv_results_empirical.pkl")
    joblib.dump(gradca_results, output_path)
    print(f"GradCA results saved to {output_path}")
    
    return gradca_results


def run_baryca_optimization(config, cv_folds, U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega, output_dir):
    """Run BaryCA optimization."""
    print("Starting BaryCA optimization...")
    
    baryca_results = {}
    
    for i, fold_info in enumerate(cv_folds):
        print(f"BaryCA Fold {i+1}/{len(cv_folds)}")
        fold_key = f'fold_{i}'
        
        # Get training data
        U_ll_train = U_ll_hat[fold_info['train']]
        U_hl_train = U_hl_hat[fold_info['train']]
        
        # Create structural matrices
        L_matrices = [torch.eye(3092, 3092) for _ in omega.items()]
        H_matrices = [torch.eye(21, 21) for _ in omega.items()]
        
        # Create full vectors
        U_ll_full = torch.cat([U_ll_train, 
                             F.one_hot(torch.randint(0, 10, (U_ll_train.shape[0],)), num_classes=10).float(),
                             F.one_hot(torch.randint(0, 10, (U_ll_train.shape[0],)), num_classes=10).float()], dim=1)
        
        U_hl_full = torch.cat([F.one_hot(torch.randint(0, 10, (U_hl_train.shape[0],)), num_classes=10).float(),
                              F.one_hot(torch.randint(0, 10, (U_hl_train.shape[0],)), num_classes=10).float(),
                              U_hl_train], dim=1)
        
        # Run BaryCA optimization
        trained_T = run_baryca_optimization_single(
            U_ll_full, U_hl_full, L_matrices, H_matrices, **config['baryca']
        )
        
        baryca_results[fold_key] = {
            'baryca_run': {
                'T_matrix': torch.tensor(trained_T),
                'test_indices': fold_info['test'] 
            }
        }
    
    # Save results
    output_path = os.path.join(output_dir, "baryca_cv_results_empirical.pkl")
    joblib.dump(baryca_results, output_path)
    print(f"BaryCA results saved to {output_path}")
    
    return baryca_results


def run_baryca_optimization_single(U_ll_full, U_hl_full, L_matrices, H_matrices, max_iter=1000, tol=1e-5, lr=1e-3, seed=42):
    """Single BaryCA optimization run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    T = torch.randn(21, 3092, requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=lr)
    
    def barycentric_objective(T, L_matrices, H_matrices):
        total_loss = 0.0
        for L_mat, H_mat in zip(L_matrices, H_matrices):
            transformed = T @ L_mat
            diff = transformed - H_mat
            total_loss += torch.norm(diff, p='fro') ** 2
        return total_loss / len(L_matrices)
    
    for iteration in tqdm(range(max_iter)):
        optimizer.zero_grad()
        loss = barycentric_objective(T, L_matrices, H_matrices)
        loss.backward()
        optimizer.step()
        
        if loss.item() < tol:
            print(f"Converged at iteration {iteration + 1}")
            break
    
    return T.detach().numpy()


def run_abslingam_optimization(config, cv_folds, Dll_samples, Dhl_samples, output_dir):
    """Run Abs-LiNGAM optimization."""
    print("Starting Abs-LiNGAM optimization...")
    
    abslingam_results = {}
    
    for i, fold_info in enumerate(cv_folds):
        print(f"Abs-LiNGAM Fold {i+1}/{len(cv_folds)}")
        fold_key = f'fold_{i}'
        
        train_idx = fold_info['train']
        
        # Get training data
        final_images = Dll_samples[None][0][train_idx]
        digits = Dll_samples[None][2][train_idx]
        colors = Dll_samples[None][3][train_idx]
        
        # Create full vectors
        images_flat = final_images.view(final_images.shape[0], -1)
        digits_onehot = F.one_hot(digits, num_classes=10).float()
        colors_onehot = F.one_hot(colors, num_classes=10).float()
        Dll_obs_full = torch.cat([images_flat, digits_onehot, colors_onehot], dim=1)
        
        Dhl_obs_full = Dhl_samples[None][train_idx]
        
        # Run Abs-LiNGAM
        abslingam_results_for_fold = run_abslingam_single(Dll_obs_full, Dhl_obs_full, **config['abslingam'])

        abslingam_results[fold_key] = {
            'Perfect': {
                'T_matrix': torch.tensor(abslingam_results_for_fold['Perfect']['T']),
                'test_indices': fold_info['test']
            },
            'Noisy': {
                'T_matrix': torch.tensor(abslingam_results_for_fold['Noisy']['T']),
                'test_indices': fold_info['test']
            }
        }
    
    # Save results
    output_path = os.path.join(output_dir, "abslingam_cv_results_empirical.pkl")
    joblib.dump(abslingam_results, output_path)
    print(f"Abs-LiNGAM results saved to {output_path}")
    
    return abslingam_results


def run_abslingam_single(Dll_obs, Dhl_obs, alpha=0.1):
    """Single Abs-LiNGAM run."""
    if isinstance(Dll_obs, torch.Tensor):
        Dll_obs = Dll_obs.numpy()
    if isinstance(Dhl_obs, torch.Tensor):
        Dhl_obs = Dhl_obs.numpy()
    
    # Perfect case
    reg_perfect = LinearRegression()
    reg_perfect.fit(Dll_obs, Dhl_obs)
    T_perfect = reg_perfect.coef_
    
    # Noisy case
    reg_noisy = Ridge(alpha=alpha)
    reg_noisy.fit(Dll_obs, Dhl_obs)
    T_noisy = reg_noisy.coef_
    
    return {
        'Perfect': {'T': T_perfect},
        'Noisy': {'T': T_noisy}
    }


def main():
    """Main optimization pipeline."""
    print("=" * 60)
    print("COLORMNIST EMPIRICAL OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    # Load data
    Dll_samples, Dhl_samples, omega, ll_model, hl_model, U_ll_hat, U_hl_hat = load_cmnist_data()
    
    # Set up cross-validation
    cv_folds = setup_cv_folds(Dll_samples[None])
    
    # Prepare parent info
    parent_info_ll_tuple = (
        Dll_samples[None][0],
        Dll_samples[None][1],
        Dll_samples[None][2],
        Dll_samples[None][3],
    )
    parent_info_hl = Dhl_samples[None][:, :-1].numpy()
    
    # Compute deterministic outputs
    print("Computing deterministic outputs...")
    det_ll_dict = {}
    for iota in tqdm(list(omega.keys()), desc="Computing Low-Level"):
        det_ll_dict[iota] = det_ll_func(ll_model, parent_info_ll_tuple, iota)
    
    det_hl_dict = {}
    for eta in tqdm(list(omega.values()), desc="Computing High-Level"):
        det_hl_dict[eta] = det_hl_func(hl_model, parent_info_hl, eta)
    
    # Create output directory
    output_dir = "data/cmnist/results_empirical"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimizations for each method
    methods = ['diroca', 'gradca', 'baryca', 'abslingam']
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"STARTING {method.upper()} OPTIMIZATION")
        print(f"{'='*60}")
        
        config_path = f"configs/{method}_cmnist.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path)
            print(f"Loaded config from {config_path}")
        else:
            print(f"Warning: Config file {config_path} not found, using defaults")
            config = {}
        
        try:
            if method == 'diroca':
                results[method] = run_diroca_optimization(config, cv_folds, U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega, output_dir)
            elif method == 'gradca':
                results[method] = run_gradca_optimization(config, cv_folds, U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega, output_dir)
            elif method == 'baryca':
                results[method] = run_baryca_optimization(config, cv_folds, U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega, output_dir)
            elif method == 'abslingam':
                results[method] = run_abslingam_optimization(config, cv_folds, Dll_samples, Dhl_samples, output_dir)
            
            print(f"✓ {method.upper()} completed successfully")
            
        except Exception as e:
            print(f"❌ {method.upper()} failed: {str(e)}")
            results[method] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print("\nResults summary:")
    for method, method_results in results.items():
        if method_results is not None:
            if method == 'diroca':
                total_hyperparams = sum(len(fold_results) for fold_results in method_results.values())
                print(f"  ✓ {method}: {len(method_results)} folds, {total_hyperparams} hyperparameter combinations")
            else:
                print(f"  ✓ {method}: {len(method_results)} folds")
        else:
            print(f"  ❌ {method}: FAILED")
    
    return results


if __name__ == "__main__":
    results = main()
