import os
import pickle
import torch
import numpy as np

# Path to your results file
path = "data/cmnist/results_empirical/cmnist_all_results.pkl"

if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    print(f"Loading {path}...")
    with open(path, "rb") as f:
        all_results = pickle.load(f)

    print(f"\nKeys in all_results: {list(all_results.keys())}")

    # Drill down into the first available fold and run
    # Usually structure is: MethodGroup -> Fold -> RunKey -> ResultDict
    for method_group, folds in all_results.items():
        if not isinstance(folds, dict): continue
        
        print(f"\n=== Inspecting Method: {method_group} ===")
        
        for fold_key, fold_data in folds.items():
            if not isinstance(fold_data, dict) or "error" in fold_data: continue
            
            print(f"  Fold: {fold_key}")
            
            for run_key, result in fold_data.items():
                if "error" in result: continue
                
                print(f"    Run: {run_key}")
                print(f"    Keys available in result: {list(result.keys())}")
                
                # Check T_matrix shape
                if "T_matrix" in result:
                    T = result["T_matrix"]
                    if isinstance(T, torch.Tensor): T = T.numpy()
                    print(f"      T_matrix shape: {T.shape}")
                
                # CHECK FOR BIAS
                bias_candidates = ["bias", "intercept", "b", "mean", "mu"]
                found_bias = False
                for k in result.keys():
                    if any(cand in k.lower() for cand in bias_candidates):
                        val = result[k]
                        shape = val.shape if hasattr(val, "shape") else "scalar/list"
                        print(f"      [POTENTIAL BIAS FOUND] Key: '{k}', Shape: {shape}")
                        print(f"      Value preview: {val}")
                        found_bias = True
                
                if not found_bias:
                    print("      [WARNING] No obvious bias key found.")

                # Stop after printing one run per method to keep output clean
                break 
            break