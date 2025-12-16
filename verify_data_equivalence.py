import joblib
import numpy as np
import os

# Path to your pack
PACK_PATH = "data/lucas/lucas_pack.pkl"

def main():
    print(f"[TEST] Loading {PACK_PATH}...")
    if not os.path.exists(PACK_PATH):
        print("Error: Pack not found.")
        return

    pack = joblib.load(PACK_PATH)
    
    # 1. Get Observational Noise (The Training Noise)
    # We use this to simulate "Counterfactual" evaluation
    U_obs = pack['ll']['iota0']['U']
    
    # 2. Pick a random intervention to test (e.g., iota1)
    # If we used iota0, they would be identical by definition.
    test_intervention = 'iota1' 
    if test_intervention not in pack['ll']:
        test_intervention = list(pack['ll'].keys())[1] # Pick second available if iota1 missing
        
    print(f"[TEST] Comparing data for intervention: {test_intervention}")
    
    # --- METHOD A: Raw Samples (Old Code) ---
    # This contains noise sampled specifically for THIS intervention
    X_raw = pack['ll'][test_intervention]['X']
    
    # --- METHOD B: Reconstructed (New Code / Grid Search) ---
    # This forces the mechanism D to use the Observational Noise
    D_intervention = pack['ll'][test_intervention]['D']
    X_reconstructed = D_intervention + U_obs
    
    # --- COMPARISON ---
    # Calculate difference
    diff = X_raw - X_reconstructed
    mse = np.mean(diff ** 2)
    max_diff = np.max(np.abs(diff))
    
    print(f"\n--- RESULTS ---")
    print(f"Shape of X: {X_raw.shape}")
    print(f"Mean Squared Difference: {mse:.6f}")
    print(f"Max Absolute Difference: {max_diff:.6f}")
    
    if mse < 1e-9:
        print("\n✅ EQUIVALENT: The generator used coupled noise. You can use X directly.")
    else:
        print("\n❌ DIFFERENT: The generator sampled new noise for interventional data.")
        print("   -> Using X_raw tests GENERALIZATION (New Noise).")
        print("   -> Using D+U_obs tests COUNTERFACTUAL (Same Noise).")
        print("\n   To reproduce the 2.42 error (ERM Success), you MUST use Method B.")

if __name__ == "__main__":
    main()