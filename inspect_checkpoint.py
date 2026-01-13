import torch
import os
import argparse

def inspect_checkpoint(ckpt_path):
    print(f"--- Inspecting: {ckpt_path} ---\n")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: File not found at {ckpt_path}")
        return

    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # 1. Print Top-Level Keys
        keys = list(checkpoint.keys())
        print(f"✅ Loaded successfully.")
        print(f"🔑 Top-level keys found: {keys}\n")
        
        # 2. Check for Hyperparameters
        if "hyper_parameters" in checkpoint:
            print("📄 FOUND 'hyper_parameters' block! Here is what is inside:")
            hparams = checkpoint["hyper_parameters"]
            # Pretty print the dictionary
            for k, v in hparams.items():
                print(f"   - {k}: {v}")
        else:
            print("⚠️ NO 'hyper_parameters' key found. You will need the separate json file.")

        # 3. Check State Dictionary (Weights)
        if "state_dict" in checkpoint:
            weight_keys = list(checkpoint["state_dict"].keys())
            print(f"\n⚖️  State Dict found with {len(weight_keys)} weight tensors.")
            # Print first 5 keys to verify naming style
            print("   First 5 weight keys (check for 'model.' prefix):")
            for k in weight_keys[:5]:
                print(f"   - {k}")
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Set default to where you likely put it
    parser.add_argument("--path", type=str, default="models/xia_checkpoints/epoch=961-step=225108.ckpt", 
                        help="Path to the .ckpt file")
    args = parser.parse_args()
    
    inspect_checkpoint(args.path)