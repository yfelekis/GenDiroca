# Load trained models and abduced noise
print("=== Loading Trained Models ===")

# Load the trained models
data_dir = 'data/cmnist'

# Load low-level U-Net model
ll_model_state = torch.load(f'{data_dir}/ll_model_unet.pth')
ll_model = ImageColorizerUNet()
ll_model.load_state_dict(ll_model_state)
ll_model.eval()
print("✓ Low-level U-Net model loaded")

# Load high-level linear model
hl_model = torch.load(f'{data_dir}/hl_model.pkl')
print("✓ High-level linear model loaded")

# Load abduced noise
U_ll_hat = torch.load(f'{data_dir}/U_ll_hat.pkl')
U_hl_hat = torch.load(f'{data_dir}/U_hl_hat.pkl')
print("✓ Abduced noise loaded")

print(f"\nModel information:")
print(f"  - Low-level model: U-Net with {sum(p.numel() for p in ll_model.parameters()):,} parameters")
print(f"  - High-level model: Linear regression")
print(f"  - Low-level noise shape: {U_ll_hat.shape}")
print(f"  - High-level noise shape: {U_hl_hat.shape}")

# Make models available for use
print("\nModels are now ready for causal abstraction experiments!")
