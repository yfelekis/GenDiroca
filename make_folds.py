# make_folds.py
import joblib
import utilities as ut

# Load aligned data
all_data = ut.load_all_data("battery")
Dll_obs = all_data['LLmodel']['data'][None]

# Rebuild folds on the aligned dataset
folds_path = "data/battery/cv_folds.pkl"
ut.prepare_cv_folds(Dll_obs, k=5, random_state=42, save_path=folds_path)

print(f"New CV folds saved to {folds_path}")
