import argparse
import os
import joblib
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Your existing codebase imports
from models import CausalBayesianNetwork, Intervention


# ============================================================
# Custom Nonlinear Additive Noise SCM Implementation
# ============================================================

class IHDPNonlinearAddSCM:
    """
    Custom implementation of Nonlinear Additive Noise SCM for IHDP data.
    Designed to work with function signatures: func(parents, u)
    where parents is a 2D array (n_samples x n_parents) and u is 1D (n_samples).
    """
    def __init__(self, causal_graph, functions, intervention=None):
        """
        Initialize the SCM.
        
        Args:
            causal_graph: CausalBayesianNetwork object
            functions: dict mapping variable names to callable functions
            intervention: Intervention object or None
        """
        self.functions = functions
        self.intervention_dict = intervention.vv() if intervention else {}

        # The .do() operation modifies the graph structure
        if self.intervention_dict:
            self.causal_graph = causal_graph.do(list(self.intervention_dict.keys()))
        else:
            self.causal_graph = causal_graph
        
        # Get topological ordering of variables
        self.variables = list(nx.topological_sort(self.causal_graph))
        self.var_index = {var: i for i, var in enumerate(self.variables)}
        self.dim = len(self.variables)

    def simulate(self, exogenous_noise):
        """
        Simulates data from the SCM by executing the functions in topological order.
        
        Args:
            exogenous_noise: 2D array (n_samples x n_variables) of exogenous noise
            
        Returns:
            2D array (n_samples x n_variables) of endogenous variable values
        """
        n_samples = exogenous_noise.shape[0]
        endogenous = np.zeros_like(exogenous_noise)

        # Iterate through variables in topological order to compute their values
        for var_name in self.variables:
            var_idx = self.var_index[var_name]

            if var_name in self.intervention_dict:
                # If the variable is intervened on, set its value directly
                endogenous[:, var_idx] = self.intervention_dict[var_name]
            else:
                # Otherwise, calculate its value from its parents using its function
                parents = list(self.causal_graph.predecessors(var_name))
                
                # Get the callable function for the current node
                func = self.functions[var_name]
                
                # Collect parent values as a 2D array (n_samples x n_parents)
                # Ensure consistent ordering: T first if present, then others sorted
                if parents:
                    # Sort parents: put "T" first if it exists, then sort the rest
                    parents_sorted = sorted([p for p in parents if p != "T"])
                    if "T" in parents:
                        parents_sorted = ["T"] + parents_sorted
                    
                    # Stack parent values as columns
                    parent_values = np.column_stack([
                        endogenous[:, self.var_index[p]] for p in parents_sorted
                    ])
                else:
                    # No parents (exogenous variable)
                    parent_values = np.zeros((n_samples, 0))
                
                # Get the exogenous noise for this variable
                u = exogenous_noise[:, var_idx]
                
                # Call function with (parents, u) as positional arguments
                endogenous[:, var_idx] = func(parent_values, u)
                
        return endogenous


# ============================================================
# Top-level wrapper classes (PICKLABLE)
# ============================================================

class XIdentityFunction:
    """ANM for X_j: X_j = U_j (no parents)."""
    def __call__(self, parents, u):
        return u


class TFunction:
    """ANM for T: T = f_T(X) + U_T, with f_T from logistic regression."""
    def __init__(self, logreg):
        self.logreg = logreg

    def __call__(self, parents, u):
        # parents: X vector
        parents = np.atleast_2d(parents)
        p = self.logreg.predict_proba(parents)[:, 1]
        return p + u


class YFunction:
    """ANM for Y: Y = f_Y(X,T) + U_Y, with f0,f1 from RFs."""
    def __init__(self, rf0, rf1):
        self.rf0 = rf0
        self.rf1 = rf1

    def __call__(self, parents, u):
        # parents: [T, x1..x25]
        parents = np.atleast_2d(parents)
        T_par = parents[:, 0]
        X_par = parents[:, 1:]
        mu0_hat = self.rf0.predict(X_par)
        mu1_hat = self.rf1.predict(X_par)
        mu = (1.0 - T_par) * mu0_hat + T_par * mu1_hat
        return mu + u


# ============================================================
# 1. Load IHDP (CEVAE-style)
# ============================================================

def load_ihdp(path_or_url: str = None) -> pd.DataFrame:
    """
    Load IHDP dataset in CEVAE format.
    If no path is provided, download ihdp_npci_1.csv from CEVAE GitHub.
    """
    if path_or_url is None:
        url = (
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/"
            "datasets/IHDP/csv/ihdp_npci_1.csv"
        )
        print(f"No --ihdp_path provided. Downloading IHDP data from:\n{url}")
        df = pd.read_csv(url, header=None)
    else:
        print(f"Loading IHDP data from: {path_or_url}")
        df = pd.read_csv(path_or_url, header=None)

    cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [
        f"x{i}" for i in range(1, 26)
    ]
    df.columns = cols
    return df


# ============================================================
# 2. Fit structural functions & compute ANM residuals
# ============================================================

def fit_structural_functions(df: pd.DataFrame, random_state: int = 0):
    """
    Fit ANM-style structural functions on IHDP.

    Variables:
      X = (x1..x25)
      T in {0,1}
      Y = y_factual

    Graph:
      X_j -> T, X_j -> Y, T -> Y
      X_j exogenous: X_j = U_{X_j}

    Use:
      - RF on mu0, mu1 to get f0(x), f1(x).
      - LogisticRegression for E[T|X].
      - f_Y(x,t) = (1-t) f0(x) + t f1(x).
    """
    x_cols = [c for c in df.columns if c.startswith("x")]
    X = df[x_cols].values
    T = df["treatment"].values.astype(float)
    y = df["y_factual"].values.astype(float)
    mu0_true = df["mu0"].values.astype(float)
    mu1_true = df["mu1"].values.astype(float)

    n, d_x = X.shape

    # ---- f0, f1 via RF ----
    rf0 = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    rf1 = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state + 1,
    )
    rf0.fit(X, mu0_true)
    rf1.fit(X, mu1_true)

    # ---- f_T via logistic regression ----
    logreg = LogisticRegression(max_iter=2000, n_jobs=-1)
    logreg.fit(X, T)
    p_T1 = logreg.predict_proba(X)[:, 1]
    fT_vals = p_T1  # deterministic part for T

    # ---- f_Y(x,t) from RFs ----
    muY_hat = (1.0 - T) * rf0.predict(X) + T * rf1.predict(X)

    # ---- Build D_obs and U_obs for [x1..x25, T, Y] ----
    D = np.zeros((n, d_x + 2))
    U = np.zeros((n, d_x + 2))

    # X_j: exogenous => X_j = U_{X_j}
    D[:, :d_x] = 0.0
    U[:, :d_x] = X

    # T: T = f_T(X) + U_T
    D[:, d_x] = fT_vals
    U[:, d_x] = T - fT_vals

    # Y: Y = f_Y(X,T) + U_Y
    D[:, d_x + 1] = muY_hat
    U[:, d_x + 1] = y - muY_hat

    # ---- Build picklable structural functions ----
    functions = {}

    # Same XIdentityFunction for all x_j (they all: X_j = U_j)
    x_fun = XIdentityFunction()
    for name in x_cols:
        functions[name] = x_fun

    # T and Y:
    functions["T"] = TFunction(logreg)
    functions["Y"] = YFunction(rf0, rf1)

    return {
        "x_cols": x_cols,
        "functions": functions,
        "D_obs": D,
        "U_obs": U,
    }


# ============================================================
# 3. Build causal graph
# ============================================================

def build_graph(x_cols):
    """
    Causal graph:
      X_j -> T, X_j -> Y, T -> Y
    """
    edges = []
    for x in x_cols:
        edges.append((x, "T"))
        edges.append((x, "Y"))
    edges.append(("T", "Y"))
    return CausalBayesianNetwork(edges)


# ============================================================
# 4. Main: generate identical LL & HL SCMs + abstraction
# ============================================================

def main(ihdp_path, out_dir, seed):
    np.random.seed(seed)

    # 4.1 Load IHDP
    df = load_ihdp(ihdp_path)

    # Align SCM-level variable names
    df["T"] = df["treatment"].astype(float)
    df["Y"] = df["y_factual"].astype(float)

    # 4.2 Fit structural functions & compute ANM decomposition
    print("Fitting structural functions and computing residuals...")
    sf = fit_structural_functions(df, random_state=seed)
    x_cols = sf["x_cols"]
    functions = sf["functions"]
    U_obs = sf["U_obs"]

    # 4.3 Build causal graph
    cbn = build_graph(x_cols)

    # 4.4 Define a simple intervention set (shared for LL & HL)
    interv_specs = {
        "iota0": "None",      # observational
        "iota1": {"T": 0.0},  # do(T=0)
        "iota2": {"T": 1.0},  # do(T=1)
    }
    interventions = {
        name: (None if spec == "None" else Intervention(spec))
        for name, spec in interv_specs.items()
    }
    Ill_relevant = list(interventions.values())
    Ihl_relevant = Ill_relevant  # identical

    # 4.5 Identity abstraction: tau = I, omega = id
    var_order = x_cols + ["T", "Y"]
    d = len(var_order)
    T_mat = np.eye(d)

    omega = {
        interventions[name]: interventions[name]
        for name in interv_specs.keys()
    }

    # 4.6 Build SCM instances & attach observational data

    LL_scm_instances = {}
    HL_scm_instances = {}
    LL_data = {}
    HL_data = {}
    LL_noise = {}
    HL_noise = {}

    # observational (iota0)
    iota0 = interventions["iota0"]
    LL_scm_instances[iota0] = IHDPNonlinearAddSCM(cbn, functions, iota0)
    HL_scm_instances[iota0] = IHDPNonlinearAddSCM(cbn, functions, iota0)

    obs_matrix = df[var_order].values
    LL_data[iota0] = obs_matrix
    HL_data[iota0] = obs_matrix.copy()

    LL_noise[iota0] = U_obs
    HL_noise[iota0] = U_obs.copy()

    # do(T=0), do(T=1): create SCM instances; data can be simulated later
    for key in ["iota1", "iota2"]:
        iota = interventions[key]
        LL_scm_instances[iota] = IHDPNonlinearAddSCM(cbn, functions, iota)
        HL_scm_instances[iota] = IHDPNonlinearAddSCM(cbn, functions, iota)

    # 4.7 Empirical exogenous distribution (for DRO later)
    mu_hat = U_obs.mean(axis=0)
    Sigma_hat = np.diag(U_obs.var(axis=0))

    # 4.8 Pack LL & HL models (IDENTICAL)
    LLmodel = {
        "graph": cbn,
        "intervention_set": Ill_relevant,
        "functions": functions,
        "noise_dist": {"mu": mu_hat, "sigma": Sigma_hat},
        "data": LL_data,
        "scm_instances": LL_scm_instances,
        "noise": LL_noise,
        "var_order": var_order,
    }

    HLmodel = {
        "graph": cbn,
        "intervention_set": Ihl_relevant,
        "functions": functions,
        "noise_dist": {"mu": mu_hat.copy(), "sigma": Sigma_hat.copy()},
        "data": HL_data,
        "scm_instances": HL_scm_instances,
        "noise": HL_noise,
        "var_order": var_order,
    }

    abstraction_data = {
        "T": T_mat,
        "omega": omega,
    }

    # 4.9 Save
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(LLmodel, os.path.join(out_dir, "LLmodel.pkl"))
    joblib.dump(HLmodel, os.path.join(out_dir, "HLmodel.pkl"))
    joblib.dump(abstraction_data, os.path.join(out_dir, "abstraction_data.pkl"))

    print(f"✓ Saved LLmodel, HLmodel, abstraction_data to '{out_dir}'")
    print("✓ LL and HL are identical. Environment shift is to be modeled later via ambiguity sets on U.")


# ============================================================
# 5. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate identical IHDP-based SCM copies (LL & HL) for "
            "environment-robust causal abstraction experiments."
        )
    )
    parser.add_argument(
        "--ihdp_path",
        type=str,
        default=None,
        help=(
            "Optional path/URL to IHDP CSV in CEVAE format. "
            "If omitted, downloads ihdp_npci_1.csv from the CEVAE repo."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/ihdp_envcopies",
        help="Output directory for LLmodel.pkl, HLmodel.pkl, abstraction_data.pkl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for RF/logistic etc.",
    )

    args = parser.parse_args()
    main(args.ihdp_path, args.out_dir, args.seed)
