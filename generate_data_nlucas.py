#!/usr/bin/env python3
"""
LUCAS (non-linear LiLUCAS) data generator
-----------------------------------------
A minimal, self-contained generator for a non-linear additive-noise SCM at the
low level (LL), an inferred linear ANM at the high level (HL), and a fixed
linear abstraction T. It outputs DIROCA-ready tuples (D, U, X) per intervention
along with the omega mapping.

Usage (example):
    python lucas_nonlinear_generator.py --seed 23 --n_per_int 10000 --out data/lucas

Saves a single joblib file at <out>/lucas_pack.pkl containing a dictionary with keys:
    - 'T': (3x6) numpy array
    - 'omega': dict mapping LL intervention names -> HL intervention names
    - 'll': { iota_name: {'X':..., 'D':..., 'U':...} }
    - 'hl': { eta_name: {'X':..., 'D':..., 'U':...} }
    - 'hl_model': {'alpha': float, 'beta': (2,), 'mu': (3,), 'sigma_diag': (3,)}
    - 'graphs': {'ll_edges': list of (parent, child), 'hl_edges': list}

Everything is numpy arrays of dtype float64. Covariances are diagonal.
"""

from __future__ import annotations
import os
import argparse
import joblib
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# ----------------------------- helpers ---------------------------------

def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# ----------------------------- spec ------------------------------------

LL_VARS = ['Smoking', 'Genetics', 'Lung Cancer', 'Allergy', 'Coughing', 'Fatigue']
HL_VARS = ["Environment'", "Genetics'", "Lung Cancer'"]

@dataclass(frozen=True)
class LucasSpec:
    ll_graph: nx.DiGraph
    hl_graph: nx.DiGraph
    f: Dict[str, callable]              # LL mechanisms as dict-style: f(pa: dict)->array
    mu_ll: np.ndarray                   # (6,)
    Sigma_ll_diag: np.ndarray           # (6,)
    T: np.ndarray                       # (3,6)
    interventions_ll: Dict[str, Optional[Dict[str, float]]]
    interventions_hl: Dict[str, Optional[Dict[str, float]]]
    omega: Dict[str, str]

@dataclass
class HLModel:
    alpha: float           # intercept for LC'
    beta: np.ndarray       # (2,) coefficients [EN', GE'] -> LC'
    mu: np.ndarray         # (3,) means of U_h on obs (EN', GE', residual_LC')
    sigma_diag: np.ndarray # (3,) diagonal variances of U_h on obs


def build_lucas(seed: int = 23) -> LucasSpec:
    rng = np.random.default_rng(seed)

    # LL graph
    Gll = nx.DiGraph()
    Gll.add_nodes_from(LL_VARS)
    edges = [
        ('Smoking', 'Lung Cancer'),
        ('Genetics', 'Lung Cancer'),
        ('Lung Cancer', 'Coughing'),
        ('Allergy', 'Coughing'),
        ('Lung Cancer', 'Fatigue'),
        ('Coughing', 'Fatigue'),
    ]
    Gll.add_edges_from(edges)

    # HL graph (EN' -> LC', GE' -> LC')
    Ghl = nx.DiGraph()
    Ghl.add_nodes_from(HL_VARS)
    hl_edges = [("Environment'", "Lung Cancer'"), ("Genetics'", "Lung Cancer'")]
    Ghl.add_edges_from(hl_edges)

    # LL mechanisms (dict-style pa)
    def f_SM(pa):
        return np.zeros_like(pa.get('Smoking', np.zeros(1)))  # roots: 0 + U
    def f_GE(pa):
        return np.zeros_like(pa.get('Genetics', np.zeros(1)))
    def f_AL(pa):
        return np.zeros_like(pa.get('Allergy', np.zeros(1)))
    def f_LC(pa):
        SM = pa['Smoking']; GE = pa['Genetics']
        return 1.3*np.tanh(0.9*SM) + 1.1*np.tanh(0.8*GE) + 0.10*SM*GE
    def f_CO(pa):
        LC = pa['Lung Cancer']; AL = pa['Allergy']
        return 0.9*softplus(0.9*LC) + 0.7*np.tanh(0.8*AL) + 0.08*(LC**2)
    def f_FA(pa):
        LC = pa['Lung Cancer']; CO = pa['Coughing']
        return 1.1*np.tanh(0.9*LC) + 0.8*np.tanh(0.7*CO) + 0.06*LC*CO

    f = {
        'Smoking': f_SM,
        'Genetics': f_GE,
        'Allergy': f_AL,
        'Lung Cancer': f_LC,
        'Coughing': f_CO,
        'Fatigue': f_FA,
    }

    # Noise parameters (same scale as LiLUCAS)
    mu_ll = np.array([0, 0, 0.1, 0.1, 0.3, 0.2], dtype=np.float64)
    Sigma_ll_diag = np.array([0.5, 2.0, 1.0, 1.5, 0.8, 1.2], dtype=np.float64)

    # Abstraction matrix T (3x6)
    T = np.array([
        [2, 1, 1, 0, 0, 0],
        [0, 2, 1, 1, 0, 1],
        [1, 0, 1, 2, 0, 1],
    ], dtype=np.float64)

    # LL interventions 
    interventions_ll = {
        'iota0': None,
        'iota1': {'Smoking': 0.0},
        'iota2': {'Smoking': 1.0},
        'iota3': {'Genetics': 0.0},
        'iota4': {'Genetics': 1.0},
        'iota5': {'Lung Cancer': 0.0},
        'iota6': {'Lung Cancer': 1.0},
        'iota7': {'Smoking': 0.0, 'Lung Cancer': 0.0},
        'iota8': {'Smoking': 1.0, 'Lung Cancer': 1.0},
        'iota9': {'Allergy': 0.0},
        'iota10': {'Allergy': 1.0},
    }

    # HL interventions (mirror semantics)
    interventions_hl = {
        'eta0': None,
        'eta1': {"Environment'": 0.0},
        'eta2': {"Environment'": 1.0},
        'eta3': {"Genetics'": 0.0},
        'eta4': {"Genetics'": 1.0},
        'eta5': {"Lung Cancer'": 0.0},
        'eta6': {"Lung Cancer'": 1.0},
        'eta7': {"Environment'": 0.0, "Lung Cancer'": 0.0},
        'eta8': {"Environment'": 1.0, "Lung Cancer'": 1.0},
        'eta9': {"Environment'": 0.0, "Genetics'": 0.0},
        'eta10': {"Environment'": 1.0, "Genetics'": 1.0},
    }

    # Omega mapping (LL iota -> HL eta)
    omega = {
        'iota0': 'eta0',
        'iota1': 'eta1',
        'iota2': 'eta2',
        'iota3': 'eta3',
        'iota4': 'eta4',
        'iota5': 'eta5',
        'iota6': 'eta6',
        'iota7': 'eta7',
        'iota8': 'eta8',
        'iota9': 'eta9',
        'iota10': 'eta10',
    }

    return LucasSpec(
        ll_graph=Gll,
        hl_graph=Ghl,
        f=f,
        mu_ll=mu_ll,
        Sigma_ll_diag=Sigma_ll_diag,
        T=T,
        interventions_ll=interventions_ll,
        interventions_hl=interventions_hl,
        omega=omega,
    )

# ----------------------------- generation --------------------------------

def _topo_lists(G: nx.DiGraph, vars_order: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
    parents = {v: list(G.predecessors(v)) for v in vars_order}
    idx = {v: i for i, v in enumerate(vars_order)}
    parent_idx = {v: [idx[p] for p in parents[v]] for v in vars_order}
    return parents, parent_idx


def sample_ll(spec: LucasSpec, iota: Optional[Dict[str, float]], n: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    vars_order = LL_VARS
    parents, parent_idx = _topo_lists(spec.ll_graph, vars_order)

    # sample exogenous noise
    U = rng.normal(loc=spec.mu_ll, scale=np.sqrt(spec.Sigma_ll_diag), size=(n, len(vars_order)))
    U = U.astype(np.float64)

    D = np.zeros_like(U, dtype=np.float64)
    X = np.zeros_like(U, dtype=np.float64)

    iota = iota or {}
    # simulate in topological order
    for v in vars_order:
        j = LL_VARS.index(v)
        if v in iota:
            D[:, j] = float(iota[v])
            U[:, j] = 0.0
            X[:, j] = D[:, j]
        else:
            if parents[v]:
                pa_vals = {p: X[:, LL_VARS.index(p)] for p in parents[v]}
                D[:, j] = spec.f[v](pa_vals)
            else:
                D[:, j] = spec.f[v]({})
            X[:, j] = D[:, j] + U[:, j]

    return {'X': X, 'D': D, 'U': U}


# ----------------------------- HL inference & sampling -------------------

def infer_hl(spec: LucasSpec, Xll_obs: np.ndarray) -> Tuple[HLModel, Dict[str, np.ndarray]]:
    """Infer HL linear ANM from observational HL = Xll_obs @ T^T.
    Roots EN', GE' have f=0; LC' = alpha + b1*EN' + b2*GE' + U_LC.
    Returns model + obs HL tuples (X,D,U) for convenience.
    """
    Xh_obs = Xll_obs @ spec.T.T  # (n,3)
    n = Xh_obs.shape[0]
    EN = Xh_obs[:, 0]
    GE = Xh_obs[:, 1]
    LC = Xh_obs[:, 2]

    # OLS with intercept: LC ~ 1 + EN + GE
    Xdesign = np.column_stack([np.ones(n), EN, GE])  # (n,3)
    theta, *_ = np.linalg.lstsq(Xdesign, LC, rcond=None)
    alpha = float(theta[0])
    beta = theta[1:].astype(np.float64)  # (2,)

    # Deterministic parts at obs
    Dh = np.zeros_like(Xh_obs, dtype=np.float64)
    Dh[:, 0] = 0.0  # EN' root
    Dh[:, 1] = 0.0  # GE' root
    Dh[:, 2] = alpha + beta[0]*EN + beta[1]*GE

    Uh = Xh_obs - Dh

    # Estimate HL noise stats from obs residuals (for all 3 coords)
    mu = Uh.mean(axis=0)
    sigma_diag = Uh.var(axis=0, ddof=0)

    model = HLModel(alpha=alpha, beta=beta, mu=mu, sigma_diag=sigma_diag)
    return model, {'X': Xh_obs, 'D': Dh, 'U': Uh}


def sample_hl(spec: LucasSpec, model: HLModel, eta: Optional[Dict[str, float]], n: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    eta = eta or {}

    # sample HL noise
    Uh = rng.normal(loc=model.mu, scale=np.sqrt(model.sigma_diag), size=(n, 3)).astype(np.float64)

    Dh = np.zeros((n, 3), dtype=np.float64)

    # EN' and GE' are roots: f=0, so D[:,0]=0, D[:,1]=0 unless intervened.
    # Apply do to EN' and GE'
    if "Environment'" in eta:
        Dh[:, 0] = float(eta["Environment'"])
        Uh[:, 0] = 0.0
    if "Genetics'" in eta:
        Dh[:, 1] = float(eta["Genetics'"])
        Uh[:, 1] = 0.0

    EN = Dh[:, 0] + Uh[:, 0]
    GE = Dh[:, 1] + Uh[:, 1]

    # LC' deterministic given EN, GE
    Dh[:, 2] = model.alpha + model.beta[0]*EN + model.beta[1]*GE

    # Apply do to LC' if present
    if "Lung Cancer'" in eta:
        Dh[:, 2] = float(eta["Lung Cancer'"])
        Uh[:, 2] = 0.0

    Xh = Dh + Uh
    return {'X': Xh, 'D': Dh, 'U': Uh}


# ----------------------------- packing for DIROCA -----------------------

def make_diroca_pack(spec: LucasSpec, model: HLModel, ll_data: Dict[str, Dict[str, np.ndarray]], hl_data: Dict[str, Dict[str, np.ndarray]]):
    return {
        'T': spec.T,
        'omega': spec.omega,
        'll': ll_data,
        'hl': hl_data,
        'hl_model': {
            'alpha': model.alpha,
            'beta': model.beta,
            'mu': model.mu,
            'sigma_diag': model.sigma_diag,
        },
        'graphs': {
            'll_edges': list(spec.ll_graph.edges()),
            'hl_edges': list(spec.hl_graph.edges()),
        }
    }

# ----------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=23)
    ap.add_argument('--n_per_int', type=int, default=5000, help='samples per intervention (LL & HL)')
    ap.add_argument('--out', type=str, default='data/lucas')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) Build spec
    spec = build_lucas(seed=args.seed)

    # 2) Generate LL observational and infer HL
    ll_obs = sample_ll(spec, iota=None, n=args.n_per_int, seed=args.seed)
    hl_model, hl_obs = infer_hl(spec, Xll_obs=ll_obs['X'])

    # 3) Generate per-intervention LL & HL using omega mapping
    ll_data = {'iota0': ll_obs}
    hl_data = {'eta0': hl_obs}

    # Use a deterministic per-intervention seed schedule for reproducibility
    # (different from global to avoid accidental coupling)
    base = args.seed * 7919  # a prime multiplier

    for iota_name, iota in spec.interventions_ll.items():
        if iota_name == 'iota0':
            continue  # already done obs
        seed_ll = base + hash(('ll', iota_name)) % (10**9)
        ll_data[iota_name] = sample_ll(spec, iota=iota, n=args.n_per_int, seed=seed_ll)

    for iota_name, eta_name in spec.omega.items():
        # ensure HL for all mapped etas
        eta = spec.interventions_hl[eta_name]
        if eta_name in hl_data:
            continue
        seed_hl = base + hash(('hl', eta_name)) % (10**9)
        hl_data[eta_name] = sample_hl(spec, hl_model, eta=eta, n=args.n_per_int, seed=seed_hl)

    # 4) Pack and save
    pack = make_diroca_pack(spec, hl_model, ll_data, hl_data)
    out_path = os.path.join(args.out, 'lucas_pack.pkl')
    joblib.dump(pack, out_path)

    # small human-friendly printout
    print('✓ LUCAS non-linear data generated')
    print(f'  seed           : {args.seed}')
    print(f'  samples/int    : {args.n_per_int}')
    print(f'  interventions  : {len(spec.interventions_ll)} LL  |  {len(spec.interventions_hl)} HL')
    print(f'  saved to       : {out_path}')


if __name__ == '__main__':
    main()
