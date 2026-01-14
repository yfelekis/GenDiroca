# Distributionally Robust Causal Abstractions (DiRoCA)

This repository contains the implementation and evaluation framework for the paper **"Distributionally Robust Causal Abstractions"**.

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yfelekis/GenDiroca.git
   cd DiRoCA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Experiments

The repository supports three main experimental settings: Linear (Synthetic), Non-Linear (nLUCAS & EBM), and High-Dimensional (Colored MNIST).

### 1. Linear Experiments

Standard synthetic experiments testing Gaussian vs. Empirical optimization. This category includes datasets with linear causal mechanisms.

#### Simple Lung Cancer (SLC)

The **SLC** dataset models causal relationships between three continuous variables: **S** (Smoking), **T** (Tar deposition), and **C** (Cancer). The low-level model has the structure S → T → C, while the high-level abstraction removes the intermediate variable T, resulting in a direct causal link S' → C'. The experiment includes 6 distinct low-level binary interventions mapped to 3 high-level interventions via the map $\omega$.

**Generate Data:**
```bash
python generate_data.py --experiment slc
```

**Run Optimization (Choose one):**
```bash
# Parametric (Gaussian) assumption
python gauss_optimization.py --experiment slc

# Non-parametric (Empirical) approach
python empirical_optimization.py --experiment slc
```

**Run Evaluation:**
```bash
# If using Gaussian optimization
python run_evaluation.py --experiment slc

# If using Empirical optimization
python run_empirical_evaluation.py --experiment slc
```

#### Linearized LUCAS (LiLUCAS)

The **LiLUCAS** dataset uses linear mechanisms to model lung cancer diagnosis scenarios. The low-level model involves six continuous variables: **SM** (Smoking), **GE** (Genetics), **LC** (Lung Cancer), **AL** (Allergy), **CO** (Coughing), and **FA** (Fatigue). The high-level abstraction groups these into three broader concepts: **EN'** (Environment), **GE'** (Genetics), and **LC'** (Lung Cancer). The experiment utilizes 21 relevant binary low-level interventions (including observational) mapped to 11 high-level interventions.

**Generate Data:**
```bash
python generate_data.py --experiment lilucas
```

**Run Optimization (Choose one):**
```bash
# Parametric (Gaussian) assumption
python gauss_optimization.py --experiment lilucas

# Non-parametric (Empirical) approach
python empirical_optimization.py --experiment lilucas
```

**Run Evaluation:**
```bash
# If using Gaussian optimization
python run_evaluation.py --experiment lilucas

# If using Empirical optimization
python run_empirical_evaluation.py --experiment lilucas
```

**Analyze Results:**
Open `huber_analysis.ipynb` or `contamination_analysis.ipynb` to visualize performance.

### 2. Non-Linear Experiments (nLUCAS & EBM)

Experiments involving non-linear mechanisms or real-world data.

#### nLUCAS (Non-Linear LUCAS)

The **nLUCAS** dataset shares the same causal structure and abstraction logic as LiLUCAS but employs non-linear relationships in the low-level model. The low-level model involves the same six variables (**SM**, **GE**, **LC**, **AL**, **CO**, **FA**), which are abstracted to the same three high-level concepts (**EN'**, **GE'**, **LC'**). Unlike LiLUCAS, nLUCAS employs a subset of 11 low-level interventions while keeping the same 11 high-level interventions.

**Generate Data:**
```bash
python generate_data_nlucas.py
```

**Optimization:**
```bash
python general_optimization.py --experiment nlucas
```

**Evaluation:**
```bash
python run_general_evaluation.py --experiment nlucas
```

**Analysis:**
Run the notebook `huber_lucas.ipynb`.

#### Electric Battery Manufacturing (EBM)

The **EBM** dataset applies the framework to a real-world scenario involving lithium-ion battery production. The dataset contains data from two experimental settings representing different levels of granularity: the low-level setting (WMG) captures the effect of a control variable (Comma Gap, **CG**) on Mass Loading outputs (**ML₁**, **ML₂**) at two distinct spatial locations, while the high-level setting (LRCS) relates the same control variable to a single aggregated output (**ML**). The low-level variables are $\mathbf{X}_L = [\text{CG}, \text{ML}_1, \text{ML}_2]^\top$, and the high-level variables are $\mathbf{X}_H = [\text{CG}, \text{ML}]^\top$. The framework uses parametric Additive Noise Models (ANMs) with linear mechanisms with intercepts to fit both datasets, performing exact abduction to recover noise terms. Interventions correspond to setting the Comma Gap to specific continuous values (e.g., 75, 100, 200 μm).

**Data Preparation:**
```bash
python generate_data_battery.py
```

**Optimization:**
```bash
python battery_optimization.py
```

**Evaluation:**
```bash
python run_battery_evaluation.py
```

**Analysis:**
Run the notebook `huber_battery.ipynb`.

### 3. High-Dimensional Experiment (Colored MNIST)

A complex vision task testing the framework on a high-dimensional, non-linear scenario. The objective is to learn a robust **linear** abstraction $T$ that aligns a complex pixel-level SCM with a disentangled latent-space SCM. Both levels operate on the same graph structure with parents **D** (Digit) and **C** (Color), which are correlated ($p=0.85$). The low-level target $X^\ell$ is instantiated as the high-dimensional image $I_P \in \mathbb{R}^{3072}$, while the high-level target $X^h$ is the compact latent code $z \in \mathbb{R}^{64}$ obtained via a pre-trained encoder $E_\phi$ from Xia et al. (2024). The experiment utilizes a set of 10 distinct interventions alongside observational data, including atomic interventions (e.g., do$(D=6)$) and joint interventions (e.g., do$(D=4, C=4)$) that break spurious correlations present in the observational distribution.

**SCM Construction:**
- **Low-Level Model ($f_L$):** U-Net with FiLM (Feature-wise Linear Modulation) layers that take a grayscale "Shape" image as input and inject causal control via parents $(D, C)$.
- **High-Level Model ($f_H$):** Vector-valued cell-means model that fits a deterministic vector $m_{d,c}$ for each $(d,c)$ combination in the discrete $10 \times 10$ parent space, with shrinkage regularization for robustness.

**Prerequisite:**:This experiment requires the pre-trained encoder $E_\phi$ to generate the ground-truth latent targets. The encoder is provided as a "frozen" TorchScript artifact in checkpoints/xia_color_mnist_rep/rep_encoder_only_traced.pt.

See checkpoints/model_training.md for the exact training command, hyperparameters, and the procedure used to extract this artifact. The original PyTorch Lightning checkpoint is preserved in the same directory for archival purposes.

*Overall:* Ensure `rep_encoder_only_traced.pt` (the pre-trained encoder) is available in the directory (see `generate_data_cmnist.py`).


**Generate Data:**
Generates the synthetic Colored MNIST dataset and extracts ground-truth latents using the pre-trained encoder.
```bash
python generate_data_cmnist.py
```

or

```bash
python cmnist_data_generation.py --encoder-ts checkpoints/xia_color_mnist_rep/rep_encoder_only_traced.pt
```

**Fit Abduction Models:**
Trains the Low-Level U-Net and High-Level Vector-Means models to abduce noise residuals.
```bash
python cmnist_fit_models.py
```

**Optimization:**
```bash
python cmnist_optimization.py
```

**Evaluation & Reporting:**
Evaluates the abstraction against Rotation, Lighting, and Noise shifts.
Output: Generates a PDF report (e.g., `cmnist_results_final_curve.pdf`).
```bash
python run_cmnist_evaluation.py
```

## Project Structure

```
DiRoCA/
├── README.md                           # This file
├── requirements.txt                    # Project dependencies
├── configs/                            # Configuration YAML files
├── third_party/                        # External dependencies (Xia et al. encoder)
├── plots/                              # Saved visualization plots
├── archive/                            # Archived scripts/results
├── checkpoints/                        # Pre-trained model artifacts
│   ├── model_training.md               # Documentation for pre-trained models
│   └── xia_color_mnist_rep/            # Checkpoint subfolder
│       ├── rep_encoder_only_traced.pt  # Frozen encoder for data generation
│       └── epoch=961-step=225108.ckpt  # Original training checkpoint (backup)

# --- Linear Experiments ---
├── generate_data.py                    # Data generation for linear SCMs
├── gauss_optimization.py               # Optimization using Gaussian distribution
├── empirical_optimization.py           # Optimization using Empirical distribution
├── run_evaluation.py                   # Evaluation for Gaussian pipeline
├── run_empirical_evaluation.py         # Evaluation for Empirical pipeline
├── huber_analysis.ipynb                # Analysis notebook for linear experiments
├── contamination_analysis.ipynb        # Analysis notebook for linear experiments

# --- Non-Linear Experiments (nLUCAS) ---
├── generate_data_nlucas.py             # nLUCAS data generation
├── nlucas_optimization.py              # Optimization script for nLUCAS/General cases
├── run_general_evaluation.py           # Evaluation script for nLUCAS/General cases
├── huber_lucas.ipynb                   # nLUCAS results analysis

# --- Real-World Surrogate (EBM) ---
├── generate_data_battery.py            # Battery dataset preparation
├── battery_optimization.py             # Battery experiment optimization
├── run_battery_evaluation.py           # Battery experiment evaluation
├── huber_battery.ipynb                 # Battery results analysis

# --- Vision Experiments (CMNIST) ---
├── generate_data_cmnist.py             # CMNIST generation & encoder extraction
├── cmnist_fit_models.py                # Fits f_L (U-Net) and f_H (Cell-Means)
├── cmnist_optimization.py              # Optimization for CMNIST
├── run_cmnist_evaluation.py            # Generates PDF report for shifts (Rotation/Lighting)

# --- Utilities & Shared Modules ---
├── models.py                           # Model architectures
├── operations.py                       # Causal interventions logic
├── opt_tools.py                        # Optimization solvers
├── math_utils.py                       # Mathematical helper functions
├── utilities.py                        # General IO and logging utils
├── evaluation_utils.py                 # Metrics and plotting helpers
├── load_encoder.py                     # Helper to load Xia et al. encoder
├── make_folds.py                       # Cross-validation utility
└── load_models_cell.py                 # Loading utilities for cell-based models
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{diroca2026,
  title={Distributionally Robust Causal Abstractions},
  author={Felekis, Yorgos and Damoulas, Theodoros and Giampouras, Paris},
  journal={arXiv preprint},
  year={2026}
}
```
