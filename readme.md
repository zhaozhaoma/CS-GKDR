## Interpretable Multimodal Brain Age Prediction via Cauchy-Schwarz Divergence-Induced Gács-Körner Common Information

## Overview
<p align="center">
  <img width="2817" height="1454" alt="image" src="https://github.com/user-attachments/assets/e837131d-3498-4778-800d-019261b7fd9a" />
</p>

| |
|:--:|
| *Framework of **C**auchy–**S**chwarz **G**ács–**K**örner **D**ecomposition for **R**egression (**CS-GKDR**). Each modality is encoded into common and modality-specific representations, which are combined to predict brain age.* |

CS-GKDR is an information-theoretic framework for interpretable multimodal brain age prediction. Rather than directly concatenating heterogeneous inputs, CS-GKDR learns a structured latent space that distinguishes information shared across modalities from information specific to each modality. The framework consists of three complementary components:

- ∩ **Common Information**  
  A shared representation encouraged to preserve information consistently recoverable from different modalities, based on an approximate multivariate Gács–Körner common-information objective.

- ⊕ **Unique Information**  
  Modality-specific representations that retain complementary signals. A total-correlation regularizer discourages statistical dependence among these representations.

- ∑ **Cauchy–Schwarz Information Measures**  
  Kernel-based estimators of mutual information and total correlation that support learning from continuous, high-dimensional representations without assuming a parametric density model.

By combining the learned common representation with all modality-specific representations, CS-GKDR (i) provides a more interpretable account of multimodal contributions and (ii) supports continuous brain age regression. 

## Key Features

| ✦ | Feature | Description |
|:--:|:--|:--|
| ① | **Multivariate Common Information** | Extends an approximate Gács–Körner common-information objective to more than two modalities, without choosing a privileged anchor modality. |
| ② | **Cauchy–Schwarz Information Estimation** | Uses CS-divergence-based estimators for mutual information and total correlation in continuous latent spaces. |
| ③ | **Common–Unique Decomposition** | Learns shared and modality-specific features with latent-space reconstruction and total-correlation regularization. |
| ④ | **Multimodal Neuroimaging Encoders** | Combines a graph-based encoder for resting-state fMRI, a 3D CNN for structural MRI, and an optional clinical-feature encoder. |
| ⑤ | **Interpretability and Evaluation** | Supports representation-dependence diagnostics, age-bias correction. |

## Installation

1. Repository preparation:

```bash
git clone https://github.com/zhaozhaoma/CS-GKDR.git
cd CS-GKDR
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. The implementation targets Python 3.12 and PyTorch 2.x. A GPU <img src="https://github.com/user-attachments/assets/1c9ba16d-ec0b-4f06-b58e-535b7050db49" width="36" alt="RTX 4090" align="middle" /> with at least **24 GB** of memory is recommended for the 3D sMRI encoder at the default batch size. 


## Data Format

> [!TIP]
> **REST-meta-MDD Dataset Access**
>
> The REST-meta-MDD dataset is available through the [official project website](https://rfmri.org/REST-meta-MDD). Researchers must complete the [Data Use Agreement](https://rfmri.org/sites/default/files/REST-meta-MDD_Data_Use_Agreement.pdf) and submit the signed form to obtain the necessary access credentials.

Once you have obtained the dataset, this project expects the following:

```text
data/REST-meta-MDD/
├── fMRI/AAL/
│   └── <subID>.npy
├── VBM/wc1/
│   └── <subID>.nii.gz
├── metadata.csv
└── REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx  # Trimodal only
```

| Input | Expected Format | Modality |
|:--|:--|:--:|
| **Resting-state fMRI** | `<subID>.npy`: AAL regional time series of shape **`[116, T]`** (116 regions × T time points), converted into functional-connectivity graphs | **fMRI** |
| **Structural MRI** | `wc1/<subID>.nii.gz`: VBM gray-matter density volume in MNI space, loaded from `data.smri_data_dir/wc1/` by default | **sMRI** |
| **Subject metadata** | `metadata.csv` with `subID`, `age`, `sex`, `site`, and `label`; `site` must be integer-coded (e.g., `17` for S17), with `sex` coded as 1 = male and 2 = female | **Metadata / Target** |
| **Clinical / phenotypic data** | Excel file with `MDD` and `Controls` sheets, processed into a 39-dimensional vector; required **only** for the trimodal model | **Optional Third Modality** |

To enable the third modality, use `--trimodal` and set `data.clinical_excel_path` to an existing Excel file. The clinical preprocessor uses the source columns `ID`, `Sex`, `Education (years)`, and `*HAMD1`–`*HAMD17` (HAMD items may be absent for controls). The 39-dimensional feature vector comprises sex one-hot encoding (2), standardized education (1), normalized HAMD-17 scores (17), and missing-value indicators (19). Imputation and scaling statistics are fitted on training subjects only.

## Model Components

| File | Purpose | Key Functionalities |
|:--|:--|:--|
| **`csgkdr/models/encoders.py`** | Modality-specific backbones | • GIN-based functional-connectivity graph encoder for **fMRI**<br>• SFCN-based volumetric encoder for **sMRI**<br>• Optional MLP encoder for clinical features |
| **`csgkdr/models/csgkdr.py`** | Core CS-GKDR model | • Common and unique representation encoders<br>• Latent reconstruction decoders<br>• Multimodal fusion and age prediction head |
| **`csgkdr/divergence.py`** | Information-theoretic estimators | • Gaussian-kernel CS quadratic mutual information<br>• CS-based total correlation across multiple representations |
| **`csgkdr/objective.py`** | Learning objective | • Regression and feature-reconstruction losses<br>• Common-information and unique-independence regularization<br>• Variance and correlation penalties against representation collapse |
| **`csgkdr/data/`** | Data processing | • fMRI graph and sMRI volume preprocessing<br>• Site-disjoint train/validation/test splits<br>• Optional clinical feature processing |
| **`csgkdr/trainer.py`** | Training orchestration | • Validation-based early stopping<br>• Checkpointing, resumption, and optimization diagnostics |
| **`csgkdr/evaluation.py`** | Evaluation and interpretability | • Age prediction metrics and bias correction<br>• Representation-dependence analysis<br>• Brain-PAD group comparisons |
| **`main.py`** | Experiment interface | • `train`, `evaluate`, `analyze`, and `summary` commands<br>• Configuration overrides and ablation variants |

Each modality is mapped to common and modality-specific latent representations. During training, the age predictor receives the **common representation from one randomly selected modality** (one selection per batch); during inference, it receives the **mean of the common representations** across modalities. In both cases, the predictor also receives the unique representations from all enabled modalities.

## Usage

1. Basic Training

```bash
python main.py train --config configs/default.json
```

2. Advanced Configuration

```bash
# Include the optional clinical modality (requires the phenotype Excel file)
python main.py train --trimodal

# Choose another predefined site split and random seed
python main.py train --split exp_04 --seed 43

# Specify validation/test sites directly
python main.py train \
    --set 'split.test_sites=["S17","S10"]' \
    --set 'split.val_sites=["S4","S9"]'

# Modify training or information-theoretic hyperparameters
python main.py train \
    --set training.n_epochs=200 \
    --set training.batch_size=10 \
    --set loss.tc=0.1 \
    --set model.common_dim=64 \
    --name csgkdr_custom
```

3. Evaluation and Analysis

```bash
# Re-evaluate a trained model
python main.py evaluate --run runs/exp_05_bi_full_seed42

# Analyze representation decomposition and Brain-PAD
python main.py analyze --run runs/exp_05_bi_full_seed42

# Aggregate metrics from multiple runs
python main.py summary --runs \
    runs/exp_05_bi_full_seed42 \
    runs/exp_05_bi_full_seed43
```

Trained checkpoints, per-subject predictions, evaluation metrics, and analysis reports are stored under `runs/<run_name>/`. 


<!--
## Citation

If you use CS-GKDR in your research, we would appreciate your citation 📃:

```bibtex
@misc{,
  title={Interpretable Multimodal Brain Age Prediction via Cauchy-Schwarz Divergence-Induced Gács-Körner Common Information},
  author={},
  year={},
  note={}
}
```
-->


---
<h1 align="center">👷 🧱 🏗️ ✨</h1>
<p align="center">Under construction</p>
