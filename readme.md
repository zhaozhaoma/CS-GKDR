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



---
<h1 align="center">👷 🧱 🏗️ ✨</h1>
<p align="center">Under construction</p>
