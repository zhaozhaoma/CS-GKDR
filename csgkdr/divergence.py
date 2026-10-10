from __future__ import annotations
from typing import Optional, Tuple

import torch


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    ||x - y||^2 for all pairs.
    x: [N,d], y: [M,d] => [N,M]
    """
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    xy = x @ y.t()
    d2 = (x2 + y2 - 2.0 * xy).clamp(min=0.0)
    return d2


@torch.no_grad()
def median_heuristic_sigma(x: torch.Tensor, y: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """
    Median heuristic for Gaussian kernel bandwidth sigma.
    """
    if y is None:
        y = x
    N = x.size(0)
    M = y.size(0)
    if N > 512:
        idx = torch.randperm(N, device=x.device)[:512]
        x = x[idx]
    if M > 512:
        idx = torch.randperm(M, device=y.device)[:512]
        y = y[idx]
    d2 = pairwise_sq_dists(x, y)
    med = torch.median(d2)
    sigma = torch.sqrt(med.clamp(min=eps))
    return sigma


def get_gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    *,
    sigma_min: float = 1e-3,
    sigma_max: float = 1e3,
    dist2_clamp: float = 1e6,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian kernel matrix K(x,y) with numerical guards.
    Returns:
      K: [N,M]
      sigma_used: scalar tensor
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if sigma is None:
        sigma = median_heuristic_sigma(x, y, eps=eps)

    sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    denom = 2.0 * (sigma * sigma).clamp(min=sigma_min * sigma_min) + eps

    d2 = pairwise_sq_dists(x, y).clamp(max=dist2_clamp)
    K = torch.exp(-d2 / denom)

    K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    return K, sigma


def cs_qmi(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigma: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    CS-based Quadratic Mutual Information (CS-QMI).
    Estimates I(X;Y) using CS divergence.
    
    CS-QMI(X;Y) = D_CS(p(x,y) || p(x)p(y))
    
    CRITICAL FIX: This implementation now follows the structure of Theorem 1 
    (Eq. 33 in paper) specialized to the two-variable case, ensuring consistency
    with cs_total_correlation.
    
    x: [N, d_x]
    y: [N, d_y]
    Returns: scalar mutual information estimate
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have same batch size, got {x.shape[0]} vs {y.shape[0]}")
    
    N = x.shape[0]
    
    # Concatenate for joint
    xy = torch.cat([x, y], dim=1)  # [N, d_x + d_y]
    
    # Compute kernels
    if sigma is None:
        sigma_x = median_heuristic_sigma(x, x, eps=eps)
        sigma_y = median_heuristic_sigma(y, y, eps=eps)
        sigma_xy = torch.sqrt(sigma_x * sigma_x + sigma_y * sigma_y)
    else:
        sigma_xy = sigma
        sigma_x = sigma
        sigma_y = sigma
    
    K_xy, _ = get_gaussian_kernel(xy, xy, sigma=sigma_xy, eps=eps)
    K_x, _ = get_gaussian_kernel(x, x, sigma=sigma_x, eps=eps)
    K_y, _ = get_gaussian_kernel(y, y, sigma=sigma_y, eps=eps)
    
    # ============================================
    # CRITICAL FIX: Use Theorem 1 structure for two variables
    # ============================================
    # Following Eq. (33) with M=2:
    # term1 = (1/N^2) Σ_{i,j} Q_1(i,j) * Q_2(i,j)
    # term2 = (1/N^4) Σ_{i1,j1,i2,j2} Q_1(i1,j1) * Q_2(i2,j2)
    # term3 = (1/N^3) Σ_{i,j1,j2} Q_1(i,j1) * Q_2(i,j2)
    
    # term1: joint self-inner-product
    v_joint = K_xy.mean().clamp(min=eps)
    
    # term2: product of marginal self-inner-products
    v_x = K_x.mean().clamp(min=eps)
    v_y = K_y.mean().clamp(min=eps)
    v_indep = (v_x * v_y).clamp(min=eps)
    
    # term3: cross inner product (CORRECTED)
    # Following the structure in cs_total_correlation's term3:
    # (1/N^{M+1}) Σ_{i,j_1,...,j_M} ∏_k Q_k(i,j_k)
    # For M=2: (1/N^3) Σ_i [Σ_{j1} Q_1(i,j1)] * [Σ_{j2} Q_2(i,j2)]
    # = (1/N) Σ_i [mean_j Q_1(i,j)] * [mean_j Q_2(i,j)]
    row_mean_x = K_x.mean(dim=1).clamp(min=eps)  # [N]
    row_mean_y = K_y.mean(dim=1).clamp(min=eps)  # [N]
    v_cross = (row_mean_x * row_mean_y).mean().clamp(min=eps)
    
    # CS-QMI formula (Eq. 35 in paper)
    # D_CS(p||q) = -log( (∫pq)^2 / (∫p^2 * ∫q^2) )
    numerator = v_cross * v_cross
    denominator = (v_joint * v_indep).clamp(min=eps)
    ratio = (numerator / denominator).clamp(min=eps)
    
    qmi = -torch.log(ratio)
    return qmi.clamp(min=0.0)


def cs_total_correlation(
    *variables: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Full multivariate Total Correlation using CS divergence (Theorem 1 in paper).
    
    TC(U^1,...,U^M) = D_CS(p(u^1,...,u^M) || ∏_m p(u^m))
    
    Implements Eq. (33):
    TC = log(term1) + log(term2) - 2*log(term3)
    
    where:
    - term1 = (1/N^2) ∑_{i,j} ∏_k Q_k(i,j)
    - term2 = (1/N^{2M}) ∑_{i_1,j_1,...,i_M,j_M} ∏_k Q_k(i_k,j_k)
    - term3 = (1/N^{M+1}) ∑_{i,j_1,...,j_M} ∏_k Q_k(i,j_k)
    
    Args:
        *variables: M tensors of shape [N, d_m]
        sigma: optional bandwidth
        eps: numerical stability constant
    
    Returns:
        scalar TC value
    """
    if len(variables) < 2:
        raise ValueError("TC requires at least 2 variables")
    
    M = len(variables)
    N = variables[0].shape[0]
    
    # Verify all have same batch size
    for i, var in enumerate(variables):
        if var.shape[0] != N:
            raise ValueError(f"All variables must have same batch size, variable {i} has {var.shape[0]} vs {N}")
    
    # Compute Gram matrices Q_k for each modality
    Q_list = []
    for k, u_k in enumerate(variables):
        if sigma is None:
            sig_k = median_heuristic_sigma(u_k, u_k, eps=eps)
        else:
            sig_k = sigma
        K_k, _ = get_gaussian_kernel(u_k, u_k, sigma=sig_k, eps=eps)
        Q_list.append(K_k)
    
    # Term 1: (1/N^2) ∑_{i,j} ∏_k Q_k(i,j)
    Q_prod = torch.ones_like(Q_list[0])
    for Q_k in Q_list:
        Q_prod = Q_prod * Q_k
    term1 = Q_prod.mean().clamp(min=eps)
    
    # Term 2: (1/N^{2M}) ∑_{i_1,j_1,...,i_M,j_M} ∏_k Q_k(i_k,j_k)
    # This is equivalent to ∏_k mean(Q_k)
    term2 = torch.tensor(1.0, device=variables[0].device, dtype=variables[0].dtype)
    for Q_k in Q_list:
        term2 = term2 * Q_k.mean().clamp(min=eps)
    term2 = term2.clamp(min=eps)
    
    # Term 3: (1/N^{M+1}) ∑_{i,j_1,...,j_M} ∏_k Q_k(i,j_k)
    # For each i, compute mean over all j's: (1/N) ∑_j Q_k(i,j)
    # Then multiply across modalities and average over i
    term3_vals = torch.ones(N, device=variables[0].device, dtype=variables[0].dtype)
    for Q_k in Q_list:
        # Mean over columns for each row
        term3_vals = term3_vals * Q_k.mean(dim=1).clamp(min=eps)
    term3 = term3_vals.mean().clamp(min=eps)
    
    # CS-TC formula (Eq 33)
    tc = torch.log(term1) + torch.log(term2) - 2.0 * torch.log(term3)
    
    return tc.clamp(min=0.0)
