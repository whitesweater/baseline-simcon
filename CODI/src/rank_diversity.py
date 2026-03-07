"""
Rank Diversity Loss for CODI Latent Tokens.

Addresses the rank collapse problem: as latent token count increases (7→16→32),
effective rank grows sublinearly (4.19→7.25→9.14), indicating tokens collapse
into a low-dimensional subspace.

Two complementary loss modes are provided:
  1. SVD Entropy Loss (default): Directly maximizes effective rank by maximizing
     the entropy of normalized singular values. Most principled approach.
  2. Cosine Diversity Loss: Penalizes pairwise cosine similarity between latent
     tokens. Simpler and helps gradient flow.

Reference:
  - Effective rank: Roy & Bhattacharyya (2007), "Effective rank: a measure
    of effective dimensionality." EURASIP J. Adv. Signal Process.
    erank(M) = exp(H(ŝ)) where ŝ_i = σ_i / Σ σ_j
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankDiversityLoss(nn.Module):
    """
    Promotes diversity / high effective rank among latent token embeddings.

    Input shape: [T, B, D] where T = num_latent_tokens, B = batch_size, D = hidden_dim.
    Internally transposed to [B, T, D] to compute per-batch losses.

    Args:
        mode: Loss mode, one of {"svd_entropy", "cosine", "combined"}.
            - "svd_entropy": Minimize 1 - H(ŝ)/log(K) where ŝ = normalized singular values.
            - "cosine": Minimize mean |cos(z_i, z_j)| for i ≠ j.
            - "combined": Weighted sum of both (weights: svd_weight + cosine_weight).
        svd_weight: Weight for svd_entropy component when mode="combined". Default 1.0.
        cosine_weight: Weight for cosine component when mode="combined". Default 0.5.
        eps: Numerical stability constant.
        center_before_svd: Whether to center the latent matrix (subtract mean) before SVD.
            Centering makes the loss measure spread of directions rather than offset.
    """

    VALID_MODES = {"svd_entropy", "cosine", "combined"}

    def __init__(
        self,
        mode: str = "svd_entropy",
        svd_weight: float = 1.0,
        cosine_weight: float = 0.5,
        eps: float = 1e-8,
        center_before_svd: bool = True,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode
        self.svd_weight = svd_weight
        self.cosine_weight = cosine_weight
        self.eps = eps
        self.center_before_svd = center_before_svd

    # ------------------------------------------------------------------
    # Core loss computations
    # ------------------------------------------------------------------

    def _svd_entropy_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        SVD-entropy based rank diversity loss.

        Args:
            X: [B, K, D] latent token matrix (K tokens, D hidden dim)

        Returns:
            Scalar loss in [0, 1]. 0 = perfectly uniform singular values (max rank),
            1 = all singular values concentrated in one direction (rank 1).
        """
        B, K, D = X.shape

        if K <= 1:
            # Single token: no diversity to promote
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)

        if self.center_before_svd:
            X = X - X.mean(dim=1, keepdim=True)

        # SVD: compute singular values only (more efficient)
        # S shape: [B, min(K, D)]
        S = torch.linalg.svdvals(X)

        # Only the first K singular values matter (K << D typically)
        S = S[:, :K]

        # Normalize to get a probability distribution
        S_sum = S.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        S_hat = S / S_sum  # [B, K]

        # Shannon entropy of normalized singular values
        # H = -Σ ŝ_i log(ŝ_i)
        log_S_hat = torch.log(S_hat + self.eps)
        entropy = -(S_hat * log_S_hat).sum(dim=-1)  # [B]

        # Normalize by max entropy: log(K)
        max_entropy = math.log(K)

        # Loss = 1 - H/H_max
        # = 0 when all singular values equal (max effective rank)
        # = 1 when rank collapses to 1
        loss = 1.0 - entropy.mean() / max_entropy

        return loss

    def _cosine_diversity_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity based diversity loss.

        Penalizes high absolute cosine similarity between pairs of latent tokens.

        Args:
            X: [B, K, D] latent token matrix

        Returns:
            Scalar loss in [0, 1]. 0 = all tokens orthogonal, 1 = all tokens parallel.
        """
        B, K, D = X.shape

        if K <= 1:
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)

        # L2 normalize along feature dimension
        X_norm = F.normalize(X, p=2, dim=-1)  # [B, K, D]

        # Pairwise cosine similarity matrix
        cos_sim = torch.bmm(X_norm, X_norm.transpose(1, 2))  # [B, K, K]

        # Mask out diagonal (self-similarity = 1)
        mask = ~torch.eye(K, dtype=torch.bool, device=X.device).unsqueeze(0)  # [1, K, K]

        # Mean absolute off-diagonal cosine similarity
        off_diag = cos_sim.masked_select(mask).reshape(B, K * (K - 1))
        loss = off_diag.abs().mean()

        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, latent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute rank diversity loss.

        Args:
            latent_embeddings: [T, B, D] tensor
                T = num latent tokens, B = batch size, D = hidden dim.

        Returns:
            Scalar loss tensor.
        """
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T, B, D], got {latent_embeddings.dim()}D. "
                f"Shape: {latent_embeddings.shape}"
            )

        # [T, B, D] -> [B, T, D]
        X = latent_embeddings.transpose(0, 1)

        if self.mode == "svd_entropy":
            return self._svd_entropy_loss(X)
        elif self.mode == "cosine":
            return self._cosine_diversity_loss(X)
        elif self.mode == "combined":
            svd_loss = self._svd_entropy_loss(X)
            cos_loss = self._cosine_diversity_loss(X)
            return self.svd_weight * svd_loss + self.cosine_weight * cos_loss
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    # Diagnostics (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_stats(self, latent_embeddings: torch.Tensor) -> dict:
        """
        Compute rank diversity statistics for logging.

        Args:
            latent_embeddings: [T, B, D] tensor

        Returns:
            Dict with keys: effective_rank, svd_entropy, svd_entropy_loss,
            cosine_mean, cosine_max, singular_values_normalized.
        """
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T, B, D], got {latent_embeddings.dim()}D"
            )

        X = latent_embeddings.transpose(0, 1)  # [B, T, D]
        B, K, D = X.shape

        stats = {}

        # SVD stats
        X_centered = X - X.mean(dim=1, keepdim=True) if self.center_before_svd else X
        S = torch.linalg.svdvals(X_centered)[:, :K]  # [B, K]
        S_sum = S.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        S_hat = S / S_sum
        entropy = -(S_hat * torch.log(S_hat + self.eps)).sum(dim=-1)  # [B]
        erank = torch.exp(entropy)  # [B]

        stats["effective_rank_mean"] = erank.mean()
        stats["effective_rank_std"] = erank.std()
        stats["effective_rank_min"] = erank.min()
        stats["effective_rank_max"] = erank.max()
        stats["svd_entropy_mean"] = entropy.mean()
        stats["svd_entropy_loss"] = 1.0 - entropy.mean() / math.log(max(K, 2))
        stats["num_tokens"] = torch.tensor(K, dtype=torch.long, device=X.device)

        # Top singular value ratio (concentration indicator)
        stats["top_sv_ratio"] = (S[:, 0] / S_sum.squeeze(-1)).mean()

        # Cosine stats
        if K > 1:
            X_norm = F.normalize(X, p=2, dim=-1)
            cos_sim = torch.bmm(X_norm, X_norm.transpose(1, 2))
            mask = ~torch.eye(K, dtype=torch.bool, device=X.device).unsqueeze(0)
            off_diag = cos_sim.masked_select(mask).reshape(B, -1)
            stats["cosine_sim_mean"] = off_diag.abs().mean()
            stats["cosine_sim_max"] = off_diag.abs().max()
        else:
            stats["cosine_sim_mean"] = torch.tensor(0.0, device=X.device)
            stats["cosine_sim_max"] = torch.tensor(0.0, device=X.device)

        return stats
