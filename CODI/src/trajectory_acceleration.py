"""
Trajectory Acceleration (Second-Order Smoothness) Constraint.

Constrains latent trajectories to avoid sudden changes in velocity by
penalizing large second-order differences (accelerations).
"""

from typing import Dict

import torch
import torch.nn as nn


# =========================
# Trajectory Acceleration Core
# =========================

class TrajectoryAccelerationCore:
    """
    Core logic for second-order smoothness (acceleration) constraint.

    Given latent tokens z_0..z_{T-1}, define velocities:
        v_k = z_{k+1} - z_k
    and accelerations:
        a_k = v_{k+1} - v_k = z_{k+2} - 2 z_{k+1} + z_k

    Loss = mean(max(0, ||a_k|| - max_acceleration))
    """

    def __init__(self, max_acceleration: float = 1.0, eps: float = 1e-8):
        self.max_acceleration = float(max_acceleration)
        self.eps = float(eps)

    def accelerations(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute accelerations for a batch of trajectories.

        Args:
            X: [B,T,D] tensor
        Returns:
            a: [B,T-2,D] tensor of accelerations
        """
        if X.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B,T,D], got {X.dim()}D")
        if X.size(1) < 3:
            return X.new_zeros((X.size(0), 0, X.size(2)))

        z_prev = X[:, :-2, :]
        z_curr = X[:, 1:-1, :]
        z_next = X[:, 2:, :]
        return z_next - 2 * z_curr + z_prev

    def loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Acceleration constraint loss.

        Args:
            X: [B,T,D] batch tensor
        Returns:
            scalar loss
        """
        if X.size(1) < 3:
            # No acceleration if fewer than 3 steps
            return X.new_tensor(0.0)

        a = self.accelerations(X)
        accel_norm = torch.linalg.norm(a, dim=-1)
        violation = torch.clamp(accel_norm - self.max_acceleration, min=0.0)
        return violation.mean()


# =========================
# nn.Module Wrapper
# =========================

class TrajectoryAccelerationLoss(nn.Module):
    """
    Training-time module for second-order smoothness (acceleration) loss.

    Input shape: [T,B,D] where T=num_latent_tokens, B=batch_size, D=hidden_dim
    Internally transposed to [B,T,D] for computation.
    """

    def __init__(self, max_acceleration: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.core = TrajectoryAccelerationCore(max_acceleration=max_acceleration, eps=eps)

    def forward(self, latent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration smoothness loss.

        Args:
            latent_embeddings: [T,B,D] tensor
        Returns:
            scalar loss tensor
        """
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T,B,D], got {latent_embeddings.dim()}D. "
                f"Shape: {latent_embeddings.shape}"
            )

        X = latent_embeddings.transpose(0, 1)  # [B,T,D]
        return self.core.loss(X)

    @torch.no_grad()
    def compute_stats(self, latent_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute acceleration statistics for logging/analysis.

        Returns:
            dict with keys:
              - accel_norm: [B,T-2] acceleration magnitudes
              - accel_max: scalar, max acceleration magnitude
              - accel_mean: scalar, mean acceleration magnitude
              - accel_std: scalar, std of acceleration magnitudes
              - violation_count: scalar, number of violations
              - violation_rate: scalar, fraction of violations
              - max_acceleration: scalar, the threshold used
        """
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T,B,D], got {latent_embeddings.dim()}D"
            )

        X = latent_embeddings.transpose(0, 1)  # [B,T,D]
        a = self.core.accelerations(X)
        if a.numel() == 0:
            zero = X.new_tensor(0.0)
            return {
                "accel_norm": a.new_zeros((X.size(0), 0)),
                "accel_max": zero,
                "accel_mean": zero,
                "accel_std": zero,
                "violation_count": zero,
                "violation_rate": zero,
                "max_acceleration": X.new_tensor(self.core.max_acceleration),
            }

        accel_norm = torch.linalg.norm(a, dim=-1)
        violations = accel_norm > self.core.max_acceleration

        return {
            "accel_norm": accel_norm,
            "accel_max": accel_norm.max(),
            "accel_mean": accel_norm.mean(),
            "accel_std": accel_norm.std(),
            "violation_count": violations.sum(),
            "violation_rate": violations.float().mean(),
            "max_acceleration": X.new_tensor(self.core.max_acceleration),
        }
