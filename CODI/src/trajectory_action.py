"""
Trajectory Least Action (Path Energy) Constraint.

Minimize path energy = kinetic (velocity magnitude) + potential (deviation from center).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class TrajectoryActionCore:
    """
    Core logic for least action (path energy) constraint.

    Energy = lambda_energy * E[||v||^2] + lambda_length * E[||z - center||^2]
    """

    def __init__(self, lambda_energy: float = 1.0, lambda_length: float = 0.1, eps: float = 1e-8):
        self.lambda_energy = float(lambda_energy)
        self.lambda_length = float(lambda_length)
        self.eps = float(eps)

    def compute_energy_terms(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute kinetic, potential, and total energy for batch trajectories.

        Args:
            X: [B,T,D]
        Returns:
            kinetic: [B]
            potential: [B]
            total: [B]
        """
        if X.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B,T,D], got {X.dim()}D")
        if X.size(1) < 2:
            kinetic = X.new_zeros((X.size(0),))
            potential = X.new_zeros((X.size(0),))
            total = X.new_zeros((X.size(0),))
            return kinetic, potential, total

        center = X.mean(dim=1, keepdim=True)  # [B,1,D]

        v = X[:, 1:, :] - X[:, :-1, :]  # [B,T-1,D]
        kinetic = (v.pow(2).sum(dim=-1)).mean(dim=-1)  # [B]

        displacement = X - center  # [B,T,D]
        potential = (displacement.pow(2).sum(dim=-1)).mean(dim=-1)  # [B]

        total = self.lambda_energy * kinetic + self.lambda_length * potential
        return kinetic, potential, total

    def loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Least action loss.

        Args:
            X: [B,T,D]
        Returns:
            scalar loss
        """
        _, _, total = self.compute_energy_terms(X)
        return total.mean()


class TrajectoryActionLoss(nn.Module):
    """
    Training-time module for least action (path energy) loss.

    Input shape: [T,B,D] where T=num_latent_tokens, B=batch_size, D=hidden_dim
    Internally transposed to [B,T,D] for computation.
    """

    def __init__(self, lambda_energy: float = 1.0, lambda_length: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.core = TrajectoryActionCore(lambda_energy=lambda_energy, lambda_length=lambda_length, eps=eps)

    def forward(self, latent_embeddings: torch.Tensor) -> torch.Tensor:
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T,B,D], got {latent_embeddings.dim()}D. "
                f"Shape: {latent_embeddings.shape}"
            )
        X = latent_embeddings.transpose(0, 1)  # [B,T,D]
        return self.core.loss(X)

    @torch.no_grad()
    def compute_stats(self, latent_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T,B,D], got {latent_embeddings.dim()}D"
            )
        X = latent_embeddings.transpose(0, 1)  # [B,T,D]
        kinetic, potential, total = self.core.compute_energy_terms(X)
        return {
            "kinetic": kinetic,
            "potential": potential,
            "total": total,
            "kinetic_mean": kinetic.mean(),
            "potential_mean": potential.mean(),
            "total_mean": total.mean(),
            "kinetic_max": kinetic.max(),
            "potential_max": potential.max(),
            "total_max": total.max(),
            "lambda_energy": X.new_tensor(self.core.lambda_energy),
            "lambda_length": X.new_tensor(self.core.lambda_length),
        }
