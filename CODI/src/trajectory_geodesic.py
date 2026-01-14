"""
Trajectory Geodesic Deviation Constraint (Hyperbolic Space).

Penalize deviation from the geodesic in Poincaré ball between the start
and end latent states.
"""

from typing import Dict

import torch
import torch.nn as nn

from src.trajectory_consistency import GeometryConfig, HyperbolicGeometry


class TrajectoryGeodesicDeviationCore:
    """
    Core logic for geodesic deviation loss in hyperbolic space.

    For Poincaré ball, the geodesic between x and y can be written as:
        gamma(t) = x ⊕ exp_0( t * log_0( (-x) ⊕ y ) )
    where ⊕ is Möbius addition, and log_0/exp_0 are maps at the origin.
    Deviation is average hyperbolic distance to the geodesic.
    """

    def __init__(self, curvature: float = -1.0, eps: float = 1e-8):
        cfg = GeometryConfig(curvature=curvature, eps=eps)
        self.geo = HyperbolicGeometry(cfg)

    def compute_geodesic(self, z_start: torch.Tensor, z_end: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Compute geodesic points between z_start and z_end in Poincaré ball.

        Args:
            z_start: [B,D]
            z_end:   [B,D]
            num_points: int
        Returns:
            geodesic: [B,T,D]
        """
        if num_points <= 1:
            return z_start.unsqueeze(1)

        z_start = self.geo.project(self.geo._safe_normalize(z_start))
        z_end = self.geo.project(self.geo._safe_normalize(z_end))

        u = self.geo.mobius_add(-z_start, z_end)  # [B,D]
        v = self.geo.log0(u)  # [B,D]

        t = torch.linspace(0.0, 1.0, steps=num_points, device=z_start.device, dtype=z_start.dtype)
        t = t.view(1, num_points, 1)
        v_t = v.unsqueeze(1) * t  # [B,T,D]
        exp_vt = self.geo.exp0(v_t)  # [B,T,D]
        geodesic = self.geo.mobius_add(z_start.unsqueeze(1), exp_vt)  # [B,T,D]
        return geodesic

    def loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Geodesic deviation loss (hyperbolic distance).

        Args:
            X: [B,T,D]
        Returns:
            scalar loss
        """
        if X.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B,T,D], got {X.dim()}D")
        if X.size(1) < 2:
            return X.new_tensor(0.0)

        Xn = self.geo.project(self.geo._safe_normalize(X))
        z_start = Xn[:, 0, :]
        z_end = Xn[:, -1, :]
        geodesic = self.compute_geodesic(z_start, z_end, Xn.size(1))  # [B,T,D]
        deviation = self.geo.distance(Xn, geodesic)  # [B,T]
        return deviation.mean()


class TrajectoryGeodesicDeviationLoss(nn.Module):
    """
    Training-time module for geodesic deviation loss (hyperbolic).

    Input shape: [T,B,D] where T=num_latent_tokens, B=batch_size, D=hidden_dim
    Internally transposed to [B,T,D] for computation.
    """

    def __init__(self, curvature: float = -1.0, eps: float = 1e-8):
        super().__init__()
        self.core = TrajectoryGeodesicDeviationCore(curvature=curvature, eps=eps)

    def forward(self, latent_embeddings: torch.Tensor) -> torch.Tensor:
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T,B,D], got {latent_embeddings.dim()}D. "
                f"Shape: {latent_embeddings.shape}"
            )
        X = latent_embeddings.transpose(0, 1)
        return self.core.loss(X)

    @torch.no_grad()
    def compute_stats(self, latent_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T,B,D], got {latent_embeddings.dim()}D"
            )
        X = latent_embeddings.transpose(0, 1)
        if X.size(1) < 2:
            zero = X.new_tensor(0.0)
            return {
                "deviation": X.new_zeros((X.size(0), 0)),
                "deviation_mean": zero,
                "deviation_max": zero,
            }
        Xn = self.core.geo.project(self.core.geo._safe_normalize(X))
        z_start = Xn[:, 0, :]
        z_end = Xn[:, -1, :]
        geodesic = self.core.compute_geodesic(z_start, z_end, Xn.size(1))
        deviation = self.core.geo.distance(Xn, geodesic)
        return {
            "deviation": deviation,
            "deviation_mean": deviation.mean(),
            "deviation_max": deviation.max(),
        }
