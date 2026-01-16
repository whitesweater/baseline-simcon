"""
Trajectory Consistency Loss for Latent Token Regularization (Euclidean Only).

Simplified version for Coconut project - only supports Euclidean geometry.
Constrains latent tokens to stay within a radius around their geometric center (Fréchet mean).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple


# =========================
# Configuration
# =========================

@dataclass
class GeometryConfig:
    """Configuration for geometry backend."""
    eps: float = 1e-8


# =========================
# Euclidean Geometry Backend
# =========================

class EuclideanGeometry:
    """Euclidean geometry: arithmetic mean + L2 distance."""
    
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg
    
    def frechet_mean(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute Fréchet mean (arithmetic mean in Euclidean space).
        
        Args:
            X: [K, D] tensor of K tokens with D dimensions
            
        Returns:
            [D] tensor - the geometric center
        """
        return X.mean(dim=0)
    
    def distance(self, X: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 distance from each point to the center.
        
        Args:
            X: [K, D] tensor of K tokens
            center: [D] tensor - the geometric center
            
        Returns:
            [K] tensor of distances
        """
        return torch.linalg.norm(X - center, dim=-1)
    
    def center_and_dist_batch(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized computation for batch.
        
        Args:
            X: [B, K, D] batch of K tokens with D dimensions
            
        Returns:
            center: [B, D] geometric center per batch
            dist: [B, K] distance from each token to its center
        """
        center = X.mean(dim=1)  # [B, D]
        dist = torch.linalg.norm(X - center.unsqueeze(1), dim=-1)  # [B, K]
        return center, dist


# =========================
# Trajectory Consistency Core
# =========================

class TrajectoryConsistencyCore:
    """
    Core logic for trajectory consistency loss.
    
    Constrains all latent tokens to lie within radius_threshold of their geometric center.
    
    Loss = mean(max(0, d(z_k, center) - radius_threshold))
    """
    
    def __init__(self, eps: float = 1e-8):
        self.cfg = GeometryConfig(eps=eps)
        self.geo = EuclideanGeometry(self.cfg)
    
    def center_and_dist(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Fréchet center and distances.
        
        Args:
            X: [B, K, D] batch of K tokens with D dimensions
            
        Returns:
            center: [B, D] geometric center per batch
            dist: [B, K] distance from each token to its center
        """
        if X.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, K, D], got {X.dim()}D")
        
        if X.numel() == 0:
            raise ValueError("Empty input tensor")
        
        return self.geo.center_and_dist_batch(X)
    
    def loss(self, X: torch.Tensor, radius_threshold: float) -> torch.Tensor:
        """
        Radius constraint loss.
        
        Args:
            X: [B, K, D] batch tensor
            radius_threshold: maximum allowed distance from center
            
        Returns:
            scalar loss = mean(ReLU(dist - threshold))
        """
        _, dist = self.center_and_dist(X)
        violation = torch.clamp(dist - radius_threshold, min=0.0)
        return violation.mean()


# =========================
# nn.Module Wrapper
# =========================

class TrajectoryConsistencyLoss(nn.Module):
    """
    Training-time module for trajectory consistency loss.
    
    Input shape: [T, B, D] where T=num_latent_tokens, B=batch_size, D=hidden_dim
    Internally transposed to [B, T, D] for computation.
    
    Usage example:
        trajectory_loss_module = TrajectoryConsistencyLoss(radius_threshold=2.0)
        
        # latent_embeddings: [num_latent, batch_size, hidden_dim]
        trajectory_loss = trajectory_loss_module(latent_embeddings)
        
        total_loss = ce_loss + trajectory_loss_factor * trajectory_loss
    """
    
    def __init__(
        self, 
        radius_threshold: float = 2.0, 
        eps: float = 1e-8
    ):
        """
        Initialize TrajectoryConsistencyLoss.
        
        Args:
            radius_threshold: Maximum allowed distance from center. 
                              Points outside this radius will incur a penalty.
            eps: Small epsilon for numerical stability.
        """
        super().__init__()
        self.core = TrajectoryConsistencyCore(eps=eps)
        self.radius_threshold = float(radius_threshold)
    
    def forward(self, latent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory consistency loss.
        
        Args:
            latent_embeddings: [T, B, D] tensor (T tokens, B batch, D dim)
            
        Returns:
            scalar loss tensor
        """
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T, B, D], got {latent_embeddings.dim()}D. "
                f"Shape: {latent_embeddings.shape}"
            )
        
        # [T, B, D] -> [B, T, D]
        X = latent_embeddings.transpose(0, 1)
        return self.core.loss(X, self.radius_threshold)
    
    @torch.no_grad()
    def compute_stats(self, latent_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute radius statistics for logging/analysis.
        
        Args:
            latent_embeddings: [T, B, D] tensor
            
        Returns:
            dict with keys:
              - center: [B, D] geometric center per batch
              - dist: [B, T] distances
              - radius_max: scalar, max distance
              - radius_mean: scalar, mean distance
              - radius_std: scalar, std of distances
              - violation_count: scalar, number of violations
              - violation_rate: scalar, fraction of violations
              - radius_threshold: scalar, the threshold used
        """
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T, B, D], got {latent_embeddings.dim()}D"
            )
        
        # [T, B, D] -> [B, T, D]
        X = latent_embeddings.transpose(0, 1)
        center, dist = self.core.center_and_dist(X)
        
        # Compute statistics
        violations = dist > self.radius_threshold
        
        return {
            "center": center,
            "dist": dist,
            "radius_max": dist.max(),
            "radius_mean": dist.mean(),
            "radius_std": dist.std(),
            "violation_count": violations.sum(),
            "violation_rate": violations.float().mean(),
            "radius_threshold": torch.tensor(self.radius_threshold, device=dist.device),
        }


# =========================
# Alternative: Per-Pass Loss
# =========================

class PerPassTrajectoryLoss(nn.Module):
    """
    Alternative loss that can be computed incrementally during forward pass.
    
    Instead of collecting all latent embeddings and computing at the end,
    this allows you to update the loss after each forward pass.
    
    Useful when memory is limited or when latent embeddings are generated
    in a streaming fashion.
    """
    
    def __init__(self, radius_threshold: float = 2.0):
        super().__init__()
        self.radius_threshold = float(radius_threshold)
        self.reset()
    
    def reset(self):
        """Reset accumulated embeddings for new batch."""
        self._embeddings = []
    
    def add_embedding(self, latent_embd: torch.Tensor):
        """
        Add a latent embedding from one pass.
        
        Args:
            latent_embd: [B, D] tensor - latent embedding from one pass
        """
        self._embeddings.append(latent_embd.detach() if not self.training else latent_embd)
    
    def compute_loss(self) -> torch.Tensor:
        """
        Compute loss from all accumulated embeddings.
        
        Returns:
            scalar loss tensor
        """
        if len(self._embeddings) < 2:
            # Need at least 2 embeddings to compute meaningful trajectory loss
            return torch.tensor(0.0, device=self._embeddings[0].device if self._embeddings else 'cpu')
        
        # Stack: [T, B, D]
        stacked = torch.stack(self._embeddings, dim=0)
        # Transpose to [B, T, D]
        X = stacked.transpose(0, 1)
        
        # Compute center and distances
        center = X.mean(dim=1)  # [B, D]
        dist = torch.linalg.norm(X - center.unsqueeze(1), dim=-1)  # [B, T]
        
        # Compute violation loss
        violation = torch.clamp(dist - self.radius_threshold, min=0.0)
        return violation.mean()


if __name__ == "__main__":
    # Quick test
    print("Testing TrajectoryConsistencyLoss (Euclidean)...")
    
    # Create dummy latent embeddings: [T=5, B=4, D=768]
    T, B, D = 5, 4, 768
    latent_embeddings = torch.randn(T, B, D)
    
    # Test TrajectoryConsistencyLoss
    loss_module = TrajectoryConsistencyLoss(radius_threshold=2.0)
    loss = loss_module(latent_embeddings)
    print(f"Loss: {loss.item():.6f}")
    
    # Test stats computation
    stats = loss_module.compute_stats(latent_embeddings)
    print(f"Radius mean: {stats['radius_mean'].item():.4f}")
    print(f"Radius max: {stats['radius_max'].item():.4f}")
    print(f"Violation rate: {stats['violation_rate'].item():.4f}")
    
    # Test PerPassTrajectoryLoss
    print("\nTesting PerPassTrajectoryLoss...")
    per_pass_loss = PerPassTrajectoryLoss(radius_threshold=2.0)
    per_pass_loss.reset()
    for t in range(T):
        per_pass_loss.add_embedding(latent_embeddings[t])
    loss2 = per_pass_loss.compute_loss()
    print(f"Per-pass loss: {loss2.item():.6f}")
    
    print("\n✓ All tests passed!")
