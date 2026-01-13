"""
Trajectory Consistency Loss for Latent Token Regularization.

Constrains latent tokens to stay within a radius around their geometric center (Fréchet mean).
Supports both Euclidean and Hyperbolic (Poincaré ball) spaces.
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


# =========================
# Configuration
# =========================

@dataclass
class GeometryConfig:
    """Configuration for geometry backends."""
    curvature: float = -1.0   # negative for hyperbolic
    eps: float = 1e-8
    max_iter: int = 50        # for Fréchet mean iteration
    step_size: float = 0.1    # gradient descent step


# =========================
# Geometry Backends (Strategy Pattern)
# =========================

class EuclideanGeometry:
    """Euclidean geometry: arithmetic mean + L2 distance."""
    
    def __init__(self, cfg: GeometryConfig):
        self.cfg = cfg
    
    def frechet_mean(self, X: torch.Tensor) -> torch.Tensor:
        """X: [K,D] -> [D]"""
        return X.mean(dim=0)
    
    def distance(self, X: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """X: [K,D], center: [D] -> [K]"""
        return torch.linalg.norm(X - center, dim=-1)
    
    def center_and_dist_batch(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized computation for batch.
        X: [B,K,D] -> center: [B,D], dist: [B,K]
        """
        center = X.mean(dim=1)  # [B,D]
        dist = torch.linalg.norm(X - center.unsqueeze(1), dim=-1)  # [B,K]
        return center, dist


class HyperbolicGeometry:
    """
    Poincaré ball model for hyperbolic geometry.
    
    Key operations:
      - project: ensure points stay in the ball
      - mobius_add: addition in hyperbolic space
      - log0/exp0: log/exp maps at origin
      - frechet_mean: iterative Karcher mean
      
    All operations support batch dimensions for efficiency.
    
    **Important**: LLM hidden states typically have large norms (tens to hundreds).
    We normalize inputs before projecting to the Poincaré ball to preserve 
    directional information and avoid numerical collapse to the origin.
    """
    
    def __init__(self, cfg: GeometryConfig):
        if cfg.curvature >= 0:
            raise ValueError("Hyperbolic space requires negative curvature (e.g., -1.0)")
        self.cfg = cfg
        # Target norm for normalization (within ball but not too close to origin)
        self._target_norm = 0.5 / math.sqrt(-cfg.curvature)
    
    @property
    def c(self) -> float:
        """Positive curvature constant (c = -curvature)."""
        return -self.cfg.curvature
    
    def _safe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize vectors to have a reasonable norm for Poincaré ball operations.
        This is crucial for LLM embeddings which have very large norms.
        
        Args:
            x: [..., D] tensor (possibly with large norms)
        Returns:
            [..., D] tensor with norm scaled to target_norm
        """
        # Handle NaN/Inf in input
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        # Avoid division by zero
        norm = torch.clamp(norm, min=self.cfg.eps)
        # Scale to target norm (within the ball)
        return x * (self._target_norm / norm)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x into the open ball of radius 1/sqrt(c).
        
        Args:
            x: [..., D] tensor
        Returns:
            [..., D] tensor guaranteed to be inside the Poincaré ball
        """
        sqrt_c = math.sqrt(self.c)
        max_norm = (1.0 / sqrt_c) - self.cfg.eps
        
        # Handle NaN/Inf
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        # Only scale down if outside the ball, never scale up
        needs_projection = norm > max_norm
        scale = torch.where(
            needs_projection,
            max_norm / (norm + self.cfg.eps),
            torch.ones_like(norm)
        )
        return x * scale
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition: x ⊕ y in Poincaré ball. x,y: [...,D] (broadcastable)"""
        c = self.c
        x, y = self.project(x), self.project(y)
        
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = torch.clamp(1 + 2 * c * xy + (c ** 2) * x2 * y2, min=self.cfg.eps)
        
        result = num / den
        # Handle potential NaN from numerical issues
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        return self.project(result)
    
    def log0(self, x: torch.Tensor) -> torch.Tensor:
        """Log map at origin: log_0(x). x: [..., D]"""
        x = self.project(x)
        sqrt_c = math.sqrt(self.c)
        
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        small = x_norm < self.cfg.eps
        
        # Clamp to avoid arctanh(1) = inf
        z = torch.clamp(sqrt_c * x_norm, min=0.0, max=1.0 - 1e-4)
        coef = torch.arctanh(z) / (sqrt_c * x_norm + self.cfg.eps)
        out = coef * x
        
        # Handle small norms and NaN
        out = torch.where(small, torch.zeros_like(out), out)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out
    
    def exp0(self, v: torch.Tensor) -> torch.Tensor:
        """Exp map at origin: exp_0(v). v: [..., D]"""
        sqrt_c = math.sqrt(self.c)
        
        # Handle NaN/Inf
        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        small = v_norm < self.cfg.eps
        
        coef = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm + self.cfg.eps)
        out = self.project(coef * v)
        out = torch.where(small, torch.zeros_like(out), out)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Poincaré distance (arcosh form, numerically stable).
        x: [...,D], y: [...,D] (broadcastable) -> [...]
        """
        c = self.c
        sqrt_c = math.sqrt(c)
        
        x, y = self.project(x), self.project(y)
        
        diff2 = torch.sum((x - y) ** 2, dim=-1)
        x2 = torch.sum(x * x, dim=-1)
        y2 = torch.sum(y * y, dim=-1)
        
        denom = torch.clamp((1 - c * x2) * (1 - c * y2), min=self.cfg.eps)
        z = torch.clamp(1 + 2 * c * diff2 / denom, min=1.0 + 1e-6)
        
        dist = (1.0 / sqrt_c) * torch.arccosh(z)
        # Handle NaN
        dist = torch.where(torch.isfinite(dist), dist, torch.zeros_like(dist))
        return dist
    
    def frechet_mean_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Batched Fréchet mean (Karcher mean) via Riemannian gradient descent.
        X: [B, K, D] -> [B, D]
        
        All operations are vectorized over the batch dimension.
        
        **Note**: Input X should already be normalized via _safe_normalize before calling.
        """
        X = self.project(X)  # [B,K,D]
        
        # Initialize: exp0(mean(log0(X))) for each batch
        log_X = self.log0(X)                      # [B,K,D]
        mean_log = log_X.mean(dim=1)              # [B,D]
        center = self.exp0(mean_log)              # [B,D]
        center = self.project(center)
        
        # Early exit if center is already NaN-free and reasonable
        if not torch.isfinite(center).all():
            # Fallback: use simple Euclidean mean projected to ball
            center = self.project(X.mean(dim=1))
        
        for i in range(self.cfg.max_iter):
            # Compute tangent vectors from center to each point
            # In Poincaré ball, -x is the reflection through origin (still in ball if x is)
            neg_center = -center.unsqueeze(1)                    # [B,1,D]
            transported = self.mobius_add(neg_center, X)         # [B,K,D]
            v = self.log0(transported)                           # [B,K,D]
            
            # Gradient: mean over K dimension
            grad = v.mean(dim=1)  # [B,D]
            
            # Check for NaN and handle
            if not torch.isfinite(grad).all():
                break
            
            # Check convergence (batch-wise)
            grad_norm = torch.linalg.norm(grad, dim=-1)  # [B]
            if grad_norm.max() < 1e-6:
                break
            
            # Update centers with gradient descent
            step = self.exp0(-self.cfg.step_size * grad)  # [B,D]
            new_center = self.mobius_add(center, step)    # [B,D]
            new_center = self.project(new_center)
            
            # Only update if result is valid
            valid = torch.isfinite(new_center).all(dim=-1, keepdim=True)
            center = torch.where(valid, new_center, center)
        
        return center
    
    def center_and_dist_batch(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fully vectorized center and distance computation for batch.
        X: [B,K,D] -> center: [B,D], dist: [B,K]
        
        **Important**: This method normalizes the input before processing to handle
        LLM embeddings with large norms.
        """
        # CRITICAL: Normalize input to fit in Poincaré ball while preserving direction
        X_normalized = self._safe_normalize(X)
        
        # Compute centers for all batches at once
        center = self.frechet_mean_batch(X_normalized)  # [B,D]
        
        # Compute distances: X[B,K,D] to center[B,1,D]
        dist = self.distance(X_normalized, center.unsqueeze(1))  # [B,K]
        
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
    
    def __init__(self, space_type: str = "euclidean", curvature: float = -1.0, eps: float = 1e-8):
        if space_type not in ("euclidean", "hyperbolic"):
            raise ValueError("space_type must be 'euclidean' or 'hyperbolic'")
        
        self.space_type = space_type
        self.cfg = GeometryConfig(curvature=curvature, eps=eps)
        
        # Select geometry backend
        if space_type == "euclidean":
            self.geo = EuclideanGeometry(self.cfg)
        else:
            self.geo = HyperbolicGeometry(self.cfg)
    
    def center_and_dist(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Fréchet center and distances.
        
        Args:
            X: [B,K,D] batch of K tokens with D dimensions
            
        Returns:
            center: [B,D] geometric center per batch
            dist:   [B,K] distance from each token to its center
        """
        if X.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B,K,D], got {X.dim()}D")
        
        if X.numel() == 0:
            raise ValueError("Empty input tensor")
        
        return self.geo.center_and_dist_batch(X)
    
    def loss(self, X: torch.Tensor, radius_threshold: float) -> torch.Tensor:
        """
        Radius constraint loss.
        
        Args:
            X: [B,K,D] batch tensor
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
    
    Input shape: [T,B,D] where T=num_latent_tokens, B=batch_size, D=hidden_dim
    Internally transposed to [B,T,D] for computation.
    """
    
    def __init__(
        self, 
        space_type: str = "euclidean", 
        radius_threshold: float = 2.0, 
        curvature: float = -1.0, 
        eps: float = 1e-8
    ):
        super().__init__()
        self.core = TrajectoryConsistencyCore(
            space_type=space_type, 
            curvature=curvature, 
            eps=eps
        )
        self.radius_threshold = float(radius_threshold)
    
    def forward(self, latent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory consistency loss.
        
        Args:
            latent_embeddings: [T,B,D] tensor (T tokens, B batch, D dim)
            
        Returns:
            scalar loss tensor
        """
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T,B,D], got {latent_embeddings.dim()}D. "
                f"Shape: {latent_embeddings.shape}"
            )
        
        # [T,B,D] -> [B,T,D]
        X = latent_embeddings.transpose(0, 1)
        return self.core.loss(X, self.radius_threshold)
    
    @torch.no_grad()
    def compute_stats(self, latent_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute radius statistics for logging/analysis.
        
        Args:
            latent_embeddings: [T,B,D] tensor
            
        Returns:
            dict with keys:
              - center: [B,D] geometric center per batch
              - dist: [B,T] distances
              - radius_max: scalar, max distance
              - radius_mean: scalar, mean distance
              - radius_std: scalar, std of distances
              - violation_count: scalar, number of violations
              - violation_rate: scalar, fraction of violations
              - radius_threshold: scalar, the threshold used
        """
        if latent_embeddings.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor [T,B,D], got {latent_embeddings.dim()}D"
            )
        
        # [T,B,D] -> [B,T,D]
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
