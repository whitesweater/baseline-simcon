import math
import math

import torch
import torch.nn as nn


class TrajectoryConsistencyCenter:
    """
    Compute trajectory consistency loss by constraining latent tokens
    to stay within a radius around their Fréchet mean (geometric center).
    
    Supports both Euclidean and Hyperbolic spaces.
    """
    
    def __init__(self, space_type="euclidean", curvature=-1.0, eps=1e-8):
        assert space_type in ["euclidean", "hyperbolic"]
        if space_type == "hyperbolic":
            assert curvature < 0, "For hyperbolic space, curvature should be negative (e.g. -1.0)."
        self.space_type = space_type
        self.curvature = curvature
        self.eps = eps

    # ---------- Hyperbolic core ops (minimal set) ----------
    def project_to_ball(self, x):
        """
        Project x into the open Poincaré ball of radius 1/sqrt(c).
        Supports shape [..., D]
        """
        c = -self.curvature  # > 0
        sqrt_c = math.sqrt(c)
        max_norm = (1.0 / sqrt_c) - self.eps  # python float

        norm = torch.linalg.norm(x, dim=-1, keepdim=True)  # [...,1]
        scale = torch.clamp(max_norm / (norm + self.eps), max=1.0)
        return x * scale

    def project_to_hyperbolic(self, x):
        """Alias for compatibility with existing callers."""
        return self.project_to_ball(x)

    def mobius_add(self, x, y):
        """
        Möbius addition in Poincaré ball. Shapes [...,D] broadcastable.
        """
        c = -self.curvature
        x = self.project_to_ball(x)
        y = self.project_to_ball(y)

        x2 = torch.sum(x * x, dim=-1, keepdim=True)  # [...,1]
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
        den = torch.clamp(den, min=self.eps)

        return self.project_to_ball(num / den)

    def log0(self, x):
        """
        Log map at origin for Poincaré ball.
        x: [...,D]
        """
        x = self.project_to_ball(x)
        c = -self.curvature
        sqrt_c = math.sqrt(c)

        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)  # [...,1]
        small = x_norm < self.eps

        z = torch.clamp(sqrt_c * x_norm, max=1.0 - 1e-4)
        coef = torch.arctanh(z) / (sqrt_c * x_norm + self.eps)
        out = coef * x
        return torch.where(small, torch.zeros_like(out), out)

    def exp0(self, v):
        """
        Exp map at origin for Poincaré ball.
        v: [...,D]
        """
        c = -self.curvature
        sqrt_c = math.sqrt(c)

        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        small = v_norm < self.eps

        coef = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm + self.eps)
        out = self.project_to_ball(coef * v)
        return torch.where(small, torch.zeros_like(out), out)

    def hyperbolic_distance(self, x, y):
        """
        Standard Poincaré distance (arcosh form), stable.
        x: [...,D], y: [D] or [...,D]
        return: [...]
        """
        c = -self.curvature
        sqrt_c = math.sqrt(c)

        x = self.project_to_ball(x)
        y = self.project_to_ball(y)

        diff2 = torch.sum((x - y) ** 2, dim=-1)  # [...]
        x2 = torch.sum(x * x, dim=-1)            # [...]
        y2 = torch.sum(y * y, dim=-1)            # [...]

        denom = (1 - c * x2) * (1 - c * y2)
        denom = torch.clamp(denom, min=self.eps)

        z = 1 + 2 * c * diff2 / denom
        z = torch.clamp(z, min=1.0 + 1e-6)  # arcosh domain
        return (1.0 / sqrt_c) * torch.arccosh(z)

    # ---------- Fréchet mean ----------
    def frechet_mean(self, X, max_iter=50, step_base=0.1):
        """
        X: [K,D]
        """
        if self.space_type == "euclidean":
            return X.mean(dim=0)

        # hyperbolic
        X = self.project_to_ball(X)
        center = self.exp0(self.log0(X).mean(dim=0))  # [D]
        center = self.project_to_ball(center)

        for _ in range(max_iter):
            v = self.log0(self.mobius_add(-center, X))  # [K,D]
            grad = v.mean(dim=0)                        # [D]

            if torch.linalg.norm(grad) < 1e-6:
                break

            center = self.mobius_add(center, self.exp0(-step_base * grad))
            center = self.project_to_ball(center)

        return center

    # ---------- Loss ----------
    def center_based_consistency_loss(self, X, radius_threshold=2.0):
        """
        X: [K,D] or [B,K,D] for batched computation
        """
        if X.numel() == 0:
            return X.new_zeros(())

        # Handle batched input [B,K,D]
        if X.dim() == 3:
            B, K, D = X.shape
            # Compute center for each batch: [B,D]
            if self.space_type == "euclidean":
                center = X.mean(dim=1)  # [B,D]
            else:
                # Fallback to loop for hyperbolic (rare case)
                centers = []
                for b in range(B):
                    centers.append(self.frechet_mean(X[b]))  # [D]
                center = torch.stack(centers)  # [B,D]
            
            # Compute distances: [B,K]
            if self.space_type == "euclidean":
                dist = torch.linalg.norm(X - center.unsqueeze(1), dim=-1)  # [B,K]
            else:
                # Vectorized hyperbolic distance
                dist = torch.stack([
                    self.hyperbolic_distance(X[b], center[b])
                    for b in range(B)
                ])  # [B,K]
            
            violation = torch.clamp(dist - radius_threshold, min=0.0)  # [B,K]
            return violation.mean()
        
        # Original single-sample case [K,D]
        center = self.frechet_mean(X)

        if self.space_type == "euclidean":
            dist = torch.linalg.norm(X - center, dim=-1)   # [K]
        else:
            dist = self.hyperbolic_distance(X, center)     # [K]

        violation = torch.clamp(dist - radius_threshold, min=0.0)
        return violation.mean()

    # ---------- Statistics helpers ----------
    def center_and_dist(self, X):
        """
        Compute Fréchet center and distances for Euclidean/Hyperbolic space.
        Supports X with shape:
          - [K,D]: single sample with K tokens
          - [B,K,D]: batched samples

        Returns:
          - center: [D] for [K,D], or [B,D] for [B,K,D]
          - dist:   [K] for [K,D], or [B,K] for [B,K,D]
        """
        if X.dim() == 2:  # [K,D]
            if self.space_type == "euclidean":
                center = X.mean(dim=0)  # [D]
                dist = torch.linalg.norm(X - center, dim=-1)  # [K]
                return center, dist
            else:
                center = self.frechet_mean(X)  # [D]
                dist = self.hyperbolic_distance(X, center)  # [K]
                return center, dist

        elif X.dim() == 3:  # [B,K,D]
            B, K, D = X.shape
            if self.space_type == "euclidean":
                center = X.mean(dim=1)  # [B,D]
                dist = torch.linalg.norm(X - center.unsqueeze(1), dim=-1)  # [B,K]
                return center, dist
            else:
                centers = []
                dists = []
                for b in range(B):
                    c_b = self.frechet_mean(X[b])  # [D]
                    centers.append(c_b)
                    d_b = self.hyperbolic_distance(X[b], c_b)  # [K]
                    dists.append(d_b)
                center = torch.stack(centers)  # [B,D]
                dist = torch.stack(dists)      # [B,K]
                return center, dist
        else:
            raise ValueError(f"Unexpected tensor dimension for center_and_dist: {X.dim()}")

    # ---------- Statistics helpers ----------
    def center_and_dist(self, X):
        """
        Compute Fréchet center and distances for Euclidean/Hyperbolic space.
        Supports X with shape:
          - [K,D]: single sample with K tokens
          - [B,K,D]: batched samples

        Returns:
          - center: [D] for [K,D], or [B,D] for [B,K,D]
          - dist:   [K] for [K,D], or [B,K] for [B,K,D]
        """
        if X.dim() == 2:  # [K,D]
            if self.space_type == "euclidean":
                center = X.mean(dim=0)  # [D]
                dist = torch.linalg.norm(X - center, dim=-1)  # [K]
                return center, dist
            else:
                center = self.frechet_mean(X)  # [D]
                dist = self.hyperbolic_distance(X, center)  # [K]
                return center, dist

        elif X.dim() == 3:  # [B,K,D]
            B, K, D = X.shape
            if self.space_type == "euclidean":
                center = X.mean(dim=1)  # [B,D]
                dist = torch.linalg.norm(X - center.unsqueeze(1), dim=-1)  # [B,K]
                return center, dist
            else:
                centers = []
                dists = []
                for b in range(B):
                    c_b = self.frechet_mean(X[b])  # [D]
                    centers.append(c_b)
                    d_b = self.hyperbolic_distance(X[b], c_b)  # [K]
                    dists.append(d_b)
                center = torch.stack(centers)  # [B,D]
                dist = torch.stack(dists)      # [B,K]
                return center, dist
        else:
            raise ValueError(f"Unexpected tensor dimension for center_and_dist: {X.dim()}")



class TrajectoryConsistencyLoss(nn.Module):
    """
    Wrapper module for trajectory consistency loss
    """
    
    def __init__(self, space_type="euclidean", radius_threshold=2.0, curvature=-1.0):
        super().__init__()
        self.core = TrajectoryConsistencyCenter(
            space_type=space_type,
            curvature=curvature
        )
        self.radius_threshold = radius_threshold
    
    def forward(self, latent_embeddings: torch.Tensor):
        """
        latent_embeddings:
          - [B,D]      : treat as K=B tokens
          - [T,B,D]    : per-sample trajectory, batched computation (OPTIMIZED)
        """
        if latent_embeddings.dim() == 2:
            return self.core.center_based_consistency_loss(
                latent_embeddings,
                self.radius_threshold
            )

        if latent_embeddings.dim() == 3:
            # Transpose to [B,T,D] for batched computation
            T, B, D = latent_embeddings.shape
            X_batched = latent_embeddings.transpose(0, 1)  # [B,T,D]
            return self.core.center_based_consistency_loss(
                X_batched,
                self.radius_threshold
            )

        raise ValueError(f"Unexpected tensor dimension: {latent_embeddings.dim()}")

    def compute_stats(self, latent_embeddings: torch.Tensor):
        """
        Compute radius statistics without affecting training loss.

        latent_embeddings:
          - [B,D]: treat as K=B tokens (single sequence)
          - [T,B,D]: trajectory across T steps for each of B samples
          - [B,K,D]: already batched steps

        Returns dict with:
          - center: [B,D] or [D]
          - dist: [B,K] or [K]
          - radius_max, radius_mean, violation_rate, radius_threshold
        """
        # Normalize to [B,K,D]
        if latent_embeddings.dim() == 2:
            X_batched = latent_embeddings.unsqueeze(0)  # [1,K,D]
        elif latent_embeddings.dim() == 3:
            # Assume [T,B,D] -> transpose to [B,T,D]
            T, B, D = latent_embeddings.shape
            # Heuristic: if first dim equals steps, transpose
            X_batched = latent_embeddings.transpose(0, 1)  # [B,T,D]
        else:
            raise ValueError(f"Unexpected tensor dimension for stats: {latent_embeddings.dim()}")

        # Compute center and distances
        center, dist = self.core.center_and_dist(X_batched)

        # Scalar statistics across batch and steps
        radius_max = dist.max()
        radius_mean = dist.mean()
        violation_rate = (dist > self.radius_threshold).float().mean()

        return {
            "center": center,
            "dist": dist,
            "radius_max": radius_max,
            "radius_mean": radius_mean,
            "violation_rate": violation_rate,
            "radius_threshold": torch.tensor(self.radius_threshold, device=radius_mean.device),
        }
