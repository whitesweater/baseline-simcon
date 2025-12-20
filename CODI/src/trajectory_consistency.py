"""
Trajectory Consistency Loss based on Fréchet Mean

Constrains all tokens to stay within a certain distance from the geometric center,
preventing reasoning from deviating from the main topic (centripetal force constraint).

References:
- "Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric Measurements"
- "Hyperbolic Image Embeddings"
- "A Discriminative Feature Learning Approach for Deep Face Recognition"
- "Fast Computation of Wasserstein Barycenters"
"""

import torch
import torch.nn as nn


class TrajectoryConsistencyCenter:
    """
    Compute trajectory consistency loss by constraining latent tokens
    to stay within a radius around their Fréchet mean (geometric center).
    
    Supports both Euclidean and Hyperbolic spaces.
    """
    
    def __init__(self, space_type='euclidean', curvature=-1.0):
        """
        Args:
            space_type: 'euclidean' or 'hyperbolic'
            curvature: curvature constant for hyperbolic space (typically -1.0)
        """
        self.space_type = space_type
        self.curvature = curvature
        self.eps = 1e-8  # numerical stability
    
    def compute_frechet_mean(self, latent_tokens):
        """
        Compute Fréchet mean of latent tokens
        
        Euclidean space: simple arithmetic mean
        Hyperbolic space: iterative Karcher mean algorithm
        
        Args:
            latent_tokens: List of tensors [num_tokens, hidden_dim]
        
        Returns:
            center: Tensor [hidden_dim]
        """
        if self.space_type == 'euclidean':
            # Euclidean space: direct averaging
            center = torch.mean(torch.stack(latent_tokens), dim=0)
            return center
            
        elif self.space_type == 'hyperbolic':
            # Hyperbolic space: iterative Fréchet mean computation
            return self.hyperbolic_frechet_mean(latent_tokens)
        else:
            raise ValueError(f"Unknown space_type: {self.space_type}")
    
    def hyperbolic_frechet_mean(self, latent_tokens, max_iter=50):
        """
        Compute Frechet mean (Karcher mean) in hyperbolic space
        using gradient descent to minimize sum of squared distances to all points
        
        Args:
            latent_tokens: List of tensors in hyperbolic space
            max_iter: maximum number of iterations
        
        Returns:
            center: Frechet mean in hyperbolic space
        """
        # Initialize: map all points to origin's tangent space, average, then map back
        # This is the geometrically correct way to initialize in hyperbolic space
        h_init = self.exp_map(torch.mean(
            torch.stack([self.log_map(h) for h in latent_tokens]), 
            dim=0
        ))
        
        center = h_init
        for iteration in range(max_iter):
            # Compute gradient: sum of tangent vectors from all points to current center
            grad = torch.zeros_like(center)
            for h in latent_tokens:
                # Tangent vector from center to h
                v = self.log_map_at(center, h)
                grad += v
            
            # Update center (move in tangent space)
            step_size = 0.1 / len(latent_tokens)
            center = self.exp_map_at(center, -step_size * grad)
            
            # Convergence check
            if torch.norm(grad) < 1e-6:
                break
        
        return center
    
    def center_based_consistency_loss(self, latent_tokens, radius_threshold=2.0):
        """
        Constrain all tokens to stay within radius_threshold from center
        
        Loss = (1/K) * Σ max(0, d(z_k, center) - radius_threshold)
        
        Args:
            latent_tokens: List of tensors [hidden_dim]
            radius_threshold: maximum allowed distance from center
        
        Returns:
            loss: scalar tensor
        """
        if len(latent_tokens) == 0:
            return torch.tensor(0.0)
        
        center = self.compute_frechet_mean(latent_tokens)
        
        total_loss = 0.0
        for z_k in latent_tokens:
            if self.space_type == 'euclidean':
                distance = torch.norm(z_k - center)
            else:  # hyperbolic
                distance = self.hyperbolic_distance(z_k, center)
            
            # Penalize distances exceeding threshold
            violation = torch.clamp(distance - radius_threshold, min=0.0)
            total_loss = total_loss + violation
        
        return total_loss / len(latent_tokens)
    
    # ============ Hyperbolic Space Operations ============
    
    def log_map(self, x):
        """
        Logarithmic map from origin: map point x to tangent space at origin
        
        For Poincaré ball model, log_0(x) = arctanh(||x||) * x / ||x||
        """
        c = -self.curvature
        x_norm = torch.norm(x)
        
        if x_norm < self.eps:
            return torch.zeros_like(x)
        
        # Simplified formula at origin
        scale = torch.arctanh(torch.clamp(torch.sqrt(torch.tensor(c)) * x_norm, max=1.0 - self.eps)) / (x_norm + self.eps)
        scale = scale / torch.sqrt(torch.tensor(c))
        
        return scale * x
    
    def exp_map(self, v):
        """
        Exponential map from origin: map tangent vector v at origin to hyperbolic space
        
        For Poincaré ball model, exp_0(v) = tanh(||v||) * v / ||v||
        """
        c = -self.curvature
        v_norm = torch.norm(v)
        
        if v_norm < self.eps:
            return torch.zeros_like(v)
        
        # Simplified formula at origin
        scale = torch.tanh(torch.sqrt(torch.tensor(c)) * v_norm) / (torch.sqrt(torch.tensor(c)) * v_norm + self.eps)
        
        return scale * v
    
    def hyperbolic_distance(self, x, y):
        """
        Compute hyperbolic distance in Poincaré ball model
        
        d(x, y) = (2/sqrt(-c)) * arctanh(sqrt(-c) * ||x - y|| / (1 - c*<x,y> - sqrt((1-c||x||²)(1-c||y||²))))
        
        Simplified for numerical stability
        """
        c = -self.curvature
        
        # Poincaré ball distance (simplified)
        diff_norm_sq = torch.sum((x - y) ** 2)
        x_norm_sq = torch.sum(x ** 2)
        y_norm_sq = torch.sum(y ** 2)
        
        # Avoid division by zero
        denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
        denominator = torch.clamp(denominator, min=self.eps)
        
        # Simplified distance formula
        numerator = c * diff_norm_sq
        fraction = numerator / denominator
        
        # arcsinh for numerical stability instead of arctanh
        distance = (1.0 / torch.sqrt(torch.tensor(c))) * torch.arcsinh(
            torch.sqrt(torch.clamp(fraction, min=0.0))
        )
        
        return distance
    
    def exp_map_at(self, p, v):
        """
        Exponential map: map tangent vector v at point p to hyperbolic space
        
        exp_p(v) = p ⊕ tanh(sqrt(c)||v||/2) * v/||v||
        where ⊕ is Möbius addition
        """
        c = -self.curvature
        v_norm = torch.norm(v)
        
        if v_norm < self.eps:
            return p
        
        # Compute scaling factor
        sqrt_c_norm = torch.sqrt(torch.tensor(c)) * v_norm / 2.0
        scale = torch.tanh(sqrt_c_norm) / (torch.sqrt(torch.tensor(c)) * v_norm + self.eps)
        
        # Scaled tangent vector
        v_scaled = scale * v
        
        # Möbius addition: p ⊕ v_scaled
        return self.mobius_add(p, v_scaled)
    
    def log_map_at(self, p, x):
        """
        Logarithmic map: map point x to tangent space at point p
        
        log_p(x) = (2/sqrt(c)) * arctanh(sqrt(c)||(-p) ⊕ x||) * ((-p) ⊕ x) / ||(-p) ⊕ x||
        """
        c = -self.curvature
        
        # Compute -p ⊕ x
        diff = self.mobius_add(-p, x)
        diff_norm = torch.norm(diff)
        
        if diff_norm < self.eps:
            return torch.zeros_like(diff)
        
        # Compute scaling with numerical safeguards
        sqrt_c_norm = torch.sqrt(torch.tensor(c)) * diff_norm
        sqrt_c_norm = torch.clamp(sqrt_c_norm, max=1.0 - self.eps)
        
        scale = (2.0 / torch.sqrt(torch.tensor(c))) * torch.arctanh(sqrt_c_norm) / (diff_norm + self.eps)
        
        return scale * diff
    
    def mobius_add(self, x, y):
        """
        Möbius addition in Poincaré ball: x ⊕ y
        
        x ⊕ y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / (1 + 2c<x,y> + c²||x||²||y||²)
        """
        c = -self.curvature
        
        x_sq = torch.sum(x ** 2)
        y_sq = torch.sum(y ** 2)
        xy = torch.sum(x * y)
        
        # Clamp to prevent numerical overflow
        x_sq = torch.clamp(x_sq, max=0.99 / c)
        y_sq = torch.clamp(y_sq, max=0.99 / c)
        
        numerator = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denominator = 1 + 2 * c * xy + (c ** 2) * x_sq * y_sq
        denominator = torch.clamp(denominator, min=self.eps)
        
        result = numerator / denominator
        
        # Project back to valid hyperbolic space
        max_norm = 1.0 / torch.sqrt(torch.tensor(c)) - self.eps
        result_norm = torch.norm(result)
        if result_norm >= max_norm:
            result = result / result_norm * (max_norm - self.eps)
        
        return result


class TrajectoryConsistencyLoss(nn.Module):
    """
    Wrapper module for trajectory consistency loss
    """
    
    def __init__(self, space_type='euclidean', radius_threshold=2.0, curvature=-1.0):
        super().__init__()
        self.consistency_computer = TrajectoryConsistencyCenter(
            space_type=space_type,
            curvature=curvature
        )
        self.radius_threshold = radius_threshold
    
    def forward(self, latent_embeddings):
        """
        Args:
            latent_embeddings: List of latent token embeddings [batch, hidden_dim]
                             or [num_latent_steps, batch, hidden_dim]
        
        Returns:
            loss: scalar tensor
        """
        # Handle different input formats
        if isinstance(latent_embeddings, list):
            tokens = latent_embeddings
        elif isinstance(latent_embeddings, torch.Tensor):
            if latent_embeddings.dim() == 3:  # [steps, batch, dim]
                # Average over batch dimension for each step
                tokens = [latent_embeddings[i].mean(dim=0) for i in range(latent_embeddings.size(0))]
            elif latent_embeddings.dim() == 2:  # [batch, dim]
                tokens = [latent_embeddings[i] for i in range(latent_embeddings.size(0))]
            else:
                raise ValueError(f"Unexpected tensor dimension: {latent_embeddings.dim()}")
        else:
            raise ValueError(f"Unexpected input type: {type(latent_embeddings)}")
        
        return self.consistency_computer.center_based_consistency_loss(
            tokens, 
            radius_threshold=self.radius_threshold
        )
