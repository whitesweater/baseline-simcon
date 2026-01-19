import torch
import torch.nn as nn

class TrajectoryConsistencyCenter(nn.Module):
    """
    轨迹一致性约束
    
    核心思想：将推理轨迹视为点云，约束所有 token 到中心的距离
    类似物理中的向心力约束，防止推理偏离主线
    """
    
    def __init__(self, space_type='euclidean', dim=None, k=1.0, eps=1e-8, max_norm=11.0):
        """
        Args:
            space_type: 'euclidean' 或 'hyperbolic'
            dim: latent token 的维度 (hyperbolic 时必须提供)
            k: 双曲空间曲率参数 (k > 0)
            eps: 数值稳定性
            max_norm: 输入的最大值限制
        """
        super().__init__()
        self.space_type = space_type
        self.k = k
        self.eps = eps
        self.max_norm = max_norm
        
        # 可学习的 scale 参数，初始化为 sqrt(dim)
        if space_type == 'hyperbolic':
            assert dim is not None, "dim must be provided for hyperbolic space"
            self.scale = nn.Parameter(torch.tensor(dim ** 0.5))
    
    # ============ 数值稳定的双曲函数 ============
    
    def cosh(self, x):
        """数值稳定的 cosh"""
        return torch.cosh(x.clamp(max=self.max_norm))
    
    def sinh(self, x):
        """数值稳定的 sinh"""
        return torch.sinh(x.clamp(max=self.max_norm))
    
    def arcosh(self, x):
        """数值稳定的 arcosh"""
        return torch.acosh(x.clamp(min=1.0 + self.eps))
    
    # ============ Lorentz 基础运算 ============
    
    def _inner(self, u, v, keepdim=False, dim=-1):
        """
        Lorentzian 内积: ⟨u,v⟩_L = -u_0·v_0 + Σ(u_i·v_i)
        """
        d = u.size(dim) - 1
        uv = u * v
        if keepdim is False:
            return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=False)
        else:
            return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=True)
    
    def hyperbolic_distance(self, x, y, dim=-1):
        """双曲距离: d(x,y) = √k · arcosh(-⟨x,y⟩_L / k)"""
        inner = self._inner(x, y, keepdim=False, dim=dim)
        return (self.k ** 0.5) * self.arcosh(-inner / self.k)
    
    def exp_map0(self, v, dim=-1):
        """
        原点处的指数映射: T_o(H^n) → H^n
        
        原点 o = (√k, 0, ..., 0)
        切向量 u = [0, v] ∈ T_o(H^n)
        
        Args:
            v: [..., d] 欧式向量
        Returns:
            x: [..., d+1] 双曲面上的点
        """
        v_norm = v.norm(dim=dim, keepdim=True).clamp(min=self.eps)
        theta = v_norm / (self.k ** 0.5)
        
        x_time = (self.k ** 0.5) * self.cosh(theta)
        x_space = self.sinh(theta) * v / v_norm
        
        return torch.cat([x_time, x_space], dim=dim)
    
    # ============ 计算中心 ============
    
    def compute_mean(self, latent_tokens):
        """
        计算 latent tokens 的中心
        
        欧式空间: 算术平均
        双曲空间: Lorentz midpoint (闭式解)
        """
        if isinstance(latent_tokens, list):
            latent_tokens = torch.stack(latent_tokens, dim=0)
        
        if self.space_type == 'euclidean':
            return torch.mean(latent_tokens, dim=0)
        
        elif self.space_type == 'hyperbolic':
            hyp_tokens = self.exp_map0(latent_tokens)
            return self.lorentz_midpoint(hyp_tokens)
    
    def lorentz_midpoint(self, x, weights=None, dim=-1):
        """
        Lorentzian Centroid (闭式解)
        
        μ = Σ(wᵢ·xᵢ) / √(-⟨Σwᵢxᵢ, Σwᵢxᵢ⟩_L / k)
        """
        N = x.shape[0]
        if weights is None:
            weights = torch.ones(N, device=x.device, dtype=x.dtype) / N
        else:
            weights = weights / (weights.sum() + self.eps)
        
        weighted_sum = torch.einsum('n,nd->d', weights, x)
        inner = self._inner(weighted_sum, weighted_sum, keepdim=False, dim=dim)
        scale = torch.sqrt((-inner / self.k).clamp(min=self.eps))
        
        return weighted_sum / scale
    
    # ============ 一致性损失 ============
    
    def center_based_consistency_loss(self, latent_tokens, radius_threshold=2.0):
        """
        约束所有 token 到中心的距离
        
        Loss = (1/N) Σ max(0, d(z_k, center) - radius_threshold)
        """
        if isinstance(latent_tokens, list):
            latent_tokens = torch.stack(latent_tokens, dim=0)
        
        if self.space_type == 'euclidean':
            center = self.compute_mean(latent_tokens)
            distances = torch.norm(latent_tokens - center, dim=-1)
        
        else:  # hyperbolic
            # 对 latent_tokens 进行 scale，控制投影范围
            scaled_tokens = latent_tokens / self.scale
            # 投影到双曲面: [N, d] → [N, d+1]
            hyp_tokens = self.exp_map0(scaled_tokens)
            # 计算中心: [d+1]
            center = self.lorentz_midpoint(hyp_tokens)
            # 维度校验
            assert center.size(-1) == hyp_tokens.size(-1), \
                f"Dimension mismatch: center {center.size(-1)} vs hyp_tokens {hyp_tokens.size(-1)}, both should be d+1"
            # 计算双曲距离: [N]
            distances = self.hyperbolic_distance(hyp_tokens, center.unsqueeze(0).expand_as(hyp_tokens))
        
        # 超出阈值的部分受惩罚
        violations = torch.clamp(distances - radius_threshold, min=0.0)
        
        return violations.mean()


# ============ 测试 ============

if __name__ == "__main__":
    torch.manual_seed(42)
    
    d = 64
    latent_tokens = torch.randn(10, d) * 0.5
    
    # 欧式
    tcc_euc = TrajectoryConsistencyCenter(space_type='euclidean')
    loss_euc = tcc_euc.center_based_consistency_loss(latent_tokens, radius_threshold=1.5)
    print(f"Euclidean loss: {loss_euc.item():.4f}")
    
    # 双曲
    tcc_hyp = TrajectoryConsistencyCenter(space_type='hyperbolic', dim=d, k=1.0)
    print(f"Initial scale: {tcc_hyp.scale.item():.4f} (sqrt({d}) = {d**0.5:.4f})")
    
    loss_hyp = tcc_hyp.center_based_consistency_loss(latent_tokens, radius_threshold=1.5)
    print(f"Hyperbolic loss: {loss_hyp.item():.4f}")
    
    # 测试梯度
    loss_hyp.backward()
    print(f"Scale gradient: {tcc_hyp.scale.grad.item():.6f}")