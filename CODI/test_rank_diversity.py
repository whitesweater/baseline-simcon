"""
Unit tests for RankDiversityLoss.

Tests:
1. Shape handling and input validation
2. SVD entropy loss correctness (identity-like → low loss, collapsed → high loss)
3. Cosine diversity loss correctness
4. Combined mode
5. Gradient flow
6. Edge cases (single token, large batch)
"""

import math
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rank_diversity import RankDiversityLoss


def test_input_validation():
    """Test that non-3D inputs raise ValueError."""
    loss_fn = RankDiversityLoss(mode="svd_entropy")
    try:
        loss_fn(torch.randn(4, 8))
        raise AssertionError("Should have raised ValueError for 2D input")
    except ValueError:
        pass
    try:
        loss_fn(torch.randn(4))
        raise AssertionError("Should have raised ValueError for 1D input")
    except ValueError:
        pass


def test_invalid_mode():
    """Test that invalid mode raises ValueError."""
    try:
        RankDiversityLoss(mode="invalid")
        raise AssertionError("Should have raised ValueError for invalid mode")
    except ValueError:
        pass


def test_svd_entropy_orthogonal_low_loss():
    """Orthogonal vectors should yield low SVD entropy loss (high effective rank)."""
    loss_fn = RankDiversityLoss(mode="svd_entropy", center_before_svd=False)
    T, B, D = 8, 4, 64
    # Create orthogonal token embeddings using QR decomposition
    X = torch.randn(B, T, D)
    Q, _ = torch.linalg.qr(X.transpose(1, 2))  # [B, D, T]
    Q = Q.transpose(1, 2)  # [B, T, D]
    # Scale to have equal norms
    Q = Q / Q.norm(dim=-1, keepdim=True) * 10.0
    # [B, T, D] -> [T, B, D]
    latent = Q.transpose(0, 1)
    loss = loss_fn(latent)
    # Orthogonal vectors → nearly uniform singular values → low loss
    assert loss.item() < 0.15, f"Orthogonal vectors should have low loss, got {loss.item()}"


def test_svd_entropy_collapsed_high_loss():
    """Collapsed vectors (all same direction) should yield high SVD entropy loss."""
    loss_fn = RankDiversityLoss(mode="svd_entropy", center_before_svd=False)
    T, B, D = 8, 4, 64
    # All tokens are the same vector (plus tiny noise for numerical stability)
    base = torch.randn(1, B, D)
    latent = base.expand(T, -1, -1) + torch.randn(T, B, D) * 1e-6
    loss = loss_fn(latent)
    # Collapsed → singular values concentrated in one direction → high loss
    assert loss.item() > 0.8, f"Collapsed vectors should have high loss, got {loss.item()}"


def test_cosine_orthogonal_low_loss():
    """Orthogonal vectors should have low cosine diversity loss."""
    loss_fn = RankDiversityLoss(mode="cosine")
    T, B, D = 8, 4, 64
    # Create near-orthogonal embeddings
    X = torch.randn(B, D, T)
    Q, _ = torch.linalg.qr(X)  # [B, D, T] orthogonal columns
    Q = Q[:, :, :T]  # [B, D, T]
    latent = Q.transpose(1, 2).transpose(0, 1).contiguous()  # [T, B, D]
    loss = loss_fn(latent)
    assert loss.item() < 0.15, f"Orthogonal tokens should have low cosine loss, got {loss.item()}"


def test_cosine_parallel_high_loss():
    """Parallel vectors should have high cosine diversity loss."""
    loss_fn = RankDiversityLoss(mode="cosine")
    T, B, D = 8, 4, 64
    base = torch.randn(1, B, D)
    latent = base.expand(T, -1, -1) + torch.randn(T, B, D) * 1e-6
    loss = loss_fn(latent)
    assert loss.item() > 0.9, f"Parallel tokens should have high cosine loss, got {loss.item()}"


def test_combined_mode():
    """Combined mode should be a weighted sum of SVD and cosine losses."""
    svd_w, cos_w = 1.0, 0.5
    loss_fn_combined = RankDiversityLoss(mode="combined", svd_weight=svd_w, cosine_weight=cos_w, center_before_svd=False)
    loss_fn_svd = RankDiversityLoss(mode="svd_entropy", center_before_svd=False)
    loss_fn_cos = RankDiversityLoss(mode="cosine")

    T, B, D = 8, 4, 64
    latent = torch.randn(T, B, D)

    combined = loss_fn_combined(latent).item()
    expected = svd_w * loss_fn_svd(latent).item() + cos_w * loss_fn_cos(latent).item()
    assert abs(combined - expected) < 1e-5, f"Combined {combined} != expected {expected}"


def test_single_token_returns_zero():
    """Single token should return 0 loss (no diversity to promote)."""
    for mode in ["svd_entropy", "cosine", "combined"]:
        loss_fn = RankDiversityLoss(mode=mode)
        latent = torch.randn(1, 4, 64)  # T=1
        loss = loss_fn(latent)
        assert loss.item() == 0.0, f"Single token should have zero loss in mode={mode}"


def test_gradient_flow():
    """Verify gradients flow back through the loss."""
    loss_fn = RankDiversityLoss(mode="svd_entropy")
    latent = torch.randn(8, 4, 64, requires_grad=True)
    loss = loss_fn(latent)
    loss.backward()
    assert latent.grad is not None, "Gradients should flow"
    assert not torch.all(latent.grad == 0), "Gradients should not be all zero"


def test_gradient_flow_cosine():
    """Verify gradients flow through cosine mode."""
    loss_fn = RankDiversityLoss(mode="cosine")
    latent = torch.randn(8, 4, 64, requires_grad=True)
    loss = loss_fn(latent)
    loss.backward()
    assert latent.grad is not None
    assert not torch.all(latent.grad == 0)


def test_gradient_flow_combined():
    """Verify gradients flow through combined mode."""
    loss_fn = RankDiversityLoss(mode="combined")
    latent = torch.randn(8, 4, 64, requires_grad=True)
    loss = loss_fn(latent)
    loss.backward()
    assert latent.grad is not None
    assert not torch.all(latent.grad == 0)


def test_compute_stats():
    """Test diagnostic stats computation."""
    loss_fn = RankDiversityLoss(mode="svd_entropy")
    latent = torch.randn(8, 4, 64)
    stats = loss_fn.compute_stats(latent)

    assert "effective_rank_mean" in stats
    assert "effective_rank_std" in stats
    assert "svd_entropy_loss" in stats
    assert "cosine_sim_mean" in stats
    assert "top_sv_ratio" in stats
    assert "num_tokens" in stats
    assert stats["num_tokens"].item() == 8

    # Effective rank should be between 1 and T
    erank = stats["effective_rank_mean"].item()
    assert 1.0 <= erank <= 8.0, f"Effective rank {erank} out of expected range [1, 8]"


def test_loss_range():
    """Loss should be in [0, 1] for svd_entropy mode."""
    loss_fn = RankDiversityLoss(mode="svd_entropy")
    for _ in range(10):
        latent = torch.randn(8, 4, 64)
        loss = loss_fn(latent)
        assert 0.0 <= loss.item() <= 1.0, f"Loss {loss.item()} outside [0, 1]"


def test_more_tokens_same_loss_direction():
    """
    Verify that random orthogonal tokens maintain low loss even at larger T.
    This is the core property we want: more tokens → still high effective rank.
    """
    loss_fn = RankDiversityLoss(mode="svd_entropy", center_before_svd=False)
    D = 128
    for T in [4, 8, 16, 32]:
        B = 2
        # Generate random orthogonal tokens
        X = torch.randn(B, D, T)
        Q, _ = torch.linalg.qr(X)
        Q = Q[:, :, :T].transpose(1, 2)  # [B, T, D]
        latent = Q.transpose(0, 1)  # [T, B, D]
        loss = loss_fn(latent)
        assert loss.item() < 0.15, f"T={T}: Orthogonal tokens should have low loss, got {loss.item()}"


if __name__ == "__main__":
    print("Running rank diversity loss tests...")
    tests = [
        ("input_validation", test_input_validation),
        ("invalid_mode", test_invalid_mode),
        ("svd_orthogonal_low", test_svd_entropy_orthogonal_low_loss),
        ("svd_collapsed_high", test_svd_entropy_collapsed_high_loss),
        ("cosine_orthogonal_low", test_cosine_orthogonal_low_loss),
        ("cosine_parallel_high", test_cosine_parallel_high_loss),
        ("combined_mode", test_combined_mode),
        ("single_token_zero", test_single_token_returns_zero),
        ("gradient_flow_svd", test_gradient_flow),
        ("gradient_flow_cosine", test_gradient_flow_cosine),
        ("gradient_flow_combined", test_gradient_flow_combined),
        ("compute_stats", test_compute_stats),
        ("loss_range", test_loss_range),
        ("scaling_tokens", test_more_tokens_same_loss_direction),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed > 0:
        sys.exit(1)
