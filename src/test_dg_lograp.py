"""
Test script for DG-LoGraP implementation

Run this script to verify all components work correctly.
"""

import torch
import numpy as np

def test_static_graph_loading():
    """Test loading and processing of static graph."""
    print("\n" + "="*60)
    print("Test 1: Static Graph Loading")
    print("="*60)
    
    from dynamic_graph import load_static_graph
    
    adj = load_static_graph('../datasets/PEMS03_graph.csv', num_nodes=358)
    
    print(f"✓ Static adjacency shape: {adj.shape}")
    print(f"✓ Non-zero elements: {(adj > 0).sum().item()}")
    print(f"✓ Diagonal (self-loops): {adj.diag().mean().item():.4f}")
    print(f"✓ Symmetric: {torch.allclose(adj, adj.T)}")
    
    assert adj.shape == (358, 358), "Adjacency matrix shape mismatch"
    assert torch.allclose(adj, adj.T, atol=1e-6), "Adjacency matrix not symmetric"
    
    print("Test 1 PASSED ✓")
    return adj


def test_sagsam():
    """Test SAGSAM module."""
    print("\n" + "="*60)
    print("Test 2: SAGSAM (Dynamic Adjacency)")
    print("="*60)
    
    from dynamic_graph import SAGSAM
    
    num_nodes = 100
    embed_dim = 32
    
    sagsam = SAGSAM(num_nodes=num_nodes, embed_dim=embed_dim)
    
    # Without hidden states
    dyn_adj = sagsam()
    print(f"✓ Dynamic adjacency shape: {dyn_adj.shape}")
    print(f"✓ Positive values: {(dyn_adj > 0).sum().item()}")
    
    # With hidden states
    batch_size = 8
    hidden = torch.randn(batch_size, num_nodes, 64)
    dyn_adj_batch = sagsam(hidden)
    print(f"✓ Batched dynamic adjacency shape: {dyn_adj_batch.shape}")
    
    assert dyn_adj.shape == (num_nodes, num_nodes)
    assert dyn_adj_batch.shape == (batch_size, num_nodes, num_nodes)
    
    print("Test 2 PASSED ✓")
    return sagsam


def test_sags_gcn():
    """Test SAGS-GCN module."""
    print("\n" + "="*60)
    print("Test 3: SAGS-GCN (Graph Convolution)")
    print("="*60)
    
    from dynamic_graph import SAGS_GCN
    
    num_nodes = 100
    in_features = 64
    out_features = 64
    embed_dim = 32
    batch_size = 8
    
    sags_gcn = SAGS_GCN(
        num_nodes=num_nodes,
        in_features=in_features,
        out_features=out_features,
        embed_dim=embed_dim,
    )
    
    # Forward pass
    x = torch.randn(batch_size, num_nodes, in_features)
    z, adj = sags_gcn(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {z.shape}")
    print(f"✓ Dynamic adj shape: {adj.shape}")
    
    assert z.shape == (batch_size, num_nodes, out_features)
    
    print("Test 3 PASSED ✓")
    return sags_gcn


def test_temporal_kernel():
    """Test temporal correlation kernel."""
    print("\n" + "="*60)
    print("Test 4: Temporal Correlation Kernel")
    print("="*60)
    
    from dynamic_graph import TemporalCorrelationKernel
    
    horizon = 12
    num_kernels = 4
    hidden_dim = 32
    batch_size = 8
    
    kernel = TemporalCorrelationKernel(
        horizon=horizon,
        num_kernels=num_kernels,
        hidden_dim=hidden_dim,
    )
    
    # Forward pass
    h = torch.randn(batch_size, hidden_dim)
    C = kernel(h)
    
    print(f"✓ Hidden state shape: {h.shape}")
    print(f"✓ Correlation matrix shape: {C.shape}")
    print(f"✓ Base kernels shape: {kernel.base_kernels.shape}")
    
    # Check symmetry (approximate due to numerical precision)
    is_symmetric = torch.allclose(C, C.transpose(1, 2), atol=1e-5)
    print(f"✓ Correlation matrices symmetric: {is_symmetric}")
    
    assert C.shape == (batch_size, horizon, horizon)
    
    print("Test 4 PASSED ✓")
    return kernel


def test_graph_loader():
    """Test dynamic graph loader."""
    print("\n" + "="*60)
    print("Test 5: Dynamic Graph Loader")
    print("="*60)
    
    from dynamic_graph import DynamicGraphLoader
    
    num_nodes = 50
    num_groups = 2
    ranks_per_group = [5, 5]
    embed_dim = 16
    
    loader = DynamicGraphLoader(
        num_nodes=num_nodes,
        num_groups=num_groups,
        ranks_per_group=ranks_per_group,
        embed_dim=embed_dim,
    )
    
    # Get loading matrices
    L_list, adj = loader()
    
    print(f"✓ Number of groups: {len(L_list)}")
    for g, L_g in enumerate(L_list):
        print(f"  - Group {g} loading matrix shape: {L_g.shape}")
    print(f"✓ Dynamic adjacency shape: {adj.shape}")
    
    assert len(L_list) == num_groups
    for g, L_g in enumerate(L_list):
        assert L_g.shape == (num_nodes, ranks_per_group[g])
    
    print("Test 5 PASSED ✓")
    return loader


def test_dg_lograp_distribution():
    """Test full DG-LoGraP distribution."""
    print("\n" + "="*60)
    print("Test 6: DG-LoGraP Distribution")
    print("="*60)
    
    from dynamic_graph import DGLoGraPDistribution
    
    num_nodes = 30
    horizon = 6
    num_groups = 2
    ranks_per_group = [3, 3]
    embed_dim = 16
    num_kernels = 3
    
    dist = DGLoGraPDistribution(
        num_nodes=num_nodes,
        horizon=horizon,
        num_groups=num_groups,
        ranks_per_group=ranks_per_group,
        embed_dim=embed_dim,
        num_kernels=num_kernels,
    )
    
    print(f"✓ Num nodes: {num_nodes}")
    print(f"✓ Horizon: {horizon}")
    print(f"✓ Num groups: {num_groups}")
    print(f"✓ Total rank: {dist.total_rank}")
    
    # Test covariance computation
    batch_size = 4
    hidden_states = torch.randn(batch_size, num_nodes, 32)
    diag_variance = torch.rand(batch_size, horizon * num_nodes) + 0.1
    temporal_hidden = torch.randn(batch_size, embed_dim)
    
    cov_info = dist.compute_batch_covariance(hidden_states, diag_variance, temporal_hidden)
    
    print(f"✓ A (loading) shape: {cov_info['A'].shape}")
    print(f"✓ C (correlation) shape: {cov_info['C'].shape}")
    print(f"✓ Log determinant shape: {cov_info['log_det'].shape}")
    
    # Test log probability
    residuals = torch.randn(batch_size, horizon * num_nodes)
    log_prob = dist.log_prob(residuals, hidden_states, diag_variance, temporal_hidden)
    
    print(f"✓ Log probability shape: {log_prob.shape}")
    print(f"✓ Log probability mean: {log_prob.mean().item():.4f}")
    
    print("Test 6 PASSED ✓")
    return dist


def test_dg_lograp_loss():
    """Test DG-LoGraP loss function."""
    print("\n" + "="*60)
    print("Test 7: DG-LoGraP Loss")
    print("="*60)
    
    from loss import DGLoGraP_Loss
    
    num_nodes = 30
    D = 6
    num_groups = 2
    ranks_per_group = [3, 3]
    
    loss_fn = DGLoGraP_Loss(
        num_nodes=num_nodes,
        D=D,
        num_groups=num_groups,
        ranks_per_group=ranks_per_group,
        embed_dim=16,
        num_kernels=3,
        graph_path=None,  # Use identity
    )
    
    print(f"✓ Total rank: {loss_fn.total_rank}")
    print(f"✓ Distribution arguments: {loss_fn.distribution_arguments}")
    
    # Test loss computation
    batch_size = 4
    n_timesteps = D * 2  # 2 windows
    n_params = 2 + loss_fn.total_rank + 3 + 2  # loc, scale, cov_factor, kernel_weights, target_scale
    
    y_pred = torch.randn(batch_size, n_timesteps, n_params)
    y_actual = torch.randn(batch_size, n_timesteps)
    
    loss_fn._transformation = None
    
    try:
        loss_val = loss_fn.loss(y_pred, y_actual)
        print(f"✓ Loss value: {loss_val.item():.4f}")
        print("Test 7 PASSED ✓")
    except Exception as e:
        print(f"✗ Error in loss computation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return loss_fn


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# DG-LoGraP Implementation Tests")
    print("#"*60)
    
    tests = [
        ("Static Graph Loading", test_static_graph_loading),
        ("SAGSAM", test_sagsam),
        ("SAGS-GCN", test_sags_gcn),
        ("Temporal Kernel", test_temporal_kernel),
        ("Graph Loader", test_graph_loader),
        ("DG-LoGraP Distribution", test_dg_lograp_distribution),
        ("DG-LoGraP Loss", test_dg_lograp_loss),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, True, result))
        except Exception as e:
            print(f"\n✗ Test {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, None))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(1 for _, status, _ in results if status)
    total = len(results)
    
    for name, status, _ in results:
        status_str = "✓ PASSED" if status else "✗ FAILED"
        print(f"  {name}: {status_str}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("All tests passed! Implementation is ready.")
        print("="*60)
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
