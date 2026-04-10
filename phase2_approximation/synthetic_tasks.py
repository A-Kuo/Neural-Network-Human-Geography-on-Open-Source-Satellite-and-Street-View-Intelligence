"""
synthetic_tasks.py — Synthetic benchmarks for GNN expressivity validation.

Task: Create a function that depends on k-hop neighborhoods.
Expected: Only GNNs with depth >= k succeed.

This validates Stone-Weierstrass theory before applying to real Chicago data.
"""

from typing import Callable, Dict, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data


def create_synthetic_income_function(
    graph: nx.Graph,
    num_hops: int = 5,
    scale: float = 100000.0,
    noise_std: float = 5000.0,
    seed: int = 42,
) -> Callable:
    """
    Create synthetic income function that requires num_hops receptive field.

    Formula:
    income[i] = sum over k-hop neighbors:
        - alpha * local_feature[i]           (immediate)
        - beta * avg_transit_5hop[i]         (5-hop dependency)
        - gamma * global_component[i]        (global property)

    Key: Function is hidden in multi-hop structure.
    - 1-layer GNN cannot access 5-hop → FAILS
    - 5-layer GNN can access 5-hop → SUCCEEDS

    Args:
        graph: NetworkX graph
        num_hops: dependency distance
        scale: income scale ($50k-$150k)
        noise_std: Gaussian noise
        seed: reproducibility

    Returns:
        Function f: node_id -> income (float)
    """
    np.random.seed(seed)
    n_nodes = graph.number_of_nodes()

    # Compute k-hop neighborhoods
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    # Node features (building density, visual features, etc.)
    node_local_features = np.random.normal(0, 1, n_nodes)

    # Transit accessibility at each hop
    transit_signals = {}
    for hop in range(1, num_hops + 1):
        hop_signal = np.zeros(n_nodes)
        for i, node in enumerate(graph.nodes()):
            neighbors_at_hop = [
                other
                for other in graph.nodes()
                if shortest_paths.get(node, {}).get(other, np.inf) == hop
            ]
            if neighbors_at_hop:
                hop_signal[i] = len(neighbors_at_hop) / (hop**1.5)
        transit_signals[hop] = hop_signal

    # Income formula: combines local + multi-hop signals
    income_base = np.zeros(n_nodes)
    for i, node in enumerate(graph.nodes()):
        # Component 1: Local features
        income_base[i] += 0.3 * scale * node_local_features[i]

        # Component 2: Transit at k-hops (multi-scale signal)
        if num_hops in transit_signals:
            income_base[i] += 0.5 * scale * transit_signals[num_hops][i]

        # Component 3: Cluster-based (global property)
        component_id = None
        for comp in nx.connected_components(graph):
            if node in comp:
                component_id = len(comp)
                break
        income_base[i] += 0.2 * scale * (component_id / n_nodes if component_id else 0)

    # Normalize to realistic range
    income_base = income_base - income_base.min()
    income_base = income_base / (income_base.max() + 1e-10)
    income_base = income_base * scale + scale / 2

    # Add noise
    noise = np.random.normal(0, noise_std, n_nodes)
    income_noisy = income_base + noise
    income_noisy = np.clip(income_noisy, 20000, 200000)

    def f(node_id: int) -> float:
        """Return synthetic income for a node."""
        node_list = list(graph.nodes())
        idx = node_list.index(node_id)
        return float(income_noisy[idx])

    f.ground_truth = income_noisy
    f.node_list = list(graph.nodes())
    f.required_hops = num_hops
    f.transit_signals = transit_signals

    return f


def evaluate_gnn_on_synthetic_task(
    gnn_model: nn.Module,
    graph: nx.Graph,
    target_fn: Callable,
    num_hops_required: int,
    num_epochs: int = 100,
    device: str = "cpu",
) -> Dict:
    """
    Train GNN and measure whether it solves k-hop task.

    Question: Does GNN depth match task difficulty (num_hops_required)?

    Args:
        gnn_model: Initialized GNN model
        graph: NetworkX graph
        target_fn: Synthetic function (from create_synthetic_income_function)
        num_hops_required: Ground truth k (income depends on k-hops)
        num_epochs: Training epochs
        device: Compute device

    Returns:
        {
            test_mse: Mean squared error,
            test_mae: Mean absolute error,
            success: bool (MAE < threshold),
            num_hops_required: Ground truth k,
            predicted: Predictions,
            ground_truth: True values,
        }
    """
    n_nodes = graph.number_of_nodes()
    node_list = list(graph.nodes())

    # Create target tensor
    y_true = torch.tensor(target_fn.ground_truth, dtype=torch.float32).unsqueeze(1).to(device)

    # Node features (dummy: random)
    x = torch.randn(n_nodes, 8, dtype=torch.float32).to(device)

    # Edge indices
    edge_list = list(graph.edges())
    if edge_list:
        edge_index = (
            torch.tensor(
                [[node_list.index(u), node_list.index(v)] for u, v in edge_list]
                + [[node_list.index(v), node_list.index(u)] for u, v in edge_list],
                dtype=torch.long,
            )
            .t()
            .contiguous()
            .to(device)
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

    # Training
    gnn_model = gnn_model.to(device)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        gnn_model.train()
        optimizer.zero_grad()
        out = gnn_model(x, edge_index)
        loss = criterion(out, y_true)
        loss.backward()
        optimizer.step()

    # Evaluate
    gnn_model.eval()
    with torch.no_grad():
        predictions = gnn_model(x, edge_index)
        test_mse = criterion(predictions, y_true).item()
        test_mae = (predictions - y_true).abs().mean().item()

    # Success: MAE < income range / 10
    income_range = y_true.max() - y_true.min()
    success = test_mae < (income_range / 10).item()

    return {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "success": success,
        "num_hops_required": num_hops_required,
        "predicted": predictions.detach().cpu().numpy(),
        "ground_truth": target_fn.ground_truth,
    }
