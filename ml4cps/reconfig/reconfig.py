"""
CPS reconfiguration is a process of finding a configuration that will bring/maintain the system in the desired state.
"""
import networkx as nx
import numpy as np
from ml4cps import tools


class Reconfigurator:
    """

    """
    def __init__(self, system_model):
        self.system_model = system_model

    def reconfigure(self, state):
        pass


def example_two_tank_system():
    edges = [("bv01", "l1", 1), ("bp21", "l1", 1), ("bp12", "l2", 1),
                     ("bv02", "l2", 1), ("bh1", "v1", 1), ("bh2", "v1", 1),
                     ("bv10", "l1", -1), ("bp12", "l1", -1), ("bp21", "l2", -1),
                     ("bv20", "l2", -1), ("bp21", "v1", -1), ("bv02", "v2", -1),
                     ("bc1", "v2", -1), ("bc2", "v2", -1)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    return G


def similarity_based_historical_baseline(
        test_input,
        test_config,
        train_inputs,
        train_configs,
        window_size: int=1
):
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")

    train_inputs = tools.window(train_inputs, window_size)
    train_configs = tools.window(train_configs, window_size)

    if isinstance(train_inputs, list):
        train_inputs = np.concatenate(train_inputs)
    if isinstance(train_configs, list):
        train_configs = np.concatenate(train_configs)

    train_mean = np.mean(train_inputs, axis=(0, 1), keepdims=True)
    train_std = np.std(train_inputs, axis=(0, 1), keepdims=True)
    train_inputs = (train_inputs - train_mean) / (train_std + 1e-8)

    current_input_array = np.asarray(test_input)
    if current_input_array.ndim == 2:
        current_input_array = current_input_array[None, :, :]
    current_input_array = current_input_array[0, 0:window_size, :]

    standardized_current_window = (current_input_array - train_mean) / (train_std + 1e-8)

    if train_configs.ndim == 2:
        train_configs = train_configs[:,-1]

    train_inputs = train_inputs[train_configs != test_config]
    train_configs = train_configs[train_configs != test_config]

    distances = np.linalg.norm(train_inputs - standardized_current_window, axis=(1,2))
    if len(distances) == 0:
        return None

    winner_ind = np.argmin(distances)
    return train_configs[winner_ind]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    SM = example_two_tank_system()

    from ml4cps.vis import plot_bipartite_graph

    plot_bipartite_graph(SM)