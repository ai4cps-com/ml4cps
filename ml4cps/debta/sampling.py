"""Gibbs sampling of constant configurations and paths with a clamped root.

Finite candidate sets use exact path and configuration blocks. Mixed continuous
configurations use an exact path block and bounded Metropolis configuration
updates. Paths are drawn using backward messages.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp



def backward_messages(log_transition, state_cost, horizon, beta):
    """Log finite-horizon goal normalizers for each configuration and source."""
    transitions = np.asarray(log_transition, dtype=float)
    costs = np.asarray(state_cost, dtype=float)
    if transitions.ndim != 3 or transitions.shape[1] != transitions.shape[2] or not len(transitions):
        raise ValueError("Expected [configuration, source, destination] log probabilities")
    if costs.shape != (transitions.shape[1],) or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Expected finite nonnegative mode costs")
    if type(horizon) is not int or horizon < 1 or not np.isfinite(beta) or beta < 0:
        raise ValueError("Invalid horizon or goal weight")
    if np.isnan(transitions).any() or np.isposinf(transitions).any():
        raise ValueError("Invalid log probabilities")
    if not np.allclose(logsumexp(transitions, axis=2), 0., atol=1e-8):
        raise ValueError("Transition rows must be normalized and nonempty")
    messages = np.zeros((len(transitions), horizon + 1, transitions.shape[1]))
    for step in range(horizon - 1, -1, -1):
        messages[:, step] = logsumexp(transitions - beta * costs[None, None, :]
                                                    + messages[:, step + 1, None, :], axis=2)
    return messages


def path_energy(log_transition, state_cost, config, path, beta):
    path = np.asarray(path, dtype=int)
    return float(-log_transition[config, path[:-1], path[1:]].sum() + beta * np.asarray(state_cost)[path[1:]].sum())


@dataclass(frozen=True)
class PathSamples:
    configuration_indices: np.ndarray
    paths: np.ndarray
    energies: np.ndarray
    configuration_values: np.ndarray | None = None

    @property
    def best_index(self):
        return min(range(len(self.energies)), key=lambda i: (
            self.energies[i], self.configuration_indices[i], tuple(self.paths[i])))


def gibbs_paths(log_transition, state_cost, initial, *, horizon=10, beta=20.,
                n_samples=500, burn_in=200, thin=1, seed=42,
                configuration_bounds=None, initial_configuration=None,
                configuration_steps=1, proposal_scale=.1, configuration_log_prior=None):
    """Sample a constant configuration and path with a fixed initial state.

    A precomputed transition table uses exact path/configuration blocks over its
    finite candidate set. A ConditionalTransitionRBM instead requires finite
    configuration_bounds and uses exact path blocks followed by coordinate-wise
    Metropolis updates. Gaussian coordinates use symmetric random-walk proposals;
    binary coordinates use lazy flips within their allowed bounds. proposal_scale is a
    fraction of each coordinate's range. Out-of-bounds proposals are rejected.

    The bounded configuration prior is uniform unless configuration_log_prior
    supplies a log density. Gaussian RBM units do not identify this prior from
    conditional transition fitting. configuration_values stores each retained
    mixed configuration, and its indices then refer to those rows. Energies
    include the negative log prior. The best retained draw approximates joint
    MAP; finite draws guarantee neither mixing nor optimality.
    """
    from .rbms import ConditionalTransitionRBM

    if isinstance(log_transition, ConditionalTransitionRBM):
        model = log_transition
        bounds = np.asarray(configuration_bounds, dtype=float)
        if (bounds.shape != (model.n_config, 2) or not np.isfinite(bounds).all()
                or (bounds[:, 0] > bounds[:, 1]).any()
                or not np.isfinite(bounds[:, 1] - bounds[:, 0]).all()):
            raise ValueError("Supply finite [lower, upper] bounds for every configuration variable")
        if type(configuration_steps) is not int or configuration_steps < 1:
            raise ValueError("configuration_steps must be a positive integer")
        for value, minimum in ((n_samples, 1), (burn_in, 0), (thin, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError("Invalid sampling budget")
        n_states = len(model.state_codes)
        if type(initial) is not int or not 0 <= initial < n_states:
            raise ValueError("Invalid clamped initial mode")
        binary = np.array([kind == "binary" for kind in model.configuration_types])
        scale = np.broadcast_to(np.asarray(proposal_scale, dtype=float), (model.n_config,))
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("proposal_scale must contain finite positive range fractions")
        choices = [np.array([bit for bit in (0., 1.) if lo <= bit <= hi])
                   if is_binary else None for is_binary, (lo, hi) in zip(binary, bounds)]
        if any(is_binary and not len(bits) for is_binary, bits in zip(binary, choices)):
            raise ValueError("Binary configuration bounds must admit zero or one")
        rng = np.random.default_rng(seed)
        if initial_configuration is None:
            config = np.array([rng.choice(bits) if is_binary else rng.uniform(lo, hi)
                               for is_binary, bits, (lo, hi) in zip(binary, choices, bounds)])
        else:
            config = np.asarray(initial_configuration, dtype=float).copy()
        if (config.shape != (model.n_config,) or not np.isfinite(config).all()
                or (config < bounds[:, 0]).any() or (config > bounds[:, 1]).any()):
            raise ValueError("Initial configuration must lie inside the supplied bounds")

        def log_prior(value):
            result = 0. if configuration_log_prior is None else float(configuration_log_prior(value.copy()))
            if np.isnan(result) or np.isposinf(result):
                raise ValueError("Configuration log prior must be finite or negative infinity")
            return result

        prior = log_prior(config)
        if not np.isfinite(prior):
            raise ValueError("Initial configuration must have positive prior density")
        table = model.log_probabilities(config[None])[0]
        # Also validates goal costs, horizon, beta, and normalized transition rows.
        backward_messages(table[None], state_cost, horizon, beta)
        costs = np.asarray(state_cost, dtype=float)
        values = np.empty((n_samples, model.n_config))
        paths = np.empty((n_samples, horizon + 1), dtype=int)
        energies = np.empty(n_samples)
        retained = 0
        for sweep in range(burn_in + n_samples * thin):
            messages = backward_messages(table[None], costs, horizon, beta)[0]
            path = np.empty(horizon + 1, dtype=int)
            path[0] = initial
            for step in range(horizon):
                logits = table[path[step]] - beta * costs + messages[step + 1]
                probabilities = np.exp(logits - logsumexp(logits))
                path[step + 1] = rng.choice(n_states, p=probabilities)
            weight = float(table[path[:-1], path[1:]].sum()) + prior
            for _ in range(configuration_steps):
                for coordinate, (lo, hi) in enumerate(bounds):
                    if lo == hi or (binary[coordinate] and len(choices[coordinate]) == 1):
                        continue
                    proposal = config.copy()
                    if binary[coordinate]:
                        proposal[coordinate] = rng.choice(choices[coordinate])
                        if proposal[coordinate] == config[coordinate]:
                            continue
                    else:
                        proposal[coordinate] += rng.normal(scale=scale[coordinate] * (hi - lo))
                    if not lo <= proposal[coordinate] <= hi:
                        continue
                    proposed_prior = log_prior(proposal)
                    if np.isneginf(proposed_prior):
                        continue
                    proposed_table = model.log_probabilities(proposal[None])[0]
                    proposed_weight = float(proposed_table[path[:-1], path[1:]].sum()) + proposed_prior
                    if np.log(rng.random()) < proposed_weight - weight:
                        config, table, prior, weight = proposal, proposed_table, proposed_prior, proposed_weight
            if sweep >= burn_in and (sweep - burn_in) % thin == 0:
                values[retained], paths[retained] = config, path
                energies[retained] = -weight + beta * costs[path[1:]].sum()
                retained += 1
        return PathSamples(np.arange(n_samples), paths, energies, values)

    if any(value is not None for value in (configuration_bounds, initial_configuration, configuration_log_prior)):
        raise ValueError("Continuous configuration settings require a ConditionalTransitionRBM")
    messages = backward_messages(log_transition, state_cost, horizon, beta)
    transitions, costs = np.asarray(log_transition), np.asarray(state_cost)
    n_configs, n_states, _ = transitions.shape
    if type(initial) is not int or not 0 <= initial < n_states:
        raise ValueError("Invalid clamped initial mode")
    for value, minimum in ((n_samples, 1), (burn_in, 0), (thin, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError("Invalid sampling budget")
    # These categorical tables depend on the model and clamp, never on a
    # posterior optimum or a true simulator trajectory.
    logits = (transitions[:, None, :, :] - beta * costs[None, None, None, :]
              + messages[:, 1:, None, :])
    logits -= logsumexp(logits, axis=3, keepdims=True)
    cumulative = np.cumsum(np.exp(logits), axis=3)
    cumulative[:, :, :, -1] = 1.
    rng = np.random.default_rng(seed)
    config = int(rng.integers(n_configs))
    indices = np.empty(n_samples, dtype=int)
    paths = np.empty((n_samples, horizon + 1), dtype=int)
    energies = np.empty(n_samples)
    retained = 0
    for sweep in range(burn_in + n_samples * thin):
        path = np.empty(horizon + 1, dtype=int)
        path[0] = initial
        for step in range(horizon):
            path[step + 1] = np.searchsorted(cumulative[config, step, path[step]], rng.random(), side="right")
        log_weights = transitions[:, path[:-1], path[1:]].sum(axis=1)
        probabilities = np.exp(log_weights - logsumexp(log_weights))
        config = int(rng.choice(n_configs, p=probabilities))
        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            indices[retained], paths[retained] = config, path
            energies[retained] = path_energy(transitions, costs, config, path, beta)
            retained += 1
    assert retained == n_samples
    return PathSamples(indices, paths, energies)
