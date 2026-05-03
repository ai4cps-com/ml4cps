import random
import warnings
import json
import numpy as np

from ml4cps import examples, tools


RECONFIG_CASE_WINDOW_SIZE = 20
RANDOM_SEED = 123

random.seed(123)


def conveyor_system_ground_truth_reconfig_model(system_state, current_config: str):
    """
    Returns the set of possible reconfiguration targets for a given system state and current configuration.
    """
    GROUND_TRUTH = {
        ("Stop", "Stop", "Stop", "Left", "Stop", "Stop"): {"Config1", "Config3"},
        ("Stop", "Stop", "Left", "Left", "Stop", "Stop"): {"Config1", "Config3"},
        ("Stop", "Left", "Stop", "Stop", "Stop", "Stop"): {"Config1", "Config3"},
        ("Stop", "Stop", "Left", "Stop", "Stop", "Stop"): {"Config1", "Config3"},
        ("Left", "Left", "Stop", "Stop", "Stop", "Stop"): {"Config1", "Config3"},
        ("Left", "Stop", "Stop", "Stop", "Stop", "Stop"): {"Config1", "Config3"},
        ("Right", "Stop", "Stop", "Stop", "Stop", "Stop"): {"Config2", "Config4"},
        ("Right", "Right", "Stop", "Stop", "Stop", "Stop"): {"Config2", "Config4"},
        ("Stop", "Right", "Stop", "Stop", "Stop", "Stop"): {"Config2", "Config4"},
        ("Stop", "Stop", "Right", "Stop", "Stop", "Stop"): {"Config2", "Config4"},
        ("Stop", "Stop", "Right", "Right", "Stop", "Stop"): {"Config2", "Config4"},
        ("Stop", "Stop", "Stop", "Right", "Stop", "Stop"): {"Config2", "Config4"},
    }

    if system_state not in GROUND_TRUTH:
        warnings.warn("System state not in the set of possible reconfiguration targets. "
                      "Please add to the set of possible reconfiguration targets.")
        return set()

    possible_configs = set(GROUND_TRUTH[system_state])
    possible_configs.discard(current_config)
    return possible_configs

if __name__ == "__main__":
    # LOAD DATA
    discrete_data, cont_data = examples.conveyor_system_sfowl(split=True)
    discrete_data, discrete_data_valid, discrete_data_test = discrete_data
    cont_data, cont_data_valid, cont_data_test = cont_data

    train_config = ['Config' + d['Path'].astype(int).astype(str) for d in discrete_data]
    valid_config = ['Config' + d['Path'].astype(int).astype(str) for d in discrete_data_valid]
    test_config = ['Config' + d['Path'].astype(int).astype(str) for d in discrete_data_test]

    mode_data = tools.encode_columns_to_string(discrete_data)
    mode_data_valid = tools.encode_columns_to_string(discrete_data_valid)
    mode_data_test = tools.encode_columns_to_string(discrete_data_test)

    all_test_indices = []
    for seq_idx, seq_df in enumerate(discrete_data_test):
        for point_idx in range(RECONFIG_CASE_WINDOW_SIZE - 1, len(seq_df)):
            all_test_indices.append((seq_idx, point_idx))

    # Sample 1000 random indices (or all if less than 1000)
    num_samples = 1000
    sampled_indices = random.sample(all_test_indices, min(num_samples, len(all_test_indices)))

    # Extract data for sampled points
    sampled_continuous_vectors = []
    sampled_configs = []
    sampled_discrete_states = []
    sampled_target_configs = []

    for seq_idx, point_idx in sampled_indices:
        # Extract continuous vector/window
        start_idx = point_idx - RECONFIG_CASE_WINDOW_SIZE + 1
        cont_signal = cont_data_test[seq_idx].iloc[start_idx:point_idx + 1].values

        sampled_continuous_vectors.append(cont_signal)

        # Extract current config
        current_config = test_config[seq_idx]
        sampled_configs.append(current_config.iloc[point_idx])

        # Extract discrete state
        discrete_state = tuple(discrete_data_test[seq_idx].iloc[point_idx].values[0:-1])
        sampled_discrete_states.append(discrete_state)

        sampled_target_configs.append(conveyor_system_ground_truth_reconfig_model(discrete_state, current_config.iloc[point_idx]))

    # Convert numpy arrays to lists for JSON serialization
    sampled_continuous_vectors_list = [vec.tolist() for vec in sampled_continuous_vectors]

    # Convert sets to lists for JSON serialization
    sampled_target_configs_list = [list(cfg) for cfg in sampled_target_configs]

    # Create the data structure to save
    data_to_save = {
        "continuous_windows": sampled_continuous_vectors_list,
        "current_configs": sampled_configs,
        "discrete_states": [list(state) for state in sampled_discrete_states],
        "target_configs": sampled_target_configs_list,
        "metadata": {
            "window_size": RECONFIG_CASE_WINDOW_SIZE,
            "num_samples": len(sampled_indices),
            "random_seed": RANDOM_SEED,
            "continuous_variable_names": list(cont_data_test[0].columns),
            "discrete_variable_names": list(discrete_data_test[0].columns[0:-1])
        }
    }

    # Save to JSON file
    with open('conveyor_system_reconfig_data.json', 'w') as f:
        json.dump(data_to_save, f, indent=1)
    
        print("Finished")

