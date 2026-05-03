import json
import random

import mlflow
import pandas as pd

from ml4cps import examples, tools
from ml4cps import reconfig


if __name__ == "__main__":
    random.seed(123)

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

    NUM_EXPERIMENTS = 10
    # Run standardized nearest-neighbor baseline

    with open("conveyor_system_reconfig_data.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    for exp in range(1, NUM_EXPERIMENTS+1):
        WINDOW_SIZE = [1, 10, 20, 30]
        with (mlflow.start_run()):
            print("Running experiment: {}".format(exp))
            win_size = WINDOW_SIZE[exp%len(WINDOW_SIZE)]
            mlflow.log_param('window_size', win_size)
            correct_nn = 0
            for i, (test_input, test_config, target_config) in enumerate(zip(cases['continuous_windows'], cases['current_configs'], cases['target_configs'])):
                predicted_config = reconfig.similarity_based_historical_baseline(test_input=test_input,
                                                                                 test_config=test_config,
                                                                                 train_inputs=cont_data,
                                                                                 train_configs=train_config,
                                                                                 window_size=win_size)
                if predicted_config in target_config:
                    correct_nn += 1
                elif len(target_config) == 0 and predicted_config is None:
                    correct_nn += 1
                print(i)

            print("SIMILARITY_NN_RECONFIG: Number correct: {}".format(correct_nn))
            mlflow.log_metric('reconfig_accuracy', correct_nn/1000)