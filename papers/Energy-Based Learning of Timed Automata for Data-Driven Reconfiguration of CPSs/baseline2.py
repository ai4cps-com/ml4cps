import json
import random
import numpy as np

from ml4cps import examples, tools, reconfig


if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)

    # LOAD DATA
    discrete_data, cont_data = examples.conveyor_system_sfowl(split=True)
    discrete_data, discrete_data_valid, discrete_data_test = discrete_data
    cont_data, cont_data_valid, cont_data_test = cont_data

    train_config = ['Config' + d['Path'].astype(int).astype(str) for d in discrete_data]

    mode_data = tools.encode_columns_to_string(discrete_data)
    mode_data_valid = tools.encode_columns_to_string(discrete_data_valid)
    mode_data_test = tools.encode_columns_to_string(discrete_data_test)

    with open("conveyor_system_reconfig_data.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    WINDOW_SIZES = [1, 10, 20]
    for window_size in WINDOW_SIZES:
        correct_nn = 0
        for i, (test_input, test_config, target_config) in enumerate(zip(cases['continuous_windows'], cases['current_configs'], cases['target_configs'])):
            predicted_config = reconfig.similarity_based_historical_baseline(test_input=test_input,
                                                                             test_config=test_config,
                                                                             train_inputs=cont_data,
                                                                             train_configs=train_config,
                                                                             window_size=window_size)
            if predicted_config in target_config:
                correct_nn += 1
            elif len(target_config) == 0 and predicted_config is None:
                correct_nn += 1
            if (i % 10) == 0:
                print(i)

        print("SIMILARITY_NN_RECONFIG: Number correct: {}".format(correct_nn))