import json
import random
import time
import mlflow
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment
import torch

from ml4cps import examples, tools, vis
from ml4cps import debta



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

    # PROPOSED APPROACH EXPERIMENTS
    columns = cont_data[0].columns
    train_data = debta.WindowedSequenceDataset(cont_data)
    valid_data = debta.WindowedSequenceDataset(cont_data_valid)
    test_data = debta.WindowedSequenceDataset(cont_data_test)
    
    mean, std = train_data.normalize()
    valid_data.normalize(mean=mean, std=std)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    NUM_EXPERIMENTS = 30
    NUM_HIDDEN = 10
    MAX_EPOCH = 50
    VERBOSE = 0
    LEARNING_RATE = [0.005, 0.01]
    WINDOW_SIZE = [1, 10, 30]
    
    for exp_i in range(1, NUM_EXPERIMENTS+1):
        with mlflow.start_run():
            model = debta.DEBTA(num_y=24,
                                num_h=NUM_HIDDEN,
                                first_hidden_size=50,
                                num_sigm_layers=3,
                                sigma=0.3,
                                device=device,
                                log_mlflow=True)
    
            model.pretrain_layers(train_data=train_data,
                                  valid_data=valid_data,
                                  lr=LEARNING_RATE,
                                  verbose=VERBOSE,
                                  max_epoch=MAX_EPOCH,
                                  batch_size=128)
    
            model.learn_latent_automaton(train_dataset=train_data, train_config=train_config, valid_dataset=valid_data,
                                         valid_config=valid_config, verbose=VERBOSE)
    
            vis.plot_cps_component(model, freq_as_edge_thickness=True, output='dash')
            model.remove_rare_transitions(min_num=20)
            fig = vis.plot_cps_plotly(model)
            tools.log_plotly_figure_to_mlflow(fig, "cps_component_reduced")
            vis.plot_cps_component(model, event_label=False, freq_as_edge_thickness=True, show_transition_timing=True, node_labels=True, init_label=True, show_transition_freq=True, output='dash', dash_port=8051)
            print(model)
            print('Check browser for the automaton visualization')
            time.sleep(10)
    
            model.evaluate_purity(
                data=train_data,
                mode_data=mode_data,
                metric_name="purity_train"
            )
    
            model.evaluate_purity(
                data=valid_data,
                mode_data=mode_data_valid,
                metric_name="purity_valid"
            )
    
            model.evaluate_purity(
                data=test_data,
                mode_data=mode_data_test,
                metric_name="purity_test"
            )
    
            # Plot true modes vs predicted modes with data
            predicted_modes_train = model.predict(None, train_data.sequences.squeeze(1).detach())
            true_modes_train = np.concatenate([mode_data[i] for i in range(len(mode_data))])
    
            # Create time indices for plotting
            time_indices = list(range(len(true_modes_train)))
    
            # Create plotly figure for true vs predicted modes
            fig_modes = go.Figure()
            fig_modes.add_trace(go.Scatter(
                x=time_indices,
                y=true_modes_train,
                mode='lines+markers',
                name='True Modes',
                marker=dict(size=4)
            ))
            fig_modes.add_trace(go.Scatter(
                x=time_indices,
                y=predicted_modes_train,
                mode='lines+markers',
                name='Predicted Modes',
                marker=dict(size=4)
            ))
            fig_modes.update_layout(
                title='True Modes vs Predicted Modes Over Time (Train)',
                xaxis_title='Time Index',
                yaxis_title='Mode',
                hovermode='closest'
            )
            tools.log_plotly_figure_to_mlflow(fig_modes, "true_vs_predicted_modes_train")
            fig_modes.show('browser')
    
            # Create contingency matrix (rows=true modes, columns=predicted clusters)
            true_modes_unique = sorted(set(true_modes_train))
            predicted_clusters_unique = sorted(set(predicted_modes_train))
    
            contingency = pd.crosstab(
                pd.Series(true_modes_train, name='True'),
                pd.Series(predicted_modes_train, name='Predicted')
            )
    
            contingency = contingency.reindex(
                index=true_modes_unique,
                columns=predicted_clusters_unique,
                fill_value=0
            )
    
            # Keep true-mode rows fixed; reorder predicted-cluster columns for best alignment.
            cm_raw = contingency.to_numpy()
            if cm_raw.size > 0 and contingency.shape[1] > 0:
                cost = cm_raw.max() - cm_raw
                row_ind, col_ind = linear_sum_assignment(cost)
                matched_pairs = sorted(zip(row_ind, col_ind), key=lambda x: x[0])
                matched_cols = [col_pos for _, col_pos in matched_pairs]
                unmatched_cols = [j for j in range(contingency.shape[1]) if j not in matched_cols]
                ordered_col_positions = matched_cols + unmatched_cols
                contingency_aligned = contingency.iloc[:, ordered_col_positions]
            else:
                contingency_aligned = contingency
    
            cm = contingency_aligned.values
    
            # Create contingency matrix heatmap
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=contingency_aligned.columns.tolist(),
                y=contingency_aligned.index.tolist(),
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            fig_cm.update_layout(
                title='Contingency Matrix: True vs Predicted Modes (Train)',
                xaxis_title='Predicted Mode',
                yaxis_title='True Mode',
                width=1400,
                height=1400
            )
            tools.log_plotly_figure_to_mlflow(fig_cm, "contingency_matrix_train")
            fig_cm.show('browser')
            with open("conveyor_system_reconfig_data.json", "r", encoding="utf-8") as f:
                cases = json.load(f)

            correct_debta = 0
            for i, (test_input, test_config, target_config) in enumerate(zip(cases['continuous_windows'], cases['current_configs'], cases['target_configs'])):
                tiny_ds = debta.WindowedSequenceDataset(pd.DataFrame(test_input))
                tiny_ds.normalize(mean=mean, std=std)
                predicted_latent_state = model.predict(None, tiny_ds.sequences.squeeze(1).detach())
                
                transitions = model.out_transitions(predicted_latent_state)
                eval_configs = dict()
                for trans in transitions:
                    timings = model._timing_count_for_transition_data(trans[3])
                    for k, v in timings.items():
                        if k in eval_configs.keys():
                            eval_configs[k] += v
                        else:
                            eval_configs[k] = v
                eval_configs.pop(test_config, None)
                best_config = max(eval_configs, key=eval_configs.get, default=None)
                if best_config in target_config:
                    correct_debta += 1
                elif len(target_config) == 0 and best_config is None:
                    correct_debta += 1

            print("DEBTA_RECONFIG: Number correct: {}".format(correct_debta))
            mlflow.log_metric('ReconfigAccuracy', correct_debta/1000)

# python -m mlflow ui --backend-store-uri tests\mlruns
