import datetime
from ml4cps import tools, vis
from ml4cps.automata.base import Automaton
import numpy as np
import pandas as pd
import os


def simple_conveyor_8_states():
    """
    Simple conveyor model with 8 discrete states is defined in:
    [1] N. Hranisavljevic, A. Maier, and O. Niggemann, “Discretization of hybrid CPPS data into timed automaton using
    restricted Boltzmann machines,” Engineering Applications of Artificial Intelligence, vol. 95, p. 103826, 2020,
    doi: https://doi.org/10.1016/j.engappai.2020.103826.

    :return: Automaton object.
    """
    def time_fun():
        return np.random.normal(1, 0.1)

    A = Automaton(id="Simple Conveyor", dt=0.01, initial_q="q1", states=["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"],
                  transitions=[dict(source="q1", destination="q2", event="Place Item", prob=1, time=time_fun),
                               dict(source="q2", destination="q1", event="Take Item", prob=1, time=time_fun),
                               dict(source="q7", destination="q8", event="Place Item", prob=1, time=time_fun),
                               dict(source="q8", destination="q7", event="Take Item", prob=1, time=time_fun),
                               dict(source="q1", destination="q4", event="Move Downward", prob=1, time=time_fun),
                               dict(source="q4", destination="q7", event="Stop Movement", prob=1, time=time_fun),
                               dict(source="q7", destination="q3", event="Move Upward", prob=1, time=time_fun),
                               dict(source="q3", destination="q1", event="Stop Movement", prob=1, time=time_fun),
                               dict(source="q2", destination="q6", event="Move Downward", prob=1, time=time_fun),
                               dict(source="q6", destination="q8", event="Stop Movement", prob=1, time=time_fun),
                               dict(source="q8", destination="q5", event="Move Upward", prob=1, time=time_fun),
                               dict(source="q5", destination="q2", event="Stop Movement", prob=1, time=time_fun)])

    def discrete_output_fun(q, xt, xk):
        state_dict = {
            "q1": [0, 0],
            "q2": [0, 0],
            "q3": [0, 1],
            "q4": [1, 0],
            "q5": [0, 1],
            "q6": [1, 0],
            "q7": [0, 0],
            "q8": [0, 0]
        }
        return np.array(state_dict[q])

    def continuous_output_fun(q, xt, xk):
        std = 0.04
        cov = np.array([[1, 0], [0, 1]])
        mean_dict = {
            "q1": [0, 0],
            "q2": [0, 0.5],
            "q3": [0.5, 0.2],
            "q4": [0.5, 0.2],
            "q5": [0.2, 0.9],
            "q6": [0.2, 0.9],
            "q7": [0, 0],
            "q8": [0, 0.5]
        }
        mean = mean_dict[q]
        f = np.random.multivariate_normal(mean, std**2 * cov)
        return f

    def time_discrete_dynamics_fun(q, p, x, u):
        return ()

    A.output_d = discrete_output_fun
    A.output_y = continuous_output_fun
    A.time_discrete_dynamics = time_discrete_dynamics_fun

    A.reinitialize(0, state=("q1", (), ()))  # state is (q, xt, xk)
    return A


class BuckConverter (Automaton):
    def __init__(self):
        super().__init__(states=['q1', 'q2', 'q3'], dt=1e-5,
                         transitions=[('q1', 'e12', 'q2'),
                                        ('q2', 'e23', 'q3'),
                                        ('q2', 'e21', 'q1'),
                                        ('q3', 'e31', 'q1')])

        self.Vs = 24
        self.VcH = 12.1
        self.VcL = 11.9

        self.a00c, self.a01c, self.a10c, self.a11c, self.b0c, self.b1c = -271.6981, -377.3585, 454.5455, -45.4545, 377.3585, 0
        self.a00o, self.a01o, self.a10o, self.a11o, self.b0o, self.b1o = -196.2264, -377.3585, 454.5455, -45.4545, 0, 0

    def time_discrete_dynamics(self, q, p, x, u):
        # x_1 is il while x_2 is v_c
        if q == 'q1':
            x_dot_1 = self.a00c * x[0] + self.a01c * x[1] + self.b0c * self.Vs
            x_dot_2 = self.a10c * x[0] + self.a11c * x[1] + self.b1c * self.Vs
        elif q == 'q2':
            x_dot_1 = self.a00o * x[0] + self.a01o * x[1] + self.b0o * self.Vs
            x_dot_2 = self.a10o * x[0] + self.a11o * x[1] + self.b1o * self.Vs
        elif q == 'q3':
            x_dot_1 = 0.0
            x_dot_2 = self.a11o * x[1] + self.b1o * self.Vs
        else:
            raise Exception(f'Not a valid discrete state: {q}')

        x_1 = x[0] + x_dot_1 * self.dt
        x_2 = x[1] + x_dot_2 * self.dt
        return x_1, x_2

    def guards(self, q, x):
        if q == 'q1':
            if x[1] >= self.VcH:
                return "e12"
        elif q == 'q2':
            if x[0] <= 0:
                return "e23"
            if x[1] <= self.VcL:
                return "e21"
        elif q == 'q3':
            if x[1] <= self.VcL:
                return "e31"
        return None


def buck_converter():
    """
    Credits to the FAMOS paper authors.

    :return:
    """

    model = BuckConverter()
    # Initial conditions to use
    x_init = np.array([[2.0, 7.0], [8.0, 2.0], [14.0, 8.0], [20.0, 14.0], [26.0, 12.0],
                       [-0.05, 12.5], [-0.05, 14.0], [-0.05, 16], [1.0, 8.0], [4.0, 4.0]])
    states_init = np.array([1, 1, 1, 2, 2, 3, 3, 3, 1, 1])

    data = []
    # Main loop over initial states
    for curr in range(len(states_init)):
        q = f"q{states_init[curr]}"
        model.reinitialize(0, state=(q, (), x_init[curr].tolist())) # state is (q, xt, xk)
        res = model.simulate(finish_time=0.02)
        data.append(res)
    return model, data


def conveyor_system_sfowl(split=False):
    """
    Conveyor system of SFOWL.
    """

    file_path = os.path.dirname(os.path.abspath(__file__))
    log1 = pd.read_csv(os.path.join(file_path, "data", "log1.csv"))
    log2 = pd.read_csv(os.path.join(file_path, "data", "log2.csv"))
    data = [log1, log2]

    # Rename columns
    for l in data:
        rename_dict = {}
        for c in l.columns:
            new_c = c
            new_c = new_c.replace('I_w_', '')
            new_c = new_c.replace('O_w_', '')
            new_c = new_c.replace('_Axis', '')
            new_c = new_c.replace('BLO', 'LH1').replace('BHL', 'LH2').replace('BHR', 'RH2').replace('BRU', 'RH1')
            new_c = new_c.replace('HAR', 'RV').replace('HAL', 'LV')
            new_c = new_c.replace('HR', 'RV').replace('HL', 'LV')
            new_c = new_c.replace('Weg', 'position').replace('Ctrl', 'ctrl')
            rename_dict[c] = new_c
        l.rename(columns=rename_dict, inplace=True)
        l.drop(columns=[c for c in l.columns if 'energie' in c], inplace=True)

    cont_cols = [c for c in data[0].columns if ('position' in c) or ('power' in c) or ('voltage' in c) or
                 ('current' in c)]
    discrete_cols = [c for c in data[0].columns if '_ctrl' in c]
    # num_bits = {c: max([math.ceil(math.log2(d[c].max())) for d in data]) for c in discrete_cols}

    # Adding the Path/Weg variable
    new_data = []
    for d in data:
        d['timestamp_new'] = (datetime.datetime(1, 1, 1)) + d['timestamp'].apply(
            lambda x: datetime.timedelta(seconds=x))
        d['timestamp'] = pd.to_datetime(d['timestamp_new'])
        d.drop(['timestamp_new'], axis=1, inplace=True)
        d.set_index('timestamp', inplace=True)

        d[cont_cols] = d[cont_cols].astype(float)
        for c in ['LH1_ctrl', 'LH2_ctrl', 'RH1_ctrl', 'RH2_ctrl']:
            d[c] = d[c].replace({0: "Stop", 1: "Right", 3: "Left"}).astype(str)
        for c in ['LV_ctrl', 'RV_ctrl']:
            d[c] = d[c].replace({2048: "Stop", 6144: "Move"}).astype(str)

        # d[discrete_cols] = d[discrete_cols].astype(str)
        # control_sig_1 = d['O_w_BRU_Axis_Ctrl_1'].to_numpy()
        # control_sig_3 = d['O_w_BRU_Axis_Ctrl_3'].to_numpy()
        ind = (((d['LH1_ctrl'] == 'Left') & (d['LH1_ctrl'].shift(-1) == 'Right')) |
               ((d['RH1_ctrl'] == 'Right') & (d['RH1_ctrl'].shift(-1) == 'Left')))
        ind = np.nonzero(ind)[0] + 1
        ind = [0] + list(ind) + [d.shape[0]]
        d["Path"] = 0.
        for n in range(len(ind) - 1):
            # cc = c.iloc[ind[n]:ind[n + 1]].copy()
            seq = d.iloc[ind[n]:min(ind[n + 1], d.shape[0]),:]
            time_diff = seq.index[-1] - seq.index[0]
            if time_diff < datetime.timedelta(seconds=7): # Path 3 or 4
                if seq['LH1_ctrl'].iloc[-1] == 'Left':
                    d.iloc[ind[n]:ind[n + 1], d.columns.get_loc('Path')] = 3
                else:
                    d.iloc[ind[n]:ind[n + 1], d.columns.get_loc('Path')] = 4
            else: # Path 1 or 2
                if seq['LH1_ctrl'].iloc[-1] == 'Left':
                    d.iloc[ind[n]:ind[n + 1], d.columns.get_loc('Path')] = 1
                else:
                    d.iloc[ind[n]:ind[n + 1], d.columns.get_loc('Path')] = 2
        # Identify where the Path value changes
        group_ids = (d['Path'] != d['Path'].shift()).cumsum()

        # Group by the change ID
        groups = [group for _, group in d.groupby(group_ids)]
        new_data += groups
    discrete_cols.append("Path")

    # reformat timestamp
    # for i, log in enumerate(data):
    #     log['timestamp_new'] = (datetime.datetime(1, 1, 1)) + log['timestamp'].apply(lambda x: datetime.timedelta(seconds=x))
    #     log['timestamp'] = pd.to_datetime(log['timestamp_new'])
    #     log.drop(['timestamp_new'], axis=1, inplace=True)
    #     log.set_index('timestamp', inplace=True)


        # series_16bit = log[col].apply(lambda x: list(format(x, f'{num_bits[col]:03d}b')))
        # binary_df = pd.DataFrame(series_16bit.tolist(), columns=[f'{col}_bit_{i}' for i in range(num_bits[col])]).astype(int)
        # data[i].drop([col], axis=1, inplace=True)
        # data[i] = pd.concat([data[i], binary_df], axis=1)
    # data = tools.encode_nominal_list_df(data, columns=discrete_cols)

    # discrete_cols = [c for c in data[0].columns if '_Ctrl' in c]

    # remove constant bits
    # constant_cols = [c for c in discrete_cols if 1 == len(set(item for sublist in ([d[c].unique() for d in data]) for item in sublist))]
    # for d in data:
    #     d.drop(columns=constant_cols, axis=1, inplace=True)
    # discrete_cols = [d for d in discrete_cols if d not in constant_cols]

    # discrete_cols = ['LH1_ctrl', 'LH2_ctrl', 'LV_ctrl', 'RV_ctrl', 'RH2_ctrl', 'RH1_ctrl', 'Path']
    discr_data = [d[discrete_cols] for d in new_data]
    cont_data = [d[cont_cols] for d in new_data]

    for d in discr_data:
        d.index -= d.index[0]
        d.index = d.index.total_seconds()
    for d in cont_data:
        d.index -= d.index[0]
        d.index = d.index.total_seconds()

    if split:
        n = len(discr_data)
        n_train = int(n * 0.70)
        n_valid = int(n * 0.15)
        discr_data = (discr_data[:n_train], discr_data[n_train:n_train+n_valid], discr_data[n_train+n_valid:])
        cont_data = (cont_data[:n_train], cont_data[n_train:n_train + n_valid], cont_data[n_train + n_valid:])
    return discr_data, cont_data


class TunnelOven (Automaton):
    def __init__(self):
        super().__init__(states=['Off', 'On'], dt=1e-1,
                         transitions=[('Of', 'On', 'On'),
                                      ('On', 'Off', 'Off')],
                         initial_q='Off')
        self.ThetaSP = 1

        # Define the transfer function of the plant (first-order system)
        K = 1.0  # System gain
        tau = 5.0  # Time constant
        plant = ctrl.TransferFunction([K], [tau, 1])  # G(s) = K / (tau * s + 1)

        # Define the PID controller
        Kp = 1.0  # Proportional gain
        Ki = 0.5  # Integral gain
        Kd = 0.1  # Derivative gain
        controller = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])  # PID controller transfer function

        # Closed-loop system: plant with feedback controller
        self.closed_loop_system = ctrl.feedback(controller * plant)

    def time_discrete_dynamics(self, q, p, x, u):
        # x_1 is il while x_2 is v_c
        if q == 'On':
            theta_sp = self.ThetaSP
        elif q == 'Off':
            theta_sp = 0
        else:
            raise Exception(f'Not a valid discrete state: {q}')

        # Time array for simulation
        time = [0, self.dt]

        # Simulate the step response (response to a step input, i.e., setpoint change)
        time, response = ctrl.step_response(self.closed_loop_system, time)
        return 0, 0


def simple_conveyor():
    """
    We model the discrete-event controller with a three-state automaton.
    The idle conveyor is in state $q_{idle}$ until an item with a mass $M$ and a destination distance $D$ is put on it.
    Then it switches to $q_{move}$ during which the conveyor is moving the item.
    It is in this state until the destination position is reached, and it switches to $q_{settle}$.
    After $T_{settle}$ amount of time it is again in $q_{idle}$.
    :return:
    """


def tunnel_oven(complexity='111'):
    model = TunnelOven()
    res = model.simulate(finish_time=10)
    return res


if __name__ == "__main__":
    # discrete_data, timestamp_col, discrete_vars = conveyor_system_sfowl("discrete")
    # discrete_data[0]


    conv = simple_conveyor_8_states()
    # vis.plot_cps_component(conv, output="dash", min_zoom=3, max_zoom=3, node_labels=True, center_node_labels=True)

    stateflow_data, discr_output_data, cont_state_data, cont_output_data, finish_time = conv.simulate(finish_time=10)
    # vis.plot_timeseries(discr_output_data, modedata=stateflow_data, showlegend=True, discrete=True).show()
    # vis.plot_timeseries(cont_output_data, modedata=stateflow_data, showlegend=True).show()
    vis.plot_timeseries([cont_output_data], showlegend=True).show('browser')
    fig = vis.plot2d(cont_output_data, x=cont_output_data.columns[-2], y=cont_output_data.columns[-1], figure=True)
    fig.update_layout(xaxis=dict(scaleanchor='y', scaleratio=1),
                      yaxis=dict(scaleanchor='x', scaleratio=1))
    fig.show('browser')


    # model = tunnel_oven(complexity='111')

    # ta, data = buck_converter()

    # data = data[0:2]
    # vis.plot_cps_component(ta, node_labels=True, output='dash')

    # vis.plot_timeseries([x[2] for x in data], modedata=[x[0] for x in data], showlegend=True).show()

    # data = conveyor_system_sfowl()
    # exit()
    # # A = timed_control()
    # # A.simulate(finish_time=500)
    #
    # A = simple_conveyor_system()
    # A.plot_cps().show()
    # ddata = A.simulate(finish_time=500)
    #
    # A = Automaton()
    # A.add_state(["s1", "s2", "s3"])
    # A.add_transition([("s1", "s2", "e1"),
    #                   ("s2", "s3", "e1"),
    #                   ("s3", "s1", "e2")])
    #
    # print(A)
    # A.plot_cps().show()

