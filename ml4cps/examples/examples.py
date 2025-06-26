import datetime
from ml4cps import Automaton, tools, vis
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

    A = Automaton(id="Simple Conveyor", dt=0.01, states=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
                  transitions=[dict(source="s1", dest="s2", event="Place Item", prob=1, time=time_fun),
                               dict(source="s2", dest="s1", event="Take Item", prob=1, time=time_fun),
                               dict(source="s7", dest="s8", event="Place Item", prob=1, time=time_fun),
                               dict(source="s8", dest="s7", event="Take Item", prob=1, time=time_fun),
                               dict(source="s1", dest="s4", event="Move Downward", prob=1, time=time_fun),
                               dict(source="s4", dest="s7", event="Stop Movement", prob=1, time=time_fun),
                               dict(source="s7", dest="s3", event="Move Upward", prob=1, time=time_fun),
                               dict(source="s3", dest="s1", event="Stop Movement", prob=1, time=time_fun),
                               dict(source="s2", dest="s6", event="Move Downward", prob=1, time=time_fun),
                               dict(source="s6", dest="s8", event="Stop Movement", prob=1, time=time_fun),
                               dict(source="s8", dest="s5", event="Move Upward", prob=1, time=time_fun),
                               dict(source="s5", dest="s2", event="Stop Movement", prob=1, time=time_fun)])

    def discrete_output_fun(q, xt, xk):
        state_dict = {
            "s1": [0, 0],
            "s2": [0, 0],
            "s3": [0, 1],
            "s4": [1, 0],
            "s5": [0, 1],
            "s6": [1, 0],
            "s7": [0, 0],
            "s8": [0, 0]
        }
        return np.array(state_dict[q])

    def continuous_output_fun(q, xt, xk):
        std = 0.04
        cov = np.array([[1, 0], [0, 1]])
        mean_dict = {
            "s1": [0, 0],
            "s2": [0, 0.5],
            "s3": [0.5, 0.2],
            "s4": [0.5, 0.2],
            "s5": [0.2, 0.9],
            "s6": [0.2, 0.9],
            "s7": [0, 0],
            "s8": [0, 0.5]
        }
        mean = mean_dict[q]
        f = np.random.multivariate_normal(mean, std**2 * cov)
        return f

    def time_discrete_dynamics_fun(q, p, x, u):
        return ()

    A.output_d = discrete_output_fun
    A.output_y = continuous_output_fun
    A.time_discrete_dynamics = time_discrete_dynamics_fun

    A.reinitialize(0, state=("s1", (), ()))  # state is (q, xt, xk)
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


def conveyor_system_sfowl(variable_type="all"):
    """
    Conveyor system of SFOWL.
    """
    columns_16bit = ['O_w_HAL_Ctrl', 'O_w_HAR_Ctrl']

    file_path = os.path.dirname(os.path.abspath(__file__))
    log1 = pd.read_csv(os.path.join(file_path, "data", "log1.csv"))
    log2 = pd.read_csv(os.path.join(file_path, "data", "log2.csv"))
    data = [log1, log2]

    cont_cols = [c for c in data[0].columns if c.lower()[-5:] != '_ctrl' and c != "timestamp" and 'energy' not in c and
                 "energie" not in c]
    discrete_cols = [c for c in data[0].columns if '_Ctrl' in c]
    # num_bits = {c: max([math.ceil(math.log2(d[c].max())) for d in data]) for c in discrete_cols}

    for d in data:
        d.drop(columns=[c for c in d.columns if "energy" in c or "energie" in c], axis=1, inplace=True)

    # reformat timestamp
    for i, log in enumerate(data):
        log['timestamp_new'] = (datetime.datetime(1, 1, 1)) + log['timestamp'].apply(lambda x: datetime.timedelta(seconds=x))
        log['timestamp'] = pd.to_datetime(log['timestamp_new'])
        log.drop(['timestamp_new'], axis=1, inplace=True)


        # series_16bit = log[col].apply(lambda x: list(format(x, f'{num_bits[col]:03d}b')))
        # binary_df = pd.DataFrame(series_16bit.tolist(), columns=[f'{col}_bit_{i}' for i in range(num_bits[col])]).astype(int)
        # data[i].drop([col], axis=1, inplace=True)
        # data[i] = pd.concat([data[i], binary_df], axis=1)
    data = tools.encode_nominal_list_df(data, columns=discrete_cols)

    discrete_cols = [c for c in data[0].columns if '_Ctrl' in c]

    # remove constant bits
    # constant_cols = [c for c in discrete_cols if 1 == len(set(item for sublist in ([d[c].unique() for d in data]) for item in sublist))]
    # for d in data:
    #     d.drop(columns=constant_cols, axis=1, inplace=True)
    # discrete_cols = [d for d in discrete_cols if d not in constant_cols]


    # Adding the Path/Weg variable
    for d in data:
        d[cont_cols] = d[cont_cols].astype(float)
        d[discrete_cols] = d[discrete_cols].astype(float)
        control_sig_1 = d['O_w_BRU_Axis_Ctrl_1'].to_numpy()
        control_sig_3 = d['O_w_BRU_Axis_Ctrl_3'].to_numpy()
        ind = np.logical_and(control_sig_1[0:-1] == 1, control_sig_3[1:] == 1)
        ind = np.nonzero(ind)[0] + 1
        ind = [0] + list(ind) + [d.shape[0]]
        d["Weg"] = 0.
        for n in range(len(ind) - 1):
            # cc = c.iloc[ind[n]:ind[n + 1]].copy()
            time_diff = d['timestamp'].iloc[min(ind[n+1], d.shape[0]-1)] - d['timestamp'].iloc[ind[n]]
            if time_diff < datetime.timedelta(seconds=13.5):
                d.iloc[ind[n]:ind[n + 1], d.columns.get_loc("Weg")] = 1
    discrete_cols.append("Weg")

    if variable_type == "discrete":
        discrete_data = [d[['timestamp'] + discrete_cols] for d in data]
        return discrete_data, "timestamp", discrete_cols
    elif variable_type == "continuous":
        cont_data = [d[['timestamp'] + cont_cols] for d in data]
        return cont_data, "timestamp", cont_cols
    else:
        return data, "timestamp", discrete_cols, cont_cols


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
    discrete_data, timestamp_col, discrete_vars = conveyor_system_sfowl("discrete")
    discrete_data[0]


    conv = simple_conveyor_8_states()
    # vis.plot_cps_component(conv, output="dash", min_zoom=3, max_zoom=3, node_labels=True, center_node_labels=True)

    stateflow_data, discr_output_data, cont_state_data, cont_output_data, finish_time = conv.simulate(finish_time=100)
    # vis.plot_timeseries(discr_output_data, modedata=stateflow_data, showlegend=True, discrete=True).show()
    vis.plot_timeseries(cont_output_data, modedata=stateflow_data, showlegend=True).show()
    fig = vis.plot2d(cont_output_data, x=cont_output_data.columns[-2], y=cont_output_data.columns[-1], figure=True)
    fig.update_layout(xaxis=dict(scaleanchor='y', scaleratio=1),
                      yaxis=dict(scaleanchor='x', scaleratio=1))
    fig.show()


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

