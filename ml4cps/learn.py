"""
    The module provides learning algorithms for creation of different kinds of automata.

    Authors:
    - Nemanja Hranisavljevic, hranisan@hsu-hh.de, nemanja@ai4cps.com
    - Tom Westermann, tom.westermann@hsu-hh.de, tom@ai4cps.com
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ml4cps import Automaton
from collections import OrderedDict
import pprint

import numpy as np


def FnShiftAndDiff(xout, udout, norm_coeff, num_var, num_ud, max_deriv, Ts):
    # Normalize xout
    xout = xout / norm_coeff

    xout_shifts = pd.DataFrame(np.tile(xout, (1, max_deriv + 1)))

    # Calculate shifted duplicates
    for shift in range(max_deriv + 1):
        # Fill with zeros for the first 'shift' rows, then shift the xout
        xout_shifts[:, shift * num_var:(shift + 1) * num_var] = np.vstack(
            (np.zeros((shift, num_var)), xout[:-shift if shift != 0 else None, :num_var])
        )

    # Calculate derivatives up to the max_deriv order
    for deriv in range(1, max_deriv + 1):
        for curr_var in range(num_var):
            pos_last_deriv = (deriv - 1) * num_var + curr_var
            # Compute the derivative and append it to xout
            derivative = np.vstack((np.zeros((deriv, 1)), np.diff(xout[deriv - 1:, pos_last_deriv], axis=0) / Ts))
            xout = np.hstack((xout, derivative))

    # Strip entries from the front of xout and xout_shifts to match the size after derivation
    xout = xout[max_deriv:]
    xout_shifts = xout_shifts[max_deriv:]

    # Normalize udout using normalization factors (derivatives not needed)
    if num_ud != 0:
        for j in range(num_ud):
            udout[:, j] = udout[:, j] / norm_coeff[num_var + j, 0]
        # Strip entries from the front of udout to match xout
        udout = udout[max_deriv:]

    return xout, udout, xout_shifts


# # Example usage
# xout = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]])
# udout = np.array([[0.5], [0.75], [1.0], [1.25]])
# norm_coeff = np.array([[2.0], [3.0], [1.0]])  # Normalization factors for both xout and udout
#
# num_var = 2  # Number of output variables
# num_ud = 1  # Number of input variables
# max_deriv = 2  # Maximum derivative order
# Ts = 0.1  # Sampling time
#
# xout, udout, xout_shifts = FnShiftAndDiff(xout, udout, norm_coeff, num_var, num_ud, max_deriv, Ts)
#
# print("xout:")
# print(xout)
# print("udout:")
# print(udout)
# print("xout_shifts:")
# print(xout_shifts)


def simple_learn_from_event_logs(data, initial=True, count_repetition=True, verbose=False):
    """
    Simple algorithm to learn a timed automaton.
    """
    # Here the state is determined by the events it emits, but only the first event is taken as transition
    if type(data) is not list:
        data = [data]

    a = Automaton(id='Simple')
    sequence = 0
    if verbose:
        print('***Timed automaton learning from event logs***')

    for d in data:
        sequence += 1
        print('Sequence #{}'.format(sequence))
        if len(d) < 2:
            print('Skipping because num events: 0')
            continue
        print('Duration: {}'.format(d.index[-1] - d.index[0]))

        event_rpt = 0
        state_event = ''

        old_event_rpt = 0
        old_state_event = ''

        t_old = d.index[0]
        if initial:
            a.add_initial_state('initial')
        for t, event in d.items():
            if state_event == event:
                event_rpt += 1
            else:
                state_event = event
                event_rpt = 0

            delta_t = t - t_old
            if old_state_event == '':
                source = 'initial'
            else:
                if count_repetition:
                    source = '{}#{}'.format(old_state_event, old_event_rpt) if old_event_rpt else old_state_event
                else:
                    source = old_state_event

            if count_repetition:
                dest = '{}#{}'.format(state_event, event_rpt) if event_rpt else state_event
            else:
                dest = state_event

            if source != 'initial' or initial:
                a.add_single_transition(source, dest, event, delta_t)
            t_old = t
            old_state_event = state_event
            old_event_rpt = event_rpt
            if verbose:
                print(source, dest, event, delta_t)
    return a


def simple_learn_from_signal_vectors(data, sig_names, drop_no_changes=False, verbose=False):
    a = Automaton()
    sequence = 0
    if verbose:
        print('***Timed automaton learning from variable changes***')

    for d in data:
        if drop_no_changes:
            d = d.loc[(d.iloc[:, 1:].diff() != 0).any(axis=1), :]
        time_col = d.columns[0]
        sequence += 1
        if verbose:
            print('Sequence #{}'.format(sequence))
            if len(d) < 2:
                print('Skipping because num events: 0')
                continue
            print('Duration: {}'.format(d[time_col].iloc[-1] - d[time_col].iloc[0]))

        previous_state = d[sig_names].iloc[:-1]
        dest_state = d[sig_names].iloc[1:]
        event = d[sig_names].diff().apply(lambda x: ' '.join(x.astype(str)).replace(".0", ""), 1).iloc[1:]
        deltat = d[time_col].diff().iloc[1:]

        for source, dest, ev, dt in zip(previous_state.itertuples(index=False, name=None),
                                        dest_state.itertuples(index=False, name=None), event, deltat):
            source = pprint.pformat(source, compact=True).replace(".0", "")
            dest = pprint.pformat(dest, compact=True).replace(".0", "")
            a.add_single_transition(source, dest, ev, dt)
    return a


def simple_learn_from_signal_updates(data, sig_names, initial=True, verbose=False):
    a = Automaton()
    sequence = 0
    if verbose:
        print('***Timed automaton learning from variable changes***')

    for d in data:
        time_col = d.columns[0]
        sequence += 1
        print('Sequence #{}'.format(sequence))
        if len(d) < 2:
            print('Skipping because num events: 0')
            continue
        print('Duration: {}'.format(d[time_col].iloc[-1] - d[time_col].iloc[0]))

        t_old = d[time_col].iloc[0]
        if initial:
            a.add_initial_state('initial')

        state = dict.fromkeys(sig_names)
        for t, signal, value in d.itertuples(index=False, name=None):
            event = f'{signal}<-{value}'
            all_values_are_set = all(value is not None for value in state.values())

            delta_t = t - t_old
            t_old = t
            source = pprint.pformat(state)
            state[signal] = value
            dest = pprint.pformat(state)

            if all_values_are_set:
                a.add_single_transition(source, dest, event, delta_t)
    return a


def build_pta(data, event_col='event', boundaries=1):
    """
    In the function build_pta the prefix tree acceptor is created by going through each sequence of events in the
    learning examples. Also, the depth, in- and out-degree of the states are set.
    """
    pta = Automaton()
    pta.add_initial_state('q0')
    for seq in data:
        if len(seq) == 0:
            continue
        if not isinstance(seq, pd.DataFrame):
            if isinstance(seq, str):
                seq = pd.Series(list(seq))
            if isinstance(seq, pd.Series):
                seq.name = event_col
            seq = pd.DataFrame(seq).reset_index(drop=False)
        old_t = seq[seq.columns[0]].iloc[0]
        curr_stat = "q0"
        time_col = seq.columns[0]
        seq = seq[[time_col, event_col]] #.iloc[1:]
        for t, event in seq.itertuples(index=False, name=None):
            dt = t - old_t
            # if event in boundaries and curr_stat != "q0":
            #     sub_event = 1 + next(ii for ii, tt in enumerate(boundaries[event]) if dt >= tt)
            #     event = event + "'" * sub_event
            dest = pta.get_transition(curr_stat, e=event)
            if dest is None:
                dest = f"q{pta.num_modes}"
            else:
                dest = dest[1]
            pta.add_single_transition(curr_stat, dest, event, timing=dt)
            curr_stat = dest
            old_t = t
        pta.add_final_state(curr_stat)
    return pta


def FnDetectChangePoints(xout, udout, xout_shifts):
    global num_var, num_ud, max_deriv, chp_depths

    # Initialize global variables
    chpoints = []  # Global changepoints
    chp_depths = np.zeros(max_deriv + 1)  # Purely for debugging
    chp_var = [None] * (num_var + num_ud)  # Local changepoints (per variable)

    # Detect change points for output variables
    for i in range(num_var):
        new_chp = findChangePoints(xout[:, i::num_var], 0, 1, xout.shape[0], max_deriv)
        chpoints = np.union1d(chpoints, new_chp)
        chp_var[i] = np.sort(new_chp)

    # Detect change points for input variables
    for i in range(num_ud):
        new_chp = findChangePoints(udout[:, i], 0, 1, udout.shape[0], 0)
        chpoints = np.union1d(chpoints, new_chp)
        chp_var[num_var + i] = new_chp

    # Filter changepoints
    xout, udout, xout_shifts, chpoints, chp_var = filterChangePoints(xout, udout, xout_shifts, chpoints, chp_var)

    # Create the trace structure
    trace = {
        'x': xout,
        'xs': xout_shifts,
        'chpoints': chpoints,
        'chpoints_per_var': chp_var,
        'ud': udout,
        'labels_num': [],
        'labels_trace': []
    }
    return trace


def computeDistance(der):
    global windowSize
    dist = np.zeros(windowSize)
    for i in range(windowSize, len(der) - windowSize):
        before = der[(i - windowSize):i]
        after = der[(i + 1):(i + windowSize + 1)]
        dist_new = np.sum(np.abs((before - before[0]) - (after - after[0])))
        dist = np.append(dist, dist_new)
    return dist


def findChangePoints(xout, depth, starting, ending, max_depth):
    global windowSize, chp_depths

    locs = []
    if depth > max_depth or ending - starting - 1 < 2 * windowSize:
        return locs

    der = xout[starting:ending, depth]
    dist = computeDistance(der)

    # Find peaks in distance to detect change points
    _, locsDist = find_peaks(dist, height=5)
    locsHere = np.sort(locsDist + starting - 1)
    locsHere = filterindx(locsHere, 1.5 * windowSize)
    chp_depths[depth] += len(locsHere)
    locs.extend(locsHere)

    locsHere = np.concatenate([[starting - windowSize // 2], locsHere, [ending + windowSize // 2]])
    for i in range(len(locsHere) - 1):
        newStart = int(locsHere[i] + windowSize / 2)
        newEnd = int(locsHere[i + 1] - windowSize / 2)
        locsNew = findChangePoints(xout, depth + 1, newStart, newEnd, max_depth)
        locs.extend(locsNew)

    if depth == 0:
        locs = np.concatenate([[1], locs, [len(der)]])

    return np.array(locs)


def filterChangePoints(xout, udout, xout_shifts, chpoints, chp_var):
    global windowSize, num_var, num_ud

    # Filter global changepoints detected on multiple output variables
    chpoints = filterindx(chpoints, windowSize)

    if chpoints[-1] - chpoints[-2] < 2 * windowSize:
        xout = xout[:chpoints[-2], :]
        xout_shifts = xout_shifts[:chpoints[-2], :]
        if num_ud != 0:
            udout = udout[:chpoints[-2], :]
        chpoints = chpoints[:-1]
        for i in range(num_var + num_ud):
            current_chps = np.array(chp_var[i])
            current_chps = current_chps[:-1]
            current_chps = np.append(current_chps, chpoints[-1])
            chp_var[i] = current_chps

    # Ensure consistency between global and local changepoint sets
    for i in range(num_var + num_ud):
        current_chps = np.array(chp_var[i])
        for j in range(len(current_chps)):
            idx = np.argmin(np.abs(chpoints - current_chps[j]))
            current_chps[j] = chpoints[idx]
        if current_chps[-2] == current_chps[-1]:
            current_chps = current_chps[:-1]
        chp_var[i] = current_chps

    return xout, udout, xout_shifts, chpoints, chp_var


def filterindx(indx, windw):
    n = 0
    while n < len(indx) - 1:
        id1 = indx[n]
        while n + 1 < len(indx) and indx[n + 1] - id1 <= windw:
            indx = np.delete(indx, n + 1)
        n += 1
    return indx


# Global variables (must be initialized elsewhere in your code)
num_var = None
num_ud = None
useTime = None


def FnTraceToTrainingData(trace, num_var, num_ud, useTime):

    # Preallocate arrays for samples
    states = np.zeros((trace['x'].shape[0] - 1, 1))
    values = np.zeros((trace['x'].shape[0] - 1, num_var + num_ud))
    timeSwitch = np.zeros((trace['x'].shape[0] - 1, 1))

    # Variables to keep track of states
    lastswitch = 1
    indxStates = 0

    # Loop through trace data to generate feature vectors and class labels
    for indx in range(trace['x'].shape[0] - 1):
        # Update index associated with system mode switch
        if indxStates + 1 < len(trace['chpoints']) and indx >= trace['chpoints'][indxStates + 1]:
            indxStates += 1
            lastswitch = indx

        # Save current states and values for the feature vector
        states[indx] = trace['labels_trace'][indxStates]
        values[indx, :num_var] = trace['x'][indx, :num_var]
        if num_ud != 0:
            values[indx, num_var:num_var + num_ud] = trace['ud'][indx, :num_ud]

        timeSwitch[indx] = indx - lastswitch

    # Create matrices containing feature vectors and corresponding class labels
    points = np.arange(states.shape[0] - 1)
    if useTime:
        X = np.hstack([states[points], values[points, :], timeSwitch[points]])
    else:
        X = np.hstack([states[points], values[points, :]])

    Y = states[points + 1]

    return X, Y, states




if __name__ == "__main__":
    from ml4cps.examples import examples
    import tools

    discrete_data, time_col, discrete_cols = examples.conveyor_system_sfowl("discrete")
    data, _, _, cont_vars = examples.conveyor_system_sfowl("all")

    discrete_data_changes = tools.remove_timestamps_without_change(discrete_data, sig_names=discrete_cols)
    discrete_data_events = tools.create_events_from_signal_vectors(discrete_data_changes, sig_names=discrete_cols)
    discrete_data_events = tools.split_data_on_signal_value(discrete_data_events, "O_w_BRU_Axis_Ctrl", 3)


    ######################## Test simple learn from signals vectors  ###################################################
    ta = simple_learn_from_signal_vectors(discrete_data_events, sig_names=discrete_cols)

    data = ta.predict_state(data, time_col_name="timestamp", discr_col_names=discrete_cols)
    exit()

    state_sequences = tools.group_data_on_discrete_state(data, state_column="StateEstimate", reset_time=True,
                                                         time_col="timestamp")
    dd = list(state_sequences.values())[4]
    tools.plot_timeseries(dd, timestamp="timestamp", iterate_colors=False).show()
    exit()

    print("Number of sequences: ", len(discrete_data_events))
    discrete_data_events[0]

    ta = build_pta(discrete_data_events)
    print(ta)
    ta.plot_cps().show("browser")
    ta.plot_cps().show()


    # ta = simple_learn_from_signal_vectors(discrete_data, sig_names=discrete_cols)
    # ta.view_plotly(show_num_occur=True)

    ################### Test PTA #######################################################################################
    print("Number of sequences: ", len(discrete_data_events))
    discrete_data_events[0]

    ta = build_pta(discrete_data_events)
    print(ta)
    ta.plot_cps().show("browser")
    ta.plot_cps().show()

    # ta = simple_learn_from_signal_vectors(discrete_data, sig_names=discrete_cols)
    # ta.view_plotly(show_num_occur=True)


    exit()

    print("Test build_pta")

    test_data1 = [[[1, 0, 0, 0, 1.3, 9.6, 14.5],
                   [2, 0, 0, 0, 1.5, 9.5, 14.4],
                   [3, 0, 0, 1, 1.8, 9.3, 14.1],
                   [4, 0, 0, 1, 2.1, 8.9, 13.6],
                   [5, 0, 0, 1, 2.2, 8.5, 13.3],
                   [6, 0, 1, 1, 2.3, 8.4, 13.2],
                   [7, 0, 1, 1, 2.4, 8.2, 13.1],
                   [8, 0, 1, 1, 2.6, 5.1, 12.9],
                   [9, 0, 0, 1, 2.9, 7.9, 12.7],
                   [10, 0, 0, 1, 3.1, 7.8, 12.6]],
                  [[1, 0, 0, 0, 1.6, 9.9, 14.9],
                   [5, 1, 0, 0, 1.3, 9.2, 14.1],
                   [6, 1, 0, 0, 1.9, 9.6, 14.7],
                   [7, 0, 1, 1, 2.5, 8.7, 13.3],
                   [8, 0, 1, 1, 2.6, 8.2, 13.5],
                   [64, 0, 0, 1, 2.7, 8.6, 13.6],
                   [88, 0, 0, 1, 2.9, 8.1, 13.7],
                   [90, 1, 0, 1, 2.6, 5.4, 12.5],
                   [140, 1, 0, 1, 2.7, 7.2, 12.6],
                   [167, 1, 1, 1, 3.7, 7.2, 12.1]],
                  [[1, 0, 0, 0, 1.3, 9.6, 14.5],
                   [2, 0, 0, 0, 1.5, 9.5, 14.4],
                   [4, 0, 0, 1, 1.8, 9.3, 14.1],
                   [6, 0, 0, 1, 2.1, 8.9, 13.6],
                   [8, 0, 0, 1, 2.2, 8.5, 13.3],
                   [11, 0, 1, 1, 2.3, 8.4, 13.2],
                   [13, 0, 1, 1, 2.4, 8.2, 13.1],
                   [14, 0, 1, 1, 2.6, 5.1, 12.9],
                   [15, 0, 0, 1, 2.9, 7.9, 12.7],
                   [17, 0, 0, 1, 3.1, 7.8, 12.6]]]

    test_data1 = [pd.DataFrame(d) for d in test_data1]
    test_data1 = tools.remove_timestamps_without_change(test_data1, sig_names=[1, 2, 3])
    test_data1 = tools.create_events_from_signal_vectors(test_data1, sig_names=[1, 2, 3])

    # test_data1 = createEventsfromDataFrame(test_data1)
    pta = build_pta(test_data1)
    pta.plot_cps().show()