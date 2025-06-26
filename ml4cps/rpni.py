""" RPNI algorithm"""
import vis
from ml4cps.automata import Automaton
from ml4cps.learn import build_pta

def rpni(positive_samples, negative_samples):

    # Process positive samples
    dfa = build_pta(positive_samples)

    # Process negative samples
    # Function to check if the DFA rejects all negative samples
    def rejects_negative_samples(mdl):
        for sample in negative_samples:
            if mdl.accepts(sample):
                return False
        return True

    # Attempt to merge states while preserving rejection of negative samples
    states = list(dfa.discrete_states)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            state1 = states[i]
            state2 = states[j]
            if dfa.is_state(state2):
                dfa.try_merge_states(state1, state2, rejects_negative_samples)
                print('Compare')

    return dfa



if __name__ == '__main__':
    # # Example usage
    # positive_samples = ["a", "ab", "abc", "aa"]
    # negative_samples = ["b", "ba", "ac", "abab"]
    #
    # dfa = rpni(positive_samples, negative_samples)
    # vis.plot_cps_component(dfa, output='dash', min_zoom=3, max_zoom=3)
    # # Test the DFA
    # print(dfa.accepts("a"))    # True
    # print(dfa.accepts("ab"))   # True
    # print(dfa.accepts("abc"))  # True
    # print(dfa.accepts("b"))    # False
    # print(dfa.accepts("ba"))   # False
    # print(dfa.accepts("ac"))   # False
    # print(dfa.accepts("abab"))  # False

    # Example usage
    positive_samples = ["a", "ab", "abc"]
    negative_samples = ["aa", "ac", "aba", "abb"]

    dfa = rpni(positive_samples, negative_samples)
    vis.plot_cps_component(dfa, output='dash', min_zoom=3, max_zoom=3)
    # Test the DFA
    print(dfa.accepts("a"))  # True
    print(dfa.accepts("ab"))  # True
    print(dfa.accepts("abc"))  # True
    print(dfa.accepts("b"))  # False
    print(dfa.accepts("ba"))  # False
    print(dfa.accepts("ac"))  # False
    print(dfa.accepts("abab"))  # False

