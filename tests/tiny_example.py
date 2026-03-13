from ml4cps import automata as au
from ml4cps import vis

A = au.Automaton()
A.add_states_from(["s1", "s2", "s3"])
A.add_initial_state("s1")
A.add_final_state(["s2", "s3"])
A.add_transitions_from([("s1", "s2", "e1"),
                        ("s2", "s3", "e1"),
                        ("s3", "s1", "e2")])

print(A)
vis.plot_cps_plotly(A).write_image('tiny_example.png')