# Getting Started

## Basic usage
```python
import ml4cps as at

A = at.Automaton()
A.add_states_from(["s1", "s2", "s3"])
A.add_transitions_from([("s1", "s2", "e1"),
                        ("s2", "s3", "e1"),
                        ("s3", "s1", "e2")])

print(A)
A.view_plotly().show()
```

