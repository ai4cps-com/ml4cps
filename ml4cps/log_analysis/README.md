# ML4CPS - log_analysis module

ML4CPS is a Python package for learning and analysis of the behavior of hybrid dynamical systems, with the focus on 
Cyber-Physical Systems (CPS).

Within the project SILK there was a need to analyze logs for security reasons. 

## Install

To install ML4CPS:

```
pip install git+https://github.com/ai4cps-com/ml4cps.git
```

to specify the version:

```
pip install git+https://github.com/ai4cps-com/ml4cps.git@0.1.12
```

## Data

In folder "data" there are several datasets which can be easily loaded using examples module.
E.g.

```python

from ml4cps import examples

discrete_data, timestamp_col, discrete_vars = examples.conveyor_system_sfowl("discrete")
```

## License

See [LICENSE](LICENSE).  
If you use this code in your research please [cite](cite.bib) our work. 
