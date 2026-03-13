# ML4CPS 

ML4CPS is a Python package for learning and analysis 
of hybrid dynamical systems, with the focus on 
Cyber-Physical Systems (CPS).
The code was developed for several research publications ([bibtex](cite.bib)).

-   Website ([ml4cps.ai4cps.com](http://ml4cps.ai4cps.com))
-   Contact ([contact@ai4cps.com](mailto:contact@ai4cps.com))


## Jupyter notebook examples

- [Conveyor system SFOWL discrete data analysis](notebooks/Conveyors_SFOWL_discrete.ipynb)
- [Conveyor system SFOWL continuous data analysis](notebooks/Conveyors_SFOWL_cont.ipynb)

## Data

In folder "data" there are several datasets which can be easily loaded using examples module.
E.g.

```python

from ml4cps import examples

discrete_data, timestamp_col, discrete_vars = examples.conveyor_system_sfowl("discrete")
```
will load a dataset of a conveyor system from the SFOWL.

## Bugs
If you find any bugs, please contact us at [bugs@ai4cps.com](mailto:bugs@ai4cps.com).


## License

See [LICENSE](LICENSE).  
If you use this code in your research, please [cite](docs/cite.bib) our work. 
