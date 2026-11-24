[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ff69b4)](https://github.com/sponsors/ai4cps-com)

# ML4CPS
ML4CPS is a Python package for learning and analysis 
of hybrid dynamical systems, with the focus on 
Cyber-Physical Systems (CPS).
The code was developed for several research publications ([bibtex](docs/cite.bib)).

-   Website ([ml4cps.ai4cps.com](http://ml4cps.ai4cps.com))
-   Contact ([contact@ai4cps.com](mailto:contact@ai4cps.com))


## Installation

Install the core package without PyTorch or MLflow:

```sh
pip install ml4cps
```

For DEBTA models and tensor-based tools, install the optional PyTorch dependency:

```sh
pip install "ml4cps[torch]"
```

For experiment tracking and MLflow model integration, install:

```sh
pip install "ml4cps[mlflow]"
```

Combine both extras with `pip install "ml4cps[torch,mlflow]"`.
Core CPS simulation, visualization, and DEBTA with `log_mlflow=False` work
without MLflow. Tracking calls require the `mlflow` extra.
The `dev` extra includes both PyTorch and MLflow for development and tests.

## Reconfiguration utilities

Enumerate candidate binary configurations with per-variable restrictions:

```python
from ml4cps.tools import get_allowed_configurations

get_allowed_configurations(3, [(0,), (0, 1), (1,)])
# ((0, 0, 1), (0, 1, 1))
```

The result is a sorted tuple without duplicates. This helper enforces the
supplied per-variable restrictions; system constraints involving combinations
of controls can be applied separately.

## DEBTA observation encoding

`codes, reconstructed = model.encode_observations(observations, mean, std)`
standardizes a batch of sensor observations using training statistics, calls
DEBTA's existing `recon` operation for rounded latent codes and reconstruction,
and restores the reconstruction to the original sensor units. Codes are packed
as integers with the first latent bit as the least significant bit. Inputs are
single observations shaped `(samples, features)`; no temporal windows are made.

## DEBTA mode alignment

`DEBTA.assign_mode_indices(codes, observations, state_codes, prototypes, mean, std,
sensor_distance_weight=1e-6)` maps encoded modes onto a fixed training vocabulary
and returns both mode indices and original unseen-code flags. Unseen codes use
Hamming distance plus the weighted, standardized squared sensor distance.
Call it on `DEBTA` or an instance; pass `sensor_distance_weight=0` to use only
Hamming distance. Exact matches retain their known indices.

## DEBTA conditional transitions and path sampling

The `ml4cps.debta` package also exposes `FunctionalAutomatonSpace`,
`ConditionalTransitionRBM`, `fit_transition_rbm`, `transition_counts`,
`transition_log_probabilities`, `transition_energies`, `transition_vector`,
`gibbs_paths`, `PathSamples`, `backward_messages`, and `path_energy`.
The model is `debta/rbms/three_way.py::ConditionalTransitionRBM`, a
`ThreeWayBinaryBinaryRBM` subclass. The shared three-way class stores a full
interaction tensor by default, or low-rank factors with `n_factors=...`.
The transition model reuses those factors for configuration/source/hidden and
configuration/destination/hidden interactions, with no unused dense tensor. Exact conditional fitting is in `rbms/training.py`;
transition data helpers are in `debta_tools.py`. Automaton and path operations
live in `automata/base.py` and `debta/sampling.py`. Import the graph
representation directly with `from ml4cps.automata import FunctionalAutomatonSpace`;
the previous DEBTA import paths remain available for checkpoint compatibility.

```python
from ml4cps.debta import fit_transition_rbm, gibbs_paths

model = fit_transition_rbm(configurations, counts, state_codes=state_codes)
log_transition = model.log_probabilities(allowed_configurations)
draws = gibbs_paths(log_transition, state_cost, initial=0, horizon=10, seed=42)
```

The RBM learns binary destination codes conditioned on binary source codes and
configuration, using factored three-way gates. Hidden units are marginalized
exactly, and likelihood normalizes over the valid destination codebook.
`gibbs_paths` alternates complete-path and configuration blocks: one constant
configuration applies to each path and its initial state remains clamped.
Backward messages and path energy calculations are shared with diagnostics.

Configuration coordinates default to binary. Set
`configuration_types=("binary", "gaussian", ...)` and fixed positive
`configuration_sigma` values when constructing or fitting `ConditionalTransitionRBM`
for mixed configurations. Binary scales are 1; Gaussian inputs use the same
`u / sigma` interactions and quadratic energy convention as `GaussianBinaryRBM`.
The shared three-way class exposes these settings as `auxiliary_types` and
`auxiliary_sigma`. PT/NPZ checkpoints retain the schema and scales.

For continuous reconfiguration, pass the transition model to `gibbs_paths`
instead of a precomputed table, together with finite `configuration_bounds`
(one lower/upper pair per coordinate). It alternates exact path draws with
bounded Metropolis configuration updates; one configuration is constant over
an entire path. Real-valued draws are returned in `configuration_values`.
The inference prior is uniform inside the bounds unless a
`configuration_log_prior` callable is supplied. Conditional transition fitting
does not identify this prior; Gaussian context units do not make the normalized
path/configuration posterior Gaussian.

These modules have no dependency on experiment repositories.

Sensor encoding continues to use DEBTA's `recon`/`predict`, configuration
enumeration uses `ml4cps.tools.get_allowed_configurations`, and DEBTA retains
its existing `save`/`load` methods. The conditional model uses exact likelihood;
the optimizer uses PyTorch autograd with L-BFGS while destination sampling is
categorical over the valid codebook. Hidden sampling, device/parameter handling,
and free energy reuse the existing RBM implementation. `parameters()` follows
the standard PyTorch API; `transition_parameters` gives legacy checkpoint names,
and `parameter_vector()` provides an optimization vector. Native PT and legacy
NPZ checkpoints are supported. Old NumPy-object pickles are migrated on load.
Training figures open only with `train_rbm(..., show_plots=True)`.

## DEBTA subclass checkpoints

DEBTA's inherited `save` and `load` also preserve subclass attributes declared
in a `checkpoint_attributes` tuple. For example, a subclass can declare
`checkpoint_attributes = ("transition_model",)` and give the attribute a class
default of `None`. No `save` or `load` overrides are required.

`DEBTA.load(filename)` restores the saved class. `Subclass.load(filename)` also
accepts a checkpoint from an ancestor DEBTA class and returns the requested
subclass. Extra attributes missing from older checkpoints use class defaults
(or `None`); unrelated sibling classes are rejected. The loader reads the file
once and accepts the former reconfiguration checkpoint format.

## Data
In the folder "data" there are several datasets that can be easily loaded using the [`ml4cps.examples`](ml4cps/examples/examples.py) module.


## Bugs
If you find any bugs, please contact us at [bugs@ai4cps.com](mailto:bugs@ai4cps.com).


## License
See [LICENSE](LICENSE).  
If you use this code in your research, please [cite](docs/cite.bib) our work. 
