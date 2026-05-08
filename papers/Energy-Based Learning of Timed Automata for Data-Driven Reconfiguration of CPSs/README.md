# Energy-Based Learning of Timed Automata for Data-Driven Reconfiguration

This folder contains the experiment scripts and data for the paper:

`Energy-Based Learning of Timed Automata for Data-Driven Reconfiguration of CPSs`

Submission status: submitted to IEEE Access in May 2026.

## Methods Compared

1. **Random reconfiguration baseline** (`baseline1.py`): This baseline randomly selects a new configuration from all
   available options (excluding the current configuration). It serves as a naive baseline to demonstrate the minimum
   expected performance without any learning or pattern recognition.

2. **Similarity-based historical baseline** (`baseline2.py`): This method uses a nearest-neighbor approach to find the
   most similar historical continuous data window from the training set and returns the corresponding configuration
   transition. It leverages historical patterns by assuming that similar system states should lead to similar
   configuration choices.

3. **Proposed energy-based method using DEBTA** (`debta_experiments.py`): This approach learns a discrete energy-based
   timed automaton (DEBTA) from training data, discovers a latent automaton structure, and uses the learned energy
   landscape to predict optimal reconfigurations. The method combines energy-based modeling with automaton learning to
   capture both continuous dynamics and discrete configuration transitions.

## Folder Contents

- `create_reconfiguration_dataset_conveyor_system.py`: generates the reconfiguration benchmark JSON
- `conveyor_system_reconfig_data.json`: generated benchmark cases (continuous windows, current config, target configs)
- `baseline1.py`: random baseline evaluation
- `baseline2.py`: nearest-neighbor/similarity baseline evaluation
- `debta_experiments.py`: DEBTA training + latent automaton learning + reconfiguration evaluation

## Prerequisites

From the repository root:

```bash
pip install -e .
```

## Reproducing the Experiments

Run these commands from this folder:

```bash
python create_reconfiguration_dataset_conveyor_system.py
python baseline1.py
python baseline2.py
python debta_experiments.py
```

Or from repo root:

```bash
python "papers/Energy-Based Learning of Timed Automata for Data-Driven Reconfiguration of CPSs/create_reconfiguration_dataset_conveyor_system.py"
python "papers/Energy-Based Learning of Timed Automata for Data-Driven Reconfiguration of CPSs/baseline1.py"
python "papers/Energy-Based Learning of Timed Automata for Data-Driven Reconfiguration of CPSs/baseline2.py"
python "papers/Energy-Based Learning of Timed Automata for Data-Driven Reconfiguration of CPSs/debta_experiments.py"
```

## Outputs

- Console metrics:
  - random baseline: `Number correct`
  - similarity baseline: `Number correct`
  - proposed method: `Number correct`
- MLflow artifacts and metrics are logged under `mlruns/`.
- The DEBTA script also opens interactive visualizations (Dash/Plotly).

## Notes

- Random seeds are set in scripts for repeatability.
- The benchmark generation script samples up to 1000 test windows (default window size: 20).
