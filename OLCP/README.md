# Online Localized Conformal Prediction

This repository contains code for reproducing the experiments in
"Online Localized Conformal Prediction."

## Files

- `cp_methods.py`: implementations of CP, LCP, ACI, DtACI, SPCI, OLCP, and OLCP-Hedge.
- `notebooks/simu.ipynb`: synthetic experiments.
- `notebooks/elec2.ipynb`: ELEC2 experiment.
- `notebooks/ILI..ipynb`: ILINet experiment.
- `notebooks/vix.ipynb`: ETF volatility experiment.
- `data/`: input data files.

## Installation

```bash
conda create -n olcp python=3.11
conda activate olcp
pip install -r requirements.txt

