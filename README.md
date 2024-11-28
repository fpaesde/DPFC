# Towards Principal Fairness in Decision Making Systems

Source code for the project Towards Principal Fairness in Decision Making Systems.

## Installation
### Requirements

- Python >= 3.6
- `pip install -r requirements.txt`

## Outcome Estimation

To replicate the experiments in Section 4.1, enter the source directory

`cd src`

And run the experiments for all simulations

`./outcome_estimation.sh`

## Decision Making

To replicate the experiments in Section 4.2, enter the source directory

`cd src`

And run the experiments for the required dataset

`python decision_evaluation_warfarin.py`

`python decision_evaluation_warfarin_dcfr.py`

`python decision_evaluation_psa.py --name FTA`

`python decision_evaluation_psa.py --name NCA`