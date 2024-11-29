# Towards Principal Fairness in Decision Making Systems

Source code for the project Towards Principal Fairness in Decision Making Systems.

## Installation
### Requirements

- Python >= 3.6
- `conda env create -f environment.yml`

## Outcome Estimation

To replicate the experiments in Section 4.1, enter the source directory

`cd src`

And run the experiments for all simulations

`./outcome_estimation.sh`

## Decision Making

To replicate the experiments in Section 4.2, enter the source directory

`cd src`

And run the experiments for the required dataset

`python hyperparameter_tuning_outcomes.py --name "warfarin"`

`python decision_evaluation_warfarin.py --name "warfarin"`

`python decision_evaluation_warfarin_dcfr.py --name "warfarin"`

`python decision_evaluation_psa.py --name FTA`

`python decision_evaluation_psa.py --name NCA`
