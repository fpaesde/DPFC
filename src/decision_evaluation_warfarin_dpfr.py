import json
from fairgbm import FairGBMClassifier
import numpy as np
import pandas as pd
import pyreadr

from lightgbm import LGBMClassifier
import optuna

from data.dataset_builder import DatasetBuilder
from outcome_estimation.t_learner import TLearner
from models.dcfr import DCFR
from utils.model_utils import Runner

fta_dataset_params = {
    "dataset": pyreadr.read_r("data/PSA/FTAdata.rda")[
        list(
            pyreadr.read_r("data/PSA/FTAdata.rda").keys()
            )[0]
        ],
    "columns": [
            'Sex',
            'White',
            'Age',
            'PendingChargeAtTimeOfOffense',
            'NCorNonViolentMisdemeanorCharge',
            'ViolentMisdemeanorCharge',
            'ViolentFelonyCharge', 
            'NonViolentFelonyCharge', 
            'PriorMisdemeanorConviction', 
            'PriorFelonyConviction', 
            'PriorViolentConviction', 
            'PriorSentenceToIncarceration', 
            'PriorFTAInPastTwoYears', 
            'PriorFTAOlderThanTwoYears'
        ],
    "decision": "D",
    "protected_attribute_name": "Sex",
    "priviledged_class": [1],
    "outcome": "Y"
}

warfarin_optimal_params = {
    "learning_rate": 0.2521521207945049,
    "n_estimators": 265,
    "max_depth": 6,
    "num_leaves": 68,
    "min_child_samples": 63,
    "subsample": 0.9090674506628127,
    "colsample_bytree": 0.7138436983158434,
    "reg_alpha": 2.35473657010083e-05,
    "reg_lambda": 0.16052639170222655,
    "multiplier_learning_rate": 0.11877008480186582,
    "constraint_fpr_tolerance": 0.030554410906873195,
    "constraint_fnr_tolerance": 0.006154759936716918,
    "stepwise_proxy_margin": 1.9963577850877554,
    "score_threshold": 0.32948983598099757,
    "global_score_threshold": 0.37443883172152953
}

warfarin_dataset_params = {
    "n_samples": 50000
}

with open("hyperparams/optimal_unfair.json", 'r') as f:
    outcome_model_config = json.load(f)

def objective(trial, dataset, df, alpha):
    model_params = {
        "fair_coeff": trial.suggest_float("fair_coeff", 0.1, 10.0),
        "lr": trial.suggest_float("lr", 0.005, 0.2),
        "aud_steps": trial.suggest_int("aud_steps", 5, 30),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "epoch": 20,
        "optim": "Adam",
        "zdim": 14,
        "tensorboard": True,
        "task": "CF"
    }
    
    result = run_single_experiment(
        dataset=dataset,
        df = df,
        model_params=model_params
    )
    
    accuracy = result["test"]["acc"]
    cf_score = result["test"]["CF"]

    print(accuracy)
    print(cf_score)
    
    return alpha * accuracy - (1 - alpha) * cf_score

def run_single_experiment(dataset, df, model_params):

    x_dim = len(df.columns) - 3
    
    # Update model_params with computed dimensions
    model_params.update({
        "xdim": x_dim,
        "ydim": 1,
        "sdim": 1,
        "encoder": [],
        "prediction": [],
        "audit": [model_params["zdim"]],
        "seed": 123
    })
    
    # Train and evaluate model
    dataset.train_test_split(split=0.15)
    decision_model = DCFR(
        config=model_params,
        n_fair=len(dataset.fair_variables)
    )

    _, _ = dataset.process()
    runner = Runner(
        dataset=dataset,
        model=decision_model,
        config=model_params
    )

    print(50*"=")
    print("lambda = ", model_params["fair_coeff"])
    print(50*"=")
    
    runner.train()
    runner.finetune("last")
    test_results = runner.test("finetune_last_best")
    
    return test_results

def optimize_hyperparameters(name, sim, dataset_params, outcome_model, alpha, n_trials=100):
    """
    Run hyperparameter optimization using Optuna for a specific alpha value.
    
    Args:
        name, sim, dataset_params, outcome_model: Original function parameters
        alpha: Weight for the objective function
        n_trials: Number of optimization trials
    
    Returns:
        dict: Best parameters and their corresponding scores
    """
    study = optuna.create_study(direction="maximize")

    # Initialize dataset
    dataset = DatasetBuilder(name=name, sim=sim)

    if not sim:
        df = dataset.load(**dataset_params)
    else:
        df, true_outcomes = dataset.load(**dataset_params)
    
    outcome_model.fit(
        df,
        decision_col=dataset.decision,
        outcome_col=dataset.outcome,
        protected_attribute=dataset.protected_attribute_name
    )
    
    outcome_estimates = outcome_model.effect(
        df.drop([dataset.decision, dataset.outcome], axis=1)
    )
    
    estimated_outcomes = pd.DataFrame({
        'estimated_y0': np.where(outcome_estimates[:, 0] < 0.5, 0, 1),
        'estimated_y1': np.where(outcome_estimates[:, 1] < 0.5, 0, 1)
    })
    
    estimated_outcomes.loc[(df[dataset.decision]==0), 'estimated_y0'] = df[dataset.decision]
    estimated_outcomes.loc[(df[dataset.decision]==1), 'estimated_y1'] = df[dataset.decision]
    
    df = dataset.update(estimated_outcomes, result_method=False)
    
    objective_func = lambda trial: objective(
        trial, dataset, df, alpha
    )

    study.optimize(objective_func, n_trials=n_trials)
    
    return {
        "alpha": alpha,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial
    }

# Initialize outcome model
base_estimator = FairGBMClassifier(**warfarin_optimal_params)
outcome_model = TLearner(
    base_estimator=base_estimator
)

# Run optimization for different alpha values
alpha_values = [0.1, 0.5, 0.9]
all_results = []

for alpha in alpha_values:
    print(f"\nOptimizing for alpha = {alpha}")
    print("=" * 50)
    
    results = optimize_hyperparameters(
        name="warfarin",
        sim=True,
        dataset_params=warfarin_dataset_params,
        outcome_model=outcome_model,
        alpha=alpha,
        n_trials=20 
    )
    
    all_results.append(results)
    
    print(f"\nResults for alpha = {alpha}:")
    print(f"Best hyperparameters: {results['best_params']}")
    print(f"Best weighted score: {results['best_value']}")