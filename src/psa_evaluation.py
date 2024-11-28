
import argparse
from fairgbm import FairGBMClassifier
import numpy as np
import pandas as pd
import pyreadr

from data.dataset_builder import DatasetBuilder
from outcome_estimation.t_learner import TLearner

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

def load_dataset(name):
    dataset_params = {
        "dataset": pyreadr.read_r(f"data/PSA/{name}data.rda")[
            list(pyreadr.read_r(f"data/PSA/{name}data.rda").keys())[0]
        ],
        "columns": [
            'Sex', 'White', 'Age', 'PendingChargeAtTimeOfOffense',
            'NCorNonViolentMisdemeanorCharge', 'ViolentMisdemeanorCharge',
            'ViolentFelonyCharge', 'NonViolentFelonyCharge',
            'PriorMisdemeanorConviction', 'PriorFelonyConviction',
            'PriorViolentConviction', 'PriorSentenceToIncarceration',
            'PriorFTAInPastTwoYears', 'PriorFTAOlderThanTwoYears'
        ],
        "decision": "D",
        "protected_attribute_name": "Sex",
        "priviledged_class": [1],
        "outcome": "Y"
    }
    return dataset_params

def initialize_and_update_dataset(name, dataset_params, warfarin_optimal_params):
    dataset = DatasetBuilder(name=name, sim=False)
    df = dataset.load(**dataset_params)
    
    base_estimator = FairGBMClassifier(**warfarin_optimal_params, verbose = -1, random_state=123)
    t_learner = TLearner(base_estimator=base_estimator)
    
    t_learner.fit(
        df,
        decision_col=dataset.decision,
        outcome_col=dataset.outcome,
        protected_attribute=dataset.protected_attribute_name
    )
    
    outcome_estimates = t_learner.effect(df.drop([dataset.decision, dataset.outcome], axis=1))
    estimated_outcomes = pd.DataFrame({
        'estimated_y0': outcome_estimates[:, 0],
        'estimated_y1': outcome_estimates[:, 1]
    })
    
    df = dataset.update(estimated_outcomes, result_method=True)
    
    return dataset, df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, choices=['FTA', 'NCA'], required=True)
    parser.add_argument("--threshold", type=int, required=True)
    args = parser.parse_args()
    
    dataset_params = load_dataset(args.name)
    dataset, df = initialize_and_update_dataset(
        args.name, dataset_params, warfarin_optimal_params
    )

    psa_rda = pyreadr.read_r("data/PSA/PSAdata.rda")
    psa_dataset = psa_rda[list(psa_rda.keys())[0]]

    decisions = np.where(psa_dataset[f"{args.name}Score"] >= args.threshold, 1, 0)
    results = dataset.analyze(df, y=decisions)
    print(results)

if __name__ == "__main__":
    main()