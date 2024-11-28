
import argparse
import json
import logging
from lightgbm import LGBMRegressor, LGBMClassifier
from fairgbm import FairGBMClassifier
import numpy as np
from sklearn.model_selection import train_test_split

from data.dataset_builder import DatasetBuilder
from outcome_estimation.s_learner import SLearner
from outcome_estimation.t_learner import TLearner
from outcome_estimation.x_learner import XLearner
from outcome_estimation.double_robust import DRLearner
from utils.ate_utils import calculate_principal_fairness


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="Dataset name")
    parser.add_argument("--hyperparam_path", type=str, required=True, help="Path to Optimal Hyperparameter tuning file")
    parser.add_argument('--samples', type=int, default=12500, help="Total number of samples")
    parser.add_argument('--save', type=bool, default=True, help="Save results")
    parser.add_argument('--verbose', type=bool, default=True, help="Print progress")
    return parser


def process_json_file(data, alpha, model):
    
    results = {}
    key = f"{alpha}_{model}"
    if key in data:
        model_params = data[key]['best_params']
        
        base_params = {
            k.replace('base_estimator_', ''): v 
            for k, v in model_params.items() 
            if k.startswith('base_estimator_')
        }
        base_estimator = FairGBMClassifier(
            random_state=123,
            verbose=-1,
            **base_params
        )
        
        refined_params = {
            k.replace('refined_estimator_', ''): v 
            for k, v in model_params.items() 
            if k.startswith('refined_estimator_')
        }
        refined_estimator = None
        if refined_params:
            refined_estimator = LGBMRegressor(
                random_state=123,
                verbose=-1,
                **refined_params
            )
        
        ps_params = {
            k.replace('ps_estimator_', ''): v 
            for k, v in model_params.items() 
            if k.startswith('ps_estimator_')
        }
        ps_estimator = None
        if ps_params:
            ps_estimator = LGBMClassifier(
                random_state=123,
                verbose=-1,
                **ps_params
            )
        
        results[key] = {
            'base_estimator': base_estimator,
            'refined_estimator': refined_estimator,
            'ps_estimator': ps_estimator
        }
    
    return results

def evaluate_model(model, test_data, true_outcomes_test, A_test):
    X_test = test_data.drop(columns=['D', 'Y'])

    optimal_decisions = np.where(
            true_outcomes_test['Y0'] < true_outcomes_test['Y1'], 1, 0
        )

    cate_estimates = model.cate(X_test)
    decisions = np.where(cate_estimates <= 0, 0, 1)

    acc = (optimal_decisions == decisions).mean()
    fair_metric = calculate_principal_fairness(decisions, true_outcomes_test, A_test)
    
    return acc, fair_metric


def main():
    parser = build_parser()
    args = parser.parse_args()

    with open(args.hyperparam_path, 'r') as f:
        optimal_params = json.load(f)

    dataset = DatasetBuilder(args.name)
    data, true_outcomes = dataset.load(n_samples=args.samples)

    train, test, _, true_outcomes_test = train_test_split(
        data, true_outcomes, test_size=0.2, random_state=123
    )

    results = {}
    alphas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    for model_name in ['S-Learner', 'T-Learner', 'X-Learner', 'DR-Learner']:
        if args.verbose:
            logging.info(f"Training model: {model_name}")
            
        for alpha in alphas:
            if args.verbose:
                logging.info(f"Alpha: {alpha}")

            optimal_learners = process_json_file(optimal_params, alpha, model_name)
            optimal_learner = optimal_learners[list(optimal_learners.keys())[0]]

            if model_name == "S-Learner":
                outcome_model = SLearner(base_estimator = optimal_learner["base_estimator"])

            elif model_name == "T-Learner":
                outcome_model = TLearner(base_estimator = optimal_learner["base_estimator"])

            elif model_name == "X-Learner":
                outcome_model = XLearner(
                    base_estimator = optimal_learner["base_estimator"],
                    refined_estimator= optimal_learner["refined_estimator"],
                    ps_estimator = optimal_learner["ps_estimator"]
                )

            elif model_name == "DR-Learner":
                outcome_model = DRLearner(
                    base_estimator = optimal_learner["base_estimator"],
                    ps_estimator = optimal_learner["ps_estimator"]
                )

            outcome_model.fit(
                train_data=train,
                decision_col=dataset.decision,
                outcome_col=dataset.outcome,
                protected_attribute=dataset.protected_attribute_name
            )

            acc, delta_pf = evaluate_model(outcome_model, test, true_outcomes_test, test[dataset.protected_attribute_name])
            
            results[f"{model_name}_{alpha}"] = {
                "acc": acc,
                "PF": delta_pf
            }

            logging.info(f"Acc: {acc}")
            logging.info(f"Delta PF: {delta_pf}")
        
    if args.save:
        with open(f'optimization_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        logging.info(f"Results saved to optimization_results.json")

if __name__ == '__main__':
    main()
