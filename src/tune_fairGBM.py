import argparse
import datetime
import logging
import json
import os
import time
from fairgbm import FairGBMClassifier, LGBMRegressor
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from outcome_estimation.s_learner import SLearner
from outcome_estimation.t_learner import TLearner
from outcome_estimation.x_learner import XLearner
from outcome_estimation.double_robust import DRLearner
from outcome_estimation.causal_forests import CausalForestEstimator
from data.dataset_builder import DatasetBuilder
from utils.ate_utils import calculate_principal_fairness

logging.getLogger('lightgbm').disabled = True
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="Dataset name")
    parser.add_argument('--samples', type=int, default=12500, help="Total number of samples")
    parser.add_argument('--save', type=bool, default=True, help="Save results")
    parser.add_argument('--verbose', type=bool, default=True, help="Print progress")
    return parser


def create_model_configs():
    def get_refined_search_space():
        return {
            'learning_rate': ('float', 0.01, 0.3, True),
            'n_estimators': ('int', 50, 300),
            'max_depth': ('int', 3, 8),
            'num_leaves': ('int', 20, 100),
            'min_child_samples': ('int', 10, 100),
            'subsample': ('float', 0.5, 1.0),
            'colsample_bytree': ('float', 0.5, 1.0),
            'reg_alpha': ('float', 1e-8, 10.0, True),
            'reg_lambda': ('float', 1e-8, 10.0, True)
        }
    def get_base_search_space():
        return {
            'learning_rate': ('float', 0.01, 0.3, True),
            'n_estimators': ('int', 50, 300),
            'max_depth': ('int', 3, 8),
            'num_leaves': ('int', 20, 100),
            'min_child_samples': ('int', 10, 100),
            'subsample': ('float', 0.5, 1.0),
            'colsample_bytree': ('float', 0.5, 1.0),
            'reg_alpha': ('float', 1e-8, 10.0, True),
            'reg_lambda': ('float', 1e-8, 10.0, True),
            'multiplier_learning_rate': ('float', 0.01, 1.0, True),
            'constraint_fpr_tolerance': ('float', 0.001, 0.1, True),
            'constraint_fnr_tolerance': ('float', 0.001, 0.1, True),
            'stepwise_proxy_margin': ('float', 0.5, 2.0),
            'score_threshold': ('float', 0.3, 0.7),
            'global_score_threshold': ('float', 0.3, 0.7)
        }

    return {
        'S-Learner': {
            'class': SLearner,
            'params': {'base_estimator': get_base_search_space()}
        },
        'T-Learner': {
            'class': TLearner,
            'params': {'base_estimator': get_base_search_space()}
        },
        'X-Learner': {
            'class': XLearner,
            'params': {
                'base_estimator': get_base_search_space(),
                'refined_estimator': get_refined_search_space(),
                'ps_estimator': get_refined_search_space()
            }
        },
        'DR-Learner': {
            'class': DRLearner,
            'params': {
                'base_estimator': get_base_search_space(),
                'ps_estimator': get_refined_search_space()
            }
        }
    }

def create_trial_params(trial, param_space, prefix=''):
    params = {}
    for param_name, param_config in param_space.items():
        full_param_name = f"{prefix}_{param_name}" if prefix else param_name
        if param_config[0] == 'float':
            params[param_name] = trial.suggest_float(full_param_name, param_config[1], param_config[2], 
                                                   log=param_config[3] if len(param_config) > 3 else False)
        elif param_config[0] == 'int':
            params[param_name] = trial.suggest_int(full_param_name, param_config[1], param_config[2])
    
    # Add common parameters
    params['verbose'] = -1
    params['random_state'] = 123
    
    return params

def create_objective(model_config, train_data, test_data, true_outcomes_test, A_test, alpha):
    def objective(trial):
        # Create estimators with trial parameters
        estimators = {}
        for estimator_name, param_space in model_config['params'].items():
            params = create_trial_params(trial, param_space, prefix=estimator_name)
            
            if estimator_name == "base_estimator":
                estimators[estimator_name] = FairGBMClassifier(**params)
            elif estimator_name == "refined_estimator":
                estimators[estimator_name] = LGBMRegressor(**params)
            else:
                estimators[estimator_name] = LGBMClassifier(**params)

        model = model_config['class'](**estimators)
        
        model.fit(
            train_data=train_data,
            decision_col='D',
            outcome_col='Y',
            protected_attribute='A'
        )
        
        mae, fair_metric = evaluate_model(model, test_data, true_outcomes_test, A_test)

        return alpha * mae + (1 - alpha) * fair_metric

    return objective

def evaluate_model(model, test_data, true_outcomes_test, A_test):
    X_test = test_data.drop(columns=['D', 'Y'])
    
    # Calculate MAE
    outcome_estimates = model.effect(X_test)
    mae_y0 = mean_absolute_error(true_outcomes_test['Y0'], outcome_estimates[:, 0])
    mae_y1 = mean_absolute_error(true_outcomes_test['Y1'], outcome_estimates[:, 1])
    mae = (mae_y0 + mae_y1) / 2
    
    # Calculate Delta CF
    cate_estimates = model.cate(X_test)
    cate_estimates = np.where(cate_estimates < -0.5, -1,
                                np.where((cate_estimates >= -0.5) & (cate_estimates <= 0.5), 0, 1))
    decisions = np.where(cate_estimates <= 0, 0, 1)
    fair_metric = calculate_principal_fairness(decisions, true_outcomes_test, A_test)
    
    return mae, fair_metric

def main():
    parser = build_parser()
    args = parser.parse_args()

    base_dir = f"../saved/{args.name}"
    os.makedirs(base_dir, exist_ok=True)
    
    existing_trials = [d for d in os.listdir(base_dir) if d.startswith('trial_')]
    trial_numbers = [int(d.split('_')[1]) for d in existing_trials] if existing_trials else [0]
    next_trial = max(trial_numbers) + 1
    trial_dir = os.path.join(base_dir, f'trial_{next_trial}')
    os.makedirs(trial_dir)

    dataset = DatasetBuilder(args.name)
    data, true_outcomes = dataset.load(n_samples=args.samples)

    if args.save:
        data.to_csv(f"{trial_dir}/data.csv")

    train, test, _, true_outcomes_test = train_test_split(
        data, true_outcomes, test_size=0.2, random_state=123
    )

    alphas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    model_configs = create_model_configs()
    
    results = {}
    
    for model_name, model_config in model_configs.items():
        if args.verbose:
            logging.info(f"Training model: {model_name}")
            print(model_name)
            
        for alpha in alphas:
            if args.verbose:
                logging.info(f"Alpha: {alpha}")
                print(alpha)

            study = optuna.create_study(direction='minimize')
            objective = create_objective(model_config, 
                                         train, 
                                         test, 
                                         true_outcomes_test, 
                                         test[dataset.protected_attribute_name], 
                                         alpha)
            
            study.optimize(objective, n_trials=50, show_progress_bar=True)

            best_estimators = {}
            for estimator_name, param_space in model_config['params'].items():
                best_params = {k[len(estimator_name)+1:]: v 
                            for k, v in study.best_params.items() 
                            if k.startswith(f"{estimator_name}_")}
                # Add common parameters
                best_params['verbose'] = -1
                best_params['random_state'] = 123
                
                if estimator_name == 'refined_estimator':
                    best_estimators[estimator_name] = LGBMRegressor(**best_params)
                elif estimator_name == "base_estimator":
                    best_estimators[estimator_name] = FairGBMClassifier(**best_params)
                else:
                    best_estimators[estimator_name] = LGBMClassifier(**best_params)

            best_model = model_config['class'](**best_estimators)
            
            start_time = time.time()
            best_model.fit(
                train_data=train,
                decision_col=dataset.decision,
                outcome_col=dataset.outcome,
                protected_attribute=dataset.protected_attribute_name
            )
            training_time = time.time() - start_time

            mae, delta_cf = evaluate_model(best_model, test, true_outcomes_test, test[dataset.protected_attribute_name])
            print(f"Results: MAE: {mae}, Delta_CF: {delta_cf}")

            results[f"{alpha}_{model_name}"] = {
                'MAE': mae,
                'Delta_CF': delta_cf,
                'training_time': training_time,
                'best_params': study.best_params,
                'optimization_completed': str(datetime.datetime.now())
            }

    if args.save:
        with open(f'{trial_dir}/optimization_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        logging.info(f"Results saved to {trial_dir}/optimization_results.json")

if __name__ == '__main__':
    main()