import argparse
import datetime
import os
from fairgbm import FairGBMClassifier
import numpy as np
import pandas as pd
import pyreadr
import optuna
import json
from sklearn.model_selection import train_test_split


from data.dataset_builder import DatasetBuilder
from outcome_estimation.s_learner import SLearner
from outcome_estimation.t_learner import TLearner
from outcome_estimation.x_learner import XLearner
from outcome_estimation.double_robust import DRLearner
from lightgbm import LGBMRegressor, LGBMClassifier
from models.dcfr import DCFR
from utils.model_utils import Runner

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
    
    params['verbose'] = -1
    params['random_state'] = 123
    
    return params


def initialize_and_update_dataset(name, dataset_params, warfarin_optimal_params):
    dataset = DatasetBuilder(name=name, sim=False)
    df = dataset.load(**dataset_params)
    
    base_estimator = FairGBMClassifier(**warfarin_optimal_params)
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
    train, test = train_test_split(df, test_size=0.1, random_state=42)
    
    return dataset, train, test

def objective_causal(trial, dataset, model_class, model_params, train_data, test_data, alpha):
    estimators = {}
    for estimator_name, param_space in model_params.items():
        params = create_trial_params(trial, param_space, prefix=estimator_name)
        if estimator_name == "base_estimator":
            estimators[estimator_name] = FairGBMClassifier(**params)
        elif estimator_name == "refined_estimator":
            estimators[estimator_name] = LGBMRegressor(**params)
        else:
            estimators[estimator_name] = LGBMClassifier(**params)
    
    model = model_class(**estimators)
    model.fit(train_data.drop(['result'], axis=1),
              decision_col='D', outcome_col='Y', protected_attribute='Sex')
    
    outcome_estimates = model.effect(test_data.drop(['result', 'D', 'Y'], axis=1))
    estimated_outcomes = pd.DataFrame({
        'estimated_y0': outcome_estimates[:, 0],
        'estimated_y1': outcome_estimates[:, 1]
    })
    decisions = np.where(
                test_data['D'] == 1,
                np.where((estimated_outcomes['estimated_y0']) < estimated_outcomes['estimated_y1'], 1, 0),
                np.where(estimated_outcomes['estimated_y1'] == 1, 1, 0)
            )
    
    results = dataset.analyze(test_data, y=decisions)
    return alpha * results["acc"] - (1 - alpha) * results["CF"]

def run_dcfr_experiment(dataset, df, model_params):
    x_dim = len(df.columns) - 3
    model_params.update({
        "xdim": x_dim,
        "ydim": 1,
        "sdim": 1,
        "encoder": [],
        "prediction": [],
        "audit": [model_params["zdim"]],
        "seed": 123
    })
    
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
    
    runner.train()
    runner.finetune("last")
    test_results = runner.test("finetune_last_best")
    
    return test_results

def objective_dcfr(trial, dataset, df, alpha):
    model_params = {
        "fair_coeff": trial.suggest_float("fair_coeff", 0.1, 10.0),
        "lr": trial.suggest_float("lr", 0.005, 0.2),
        "aud_steps": trial.suggest_int("aud_steps", 5, 30),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "epoch": 50,
        "optim": "Adam",
        "zdim": 14,
        "tensorboard": True,
        "task": "CF"
    }
    
    result = run_dcfr_experiment(dataset, df, model_params)
    return alpha * result["test"]["acc"] - (1 - alpha) * result["test"]["CF"]

def save_results(results, dataset_name, trial_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dict = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "causal_models": {},
        "dcfr_models": {}
    }
    
    os.makedirs(trial_dir, exist_ok=True)
    
    for alpha in [0.3, 0.5, 0.9]:
        results_dict["causal_models"][f"alpha_{alpha}"] = {}
        
        for model_name in ["S-Learner", "T-Learner", "X-Learner", "DR-Learner"]:
            if f"{model_name}_{alpha}" in results:
                model_results = results[f"{model_name}_{alpha}"]
                results_dict["causal_models"][f"alpha_{alpha}"][model_name] = model_results
        
        if f"DCFR_{alpha}" in results:
            results_dict["dcfr_models"][f"alpha_{alpha}"] = results[f"DCFR_{alpha}"]
    
    output_file = os.path.join(trial_dir, f"{dataset_name}_optimization_results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, choices=['FTA', 'NCA'], required=True)
    args = parser.parse_args()
    
    dataset_params = load_dataset(args.name)
    dataset, train_data, test_data = initialize_and_update_dataset(
        args.name, dataset_params, warfarin_optimal_params
    )
    
    alpha_values = [0.3, 0.5, 0.9]
    model_configs = create_model_configs()
    results = {}

    for alpha in alpha_values:
        for model_name, model_config in model_configs.items():
            study = optuna.create_study(direction="maximize")
            objective = lambda trial: objective_causal(
                trial, dataset, model_config['class'], model_config['params'],
                train_data, test_data, alpha
            )
            study.optimize(objective, n_trials=200, show_progress_bar=True)

            best_estimators = {}
            for estimator_name, param_space in model_config['params'].items():
                best_params = {k[len(estimator_name)+1:]: v 
                             for k, v in study.best_params.items() 
                             if k.startswith(f"{estimator_name}_")}
                best_params['verbose'] = -1
                best_params['random_state'] = 123
                
                if estimator_name == "base_estimator":
                    best_estimators[estimator_name] = FairGBMClassifier(**best_params)
                elif estimator_name == "refined_estimator":
                    best_estimators[estimator_name] = LGBMRegressor(**best_params)
                else:
                    best_estimators[estimator_name] = LGBMClassifier(**best_params)

            best_model = model_config['class'](**best_estimators)
            best_model.fit(
                train_data.drop(['result'], axis=1),
                decision_col='D',
                outcome_col='Y',
                protected_attribute='Sex'
            )
            
            outcome_estimates = best_model.effect(test_data.drop(['result', 'D', 'Y'], axis=1))
            estimated_outcomes = pd.DataFrame({
                'estimated_y0': outcome_estimates[:, 0],
                'estimated_y1': outcome_estimates[:, 1]
            })
            decisions = np.where(
                        test_data['D'] == 1,
                        np.where((estimated_outcomes['estimated_y0']) < estimated_outcomes['estimated_y1'], 1, 0),
                        np.where(estimated_outcomes['estimated_y1'] == 1, 1, 0)
                    )
            final_metrics = dataset.analyze(test_data, y=decisions)
            
            results[f"{model_name}_{alpha}"] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "accuracy": final_metrics["acc"],
                "cf_score": final_metrics["CF"],
                "DP": final_metrics["DP"],
                "EO": final_metrics["EO"]
            }
            
            print(f"Model: {model_name}, Alpha: {alpha}")
            print(f"Best params: {study.best_params}")
            print(f"Best value: {study.best_value}")
            print(f"Final accuracy: {final_metrics['acc']}")
            print(f"Final CF score: {final_metrics['CF']}")

    for alpha in alpha_values:
        study = optuna.create_study(direction="maximize")
        objective = lambda trial: objective_dcfr(trial, dataset, train_data, alpha)
        study.optimize(objective, n_trials=100)
        
        best_params = study.best_params
        best_model_results = run_dcfr_experiment(dataset, train_data, {
            **best_params,
            "epoch": 50,
            "optim": "Adam",
            "zdim": 14,
            "tensorboard": True,
            "task": "CF"
        })
        
        results[f"DCFR_{alpha}"] = {
            "best_params": best_params,
            "best_value": study.best_value,
            "accuracy": best_model_results["test"]["acc"],
            "cf_score": best_model_results["test"]["CF"],
            "DP": final_metrics["DP"],
            "EO": final_metrics["EO"]
        }
        
        print(f"DCFR Model, Alpha: {alpha}")
        print(f"Best params: {best_params}")
        print(f"Best value: {study.best_value}")
        print(f"Final accuracy: {best_model_results['test']['acc']}")
        print(f"Final CF score: {best_model_results['test']['CF']}")
    
    # Save results
    trial_dir = f"results/{args.name}"
    save_results(results, args.name, trial_dir)

if __name__ == "__main__":
    main()