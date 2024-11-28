import json
import logging
import argparse
import os
import pickle
import warnings

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor
from fairgbm import FairGBMClassifier

from outcome_estimation.s_learner import SLearner
from outcome_estimation.t_learner import TLearner
from outcome_estimation.x_learner import XLearner
from outcome_estimation.double_robust import DRLearner
from outcome_estimation.causal_forests import CausalForestEstimator

from data.dataset_builder import DatasetBuilder
from utils.ate_utils import *
from utils.plot_utils import plot_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)

with open("hyperparams/optimal_unfair.json", 'r') as f:
    outcome_model_config = json.load(f)

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help="Dataset to use")
    parser.add_argument('--samples', type=int, default=12500, nargs=1)
    parser.add_argument('--save', type=bool, default=True, nargs=1)
    parser.add_argument('--model', type=str, default="boost", nargs=1)
    parser.add_argument("--mlr", type=float, default=0.1, nargs=1)

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    base_dir = f"../results/{args.name}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    existing_trials = [d for d in os.listdir(base_dir) if d.startswith('trial_')]
    trial_numbers = [int(d.split('_')[1]) for d in existing_trials] if existing_trials else [0]
    next_trial = max(trial_numbers) + 1

    trial_dir = os.path.join(base_dir, f'trial_{next_trial}')
    os.makedirs(trial_dir)

    dataset = DatasetBuilder(args.name)
    data, true_outcomes = dataset.load(n_samples = args.samples)

    if args.save:
        data.to_csv(f"{trial_dir}/data.csv")

    train, test, _, true_outcomes_test = train_test_split(
        data, true_outcomes, test_size = 0.2, random_state = 123
    )

    D_test = test['D']
    test = test.drop(columns=['D', 'Y'])

    train_sizes = [100, 500, 1000, 2500, 5000, 10000]
    
    if args.model == "boost":
        if args.name == "warfarin":
            base_estimator = LGBMClassifier(
                **outcome_model_config["base_estimator_params"],
                verbose = -1,
                random_state=123
            )

            ps_estimator = LGBMClassifier(
                **outcome_model_config["ps_estimator_params"],
                verbose = -1,
                random_state=123
            )
        else:
            base_estimator = LGBMClassifier(
                max_depth=2,
                learning_rate=0.1,
                n_estimators=100,
                min_split_gain=0.001,
                random_state=123,
                verbose=-1
                )
            ps_estimator = LGBMRegressor(
                max_depth=2,
                learning_rate=0.1,
                n_estimators=100,
                min_split_gain=0.001,
                random_state=123,
                verbose=-1
                )  

        refined_estimator = LGBMRegressor(
        max_depth=2,
        learning_rate=0.1,
        n_estimators=100,
        min_split_gain=0.001,
        random_state=123,
        verbose=-1
        )
    else:
        if args.name == "warfarin":
            base_estimator = FairGBMClassifier(
                **outcome_model_config["base_estimator_params"],
                multiplier_learning_rate=args.mlr,
                constraint_type="FPR",
                verbose = -1,
                random_state=123
            )

            ps_estimator = LGBMClassifier(
                **outcome_model_config["ps_estimator_params"],
                verbose = -1,
                random_state=123
            )
        else:
            base_estimator = FairGBMClassifier(
                max_depth=2,
                learning_rate=0.1,
                n_estimators=100,
                min_split_gain=0.001,
                random_state=123,
                multiplier_learning_rate=args.mlr,
                constraint_type="FPR",
                verbose=-1
                )
            ps_estimator = LGBMRegressor(
                max_depth=2,
                learning_rate=0.1,
                n_estimators=100,
                min_split_gain=0.001,
                random_state=123,
                verbose=-1
                )  

        refined_estimator = LGBMRegressor(
        max_depth=2,
        learning_rate=0.1,
        n_estimators=100,
        min_split_gain=0.001,
        random_state=123,
        verbose=-1
        )

    models = {
        'S-Learner': SLearner(base_estimator=base_estimator),
        'T-Learner': TLearner(base_estimator=base_estimator),
        'X-Learner': XLearner(base_estimator=base_estimator, refined_estimator=refined_estimator, ps_estimator=ps_estimator),
        'DR-Learner': DRLearner(base_estimator=base_estimator, ps_estimator=LGBMClassifier(**outcome_model_config['base_estimator_params'], verbose=-1))
    }

    if args.name != "sim_confounding":
        models['Causal-Forest'] = CausalForestEstimator()

    results = train_and_evaluate_models_with_repetitions(
            models = models,
            X_train = train,
            X_test = test,
            true_outcomes_test = true_outcomes_test,
            train_sizes=train_sizes,
            n_repetitions = 1
        )

    if args.save:
        with open(f"{trial_dir}/results.pkl", 'wb') as f:
            pickle.dump(results, f)
    
    plot_results(results, train_sizes, true_outcomes['Y0'], true_outcomes['Y1'], D_test, save_path=trial_dir)

if __name__ == '__main__':
    main()