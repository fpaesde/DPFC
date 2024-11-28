import argparse
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

from data.dataset_builder import DatasetBuilder
from outcome_estimation.double_robust import DRLearner
from outcome_estimation.s_learner import SLearner
from outcome_estimation.t_learner import TLearner
from outcome_estimation.x_learner import XLearner

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="Dataset name")
    parser.add_argument('--samples', type=int, default=12500, help="Total number of samples")
    parser.add_argument('--save', type=bool, default=True, help="Save results")
    parser.add_argument('--verbose', type=bool, default=True, help="Print progress")
    parser.add_argument('--legend', type=bool, default=False, help='Make legend in plots')
    return parser

def get_latest_trial(dataset_name):
    base_path = f"../saved/{dataset_name}"
    trials = [d for d in os.listdir(base_path) if d.startswith('trial_')]
    latest_trial = max([int(t.split('_')[1]) for t in trials])
    return f"trial_{latest_trial}"

def load_optimal_params(dataset_name, model_name):
    trial_dir = get_latest_trial(dataset_name)
    path = f"../saved/{dataset_name}/{trial_dir}/optimization_results.json"
    with open(path, 'r') as f:
        results = json.load(f)
    return results[f"10000_{model_name}"]["best_params"]

def create_estimator(params, prefix, estimator_class):
    filtered_params = {}
    for key, value in params.items():
        if key.startswith(prefix):
            filtered_params[key.replace(f"{prefix}_", "")] = value
    return estimator_class(**filtered_params, verbose=-1, random_state=123)

def evaluate_confidence_intervals(ci_estimates, true_outcomes):

    y0_true = true_outcomes['Y0'].values
    y1_true = true_outcomes['Y1'].values
    
    y0_lower, y1_lower = ci_estimates[0][:, 0], ci_estimates[0][:, 1]
    y0_upper, y1_upper = ci_estimates[1][:, 0], ci_estimates[1][:, 1]
    
    y0_coverage = np.mean((y0_true >= y0_lower) & (y0_true <= y0_upper))
    y1_coverage = np.mean((y1_true >= y1_lower) & (y1_true <= y1_upper))
    
    y0_length = np.mean(y0_upper - y0_lower)
    y1_length = np.mean(y1_upper - y1_lower)
    
    coverage = (y0_coverage + y1_coverage) / 2
    ci_length = (y0_length + y1_length) / 2
    
    return coverage, ci_length

def plot_coverage_length(results, make_legend = True, save_path=None):

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.usetex'] = True
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    colors = sns.color_palette("husl", len(results) + 1)
    markers = ['s', 'o', 'D', '^']
    
    for idx, (model_name, coverage, ci_length) in enumerate(results):
        ax.scatter(ci_length, coverage, 
                  color=colors[idx + 1],
                  marker=markers[idx],
                  s=120,
                  label=model_name,
                  alpha=0.8,
                  edgecolor='white',
                  linewidth=1.5,
                  zorder=3)

    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.6, zorder=2)
    
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0.0, 1.0)
    
    ax.set_xlabel('Average Confidence Interval Length', fontsize=11)
    ax.set_ylabel('Average Coverage', fontsize=11)
    
    ax.grid(True, linestyle=':', alpha=0.3, zorder=1)
    ax.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        

    ax.tick_params(axis='both', which='major', labelsize=9)

    if make_legend:
        legend = ax.legend(frameon=True, 
                        fancybox=True,
                        framealpha=0.95,
                        edgecolor='gray',
                        fontsize=9,
                        loc="upper right",
                        title='Models',
                        title_fontsize=10)
        legend.get_frame().set_linewidth(0.8)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
    
    return fig, ax

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Initialize dataset
    dataset = DatasetBuilder(args.name)
    data, true_outcomes = dataset.load(n_samples=args.samples)

    # Train-test split
    train, test, _, true_outcomes_test = train_test_split(
        data, true_outcomes, test_size=0.2, random_state=123
    )

    model_classes = ['S-Learner', 'T-Learner', 'X-Learner', 'DR-Learner']
    results = []

    for idx, model_class_name in enumerate(model_classes):
        if args.verbose:
            print(f"Processing {model_class_name}...")

        params = load_optimal_params(args.name, model_class_name)

        base_estimator = create_estimator(params, "base_estimator", LGBMClassifier)

        ps_estimator = None
        if any("ps_estimator" in key for key in params.keys()):
            ps_estimator = create_estimator(params, "ps_estimator", LGBMClassifier)

        refined_estimator = None
        if any("refined_estimator" in key for key in params.keys()):
            refined_estimator = create_estimator(params, "refined_estimator", LGBMRegressor)

        if model_class_name == "S-Learner":
            model = SLearner(base_estimator=base_estimator)
        elif model_class_name == "T-Learner":
            model = TLearner(base_estimator=base_estimator)
        elif model_class_name == "X-Learner":
            model = XLearner(base_estimator=base_estimator, refined_estimator=refined_estimator, ps_estimator=ps_estimator)
        else:
            model = DRLearner(base_estimator=base_estimator, residual_model=refined_estimator, ps_estimator=ps_estimator)

        model.fit(
            train_data=train,
            decision_col='D',
            outcome_col='Y',
            protected_attribute='A'
        )

        ci_estimates = model.bootstrap_confidence_intervals(test.drop(columns = ["D", "Y"]), n_bootstraps=1000)

        coverage, ci_length = evaluate_confidence_intervals(ci_estimates, true_outcomes_test)
        results.append((model_class_name, coverage, ci_length))

        if args.verbose:
            print(f"Coverage: {coverage:.3f}")
            print(f"CI Length: {ci_length:.3f}")

    fig, ax = plot_coverage_length(results, args.legend, f'../results/{args.name}_coverage_analysis.png')

if __name__ == "__main__":
    main()