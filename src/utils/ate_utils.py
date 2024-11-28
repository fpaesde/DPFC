import copy
import logging
import time
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from outcome_estimation.causal_forests import CausalForestEstimator

def calculate_principal_fairness(decisions, true_outcomes, A_test):
    """
    Calculate principal fairness metric
    
    Parameters:
    -----------
    decisions : numpy.ndarray
        Model's predicted decisions
    true_outcomes : numpy.ndarray
        Array with two columns representing true outcomes
    A_test : numpy.ndarray
        Protected attribute values (binary)
    forest : bool, optional
        Flag for forest-based calculations (default=False)
    
    Returns:
    --------
    float
        Principal fairness metric
    """

    df = pd.DataFrame({
        'decision': decisions,
        'outcome_1': true_outcomes['Y0'],
        'outcome_2': true_outcomes['Y1'],
        'protected': A_test
    })

    count = (df.groupby(['outcome_1', 'outcome_2', 'protected'])
             .size()
             .reset_index(name='count'))
    count_decisions = (df.groupby(['outcome_1', 'outcome_2', 'protected'])
                      ['decision']
                      .sum()
                      .reset_index(name='count_decisions'))
    
    merged = pd.merge(count, count_decisions, 
                     on=['outcome_1', 'outcome_2', 'protected'])
    merged['ratio'] = merged['count_decisions'] / merged['count']
    
    merged['ratio'] = (2 * merged['protected'] - 1) * merged['ratio']
    result = (merged.groupby(['outcome_1', 'outcome_2'])
             .sum()['ratio']
             .reset_index(drop=True)
             .values)
    outcome_props = (df.groupby(['outcome_1', 'outcome_2'])
                    .size()
                    .reset_index(drop=True)
                    .values)
    outcome_props = outcome_props / np.sum(outcome_props)
    fair_metric = np.sum(np.abs(result) * outcome_props)
    
    return fair_metric
    
def evaluate_model(model, X_test, true_outcomes):
    """
    Evaluate model performance comparing predicted potential outcomes with true outcomes.
    
    Parameters:
    -----------
    model : estimator object
        The fitted model
    X_test : array-like
        Test features
    true_outcomes : pd.DataFrame
        DataFrame with columns 'Y0' and 'Y1' containing true potential outcomes
    A_test: protected attribute, pd.DataFrame
        
    Returns
    --------
    dict with evaluation metrics
    """
    # Get predictions for both potential outcomes
    outcome_estimates = model.effect(X_test)
    cate_estimates = model.cate(X_test)
    mse_cate = mean_squared_error(true_outcomes['Y1'] - true_outcomes['Y0'], cate_estimates)
    cate_estimates = np.where(cate_estimates < -0.5, -1,
                         np.where((cate_estimates >= -0.5) & (cate_estimates <= 0.5), 0, 1))
    y0_pred = outcome_estimates[:, 0]
    y1_pred = outcome_estimates[:, 1]
    
    # Calculate metrics for Y0
    mse_y0 = mean_squared_error(true_outcomes['Y0'], y0_pred)
    mae_y0 = mean_absolute_error(true_outcomes['Y0'], y0_pred)
    rmae_y0 = np.sqrt(mae_y0)
    r2_y0 = r2_score(true_outcomes['Y0'], y0_pred)
    
    # Calculate metrics for Y1
    mse_y1 = mean_squared_error(true_outcomes['Y1'], y1_pred)
    mae_y1 = mean_absolute_error(true_outcomes['Y1'], y1_pred)
    rmae_y1 = np.sqrt(mse_y1)
    r2_y1 = r2_score(true_outcomes['Y1'], y1_pred)
    
    # Calculate average metrics
    mse = (mse_y0 + mse_y1) / 2
    rmae = (rmae_y0 + rmae_y1) / 2
    mae = (mae_y0 + mae_y1) / 2
    r2 = (r2_y0 + r2_y1) / 2
    
    # Calculate residuals for both outcomes
    residuals = np.column_stack([
        true_outcomes['Y0'] - y0_pred,
        true_outcomes['Y1'] - y1_pred
    ])

    # Convert CATE to decisions (D=1 if CATE > 0, D=0 if CATE ≤ 0)
    decisions = (cate_estimates > 0).astype(int)
    fair_metric = calculate_principal_fairness(decisions, true_outcomes, X_test['A'])

    return {
        'MAE_Y1': mae_y1,
        'MAE': mae,
        'RMAE': rmae,
        'R2': r2,
        'outcome_estimates': outcome_estimates,
        'residuals': residuals,
        'cate': cate_estimates,
        'cate_mse': mse_cate,
        'fairness': fair_metric,
    }


def train_and_evaluate_models_with_repetitions(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: np.ndarray,
    true_outcomes_test: pd.DataFrame,
    decision_col: str = 'D',
    outcome_col: str = 'Y',
    protected_attribute_col: str = 'A',
    train_sizes: list = [],
    n_repetitions: int = 30,
    verbose: bool = True
):
    """
    Train and evaluate multiple models with different training set sizes multiple times.
    Returns mean and variance of metrics across repetitions.
    """
    if len(train_sizes) == 0:
        train_sizes = [X_train.shape[0]]

    # Initialize results structure
    results = {
        model_name: {
            size: {
                'mean': {},
                'var': {},
                'all_runs': []
            } for size in train_sizes
        } for model_name in models.keys()
    }

    for rep in range(n_repetitions):
        if verbose:
            logging.info(f"Running repetition {rep + 1}/{n_repetitions}")

        for size in train_sizes:
            if verbose:
                logging.info(f"Training with {size} samples")
            
            X_train_subset = X_train.iloc[:size]

            for name, model in models.items():
                if verbose:
                    logging.info(f"Training and evaluating {name}")

                model_copy = copy.deepcopy(model)
                
                start = time.time()
                model_copy.fit(
                    train_data = X_train_subset,
                    decision_col = decision_col,
                    outcome_col = outcome_col,
                    protected_attribute = protected_attribute_col
                )
                train_time = time.time() - start

                eval_results = evaluate_model(model_copy, X_test, true_outcomes_test)

                eval_results['TrainTime'] = train_time
                logging.info(f"MAE: {eval_results["MAE"]}")
                results[name][size]['all_runs'].append(eval_results)

    for name in models.keys():
        for size in train_sizes:
            runs = results[name][size]['all_runs']
            results[name][size]['mean'] = {}
            results[name][size]['var'] = {}
            
            metrics = ['MAE_Y1', 'RMAE', 'MAE', 'R2', 'TrainTime', 'cate_mse', 'fairness']
            for metric in metrics:
                values = [run[metric] for run in runs]
                results[name][size]['mean'][metric] = np.mean(values)
                results[name][size]['var'][metric] = np.var(values)
            
            results[name][size]['mean']['outcome_estimates'] = np.mean([run['outcome_estimates'] for run in runs], axis=0)
            results[name][size]['var']['outcome_estimates'] = np.var([run['outcome_estimates'] for run in runs], axis=0)
            results[name][size]['mean']['residuals'] = np.mean([run['residuals'] for run in runs], axis=0)
            results[name][size]['var']['residuals'] = np.var([run['residuals'] for run in runs], axis=0)
            results[name][size]['mean']['cate'] = np.mean([run['cate'] for run in runs], axis=0)
    return results



