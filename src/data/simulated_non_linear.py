import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class SimulatedDatasetNonLinear():
    def __init__(self):
        self.data = None
        
    def generate_correlation_matrix(self, d=20):
        return np.eye(d)
        
    def load(self, n_samples=10000, d=20, propensity_score = 0.2, random_state=123) -> pd.DataFrame:
        """
        Create a simulated dataset with complex non-linear responses:
        - e(x) = 0.5 (treatment probability)
        - μ1(x) = 1/2 * ς(x1)ς(x2) + a_impact * A
        - μ0(x) = -1/2 * ς(x1)ς(x2) + a_impact * A
        where ς(x) = 2/(1 + exp(-12(x-1/2)))
        - A: binary feature with correlation-dependent impact
        """
            
        np.random.seed(random_state)
        self.d = d
        
        # Generate feature matrix X
        Sigma = self.generate_correlation_matrix(d)
        X = multivariate_normal.rvs(mean=np.zeros(d), cov=Sigma, size=n_samples)
        
        # Generate binary feature A
        A = np.random.binomial(n=1, p=0.5, size=n_samples)
        
        # Generate error terms
        eps_0 = np.random.normal(0, 0.1, n_samples)
        eps_1 = np.random.normal(0, 0.1, n_samples)
        beta_0 = np.random.uniform(-5, 1, d)
        beta_1 = np.random.uniform(-1, 5, d)
        
        # Generate potential outcomes with A's impact
        mu_0 = 2 * (1 / (1 + np.exp(-np.dot(X, beta_0)))) - 1.5
        Y_0 = (mu_0 + eps_0 > 0).astype(int)  # Binary outcome
        
        mu_1 = 2 * (1 / (1 + np.exp(-np.dot(X, beta_1)))) - 0.5
        Y_1 = (mu_1 + eps_1 > 0).astype(int)  # Binary outcome
        
        # Generate treatment assignment
        e_x = np.full(n_samples, propensity_score)  # constant propensity score
        D = np.random.binomial(n=1, p=e_x)
        
        # Generate observed outcome
        Y = D * Y_1 + (1 - D) * Y_0
        
        # Create DataFrame
        self.data = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(d)])
        self.data['A'] = A
        self.data['D'] = D
        self.data['Y'] = Y
        self.data['Y0'] = Y_0
        self.data['Y1'] = Y_1
        
        # Store additional information
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        
        return self.data
    
    def preprocess_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data into required format.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        self.true_outcomes = pd.DataFrame({
            'Y0': self.data['Y0'],
            'Y1': self.data['Y1']
        })
        self.data = self.data.drop(columns=['Y0', 'Y1'])
        
        return self.data, self.true_outcomes
    
    def get_true_responses(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the true response functions (before binary transformation).
        """
        if not hasattr(self, 'mu_0') or not hasattr(self, 'mu_1'):
            raise ValueError("Response functions not available. Call load() first.")
        
        return self.mu_0, self.mu_1
