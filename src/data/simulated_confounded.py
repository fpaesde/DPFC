import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, beta

class SimulatedDatasetBetaConfounded():
    def __init__(self):
        self.data = None
        
    def generate_correlation_matrix(self, d=20):
        return np.eye(d)
    
    def propensity_score(self, x1, a):
        """
        Calculate propensity score based on x1 and binary feature A
        """
        beta_pdf_values = beta.pdf(x1, a=2, b=4)
        a_impact = 0.1 * a
        base_propensity = 0.25 * (1 + beta_pdf_values)
        # Ensure propensity stays between 0 and 1
        return np.clip(base_propensity + a_impact, 0.01, 0.99)
        
    def load(self, n_samples=10000, d=20, random_state=123, propensity_score = 0.2) -> pd.DataFrame:
            
        np.random.seed(random_state)
        self.d = d

        eps_0 = np.random.normal(0, 1, n_samples)
        eps_1 = np.random.normal(0, 1, n_samples)
        
        Sigma = self.generate_correlation_matrix(d)
        X = multivariate_normal.rvs(mean=np.zeros(d), cov=Sigma, size=n_samples)
        A = np.random.binomial(n=1, p=0.2, size=n_samples)
        
        mu_0 = 2 * X[:, 0] - 1
        Y_0 = (mu_0  + eps_0> 0).astype(int)
        
        # μ1(x) = μ0(x)
        mu_1 = 2 * X[:, 0] - 1
        Y_1 = (mu_1  + eps_1> 0).astype(int)  # Same outcomes for treated and control
        
        # Generate treatment assignment with beta-confounded propensity
        e_x = self.propensity_score(X[:, 0], A)
        D = np.where(e_x <= 0.5, 0, 1)
        
        # Generate observed outcome
        Y = D * Y_1 + (1 - D) * Y_0
        
        # Create DataFrame
        self.data = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(d)])
        self.data['A'] = A
        self.data['D'] = D
        self.data['Y'] = Y
        self.data['Y0'] = Y_0
        self.data['Y1'] = Y_1
        
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
