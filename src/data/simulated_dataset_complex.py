import numpy as np
import pandas as pd


class SimulatedDatasetComplex():
    def __init__(self):
        self.data = None
        
    def generate_correlation_matrix(self, d=3):
        return np.eye(d)
        
    def load(self, n_samples=1000, d=3, propensity_score = 0.5, random_state=123) -> pd.DataFrame:
        """
        Create a simplified linear simulated dataset with binary outcomes:
        - e(x) = 0.5 (constant treatment probability)
        - Y0 = 1 if xᵀβ0 > 0, 0 otherwise
        - Y1 = 1 if xᵀβ1 > 0, 0 otherwise
        where β0 ~ Unif[-1/√d, 1/√d]³ and β1 ~ Unif[-1/√d, 1/√d]³
        - A: binary feature with correlation-dependent impact
        """
            
        np.random.seed(random_state)
        self.d = d
        
        # Generate feature matrix X
        Sigma = self.generate_correlation_matrix(d)
        X = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n_samples)

        # Generate binary feature A
        A = np.random.binomial(n=1, p=0.2, size=n_samples)

        # Generate error terms
        eps_0 = np.random.normal(0, 0.1, n_samples)
        eps_1 = np.random.normal(0, 0.1, n_samples)
        
        # Generate β coefficients
        beta_scale = 1 / np.sqrt(d)
        beta_0 = np.random.uniform(-beta_scale, beta_scale, d)
        beta_1 = np.random.uniform(-beta_scale, beta_scale, d)
        
        # Generate binary potential outcomes with A's impact
        mu_0 = np.dot(X, beta_0)
        Y_0 = (mu_0 + eps_0 > 0).astype(int)
        
        mu_1 = np.dot(X, beta_1)
        Y_1 = (mu_1 + eps_1 > 0).astype(int)
        
        # Generate treatment assignment
        e_x = np.full(n_samples, propensity_score)
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
        
        # Store the true parameters
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        
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
    
    def get_true_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, 'beta_0') or not hasattr(self, 'beta_1'):
            raise ValueError("Parameters not available. Call load() first.")
        return self.beta_0, self.beta_1