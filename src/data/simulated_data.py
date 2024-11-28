import numpy as np
import pandas as pd


class SimpleSimulatedDataset():
    def __init__(self):
        self.data = None
        
    def load(self, n_samples=1000, d=20, propensity_score = 0.1, random_state=123) -> pd.DataFrame:
        """
        Create a simplified simulated dataset
        """

        self.d = d
        np.random.seed(random_state)
        
        Sigma = np.eye(d)
        X = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n_samples)
        A = np.random.binomial(n=1, p=0.2, size=n_samples)
        
        eps_0 = np.random.normal(0, 0.1, n_samples)
        eps_1 = np.random.normal(0, 0.1, n_samples)

        beta_scale = 1 / np.sqrt(d)
        beta = np.random.uniform(-beta_scale, beta_scale, d)
        z = np.dot(X, beta)
        print("New")

        mu_0 = z + 0.5 * (X[:, 0] > 0) - 0.5 * (X[:, 1] <= 0)
        mu_1 = z + 0.9 * ((X[:, 0] > 0) | (X[:, 1] > 0)) + 0.3

        Y_0 = (mu_0 + eps_0 > 0).astype(int)
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