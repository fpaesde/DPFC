import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from numpy.random import binomial, multivariate_normal, normal, uniform

def mu_0_function(X, G):
    # Baseline healing probability without treatment
    base_prob = 0.15  # Very low baseline probability without treatment
    
    # Major factors affecting baseline outcome
    age_effect = -0.05 * np.tanh((X[:, 0] - 65) / 30)
    liver_effect = 0.05 * np.sin(X[:, 2] * np.pi / 10)
    vitamin_k_effect = 0.05 * np.log1p(X[:, 3])
    med_interaction = -0.02 * X[:, 4]
    inr_effect = -0.05 * np.square(X[:, 6] - 2.5)
    
    # Combine effects
    total_effect = base_prob + age_effect + liver_effect + vitamin_k_effect + med_interaction + inr_effect

    gene_effect = 0.4 * G
    
    return np.clip(total_effect + gene_effect, 0.05, 0.95)

def mu_1_function(X, G):
    # Treatment effect with warfarin
    base_prob = 0.95  # Very high baseline probability with treatment
    
    # Major factors affecting treatment outcome
    age_effect = -0.05 * np.tanh((X[:, 0] - 65) / 30)
    weight_effect = -0.05 * np.square((X[:, 1] - 75) / 50)
    liver_effect = 0.05 * np.sin(X[:, 2] * np.pi / 10)
    vitamin_k_effect = -0.05 * np.log1p(X[:, 3])
    med_interaction = -0.05 * X[:, 4]
    bp_effect = -0.05 * np.square((X[:, 5] - 120) / 50)
    inr_effect = -0.05 * np.square(X[:, 6] - 2.5)
    
    # Combine effects
    total_effect = base_prob + age_effect + weight_effect + liver_effect + \
                  vitamin_k_effect + med_interaction + bp_effect + inr_effect
    
    # Gene effect (extremely strong negative impact)
    gene_impact = 0.5 * G  # Massive reduction in effectiveness when G=1
    
    return np.clip(total_effect * (1 - gene_impact), 0.05, 0.95)

def get_propensity_score(X, A):
    # Linear combination of factors
    z = (
        -2 * A +  # Strong negative effect for Asian ethnicity
        -0.1 * (X[:, 0] - 65)/30 +  # Age
        0.1 * (X[:, 1] - 75)/15 +   # Weight
        0.1 * X[:, 2]/10 +         # Liver function
        -0.1 * X[:, 4]/10 +         # Medication count
        0.1 * (X[:, 5] - 120)/50    # Blood pressure
    )
    
    # Convert to probability using sigmoid function
    return 1 / (1 + np.exp(-z))

class WarfarinDataset():
    def __init__(self):
        super().__init__()
        self.data = None

    def load(self, n_samples=10000, seed=123) -> pd.DataFrame:
        """
        Load the dataset and return it as a pandas DataFrame with expanded features.
        
        Args:
            n_samples (int): Number of samples to generate.
            seed (int): Random seed for reproducibility.
        
        Returns:
            pd.DataFrame: The loaded dataset with additional correlated features.
        """
        self.n_samples = n_samples
        np.random.seed(seed)

        # Generate attribute feature A
        A = np.random.binomial(1, 0.25, n_samples)

        # Generate hidden feature G based on A (Rule 1)
        G = np.zeros(n_samples)
        for i in range(n_samples):
            if A[i] == 1:
                G[i] = np.random.binomial(1, 0.8)

            else:
                G[i] = np.random.binomial(1, 0.15)

        X = np.zeros((n_samples, 18))

        # X1: Age (years) - affects metabolism
        X[:, 0] = np.random.normal(65, 15, n_samples)
        X[:, 0] = np.clip(X[:, 0], 18, 95)

        # X2: Weight (kg) - affects dosing
        X[:, 1] = np.random.normal(75, 15, n_samples)
        X[:, 1] = np.clip(X[:, 1], 45, 120)

        # X3: Liver function score (0-10)
        X[:, 2] = np.random.normal(7, 2, n_samples)
        X[:, 2] = np.clip(X[:, 2], 0, 10)

        # X4: Vitamin K intake score (0-10)
        X[:, 3] = np.random.normal(5, 2, n_samples)
        X[:, 3] = np.clip(X[:, 3], 0, 10)

        # X5: Concurrent medications count (0-10)
        X[:, 4] = np.random.poisson(3, n_samples)
        X[:, 4] = np.clip(X[:, 4], 0, 10)

        # X6: Systolic blood pressure
        X[:, 5] = np.random.normal(130, 20, n_samples)
        X[:, 5] = np.clip(X[:, 5], 90, 180)

        # X7: International Normalized Ratio (INR) baseline
        X[:, 6] = np.random.normal(2.5, 0.5, n_samples)
        X[:, 6] = np.clip(X[:, 6], 1, 5)

        # X8-X18: Other clinical factors (less important)
        X[:, 7:] = np.random.normal(0, 1, (n_samples, 11))

        mu_0 = mu_0_function(X, G)
        mu_1 = mu_1_function(X, G)
        e_x = get_propensity_score(X, A)

        W = np.random.binomial(1, e_x)

        #### 7. Final outcome
        eps_0 = np.random.normal(0, 0.1, n_samples)  # Reduced noise for more deterministic outcomes
        eps_1 = np.random.normal(0, 0.1, n_samples)

        Y0 = (mu_0 + eps_0 > 0.5).astype(int)
        Y1 = (mu_1 + eps_1 > 0.5).astype(int)
        Y = W * Y1 + (1 - W) * Y0        

        # Create DataFrame
        self.data = pd.DataFrame(X, columns=[f'X{i}' for i in range(18)])
        self.data['A'] = A
        self.data['D'] = W
        self.data['Y'] = Y
        self.data['Y0'] = Y0
        self.data['Y1'] = Y1
        
        return self.data
    

    def preprocess_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the loaded data and return 5 pandas DataFrames.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                A tuple containing 5 preprocessed DataFrames (X, D, Y, Y0, Y1).
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() method first.")
        
        # Standardize features
        scaler = StandardScaler()
        self.data[[f'X{i}' for i in range(18)]] = scaler.fit_transform(self.data[[f'X{i}' for i in range(18)]])
        self.true_outcomes = pd.DataFrame({
            'Y0': self.data['Y0'],
            'Y1': self.data['Y1']
        })
        self.data = self.data.drop(columns=['Y0', 'Y1'])
        
        return self.data, self.true_outcomes
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded data. If data hasn't been loaded yet, load it.
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if self.data is None:
            self.data = self.load()
        return self.data
    

