import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

class CausalForestEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.model_cf_ = None
        self.scaler_ = StandardScaler()
        self.model_y = None  # Outcome model

    def fit(self, train_data, decision_col, outcome_col, protected_attribute):
        """
        Fit the Causal Forest model using the EconML library.
        """

        self.outcome_col = outcome_col
        self.decision_col = decision_col
        self.protected_attribute = protected_attribute
        self.train = train_data

        Y = np.asarray(train_data[outcome_col])
        D = np.asarray(train_data[decision_col])
        A = np.asarray(train_data[protected_attribute])
        X = train_data.drop([decision_col, outcome_col], axis = 1)
        X = np.asarray(X)

        self.model_y = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        model_t = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=self.random_state)

        self.model_cf_ = CausalForestDML(
            model_y=self.model_y,
            model_t=model_t,
            discrete_treatment=True,
            n_estimators=100,
            random_state=self.random_state
        )

        self.model_cf_.fit(Y, D, X=X)

        control_idx = (D == 0)
        self.model_y_c = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.model_y_c.fit(X[control_idx], Y[control_idx])

        return self

    def effect(self, X):
        """
        Estimate the potential outcomes for the covariates X.

        Parameters:
            X (array-like): Covariate matrix.

        Returns:
            np.ndarray: Array with two columns [y0, y1] representing
                        the predicted probabilities under control and treatment.
        """
        X = np.asarray(X)

        te = self.model_cf_.effect(X)

        y0 = self.model_y_c.predict(X)

        y1 = y0 + te

        y0 = np.clip(y0, 0, 1)
        y1 = np.clip(y1, 0, 1)

        return np.column_stack([y0, y1])
    
    def cate(self, X):
        X = np.asarray(X)

        te = self.model_cf_.effect(X)
        return te

    def predict(self, X, D):
        """
        Predict the potential outcome for given treatment assignments D.

        Parameters:
            X (array-like): Covariate matrix.
            D (array-like): Treatment assignments (0 or 1).

        Returns:
            np.ndarray: Predicted outcomes corresponding to the treatment assignments.
        """
        outcomes = self.effect(X)
        return outcomes[np.arange(len(D)), D.astype(int)]
    
    def bootstrap_confidence_intervals(self, X, n_bootstraps=1000, alpha=0.05):
        n_samples = X.shape[0]
        preds = np.zeros((n_bootstraps, n_samples, 2))
        
        for i in range(n_bootstraps):
            bootstrap_data = resample(self.train)
            
            model_boot = CausalForestEstimator()
            
            model_boot.fit(
                train_data=bootstrap_data,
                decision_col=self.decision_col,
                outcome_col=self.outcome_col,
                protected_attribute=self.protected_attribute
            )
            
            preds[i] = model_boot.effect(X)

        lower_bound = np.percentile(preds, 100 * (alpha / 2), axis=0)
        upper_bound = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
        
        return lower_bound, upper_bound