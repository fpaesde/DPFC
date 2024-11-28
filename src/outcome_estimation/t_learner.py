from fairgbm import FairGBMClassifier
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import copy

from sklearn.utils import resample
from tqdm import tqdm

class TLearner(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self,
        base_estimator: BaseEstimator,
        random_state: int = None
    ):
        self.base_estimator = base_estimator
        self.random_state = random_state

        try:
            self.model_0_ = clone(base_estimator)
            self.model_1_ = clone(base_estimator)
        except:
            self.model_0_ = copy.deepcopy(base_estimator)
            self.model_1_ = copy.deepcopy(base_estimator)
            
    def fit(self, train_data, decision_col, outcome_col, protected_attribute):
        """
        Fit separate models for treatment and control groups.
        
        Parameters
        ----------
        Y : array-like of shape (n_samples,)
            The observed outcomes
        D : array-like of shape (n_samples,)
            The treatment assignments (0 or 1)
        X : array-like of shape (n_samples, n_features)
            The feature matrix
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
        
        control_idx = (D == 0)
        treated_idx = (D == 1)
        
        X_0 = X[control_idx]
        Y_0 = Y[control_idx]
        A_0 = A[control_idx]
        X_1 = X[treated_idx]
        Y_1 = Y[treated_idx]
        A_1 = A[treated_idx]
        
        # Fit models for control and treated groups
        if len(Y_0) > 0:
            if isinstance(self.model_0_, FairGBMClassifier):
                self.model_0_.fit(X_0, Y_0, constraint_group = A_0)
            else:
                self.model_0_.fit(X_0, Y_0)
        if len(Y_1) > 0:
            if isinstance(self.model_1_, FairGBMClassifier):
                self.model_1_.fit(X_1, Y_1, constraint_group = A_1)
            else:
                self.model_1_.fit(X_1, Y_1)
            
        return self
    
    def effect(self, X):
        """
        Predict potential outcomes Y(0) and Y(1).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix
            
        Returns
        -------
        array-like of shape (n_samples, 2)
            Predicted potential outcomes [Y(0), Y(1)]
        """

        X = np.asarray(X)
        
        if hasattr(self.model_0_, "predict_proba") and hasattr(self.model_1_, "predict_proba"):
            y0 = self.model_0_.predict_proba(X)[:, 1]
            y1 = self.model_1_.predict_proba(X)[:, 1]
        else:
            y0 = self.model_0_.predict(X)
            y1 = self.model_1_.predict(X)
            
        return np.column_stack([y0, y1])
    
    def cate(self, X):
        """
        Predict CATE from the outcome values Y(1) and Y(0)

        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix
            
        Returns
        -------
        array-like of shape (n_samples, 2)
            Predicted potential outcomes [Y(0), Y(1)]
        """

        potential_outcomes = self.effect(X)
        return potential_outcomes[:, 1] - potential_outcomes[:, 0]
    
    def predict(self, X, D):
        """
        Predict outcomes for specific treatment assignments.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix
        D : array-like of shape (n_samples,)
            The treatment assignments
            
        Returns
        -------
        array-like of shape (n_samples,)
            Predicted outcomes
        """
        outcomes = self.effect(X)
        return outcomes[np.arange(len(D)), D.astype(int)]
    
    def bootstrap_confidence_intervals(self, X, n_bootstraps=1000, alpha=0.05):
        n_samples = X.shape[0]
        preds = np.zeros((n_bootstraps, n_samples, 2))
        
        for i in tqdm(range(n_bootstraps)):
            bootstrap_data = resample(self.train)
            
            model_boot = TLearner(
                base_estimator=self.base_estimator,
                random_state=self.random_state
            )
            
            model_boot.fit(
                train_data=bootstrap_data,
                decision_col=self.decision_col,
                outcome_col=self.outcome_col,
                protected_attribute=self.protected_attribute
            )
            
            # Get predictions for this bootstrap sample
            preds[i] = model_boot.effect(X)

        model_boot.fit(self.train,
                           self.decision_col,
                           self.outcome_col,
                           self.protected_attribute)
    
        estimates = model_boot.effect(X)
        sigma = np.std(preds, axis=0)
        z_alpha = stats.norm.ppf(1 - alpha/2)

        lower_bound = estimates - abs(z_alpha*sigma)
        upper_bound = estimates + abs(z_alpha*sigma)
        
        return lower_bound, upper_bound