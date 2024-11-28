from fairgbm import FairGBMClassifier
import numpy as np

from scipy import stats
from sklearn import clone
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm


class SLearner(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        base_estimator: BaseEstimator,
        random_state: int = None
    ):
        
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.model_ = base_estimator
            
    def fit(self, train_data, decision_col, outcome_col, protected_attribute = 'A'):
        """
        Fit separate models for treatment and control groups.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix
        """
        self.outcome_col = outcome_col
        self.decision_col = decision_col
        self.train = train_data
        
        Y = np.asarray(train_data[outcome_col])
        A = np.asarray(train_data[protected_attribute])
        X = train_data.drop([outcome_col], axis = 1)
        X = np.asarray(X)

        if isinstance(self.model_, FairGBMClassifier):
            self.model_.fit(X, Y, constraint_group = A)
        else:
            self.model_.fit(X, Y)

        return self
    
    def effect(self, X):
        """
        Predict potential outcomes Y(0) and Y(1).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features - 1)
            The feature matrix
            
        Returns
        -------
        array-like of shape (n_samples, 2)
            Predicted potential outcomes [Y(0), Y(1)]
        """

        X_augmented_0 = np.column_stack([X, np.zeros(X.shape[0])])
        X_augmented_1 = np.column_stack([X, np.ones(X.shape[0])])
        
        if hasattr(self.model_, "predict_proba"):
            y0 = self.model_.predict_proba(X_augmented_0)[:, 1]
            y1 = self.model_.predict_proba(X_augmented_1)[:, 1]
        else:
            y0 = self.model_.predict(X_augmented_0)
            y1 = self.model_.predict(X_augmented_1)
            
        return np.column_stack([y0, y1])
    
    def cate(self, X):
        potential_outcomes = self.effect(X)
        return potential_outcomes[:, 1] - potential_outcomes[:, 0]
    
    def bootstrap_confidence_intervals(self, X, n_bootstraps=1000, alpha=0.05):

        n_samples = X.shape[0]
        preds = np.zeros((n_bootstraps, n_samples, 2))
        
        for i in tqdm(range(n_bootstraps)):
            X_resampled= resample(self.train)
            
            model_boot = SLearner(base_estimator=self.base_estimator)
            model_boot.fit(X_resampled,
                           self.decision_col,
                           self.outcome_col)
                
            preds[i] = model_boot.effect(X)

        model_boot = SLearner(base_estimator=self.base_estimator)
        model_boot.fit(self.train,
                        self.decision_col,
                        self.outcome_col)
    
        estimates = model_boot.effect(X)
        sigma = np.std(preds, axis=0)
        z_alpha = stats.norm.ppf(1 - alpha/2)

        lower_bound = estimates - abs(z_alpha*sigma)
        upper_bound = estimates + abs(z_alpha*sigma)
        
        return lower_bound, upper_bound
    
