import copy
from fairgbm import FairGBMClassifier
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample
from tqdm import tqdm

class XLearner(BaseEstimator, ClassifierMixin):
  
    def __init__(
        self,
        base_estimator: BaseEstimator,
        refined_estimator: BaseEstimator,
        ps_estimator: BaseEstimator,
        random_state: int = None
    ):
        self.base_estimator = base_estimator
        self.refined_estimator = refined_estimator
        self.ps_estimator = ps_estimator
        self.random_state = random_state
        
        self.model_t_ = clone(base_estimator)
        self.model_c_ = clone(base_estimator)

        self.model_t_refined_ = clone(refined_estimator)
        self.model_c_refined_ = clone(refined_estimator)
        self.propensity_model_ = clone(ps_estimator)

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
        
        # Split data into treatment and control groups
        treated_idx = (D == 1)
        control_idx = (D == 0)
        
        X_0 = X[control_idx]
        Y_0 = Y[control_idx]
        A_0 = A[control_idx]
        X_1 = X[treated_idx]
        Y_1 = Y[treated_idx]
        A_1 = A[treated_idx]
        
        # Stage 1: Base outcome models
        if len(Y_0) > 0:
            if isinstance(self.model_c_, FairGBMClassifier):
                self.model_c_.fit(X_0, Y_0, constraint_group=A_0)
            else:
                self.model_c_.fit(X_0, Y_0)
                
        if len(Y_1) > 0:
            if isinstance(self.model_t_, FairGBMClassifier):
                self.model_t_.fit(X_1, Y_1, constraint_group=A_1)
            else:
                self.model_t_.fit(X_1, Y_1)
        
        if hasattr(self.model_t_, "predict_proba"):
            imputed_control_t = self.model_c_.predict_proba(X_1)[:, 1]
            imputed_treat_c = self.model_t_.predict_proba(X_0)[:, 1]
        else:
            imputed_control_t = self.model_c_.predict(X_1)
            imputed_treat_c = self.model_t_.predict(X_0)

        if len(Y_0) > 0:
            if isinstance(self.model_c_refined_, FairGBMClassifier):
                self.model_c_refined_.fit(X_0, imputed_treat_c - Y_0 , constraint_group=A_0)
            else:
                self.model_c_refined_.fit(X_0, imputed_treat_c - Y_0)
            
        if len(Y_1) > 0:
            
            if isinstance(self.model_t_refined_, FairGBMClassifier):
                self.model_t_refined_.fit(X_1, Y_1 - imputed_control_t, constraint_group=A_1)
            else:
                self.model_t_refined_.fit(X_1, Y_1 - imputed_control_t)
            
        # Fit propensity score model
        self.propensity_model_.fit(X, D)
        
        return self
    
    def effect(self, X):
        X = np.asarray(X)
        cate = self.cate(X)
        
        # Get base predictions
        if hasattr(self.model_t_, "predict_proba"):
            y0 = self.model_c_.predict_proba(X)[:, 1]
        else:
            y0 = self.model_c_.predict(X)
            
        # Calculate y1 based on y0 and cate
        y1 = y0 + cate
        
        # Ensure predictions are in [0, 1]
        y0 = np.clip(y0, 0, 1)
        y1 = np.clip(y1, 0, 1)
        
        return np.column_stack([y0, y1])
    
    def cate(self, X):

        X = np.asarray(X)
        
        # Get propensity scores
        g = self.propensity_model_.predict(X)
        g = np.clip(g, 0.001, 0.999)

        tau_1 = self.model_t_refined_.predict(X)
        tau_0 = self.model_c_refined_.predict(X)
            
        cate = g * tau_1 + (1 - g) * tau_0
        
        return cate

    
    def predict(self, X, D):
        """
        Predict outcomes for specific treatment assignments.
        """
        outcomes = self.effect(X)
        return outcomes[np.arange(len(D)), D.astype(int)]
    
    def bootstrap_confidence_intervals(self, X, n_bootstraps=1000, alpha=0.05):
        n_samples = X.shape[0]
        preds = np.zeros((n_bootstraps, n_samples, 2))
        
        for i in tqdm(range(n_bootstraps)):
            bootstrap_data = resample(self.train)
            
            model_boot = XLearner(
                base_estimator=self.base_estimator,
                refined_estimator=self.refined_estimator,
                ps_estimator=self.ps_estimator,
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