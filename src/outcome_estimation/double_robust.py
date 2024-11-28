import copy
from fairgbm import FairGBMClassifier
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
from typing import Optional

from sklearn.utils import resample

class DRLearner(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        ps_estimator: Optional[BaseEstimator] = None,
        residual_model: Optional[BaseEstimator] = None,
        random_state: Optional[int] = None
    ):
        self.base_estimator = base_estimator
        self.residual_model = residual_model
        self.ps_estimator = ps_estimator
        self.residual_model = residual_model
        self.random_state = random_state

        # Use classifiers for outcome models
        if self.base_estimator is None:
            base_clf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            try:
                base_clf = clone(self.base_estimator)
            except:
                base_clf = copy.deepcopy(self.base_estimator)

        if self.ps_estimator is None:
            ps_clf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            try:
                ps_clf = clone(self.ps_estimator)
            except:
                ps_clf = copy.deepcopy(self.ps_estimator)

        
        if self.residual_model is None:
            res_clf = LGBMRegressor(
            n_estimators=100, random_state=self.random_state)
        else:
            try:
                res_clf = clone(self.residual_model)
            except:
                res_clf = copy.deepcopy(self.residual_model)
        try:
            self.model_t_ = clone(base_clf)
            self.model_c_ = clone(base_clf)
            self.residual_model0_ = clone(res_clf)
            self.residual_model1_ = clone(res_clf)
            self.ps_model_ = clone(ps_clf)
        except:
            self.model_t_ = copy.deepcopy(base_clf)
            self.model_c_ = copy.deepcopy(base_clf)
            self.residual_model0_ = copy.deepcopy(res_clf)
            self.residual_model1_ = copy.deepcopy(res_clf)
            self.ps_model_ = copy.deepcopy(ps_clf)            

    def fit(self, train_data, decision_col, outcome_col, protected_attribute):
        """
        Fit the DR Learner using the following steps:
        1. Fit propensity score model
        2. Fit outcome models for treated and control groups
        3. Compute residuals and fit residual models
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

        self.ps_model_.fit(X, D)
        ps = self.ps_model_.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.01, 0.99) 


        treated_idx = (D == 1)
        control_idx = (D == 0)

        if np.any(treated_idx):
            if isinstance(self.model_t_, FairGBMClassifier) and A is not None:
                self.model_t_.fit(X[treated_idx], Y[treated_idx], constraint_group=A[treated_idx])
            else:
                self.model_t_.fit(X[treated_idx], Y[treated_idx])

        if np.any(control_idx):
            if isinstance(self.model_c_, FairGBMClassifier) and A is not None:
                self.model_c_.fit(X[control_idx], Y[control_idx], constraint_group=A[control_idx])
            else:
                self.model_c_.fit(X[control_idx], Y[control_idx])

        mu0_obs, mu1_obs = self._get_outcome_predictions(X)
        w1 = D / ps 
        w0 = (1 - D) / (1 - ps) 
        residuals1 = w1 * (Y - mu1_obs)
        residuals0 = w0 * (Y - mu0_obs)
            
        if isinstance(self.residual_model1_, FairGBMClassifier) and A is not None:
            self.residual_model1_.fit(X, residuals1, constraint_group=A)
        else:
            self.residual_model1_.fit(X, residuals1)
            
        if isinstance(self.residual_model0_, FairGBMClassifier) and A is not None:
            self.residual_model0_.fit(X, residuals0, constraint_group=A)
        else:
            self.residual_model0_.fit(X, residuals0)

        # Store observed data
        self.Y_obs_ = Y
        self.D_obs_ = D
        self.X_obs_ = X
        self.ps_obs_ = ps

        return self

    def _get_outcome_predictions(self, X):
        mu1 = self.model_t_.predict_proba(X)[:, 1]
        mu0 = self.model_c_.predict_proba(X)[:, 1]
        return mu0, mu1

    def effect(self, X):
        X = np.asarray(X)

        mu0_new, mu1_new = self._get_outcome_predictions(X)

        correction1 = self.residual_model1_.predict(X)
        correction0 = self.residual_model0_.predict(X)

        y1 = mu1_new + correction1
        y0 = mu0_new + correction0

        y0 = np.clip(y0, 0, 1)
        y1 = np.clip(y1, 0, 1)

        return np.column_stack([y0, y1])
    
    def cate(self, X):

        X = np.asarray(X)
        
        mu0, mu1 = self._get_outcome_predictions(X)
        
        correction1 = self.residual_model1_.predict(X)
        correction0 = self.residual_model0_.predict(X)
        
        cate_estimates = (mu1 + correction1) - (mu0 + correction0)
        
        return cate_estimates
    
    def predict(self, X, D):
        """
        Predict the potential outcome for given treatment assignments D.
        """
        outcomes = self.effect(X)
        return outcomes[np.arange(len(D)), D.astype(int)]

    def bootstrap_confidence_intervals(self, X, n_bootstraps=1000, alpha=0.05):
        n_samples = X.shape[0]
        preds = np.zeros((n_bootstraps, n_samples, 2))
        
        for i in range(n_bootstraps):
            bootstrap_data = resample(self.train)
            
            model_boot = DRLearner(
                base_estimator=self.base_estimator,
                residual_model=self.residual_model,
                ps_estimator=self.ps_estimator,
                random_state=self.random_state
            )
            
            model_boot.fit(
                train_data=bootstrap_data,
                decision_col=self.decision_col,
                outcome_col=self.outcome_col,
                protected_attribute=self.protected_attribute
            )
            
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