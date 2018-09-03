from sklearn import linear_model
from scipy import stats
import numpy as np
import pandas as pd


class PLinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model parameters (betas).
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def fit(self, X, y, n_jobs=1):
        self = super(PLinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(
            X.shape[0] - X.shape[1]
        )
        self.se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])

        self.t = self.coef_ / self.se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))

        self.summary = pd.DataFrame()
        self.summary["coefficients"] = self.coef_
        self.summary["standard Errors"] = self.se.reshape(-1)
        self.summary["t statistic"] = self.t.reshape(-1)
        self.summary["p values"] = self.p.reshape(-1)

        return self

