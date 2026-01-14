import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor
from sklearn.utils.validation import check_is_fitted, validate_data
from joblib import Parallel, delayed


def _fit_model(estimator, X, y):
    return estimator.fit(X, y)


class ElectricSequentialRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator1=None, estimator2=None, pivot_col_index=-1, n_jobs=None):
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        self.pivot_col_index = pivot_col_index
        self.n_jobs = n_jobs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        return tags

    def fit(self, X, y):
        assert self.estimator1 is not None and self.estimator2 is not None, 'Must specify an estimator'
        assert is_regressor(self.estimator1) and is_regressor(self.estimator2), 'Estimator must be a regressor'
        assert self.pivot_col_index is not None and isinstance(self.pivot_col_index, int), 'pivot_col_index must be an integer'
        assert self.n_jobs is None or isinstance(self.n_jobs, int), 'n_jobs must be an integer'

        X, y = validate_data(self, X, y, multi_output=True)
        assert y.ndim >= 2 and y.shape[1] >= 2, 'y must have at least 2 columns'
        self.n_outputs_ = y.shape[1]

        idx = self.pivot_col_index if self.pivot_col_index >= 0 else y.shape[1] + self.pivot_col_index
        self.pivot_real_col_index_ = idx
        y1 = y[:, idx].reshape(-1, 1)
        mask = np.ones(y.shape[1], dtype=bool)
        mask[idx] = False
        y2 = y[:, mask]
        if y2.shape[1] == 1:
            y2 = y2.ravel()

        models = [clone(self.estimator1), clone(self.estimator2)]
        parallel_results = Parallel(n_jobs=self.n_jobs)(
            [
                delayed(_fit_model)(models[0], X, y1.ravel()),
                delayed(_fit_model)(models[1], np.hstack([X, y1]), y2),
            ]
        )
        self.model1_, self.model2_ = parallel_results

        return self

    def predict(self, X):
        check_is_fitted(self, ['model1_', 'model2_'])

        in_data1 = validate_data(self, X, reset=False)
        predictions1 = self.model1_.predict(in_data1).reshape(-1, 1)
        in_data2 = np.hstack([in_data1, predictions1])
        predictions2 = self.model2_.predict(in_data2)
        if predictions2.ndim == 1:
            predictions2 = predictions2.reshape(-1, 1)

        n_samples = X.shape[0]
        predictions = np.empty((n_samples, self.n_outputs_), dtype=predictions1.dtype)
        predictions[:, self.pivot_real_col_index_] = predictions1.ravel()
        mask = np.ones(self.n_outputs_, dtype=bool)
        mask[self.pivot_real_col_index_] = False
        predictions[:, mask] = predictions2
        return predictions
