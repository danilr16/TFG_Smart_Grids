import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor
from sklearn.utils.validation import check_is_fitted, validate_data


class ElectricSequentialRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator=None, pivot_col_index=-1):
        assert estimator is not None, 'Must specify an estimator'
        assert is_regressor(estimator), 'Estimator must be a regressor'
        self.estimator = estimator
        assert pivot_col_index is not None and isinstance(pivot_col_index, int), 'pivot_col_index must be an integer'
        self.pivot_col_index = pivot_col_index

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        return tags

    def fit(self, X, y):
        X, y = validate_data(self, X, y, multi_output=True)
        assert y.ndim >= 2 and y.shape[1] >= 2, 'y must have at least 2 columns'
        self.n_outputs_ = y.shape[1]

        idx = self.pivot_col_index if self.pivot_col_index >= 0 else y.shape[1] + self.pivot_col_index
        self.pivot_real_col_index_ = idx
        y1 = y[:, idx].reshape(-1, 1)
        mask = np.ones(y.shape[1], dtype=bool)
        mask[idx] = False
        y2 = y[:, mask]

        self.model1_ = clone(self.estimator)
        self.model1_.fit(X, y1.ravel())
        self.model2_ = clone(self.estimator)
        self.model2_.fit(np.hstack([X, y1]), y2)

        return self

    def predict(self, X):
        check_is_fitted(self, ['model1_', 'model2_'])

        in_data1 = validate_data(self, X, reset=False)
        predictions1 = self.model1_.predict(in_data1).reshape(-1, 1)
        in_data2 = np.hstack([in_data1, predictions1])
        predictions2 = self.model2_.predict(in_data2)

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_outputs_))
        predictions[:, self.pivot_real_col_index_] = predictions1.ravel()
        mask = np.ones(self.n_outputs_, dtype=bool)
        mask[self.pivot_real_col_index_] = False
        predictions[:, mask] = predictions2
        return predictions
