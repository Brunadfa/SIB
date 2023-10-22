import numpy as np


from si.Data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    """
        The RidgeRegressionLeastSquares is a linear model using L2 regularization with the Least Squares method.

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter (lambda)
        scale: bool
            Whether to scale the data or not

        Attributes
        ----------
        theta: np.array
            The model parameters, namely the coefficients of the linear model.
        theta_zero: float
            The model parameter, namely the intercept of the linear model.
        mean: np.array
            The mean of the dataset for each feature.
        std: np.array
            The standard deviation of the dataset for each feature.
        """

    def __init__(self, l2_penalty: float, scale: bool = True):  #alpha: float = 0.001
        """
        Initialize the RidgeRegressionLeastSquares model.

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter (lambda).
        scale: bool
            Whether to scale the data or not.
        """
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model.
        """
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = X.shape

        # Add an intercept term to X
        X = np.c_[np.ones(m), X]

        # Compute the penalty matrix
        penalty_matrix = self.l2_penalty * np.eye(n + 1)
        penalty_matrix[0, 0] = 0

        # Compute the model parameters using Ridge Regression with Least Squares
        X_transpose = X.T
        inverse_matrix = np.linalg.inv(X_transpose.dot(X) + penalty_matrix)
        self.theta = inverse_matrix.dot(X_transpose).dot(dataset.y)
        self.theta_zero = self.theta[0]
        self.theta = self.theta[1:]

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of.

        Returns
        -------
        predictions: np.array
            The predictions of the dataset.
        """
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, _ = X.shape

        # Add an intercept term to X
        X = np.c_[np.ones(m), X]

        # Compute the predicted Y
        predictions = X.dot(np.hstack([self.theta_zero, self.theta]))

        return predictions

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on.

        Returns
        -------
        mse: float
            The Mean Square Error of the model.
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)


# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares(alpha=2.0)   #nao percebi esta parte
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=2.0)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))