from typing import Literal

import numpy as np

from si.Data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """
       A class representing a Random Forest classifier.

       Parameters:
           n_estimators (int): The number of decision trees to use in the ensemble.
           max_features (int): The maximum number of features to use per tree. If None, it defaults to sqrt(n_features).
           min_sample_split (int): The minimum number of samples required to split an internal node.
           max_depth (int): The maximum depth of the decision trees in the ensemble.
           mode (Literal['gini', 'entropy']): The impurity calculation mode for information gain (either 'gini' or 'entropy').
           seed (int): The random seed to ensure reproducibility.

       Estimated Parameters:
           trees (list of tuples): List of decision trees and their respective features used for training.

       Methods:
           - fit(dataset: Dataset) -> RandomForestClassifier:
               Fits the Random Forest classifier to a given dataset.

           - predict(dataset: Dataset) -> np.ndarray:
               Predicts labels for a given dataset using the ensemble of decision trees.

           - score(dataset: Dataset) -> float:
               Computes the accuracy of the model's predictions on a dataset.
       """
    def __init__(self, n_estimators: int = 100, max_features: int = None, min_sample_split: int = 2,
                 max_depth: int = 10, mode: Literal['gini', 'entropy'] = 'gini', seed: int = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
                Fits the Random Forest classifier to a given dataset.

                Parameters:
                    dataset (Dataset): The dataset to fit the model to.

                Returns:
                    RandomForestClassifier: The fitted model.
                """
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.X.shape
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features

        for _ in range(self.n_estimators):
            # Create a bootstrap dataset
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            bootstrap_dataset = Dataset(X=dataset.X[sample_indices][:, feature_indices],
                                         y=dataset.y[sample_indices],
                                         features=dataset.features[feature_indices],
                                         label=dataset.label)

            # Create and train a decision tree
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split,
                                          max_depth=self.max_depth,
                                          mode=self.mode)
            tree.fit(bootstrap_dataset)

            # Append the features used and the trained tree
            self.trees.append((feature_indices, tree))

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
                Predicts labels for a given dataset using the ensemble of decision trees.

                Parameters:
                    dataset (Dataset): The dataset to make predictions for.

                Returns:
                    np.ndarray: The predicted labels.

                Note:
                    The predictions are obtained by aggregating the predictions of individual decision trees in the ensemble.
                """
        predictions = []
        for x in dataset.X:
            tree_predictions = []
            for feature_indices, tree in self.trees:
                x_subset = x[feature_indices]
                subset_dataset = Dataset(X=[x_subset], features=dataset.features, label=dataset.label)
                tree_predictions.append(tree.predict(subset_dataset))
            most_common_prediction = max(set(tree_predictions), key=tree_predictions.count)
            predictions.append(most_common_prediction)
        return np.array(predictions)

    def score(self, dataset: Dataset) -> float:
        """
                Computes the accuracy of the model's predictions on a dataset.

                Parameters:
                    dataset (Dataset): The dataset to calculate the accuracy on.

                Returns:
                    float: The accuracy of the model on the dataset.

                Note:
                    The accuracy is calculated by comparing the model's predictions with the true labels in the dataset.
                """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)
