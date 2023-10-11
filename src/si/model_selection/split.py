from typing import Tuple
import numpy as np
from si.Data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset, test_size, random_state=None):
    """
        Split a Dataset object into stratified training and testing datasets.

        Parameters:
        - dataset (Dataset): The Dataset object to split.
        - test_size (float): The size of the testing Dataset (e.g., 0.2 for 20%).
        - random_state (int): Seed for generating permutations. (optional)

        Returns:
        - Tuple: A tuple containing the stratified train and test Dataset objects.
        """
    if not 0 < test_size < 1:
        raise ValueError("test_size should be a float between 0 and 1.")

    if random_state is not None:
        np.random.seed(random_state)

    unique_labels, label_counts = np.unique(dataset.y, return_counts=True)
    train_indices, test_indices = [], []

    for label in unique_labels:
        label_indices = np.where(dataset.y == label)[0]
        num_test_samples = int(label_counts[label] * test_size)

        shuffled_indices = np.random.permutation(label_indices)
        test_indices.extend(shuffled_indices[:num_test_samples])
        train_indices.extend(shuffled_indices[num_test_samples:])

    train_dataset = Dataset(X=dataset.X[train_indices], y=dataset.y[train_indices])
    test_dataset = Dataset(X=dataset.X[test_indices], y=dataset.y[test_indices])

    return train_dataset, test_dataset
