from typing import Tuple

import numpy as np
from src.si.Data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size:float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    np.random.send(random_state)
    n_samples = dataset.samples()[0]
    n_test = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)
    test_samples =permutations[:n_test]
    train_samples = permutations[n_test:]
    train = Dataset(X=dataset.X[train_samples], y=dataset.y[train_samples], features=dataset.features, label= dataset.label)
    test = Dataset(X=dataset.X[test_samples], y=dataset.y[test_samples], features=dataset.features, label= dataset.label)

