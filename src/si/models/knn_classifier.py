from typing import Union
import numpy as np
from src.si.Data.dataset import Dataset
from src.si.statistics import euclidean_distance
from typing import Callable

class KNNClassifier:
    def __int__(self: k, int = 1, distance: Callable = euclidean_distance):
        self.k = k
        self.distance =  distance
        self.dataset = None

    def fit(self, dataset: Dataset) -> KNNClassifier:
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        distances= self.distance(sample, self.dataset.X)
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        k_nearest_neighbors_labels = self.dataset[k_nearest_neighbors]
        label, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        return label(np.argmax(counts))

    def predict(self, dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_label, axis=1, Dataset)

    def score(self, dataset: Dataset) ->