import numpy as np
import pandas as pd
from typing import List


class Dataset:

    def _init_(self, X:np.ndarray, y: np.ndarray = None, features: List[str] =  None,
               label: List[str] =  None) -> None:
        """
        Parameters

        :param X: matrix/table of features (independent variables)
        :param y: vector of the dependent variable
        :param features: vector of feature names
        :param label: name of the dependent variable
        :return:
        """
        if X is None:
            raise ValueError ('x must be defined')

        if label is None:
            label = 'y'

        if y is not None and X.shape[0] != len(y):
            raise ValueError('The number of labels must be the same as the number of samples')

        if features is None:
            features = [f"feat_{i}" for i in range(X.shape[0])]


        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        """
        shape of X
        Returns
        _______
        returns the shape of X.

        """
        return self.X.shape

    def has_label(self):
        if self.y is None:
            return False

    def get_classes(self):
        return np.unique(self.y)

    def get_mean(self):
        means = np.nanmean(self.X, axis=0) #coluna é o e linha é 1

    def get_variance(self):
        return np.nanvar(self.X, axis=0)

    def get_median(self):
        return np.nanmedian(self.X, axis=0)

    def get_min(self):
        return np.nanmin(self.X, axis=0)

    def get_max(self):
        return np.nanmax(self.X, axis=0)

    def summary(self):
        metrics = {"mean": self.get_mean(),
                   "media": self.get_median(),
                   "var": self.get_variance(),
                   "max": self.get_max(),
                   "min": self.get_min()}

        data = pd.DataFrame.from_dict(metrics, orient="index")
        return data

    def dropna(self):
        #Procurar os índices das linhas (amostras) que contêm valores NaN em qualquer caraterística
        nan_indices = np.isnan(self.X).any(axis=1)

       # Remover as linhas com valores NaN da matriz de características (X) e atualizar o vetor y
        self.X = self.X[~nan_indices]
        self.y = self.y[~nan_indices]

    def fillna(self, value="mean"):
        if value == "mean":
            feature_means = np.nanmean(self.X, axis=0)
            self.X[np.isnan(self.X)] = feature_means[np.isnan(self.X)]
        elif value == "median":
            feature_medians = np.nanmedian(self.X, axis=0)
            self.X[np.isnan(self.X)] = feature_medians[np.isnan(self.X)]
        else:
            self.X[np.isnan(self.X)] = value

    def remove_by_index(self, index):
        if index < 0 or index >= len(self.X):
            raise ValueError("Index invalido. O índice deve estar dentro do intervalo de amostras disponíveis")

        self.X = np.delete(self.X, index, axis=0)
        self.y = np.delete(self.y, index)

if __name__ == '__main__':
    X = np.array([[1,2,3], [4,5,6]])
    y = np.array([1,0])
    data = Dataset(X=X, y=y)
    metrics = data.summary()
    print(metrics)