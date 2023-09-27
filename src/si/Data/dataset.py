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
        """
        Verifica se o dataset é supervisionado (tem labels (vetor y)) ou não supervisionado (não tem labels).

        return: Booleano: True se tem label; False se não tem label.
        """
        if self.y is None:
            return False

    def get_classes(self):
        """
        Retorna as classes do dataset.

        return: Lista com os valores únicos do dataset.
        """
        return np.unique(self.y)

    def get_mean(self):
        """
        Calcula a média de cada feature.

        return: Array com as médias das features.
        """
        means = np.nanmean(self.X, axis=0) #coluna é 0 e linha é 1

    def get_variance(self):

        """
        Calcula a variância de cada feature.

        return: Array com as variâncias das features.
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self):
        """
        Calcula a mediana de cada feature.

        return: Array com as medianas das features.
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self):
        """
        Calcula o mínimo.

        return: Lista com os valores mínimos das features.
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self):
        """
        Calcula o máximo de cada feature.

        return: Lista com os valores máximos das features.
        """
        return np.nanmax(self.X, axis=0)

    def summary(self):
        """
        Cria um dataframe com os valores do summary (média, mediana, variância, mínimo e máximo) das features.

        return: Pandas dataframe com o summary das features.
        """
        metrics = {"mean": self.get_mean(),
                   "media": self.get_median(),
                   "var": self.get_variance(),
                   "max": self.get_max(),
                   "min": self.get_min()}

        data = pd.DataFrame.from_dict(metrics, orient="index")
        return data

    def dropna(self):

        """
        Remove observações que contenham pelo menos um valor nulo (NaN).
        """
        #Procurar os índices das linhas (amostras) que contêm valores NaN em qualquer caraterística
        nan_indices = np.isnan(self.X).any(axis=1)

       # Remover as linhas com valores NaN da matriz de características (X) e atualizar o vetor y
        self.X = self.X[~nan_indices]
        self.y = self.y[~nan_indices]

    def fillna(self, value="mean"):
        """
        Substitui os valores nulos.

        param value: float or "mean" or "median"
        """

        if value == "mean":
            feature_means = np.nanmean(self.X, axis=0)
            self.X[np.isnan(self.X)] = feature_means[np.isnan(self.X)]
        elif value == "median":
            feature_medians = np.nanmedian(self.X, axis=0)
            self.X[np.isnan(self.X)] = feature_medians[np.isnan(self.X)]
        else:
            self.X[np.isnan(self.X)] = value

    def remove_by_index(self, index):
        """
            Remove uma amostra do conjunto de dados com base no seu índice.

            Parâmetros:
            ----------
            index : int
                O índice da amostra a ser removida. Deve estar dentro do intervalo [0, len(self.X)).

            return
            ------
            ValueError : se o índice estiver fora do intervalo válido.
        """
        if index < 0 or index >= len(self.X):
            raise ValueError("Index invalido. O índice deve estar dentro do intervalo de amostras disponíveis")

        self.X = np.delete(self.X, index, axis=0)
        self.y = np.delete(self.y, index)


if __name__ == '__main__':
    x = np.array([[1, 2, 3],
                  [4, 5, 6]])
    y = np.array([1, 2])
    features = ['A', 'B', 'C']
    label = 'y'
    dataset = Dataset(X=x, y=y, features=features, label=label) # S
    dataset_naosuperv = Dataset(X=x, y=None, features=features, label=label) # NS
    print("[S] Shape: ", dataset.shape())
    print("[S] É supervisionado: ", dataset.has_label())
    print("[NS] É supervisionado: ", dataset_naosuperv.has_label())
    print("[S] Classes: ", dataset.get_classes())
    # print(dataset.get_mean())
    # print(dataset.get_variance())
    # print(dataset.get_median())
    # print(dataset.get_min())
    # print(dataset.get_max())
    print("[S] Summary:\n", dataset.summary())
