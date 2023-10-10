import numpy as np

def accuracy(y_true: np.ndarray, y_pred:np.ndarray) -> float:
    """
       Método que calcula o valor do erro entre os valores reais e valores estimados de Y.

       :param y_true: valores/labels reais do dataset.
       :param y_pred: valores/labels estimados do dataset.

       :return: Valor do erro entre o y_true e o y_pred -> accuracy do modelo
       """
    return np.sum(y_true == y_pred) / len(y_true)


