import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_and_split_iris(test_size=0.3, random_state=1):
    """Carga el dataset de Iris y lo divide en entrenamiento y prueba."""
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test