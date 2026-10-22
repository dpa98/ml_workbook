from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_knn(X_train_std, y_train, n_neighbors=5, p=2):
    """
    Entrena un clasificador de k-vecinos más cercanos.
    p=2 corresponde a distancia euclidiana; p=1 a Manhattan.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
    knn.fit(X_train_std, y_train)
    return knn

def evaluate_model(model, X_test_std, y_test):
    """Evalúa el modelo y devuelve errores y precisión."""
    y_pred = model.predict(X_test_std)
    misclassified = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    return misclassified, accuracy