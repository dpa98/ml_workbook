from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X_train_std, y_train, C=1.0, kernel='rbf', random_state=1):
    """
    Entrena una Máquina de Vectores de Soporte (SVC).
    C es el parámetro de regularización (valores altos = margen más estricto).
    kernel: 'linear', 'rbf', 'poly' o 'sigmoid'.
    """
    svm = SVC(C=C, kernel=kernel, random_state=random_state)
    svm.fit(X_train_std, y_train)
    return svm

def evaluate_model(model, X_test_std, y_test):
    """Evalúa el modelo y devuelve errores y precisión."""
    y_pred = model.predict(X_test_std)
    misclassified = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    return misclassified, accuracy