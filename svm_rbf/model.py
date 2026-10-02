from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm_rbf(X_train_std, y_train, C=1.0, gamma=0.1, random_state=1):
    """
    Entrena una Máquina de Vectores de Soporte con kernel RBF.
    C es el parámetro de regularización (valores altos = margen más estricto).
    gamma es el coeficiente del kernel RBF: un valor pequeño (p. ej. 0.1)
    produce una frontera de decisión más suave; valores grandes tienden
    a sobreajustar (frontera muy irregular).
    """
    svm = SVC(C=C, kernel='rbf', gamma=gamma, random_state=random_state)
    svm.fit(X_train_std, y_train)
    return svm

def evaluate_model(model, X_test_std, y_test):
    """Evalúa el modelo y devuelve errores y precisión."""
    y_pred = model.predict(X_test_std)
    misclassified = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    return misclassified, accuracy