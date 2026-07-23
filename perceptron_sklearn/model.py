from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def train_perceptron(X_train_std, y_train, eta0=0.1, random_state=1):
    """Entrena un modelo Perceptrón."""
    ppn = Perceptron(eta0=eta0, random_state=random_state)
    ppn.fit(X_train_std, y_train)
    return ppn

def evaluate_model(model, X_test_std, y_test):
    """Evalúa el modelo y devuelve errores y precisión."""
    y_pred = model.predict(X_test_std)
    misclassified = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    return misclassified, accuracy
