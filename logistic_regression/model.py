from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_regression(X_train_std, y_train, C=100, solver='lbfgs', random_state=1):
    """
    Entrena un modelo de Regresión Logística.
    C es el parámetro de regularización inversa (valores más bajos = mayor regularización).
    """
    lr = LogisticRegression(C=C, solver=solver, random_state=random_state)
    lr.fit(X_train_std, y_train)
    return lr

def evaluate_model(model, X_test_std, y_test):
    """Evalúa el modelo y devuelve errores y precisión."""
    y_pred = model.predict(X_test_std)
    misclassified = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    return misclassified, accuracy
