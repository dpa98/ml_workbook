from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_random_forest(X_train_std, y_train, n_estimators=25, max_depth=None, random_state=1):
    """
    Entrena un Bosque Aleatorio (ensemble de árboles de decisión).
    n_estimators es la cantidad de árboles del bosque (más árboles = mayor
    estabilidad, a costa de más cómputo).
    max_depth limita la profundidad de cada árbol; None permite crecer sin límite.
    """
    forest = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=random_state,
                                    n_jobs=-1)
    forest.fit(X_train_std, y_train)
    return forest

def evaluate_model(model, X_test_std, y_test):
    """Evalúa el modelo y devuelve errores y precisión."""
    y_pred = model.predict(X_test_std)
    misclassified = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    return misclassified, accuracy
