from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_decision_tree(X_train_std, y_train, max_depth=None, criterion='gini', random_state=1):
    """
    Entrena un Árbol de Decisión.
    max_depth limita la profundidad del árbol (prevención de overfitting);
    None permite crecer sin límite.
    criterion: 'gini' o 'entropy' (criterio de división).
    """
    tree = DecisionTreeClassifier(max_depth=max_depth,
                                  criterion=criterion,
                                  random_state=random_state)
    tree.fit(X_train_std, y_train)
    return tree

def evaluate_model(model, X_test_std, y_test):
    """Evalúa el modelo y devuelve errores y precisión."""
    y_pred = model.predict(X_test_std)
    misclassified = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)
    return misclassified, accuracy