import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_and_split_iris
from preprocessing import standardize_data
from model import train_random_forest, evaluate_model
from visualization import plot_decision_regions

def parse_args():
    """Configura la interfaz de línea de comandos (CLI) para probar hiperparámetros rápidamente."""
    parser = argparse.ArgumentParser(description="Flujo de trabajo modular para Bosques Aleatorios")
    parser.add_argument('--n_estimators', type=int, default=25,
                        help='Número de árboles del bosque (ej. 10, 100, 500). '
                             'Por defecto es 25.')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Profundidad máxima de cada árbol (ej. 2, 3, 4). '
                             'Por defecto None (crece sin límite).')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proporción del conjunto de prueba. Por defecto es 0.3.')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"--- Ejecutando Bosque Aleatorio (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, test_size={args.test_size}) ---")

    X_train, X_test, y_train, y_test = load_and_split_iris(test_size=args.test_size)
    X_train_std, X_test_std, scaler = standardize_data(X_train, X_test)

    forest = train_random_forest(X_train_std, y_train,
                                 n_estimators=args.n_estimators,
                                 max_depth=args.max_depth)

    misclassified, accuracy = evaluate_model(forest, X_test_std, y_test)
    print(f'\nResultados:')
    print(f' -> Ejemplos mal clasificados: {misclassified}')
    print(f' -> Precisión (Accuracy): {accuracy:.3f}')

    print('\nImportancia de características:')
    for i, imp in enumerate(forest.feature_importances_):
        print(f'  Feature {i}: {imp:.3f}')

    print("\nGenerando visualización...")
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # Aplicando una estética visual minimalista clara (fondo crema/blanco oscuro)
    fig = plt.figure(facecolor='#FDFBF7')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#FDFBF7')

    test_idx_start = len(y_train)
    plot_decision_regions(X=X_combined_std,
                          y=y_combined,
                          classifier=forest,
                          test_idx=range(test_idx_start, len(y_combined)))

    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig('random_forest_regions_output.png', facecolor=fig.get_facecolor())
    print("¡Gráfico guardado como 'random_forest_regions_output.png'!")

if __name__ == '__main__':
    main()