import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_and_split_iris
from preprocessing import standardize_data
from model import train_svm_rbf, evaluate_model
from visualization import plot_decision_regions

def parse_args():
    """Configura la interfaz de línea de comandos (CLI) para probar hiperparámetros rápidamente."""
    parser = argparse.ArgumentParser(description="Flujo de trabajo modular para SVM con kernel RBF")
    parser.add_argument('--c_param', type=float, default=1.0,
                        help='Parámetro de regularización C (ej. 0.1, 1.0, 100.0). Por defecto es 1.0.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Coeficiente gamma del kernel RBF. Valor pequeño = frontera más suave. '
                             'Por defecto es 0.1.')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proporción del conjunto de prueba. Por defecto es 0.3.')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"--- Ejecutando SVM RBF (C={args.c_param}, gamma={args.gamma}, "
          f"test_size={args.test_size}) ---")

    X_train, X_test, y_train, y_test = load_and_split_iris(test_size=args.test_size)
    X_train_std, X_test_std, scaler = standardize_data(X_train, X_test)

    svm = train_svm_rbf(X_train_std, y_train, C=args.c_param, gamma=args.gamma)

    misclassified, accuracy = evaluate_model(svm, X_test_std, y_test)
    print(f'\nResultados:')
    print(f' -> Ejemplos mal clasificados: {misclassified}')
    print(f' -> Precisión (Accuracy): {accuracy:.3f}')

    print(f'\nNúmero de vectores de soporte: {len(svm.support_vectors_)}')

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
                          classifier=svm,
                          test_idx=range(test_idx_start, len(y_combined)))

    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig('svm_rbf_regions_output.png', facecolor=fig.get_facecolor())
    print("¡Gráfico guardado como 'svm_rbf_regions_output.png'!")

if __name__ == '__main__':
    main()