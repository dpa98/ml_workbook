import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_and_split_iris
from preprocessing import standardize_data
from model import train_logistic_regression, evaluate_model
from visualization import plot_decision_regions

def parse_args():
    """Configura la interfaz de línea de comandos (CLI) para probar hiperparámetros rápidamente."""
    parser = argparse.ArgumentParser(description="Flujo de trabajo modular para Regresión Logística")
    parser.add_argument('--c_param', type=float, default=100.0, 
                        help='Parámetro de regularización C (ej. 0.1, 1.0, 100.0). Por defecto es 100.0.')
    parser.add_argument('--test_size', type=float, default=0.3, 
                        help='Proporción del conjunto de prueba. Por defecto es 0.3.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"--- Ejecutando Regresión Logística (C={args.c_param}, test_size={args.test_size}) ---")
    
    X_train, X_test, y_train, y_test = load_and_split_iris(test_size=args.test_size)
    X_train_std, X_test_std, scaler = standardize_data(X_train, X_test)
    
    lr = train_logistic_regression(X_train_std, y_train, C=args.c_param)
    
    misclassified, accuracy = evaluate_model(lr, X_test_std, y_test)
    print(f'\nResultados:')
    print(f' -> Ejemplos mal clasificados: {misclassified}')
    print(f' -> Precisión (Accuracy): {accuracy:.3f}')
    
    # Demostración de predicción probabilística (La gran ventaja sobre el Perceptrón)
    print(f'\nEjemplo de predicción probabilística (primeras 3 muestras de prueba):')
    probabilidades = lr.predict_proba(X_test_std[:3, :])
    for i, prob in enumerate(probabilidades):
        print(f' Muestra {i+1}: P(Clase 0)={prob[0]:.3f}, P(Clase 1)={prob[1]:.3f}, P(Clase 2)={prob[2]:.3f}')
    
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
                          classifier=lr,
                          test_idx=range(test_idx_start, len(y_combined)))
    
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    plt.savefig('logistic_regions_output.png', facecolor=fig.get_facecolor())
    print("¡Gráfico guardado como 'logistic_regions_output.png'!")

if __name__ == '__main__':
    main()
