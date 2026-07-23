import numpy as np
import matplotlib.pyplot as plt

# Importar nuestros módulos
from data_loader import load_and_split_iris
from preprocessing import standardize_data
from model import train_perceptron, evaluate_model
from visualization import plot_decision_regions

def main():
    print("1. Cargando y dividiendo los datos...")
    X_train, X_test, y_train, y_test = load_and_split_iris()
    
    print("2. Estandarizando las características...")
    X_train_std, X_test_std, scaler = standardize_data(X_train, X_test)
    
    print("3. Entrenando el modelo Perceptrón...")
    ppn = train_perceptron(X_train_std, y_train)
    
    print("4. Evaluando el modelo...")
    misclassified, accuracy = evaluate_model(ppn, X_test_std, y_test)
    print(f'   Ejemplos mal clasificados: {misclassified}')
    print(f'   Precisión (Accuracy): {accuracy:.3f}')
    
    print("5. Generando visualización de las regiones de decisión...")
    # Combinar datos de entrenamiento y prueba para la visualización final
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    # Rango 105 a 150 corresponde a los ejemplos de prueba (ya que son 45 de los 150)
    plot_decision_regions(X=X_combined_std, 
                          y=y_combined,
                          classifier=ppn,
                          test_idx=range(105, 150))
    
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # Guardar la imagen en disco
    plt.savefig('decision_regions_output.png')
    print("¡Gráfico guardado como 'decision_regions_output.png'!")
    # plt.show() # Descomentar para ver la ventana interactiva localmente

if __name__ == '__main__':
    main()
