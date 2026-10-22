# Proyecto de K-vecinos más cercanos (KNN) Modular

Este repositorio implementa un clasificador no paramétrico (K-vecinos más cercanos) sobre el conjunto de datos Iris,
replicando la estructura del proyecto perceptrón pero cambiando la lógica del modelo. Al igual que los demás flujos
de tensorhub, está dividido en scripts limpios e independientes orquestados a través de una interfaz de línea de comandos.

## Estructura
- `data_loader.py`: Gestión de datos.
- `preprocessing.py`: Estandarización de características.
- `model.py`: Implementación de `KNeighborsClassifier` y métricas.
- `visualization.py`: Función de trazado.
- `main.py`: Script principal con CLI.

## Uso mediante CLI
El script principal utiliza `argparse`, lo que te permite ajustar hiperparámetros directamente desde la terminal
sin necesidad de abrir y editar el código.

Ejecución estándar:
`python main.py`

Modificando el número de vecinos k y el tamaño del set de prueba:
`python main.py --k 3 --test_size 0.4`
`python main.py --k 10`