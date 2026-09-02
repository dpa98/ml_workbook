# Proyecto de Bosque Aleatorio (Random Forest) Modular

Este repositorio implementa un clasificador de bosque aleatorio (ensemble de árboles de decisión)
sobre el conjunto de datos Iris, replicando la estructura de los proyectos decision_tree, knn y svm.
Al igual que los demás flujos de tensorhub, está dividido en scripts limpios e independientes
orquestados a través de una interfaz de línea de comandos.

## Estructura
- `data_loader.py`: Gestión de datos.
- `preprocessing.py`: Estandarización de características.
- `model.py`: Implementación de `RandomForestClassifier` y métricas.
- `visualization.py`: Función de trazado.
- `main.py`: Script principal con CLI.

## Uso mediante CLI
El script principal utiliza `argparse`, lo que te permite ajustar hiperparámetros directamente desde la
terminal sin necesidad de abrir y editar el código.

Ejecución estándar (25 árboles):
`python main.py`

Modificando la cantidad de árboles y la profundidad:
`python main.py --n_estimators 500`
`python main.py --n_estimators 10 --max_depth 3 --test_size 0.4`