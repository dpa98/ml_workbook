# Proyecto de Árbol de Decisión Modular

Este repositorio implementa un clasificador de árbol de decisión sobre el conjunto de datos Iris,
replicando la estructura de los proyectos knn, svm y perceptrón pero cambiando la lógica del modelo.
Al igual que los demás flujos de tensorhub, está dividido en scripts limpios e independientes
orquestados a través de una interfaz de línea de comandos.

## Estructura
- `data_loader.py`: Gestión de datos.
- `preprocessing.py`: Estandarización de características.
- `model.py`: Implementación de `DecisionTreeClassifier` y métricas.
- `visualization.py`: Función de trazado.
- `main.py`: Script principal con CLI.

## Uso mediante CLI
El script principal utiliza `argparse`, lo que te permite ajustar hiperparámetros directamente desde la
terminal sin necesidad de abrir y editar el código.

Ejecución estándar (árbol sin límite de profundidad):
`python main.py`

Limitando la profundidad del árbol y cambiando el criterio de división:
`python main.py --max_depth 3 --criterion entropy`
`python main.py --max_depth 2 --test_size 0.4`